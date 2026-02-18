"""batch_generate, cancel_batch, and progress monitoring tool implementations."""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..client import VLLMClient
from ..models import ModelRegistry
from .generate import generate_image

logger = logging.getLogger(__name__)


@dataclass
class BatchState:
    """Tracks the state of an active batch generation."""

    batch_id: str
    total: int = 0
    prompts: list[str] = field(default_factory=list)
    tasks: list[asyncio.Task] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    completed_count: int = 0
    failed_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cancelled: bool = False
    finished: bool = False
    final_result: dict[str, Any] | None = None
    total_time: float | None = None


# Module-level dict tracking active batches
active_batches: dict[str, BatchState] = {}


def _compute_max_concurrent(
    registry: ModelRegistry,
    model_name: str,
    user_override: int | None,
) -> int:
    """Determine max concurrent generations based on GPU/model estimates."""
    if user_override is not None:
        return max(1, user_override)

    profile = registry.get_profile(model_name)
    estimated_vram = profile["estimated_vram_gb"]
    # Assume a typical 24GB card unless we know better
    assumed_total_vram = 32.0
    available = assumed_total_vram - estimated_vram
    per_image_overhead = 2.0
    concurrent = max(1, math.floor(available / per_image_overhead))
    return min(concurrent, 4)


async def batch_generate(
    client: VLLMClient,
    registry: ModelRegistry,
    output_dir: str,
    prompts: list[str],
    width: int | None = None,
    height: int | None = None,
    aspect_ratio: str | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seeds: list[int] | None = None,
    max_concurrent: int | None = None,
    format: str = "png",
) -> dict[str, Any]:
    """Start a batch image generation in the background.

    Returns immediately with a batch_id. Use check_batch_progress to poll
    for results every ~50 seconds.
    """
    if not prompts:
        return {"status": "error", "error": "No prompts provided."}
    if len(prompts) > 20:
        return {
            "status": "error",
            "error": f"Too many prompts ({len(prompts)}). Maximum is 20.",
        }
    if seeds is not None and len(seeds) != len(prompts):
        return {
            "status": "error",
            "error": (
                f"Seeds list length ({len(seeds)}) must match "
                f"prompts list length ({len(prompts)})."
            ),
        }

    # Determine model
    model_name = registry.cached_model_name
    if not model_name:
        model_name = await client.get_loaded_model_name()
        if not model_name:
            return {
                "status": "error",
                "error": (
                    f"Cannot determine loaded model from vLLM server at "
                    f"{client.base_url}. Is the server running?"
                ),
            }
        registry.cached_model_name = model_name

    concurrency = _compute_max_concurrent(registry, model_name, max_concurrent)

    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    batch_state = BatchState(
        batch_id=batch_id,
        total=len(prompts),
        prompts=list(prompts),
    )
    active_batches[batch_id] = batch_state

    # Fire off the background runner — does NOT block the MCP response
    asyncio.create_task(
        _run_batch_background(
            batch_state=batch_state,
            client=client,
            registry=registry,
            output_dir=output_dir,
            prompts=prompts,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seeds=seeds,
            concurrency=concurrency,
            format=format,
        )
    )

    return {
        "status": "started",
        "batch_id": batch_id,
        "total": len(prompts),
        "max_concurrent": concurrency,
        "message": (
            f"Batch '{batch_id}' started with {len(prompts)} images. "
            "WAIT 50 SECONDS, then call check_batch_progress(batch_id) to check status. "
            "Do NOT call it immediately — images need time to generate."
        ),
    }


async def _run_batch_background(
    batch_state: BatchState,
    client: VLLMClient,
    registry: ModelRegistry,
    output_dir: str,
    prompts: list[str],
    width: int | None,
    height: int | None,
    aspect_ratio: str | None,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    seeds: list[int] | None,
    concurrency: int,
    format: str,
) -> None:
    """Run the actual batch generation in the background."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _generate_one(index: int, prompt: str, seed: int | None) -> dict[str, Any]:
        async with semaphore:
            if batch_state.cancelled:
                return {"index": index, "status": "cancelled", "prompt": prompt}
            result = await generate_image(
                client=client,
                registry=registry,
                output_dir=output_dir,
                prompt=prompt,
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                format=format,
            )
            result["index"] = index

            if result.get("status") == "success":
                batch_state.completed_count += 1
            else:
                batch_state.failed_count += 1

            return result

    start = time.monotonic()

    tasks = [
        asyncio.create_task(
            _generate_one(i, p, seeds[i] if seeds else None)
        )
        for i, p in enumerate(prompts)
    ]
    batch_state.tasks = tasks

    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.monotonic() - start

    # Process results
    processed: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            processed.append({
                "index": i,
                "status": "error",
                "error": str(r),
                "prompt": prompts[i],
            })
            failed += 1
        elif isinstance(r, dict) and r.get("status") == "success":
            processed.append({
                "index": r.get("index", i),
                "prompt": prompts[i],
                "file_path": r.get("file_path"),
                "generation_time_seconds": r.get("generation_time_seconds"),
                "seed": r.get("parameters_used", {}).get("seed"),
                "status": "success",
            })
            succeeded += 1
        elif isinstance(r, dict) and r.get("status") == "cancelled":
            processed.append(r)
        else:
            processed.append({
                "index": i,
                "status": "error",
                "error": r.get("error", "Unknown error") if isinstance(r, dict) else str(r),
                "prompt": prompts[i],
            })
            failed += 1

    batch_state.results = processed
    batch_state.total_time = round(total_time, 2)
    batch_state.final_result = {
        "status": "completed",
        "batch_id": batch_state.batch_id,
        "total_time_seconds": round(total_time, 2),
        "images_generated": succeeded,
        "images_failed": failed,
        "max_concurrent": concurrency,
        "results": processed,
    }
    batch_state.finished = True
    logger.info("Batch %s completed: %d succeeded, %d failed in %.1fs",
                batch_state.batch_id, succeeded, failed, total_time)


async def cancel_batch(batch_id: str) -> dict[str, Any]:
    """Cancel a running batch generation."""
    if batch_id not in active_batches:
        return {
            "status": "error",
            "error": (
                f"Batch '{batch_id}' not found. "
                "It may have already completed or never existed."
            ),
        }

    batch_state = active_batches[batch_id]
    batch_state.cancelled = True

    completed = 0
    cancelled = 0
    files_saved: list[str] = []

    for task in batch_state.tasks:
        if task.done():
            result = task.result()
            if isinstance(result, dict) and result.get("status") == "success":
                completed += 1
                if result.get("file_path"):
                    files_saved.append(result["file_path"])
        else:
            task.cancel()
            cancelled += 1

    active_batches.pop(batch_id, None)

    return {
        "batch_id": batch_id,
        "status": "cancelled",
        "completed": completed,
        "cancelled": cancelled,
        "files_saved": files_saved,
    }


async def check_batch_progress(batch_id: str) -> dict[str, Any]:
    """Check the progress of a running batch generation.

    Returns full results once the batch is finished, then removes it from
    the active list.
    """
    if batch_id not in active_batches:
        return {
            "status": "not_found",
            "batch_id": batch_id,
            "message": (
                f"Batch '{batch_id}' not found. "
                "It may have already completed, been cancelled, or never existed."
            ),
        }

    batch_state = active_batches[batch_id]

    # If the batch is finished, return the full results and clean up
    if batch_state.finished and batch_state.final_result is not None:
        active_batches.pop(batch_id, None)
        return batch_state.final_result

    elapsed = (datetime.now(timezone.utc) - batch_state.created_at).total_seconds()

    completed = batch_state.completed_count
    failed = batch_state.failed_count
    cancelled_count = sum(1 for t in batch_state.tasks if t.cancelled())
    in_progress = batch_state.total - completed - failed - cancelled_count

    # Build individual image statuses
    image_statuses: list[dict[str, Any]] = []
    for i, task in enumerate(batch_state.tasks):
        prompt_text = batch_state.prompts[i] if i < len(batch_state.prompts) else "unknown"
        if task.done():
            if task.cancelled():
                image_statuses.append({
                    "index": i,
                    "status": "cancelled",
                    "prompt": prompt_text[:80],
                })
            else:
                try:
                    result = task.result()
                    image_statuses.append({
                        "index": i,
                        "status": result.get("status", "unknown"),
                        "prompt": prompt_text[:80],
                        "file_path": result.get("file_path"),
                    })
                except Exception as e:
                    image_statuses.append({
                        "index": i,
                        "status": "error",
                        "prompt": prompt_text[:80],
                        "error": str(e),
                    })
        else:
            image_statuses.append({
                "index": i,
                "status": "in_progress",
                "prompt": prompt_text[:80],
            })

    return {
        "status": "in_progress",
        "batch_id": batch_id,
        "total": batch_state.total,
        "completed": completed,
        "failed": failed,
        "in_progress": in_progress,
        "cancelled": cancelled_count,
        "elapsed_seconds": round(elapsed, 1),
        "is_cancelled": batch_state.cancelled,
        "images": image_statuses,
    }


async def list_active_batches() -> dict[str, Any]:
    """List all currently active batch generations."""
    if not active_batches:
        return {
            "status": "success",
            "active_batches": [],
            "message": "No active batch generations.",
        }

    batches = []
    for batch_id, state in active_batches.items():
        elapsed = (datetime.now(timezone.utc) - state.created_at).total_seconds()
        batches.append({
            "batch_id": batch_id,
            "total": state.total,
            "completed": state.completed_count,
            "failed": state.failed_count,
            "elapsed_seconds": round(elapsed, 1),
            "is_cancelled": state.cancelled,
        })

    return {
        "status": "success",
        "active_batches": batches,
    }
