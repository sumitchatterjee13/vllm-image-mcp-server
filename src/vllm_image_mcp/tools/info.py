"""get_model_info, list_presets, and estimate_generation tool implementations."""

from __future__ import annotations

import math
from typing import Any

from ..client import VLLMClient
from ..models import ModelRegistry


async def get_model_info(
    client: VLLMClient,
    registry: ModelRegistry,
) -> dict[str, Any]:
    """Get information about the currently loaded model and its recommended parameters."""
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

    profile = registry.get_profile(model_name)
    default_res = profile["default_resolution"]

    aspect_ratios = {
        k: f"{v[0]}x{v[1]}" for k, v in profile["supported_aspect_ratios"].items()
    }

    result: dict[str, Any] = {
        "model_name": model_name,
        "model_type": profile["type"],
        "recommended_parameters": {
            "num_inference_steps": profile["default_steps"],
            "guidance_scale": profile["default_guidance_scale"],
            "default_resolution": f"{default_res[0]}x{default_res[1]}",
        },
        "supports_negative_prompt": profile["supports_negative_prompt"],
        "supported_aspect_ratios": aspect_ratios,
        "estimated_vram_gb": profile["estimated_vram_gb"],
        "in_registry": registry.is_known(model_name),
    }
    if "prompt_guidance" in profile:
        result["prompt_guidance"] = profile["prompt_guidance"]

    return result


async def list_presets(
    client: VLLMClient,
    registry: ModelRegistry,
) -> dict[str, Any]:
    """List available resolution presets and aspect ratios for the current model."""
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

    profile = registry.get_profile(model_name)
    presets = {
        k: f"{v[0]}x{v[1]}" for k, v in profile["supported_aspect_ratios"].items()
    }

    return {
        "model": model_name,
        "resolution_multiple": profile["resolution_multiple"],
        "max_megapixels": profile["max_resolution_mp"],
        "presets": presets,
    }


async def estimate_generation(
    client: VLLMClient,
    registry: ModelRegistry,
    num_images: int = 1,
    width: int | None = None,
    height: int | None = None,
    num_inference_steps: int | None = None,
) -> dict[str, Any]:
    """Estimate generation time and resource usage without generating."""
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

    profile = registry.get_profile(model_name)
    default_res = profile["default_resolution"]
    w = width if width is not None else default_res[0]
    h = height if height is not None else default_res[1]
    steps = num_inference_steps if num_inference_steps is not None else profile["default_steps"]

    mp_per_image = (w * h) / 1_000_000

    # Estimate time based on model type and steps
    if profile["type"] == "distilled":
        # Distilled models are very fast: ~0.3-0.5s per step at 1MP
        time_per_image = steps * 0.4 * (mp_per_image / 1.0)
    else:
        # Standard models: ~0.5-1.0s per step at 1MP
        time_per_image = steps * 0.7 * (mp_per_image / 1.0)

    # Estimate concurrency
    estimated_vram = profile["estimated_vram_gb"]
    assumed_total = 32.0
    available = assumed_total - estimated_vram
    per_image_overhead = 2.0 * (mp_per_image / 1.0)  # Scale with resolution
    recommended_concurrent = max(1, min(4, math.floor(available / per_image_overhead)))

    # Total time with concurrency
    batches_needed = math.ceil(num_images / recommended_concurrent)
    estimated_total = batches_needed * time_per_image

    # Peak VRAM
    peak_vram = estimated_vram + (per_image_overhead * min(num_images, recommended_concurrent))

    notes_parts = []
    if profile["type"] == "distilled":
        notes_parts.append(f"Distilled model, {steps} steps.")
    else:
        notes_parts.append(f"Standard model, {steps} steps.")
    notes_parts.append(
        "First generation may be slower due to torch.compile warmup."
    )

    return {
        "estimated_time_per_image_seconds": round(time_per_image, 1),
        "estimated_total_time_seconds": round(estimated_total, 1),
        "estimated_peak_vram_gb": round(peak_vram, 1),
        "recommended_concurrent": recommended_concurrent,
        "resolution": f"{w}x{h}",
        "total_megapixels_per_image": round(mp_per_image, 2),
        "notes": " ".join(notes_parts),
    }
