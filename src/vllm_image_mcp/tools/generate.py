"""generate_image tool implementation."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

from ..client import VLLMClient
from ..models import ModelRegistry
from ..utils import save_base64_image

logger = logging.getLogger(__name__)


async def generate_image(
    client: VLLMClient,
    registry: ModelRegistry,
    output_dir: str,
    prompt: str,
    negative_prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    aspect_ratio: str | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    filename: str | None = None,
    format: str = "png",
) -> dict[str, Any]:
    """Generate a single image from a text prompt.

    Returns a result dict with status, file_path, timing, and parameters used.
    """
    # Get the loaded model name
    model_name = registry.cached_model_name
    if not model_name:
        model_name = await client.get_loaded_model_name()
        if not model_name:
            return {
                "status": "error",
                "error": (
                    f"Cannot determine loaded model from vLLM server at "
                    f"{client.base_url}. Is the server running? "
                    "Check with the server_health tool."
                ),
            }
        registry.cached_model_name = model_name

    # Resolve parameters
    params = registry.resolve_parameters(
        model_name=model_name,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    # Use a random seed if none provided
    actual_seed = params.seed if params.seed is not None else random.randint(0, 2**32 - 1)

    # Build the API payload
    payload: dict[str, Any] = {
        "model": params.model,
        "prompt": params.prompt,
        "size": f"{params.width}x{params.height}",
        "response_format": "b64_json",
        "n": 1,
        "extra_body": {
            "num_inference_steps": params.num_inference_steps,
            "guidance_scale": params.guidance_scale,
            "seed": actual_seed,
        },
    }
    if params.negative_prompt:
        payload["extra_body"]["negative_prompt"] = params.negative_prompt

    # Send request and time it
    try:
        start = time.monotonic()
        response = await client.generate_image(payload)
        generation_time = time.monotonic() - start
    except httpx.ConnectError:
        return {
            "status": "error",
            "error": (
                f"Cannot connect to vLLM server at {client.base_url}. "
                "Is the container running? Check with the server_health tool."
            ),
        }
    except httpx.HTTPStatusError as e:
        error_detail = ""
        try:
            error_detail = e.response.json().get("message", e.response.text)
        except Exception:
            error_detail = e.response.text
        if "out of memory" in error_detail.lower() or "oom" in error_detail.lower():
            return {
                "status": "error",
                "error": (
                    "Server returned out-of-memory error. Try reducing resolution "
                    "or use the gpu_status tool to check available memory."
                ),
            }
        return {
            "status": "error",
            "error": f"vLLM server error ({e.response.status_code}): {error_detail}",
        }

    # Extract and save the image
    try:
        image_data = response["data"][0]["b64_json"]
    except (KeyError, IndexError):
        return {
            "status": "error",
            "error": "Unexpected response format from vLLM server. No image data found.",
        }

    file_path = save_base64_image(
        b64_data=image_data,
        output_dir=output_dir,
        filename=filename,
        prompt=prompt,
        format=format,
    )

    result: dict[str, Any] = {
        "status": "success",
        "file_path": str(file_path),
        "generation_time_seconds": round(generation_time, 2),
        "parameters_used": {
            "model": params.model,
            "prompt": params.prompt,
            "width": params.width,
            "height": params.height,
            "steps": params.num_inference_steps,
            "guidance_scale": params.guidance_scale,
            "seed": actual_seed,
            "format": format,
        },
    }
    if params.notes:
        result["notes"] = params.notes

    return result
