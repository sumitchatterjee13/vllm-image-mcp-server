"""gpu_status and server_health tool implementations."""

from __future__ import annotations

import math
from typing import Any

from ..client import VLLMClient
from ..models import ModelRegistry


async def gpu_status(
    client: VLLMClient,
    registry: ModelRegistry,
) -> dict[str, Any]:
    """Check GPU memory usage and availability for generation capacity planning.

    Since vLLM-Omni may not expose detailed GPU metrics, this returns estimates
    based on the model registry profile.
    """
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
    estimated_model_vram = profile["estimated_vram_gb"]

    # We can't directly query GPU stats from vLLM in most cases,
    # so we provide estimates based on the model profile.
    # Assume a reasonable GPU size for capacity planning.
    assumed_total_vram = 32.0  # Conservative assumption

    estimated_free = assumed_total_vram - estimated_model_vram
    per_image_overhead = 2.0
    recommended_concurrent = max(1, min(4, math.floor(estimated_free / per_image_overhead)))

    # Check if server is actually healthy
    is_healthy, _ = await client.health_check()

    return {
        "gpu_name": "Unknown (estimated)",
        "total_vram_gb": assumed_total_vram,
        "estimated_model_vram_gb": estimated_model_vram,
        "estimated_free_vram_gb": round(estimated_free, 1),
        "recommended_max_concurrent": recommended_concurrent,
        "server_url": client.base_url,
        "server_status": "healthy" if is_healthy else "unreachable",
        "note": (
            "GPU stats are estimated from the model profile. "
            "Actual values may differ based on your hardware and quantization."
        ),
    }


async def server_health(
    client: VLLMClient,
) -> dict[str, Any]:
    """Check if the vLLM-Omni server is running, ready, and responsive."""
    is_healthy, latency_ms = await client.health_check()

    model_loaded = None
    if is_healthy:
        model_loaded = await client.get_loaded_model_name()

    version = None
    if is_healthy:
        version = await client.get_server_version()

    status = "healthy" if is_healthy and model_loaded else "degraded" if is_healthy else "unreachable"

    result: dict[str, Any] = {
        "server_url": client.base_url,
        "status": status,
        "latency_ms": round(latency_ms, 1) if latency_ms else None,
    }
    if model_loaded:
        result["model_loaded"] = model_loaded
    if version:
        result["version"] = version
    if not is_healthy:
        result["error"] = (
            f"Cannot connect to vLLM server at {client.base_url}. "
            "Is the container running?"
        )

    return result
