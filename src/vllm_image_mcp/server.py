"""MCP server entry point with tool registrations for vLLM image generation."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from mcp.server.fastmcp import FastMCP

from .client import VLLMClient
from .models import ModelRegistry
from .tools import batch as batch_tools
from .tools import generate as generate_tools
from .tools import info as info_tools
from .tools import system as system_tools

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM Image MCP Server")
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:6655",
        help="Base URL of the vLLM-Omni server (default: http://localhost:6655)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent generations for batch (default: auto-detect)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    return parser.parse_args(argv)


def create_server(args: argparse.Namespace) -> FastMCP:
    """Create and configure the MCP server with all tools registered."""
    vllm_client = VLLMClient(base_url=args.vllm_url, timeout=args.timeout)
    registry = ModelRegistry()

    mcp = FastMCP(
        "vllm-image",
        instructions=(
            "AI-powered image generation server using vLLM-Omni. "
            "Supports single and batch image generation with automatic "
            "parameter tuning based on the loaded model.\n\n"
            "BEFORE YOUR FIRST GENERATION: Call get_model_info to learn the loaded model's "
            "name and its prompt_guidance. Different models require very different prompting "
            "strategies. The prompt_guidance field contains model-specific rules you MUST follow.\n\n"
            "IMPORTANT RULES:\n"
            "- You must always provide output_dir when generating images. "
            "Choose a path appropriate for the user's project.\n"
            "- Write prompts as natural language prose, NEVER as comma-separated tags.\n"
            "- Always specify lighting in your prompts — it has the biggest impact on quality.\n"
            "- Do NOT use quality tags like '8k, masterpiece, best quality' — they are counterproductive.\n"
            "- Subject description should come first in the prompt (first 10-20 words matter most).\n"
            "- 30-80 words is the ideal prompt length for most models.\n"
            "- Choose output format based on context: 'webp' for web projects (smallest files), "
            "'jpg' for general use (good compression), 'png' for lossless quality (default).\n"
            "- FLUX Klein models support up to 4K resolution (3840x2160). Use '16:9_4k' or "
            "'9:16_4k' aspect ratios for maximum detail when the user needs high-resolution output.\n"
            "- For batch generations, use check_batch_progress to poll every ~50 seconds "
            "to monitor progress and prevent timeouts."
        ),
    )

    # ── Tool 1: generate_image ──────────────────────────────────────────

    @mcp.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
        }
    )
    async def generate_image(
        prompt: str,
        output_dir: str,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        aspect_ratio: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        filename: str | None = None,
        format: str = "png",
    ) -> str:
        """Generate a single image from a text prompt.

        PROMPT WRITING RULES (call get_model_info first for model-specific guidance):
        - Write as natural language prose, NOT comma-separated tags.
        - Put the subject first (first 10-20 words carry the most weight).
        - Always describe lighting explicitly.
        - 30-80 words ideal. Do NOT use quality tags like '8k, masterpiece'.
        - For photorealism: reference real cameras/lenses ('Shot on Sony A7IV, 85mm f/2.0').
        - For text in images: use quotation marks ('The sign reads "OPEN"').

        Args:
            prompt: Natural language description of the image. Describe the subject first, then environment, style, and lighting. Be specific and descriptive.
            output_dir: Directory to save the image. Provide a path appropriate for the user's project (e.g. "./assets/images", "./public/img", "./output").
            negative_prompt: What to avoid (only works with Qwen-Image models, ignored by FLUX). For Qwen, good defaults: 'blurry, low quality, distorted, watermark, oversaturated'.
            width: Image width in pixels (must be multiple of 16).
            height: Image height in pixels (must be multiple of 16).
            aspect_ratio: Shortcut for resolution, e.g. "16:9", "1:1", "9:16", "16:9_2k", "16:9_4k" (overrides width/height). FLUX Klein supports up to 4K.
            num_inference_steps: Number of diffusion steps (fewer = faster, more = higher quality).
            guidance_scale: CFG scale. For Qwen-Image: raise to 7.0 for text-heavy images.
            seed: Random seed for reproducibility.
            filename: Output filename (auto-generated if omitted).
            format: Output image format. Options: "png" (lossless, default), "jpg" (lossy, quality=95, good for general use), "webp" (lossy, quality=90, smallest files, ideal for web projects). Choose based on context.
        """
        result = await generate_tools.generate_image(
            client=vllm_client,
            registry=registry,
            output_dir=output_dir,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            filename=filename,
            format=format,
        )
        return json.dumps(result, indent=2)

    # ── Tool 2: batch_generate ──────────────────────────────────────────

    @mcp.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
        }
    )
    async def batch_generate(
        prompts: list[str],
        output_dir: str,
        width: int | None = None,
        height: int | None = None,
        aspect_ratio: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seeds: list[int] | None = None,
        max_concurrent: int | None = None,
        format: str = "png",
    ) -> str:
        """Generate multiple images from multiple prompts concurrently.

        Each prompt should follow the same rules as generate_image: natural language prose,
        subject first, explicit lighting, 30-80 words each, no quality tags.

        For long-running batches, use check_batch_progress to poll progress every ~50 seconds
        to prevent timeouts. The batch_id is returned in the response.

        Args:
            prompts: List of natural language text prompts (max 20). Each prompt should be descriptive prose, not comma-separated tags.
            output_dir: Directory to save all images. Provide a path appropriate for the user's project.
            width: Width for all images in pixels.
            height: Height for all images in pixels.
            aspect_ratio: Aspect ratio shortcut for all images. FLUX Klein supports up to 4K (e.g. "16:9_4k").
            num_inference_steps: Steps for all images.
            guidance_scale: CFG scale for all images.
            seeds: List of seeds (must match prompts length if provided).
            max_concurrent: Override concurrent limit (auto-detected if omitted).
            format: Output image format for all images. Options: "png" (default), "jpg", "webp". Choose based on context.
        """
        concurrent = max_concurrent if max_concurrent is not None else args.max_concurrent
        result = await batch_tools.batch_generate(
            client=vllm_client,
            registry=registry,
            output_dir=output_dir,
            prompts=prompts,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seeds=seeds,
            max_concurrent=concurrent,
            format=format,
        )
        return json.dumps(result, indent=2)

    # ── Tool 3: get_model_info ──────────────────────────────────────────

    @mcp.tool(
        annotations={"readOnlyHint": True}
    )
    async def get_model_info() -> str:
        """Get information about the currently loaded model, its recommended parameters, and prompt writing guidance.

        IMPORTANT: Call this before your first image generation to get model-specific
        prompt_guidance. Different models (FLUX vs Qwen) require very different prompting strategies.

        Returns the model name, type, default parameters, supported aspect ratios,
        VRAM estimate, and detailed prompt_guidance with rules and examples.
        """
        result = await info_tools.get_model_info(
            client=vllm_client,
            registry=registry,
        )
        return json.dumps(result, indent=2)

    # ── Tool 4: gpu_status ──────────────────────────────────────────────

    @mcp.tool(
        annotations={"readOnlyHint": True}
    )
    async def gpu_status() -> str:
        """Check GPU memory usage and availability for generation capacity planning.

        Returns estimated VRAM usage, free memory, and recommended max concurrent
        generations.
        """
        result = await system_tools.gpu_status(
            client=vllm_client,
            registry=registry,
        )
        return json.dumps(result, indent=2)

    # ── Tool 5: list_presets ────────────────────────────────────────────

    @mcp.tool(
        annotations={"readOnlyHint": True}
    )
    async def list_presets() -> str:
        """List available resolution presets and aspect ratios for the current model.

        Returns all supported aspect ratios with their pixel dimensions, resolution
        constraints, and max megapixels.
        """
        result = await info_tools.list_presets(
            client=vllm_client,
            registry=registry,
        )
        return json.dumps(result, indent=2)

    # ── Tool 6: server_health ───────────────────────────────────────────

    @mcp.tool(
        annotations={"readOnlyHint": True}
    )
    async def server_health() -> str:
        """Check if the vLLM-Omni server is running, ready, and responsive.

        Returns server status, loaded model, response latency, and version.
        """
        result = await system_tools.server_health(
            client=vllm_client,
        )
        return json.dumps(result, indent=2)

    # ── Tool 7: estimate_generation ─────────────────────────────────────

    @mcp.tool(
        annotations={"readOnlyHint": True}
    )
    async def estimate_generation(
        num_images: int = 1,
        width: int | None = None,
        height: int | None = None,
        num_inference_steps: int | None = None,
    ) -> str:
        """Estimate generation time and resource usage without actually generating.

        Args:
            num_images: Number of images to estimate for.
            width: Image width in pixels.
            height: Image height in pixels.
            num_inference_steps: Steps per image.
        """
        result = await info_tools.estimate_generation(
            client=vllm_client,
            registry=registry,
            num_images=num_images,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
        )
        return json.dumps(result, indent=2)

    # ── Tool 8: cancel_batch ────────────────────────────────────────────

    @mcp.tool(
        annotations={
            "readOnlyHint": False,
            "destructiveHint": True,
        }
    )
    async def cancel_batch(batch_id: str) -> str:
        """Cancel a running batch generation.

        Args:
            batch_id: The batch ID returned by batch_generate.
        """
        result = await batch_tools.cancel_batch(batch_id)
        return json.dumps(result, indent=2)

    # ── Tool 9: check_batch_progress ───────────────────────────────────

    @mcp.tool(
        annotations={"readOnlyHint": True}
    )
    async def check_batch_progress(batch_id: str) -> str:
        """Check the progress of a running batch generation.

        Use this to monitor long-running batch generations. Poll every ~50 seconds
        to track progress without waiting for the full batch to complete.

        Args:
            batch_id: The batch ID returned by batch_generate.
        """
        result = await batch_tools.check_batch_progress(batch_id)
        return json.dumps(result, indent=2)

    # ── Tool 10: list_active_batches ───────────────────────────────────

    @mcp.tool(
        annotations={"readOnlyHint": True}
    )
    async def list_active_batches() -> str:
        """List all currently running batch generations and their IDs.

        Use this to find batch IDs for progress monitoring with check_batch_progress.
        """
        result = await batch_tools.list_active_batches()
        return json.dumps(result, indent=2)

    return mcp


def main(argv: list[str] | None = None) -> None:
    """Entry point for the vLLM Image MCP server."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting vLLM Image MCP server")
    logger.info("  vLLM URL: %s", args.vllm_url)
    logger.info("  Timeout: %ds", args.timeout)

    server = create_server(args)
    server.run()


if __name__ == "__main__":
    main()
