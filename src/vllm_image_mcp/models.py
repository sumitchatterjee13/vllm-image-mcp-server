"""Model registry and parameter resolver for known vLLM image generation models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

FALLBACK_PROMPT_GUIDANCE = (
    "Write prompts as natural language prose, not comma-separated tags. "
    "Structure: subject first, then style, environment, lighting, and details. "
    "Be specific and descriptive (30-80 words ideal). "
    "Always specify lighting. Avoid vague words like 'beautiful' or 'amazing'. "
    "Do not use quality tags like '8k, masterpiece, best quality' — modern models "
    "default to high quality. Do not describe sequential actions — pick one moment."
)

FALLBACK_DEFAULTS = {
    "type": "standard",
    "default_steps": 28,
    "default_guidance_scale": 5.0,
    "supports_negative_prompt": True,
    "default_resolution": (1024, 1024),
    "max_resolution_mp": 4.0,
    "resolution_multiple": 16,
    "estimated_vram_gb": 20,
    "supported_aspect_ratios": {
        "1:1": (1024, 1024),
    },
    "prompt_guidance": FALLBACK_PROMPT_GUIDANCE,
}

_FLUX_KLEIN_PROMPT_GUIDANCE = (
    "FLUX.2 Klein is a distilled DiT model. Write prompts as NATURAL LANGUAGE PROSE, "
    "never comma-separated tags. Structure: subject first (most important — first 10-20 "
    "words carry the most weight), then environment, style, and lighting.\n\n"
    "RESOLUTION OPTIONS:\n"
    "- Standard (default): 1024x1024, 1024x576, etc. — fast, good for drafts and general use.\n"
    "- 2K: Use aspect ratios like '1:1_2k', '16:9_2k', '4:3_2k' for higher detail (e.g., 2560x1440).\n"
    "- 4K: Use '16:9_4k' or '9:16_4k' for maximum detail (3840x2160). "
    "4K takes significantly longer and uses more VRAM. Use only when the user explicitly needs "
    "high-resolution output (desktop wallpapers, print-quality images, large posters).\n\n"
    "CRITICAL RULES:\n"
    "- 30-80 words is the ideal length. Every word must contribute visual information.\n"
    "- ALWAYS specify lighting — it has the single greatest impact on quality. "
    "Example: 'soft diffused natural light from a large window camera-left' not just 'good lighting'.\n"
    "- DO NOT use quality tags like '8k, masterpiece, best quality, ultra HD' — they waste tokens. "
    "The model defaults to high quality.\n"
    "- DO NOT use comma-separated tag lists. Write flowing descriptive sentences.\n"
    "- DO NOT send negative prompts — they are ignored at CFG 1.0.\n"
    "- DO NOT describe sequential actions — images are a single moment.\n"
    "- DO NOT mix conflicting styles (e.g. 'photorealistic oil painting').\n"
    "- Use camera/lens references for photorealism: 'Shot on Sony A7IV, 85mm f/2.0'.\n"
    "- Use emphasis phrases instead of weight syntax: 'prominently featuring', "
    "'with particular attention to', 'the focal point is'.\n"
    "- For text in images, use quotation marks: 'The sign reads \"OPEN\"'.\n\n"
    "GOOD PROMPT EXAMPLE:\n"
    "\"Professional studio product shot on polished concrete. A minimalist ceramic coffee mug "
    "with matte black finish, steam rising from hot coffee, centered in frame. Ultra-realistic "
    "commercial photography. Three-point softbox lighting, diffused highlights, no harsh shadows. "
    "Shot with 85mm lens at f/5.6.\"\n\n"
    "BAD PROMPT EXAMPLE:\n"
    "\"coffee mug, black, steam, studio, professional, 8k, masterpiece, best quality, "
    "ultra detailed, sharp focus\""
)

_QWEN_IMAGE_PROMPT_GUIDANCE = (
    "Qwen-Image is a standard DiT model with excellent text rendering and strong prompt adherence. "
    "Write prompts as concise, structured natural language (1-3 sentences ideal). "
    "The model interprets prompts very literally — be precise about what you want.\n\n"
    "CRITICAL RULES:\n"
    "- Structure: [Subject] + [Visual Style] + [Environment] + [Lighting] + [Extra Details].\n"
    "- ALWAYS specify lighting — it is the highest-impact element.\n"
    "- USE negative_prompt — it improves results. Good defaults: "
    "'blurry, low quality, distorted, watermark, oversaturated, artificial, plastic-looking'.\n"
    "- For portraits add: 'extra fingers, deformed hands, unnatural proportions' to negative.\n"
    "- For text in images: wrap text in double quotes and specify font style. "
    "Example: '\"Aurora Festival 2026\" in bold sans-serif lettering'. "
    "Raise guidance_scale to 7.0 for text-heavy images.\n"
    "- For photorealism: use 'photograph' instead of 'photorealistic' or '3d render'.\n"
    "- DO NOT use vague words like 'beautiful', 'amazing', 'nice' — they add noise.\n"
    "- DO NOT contradict yourself (e.g. 'bright sunny day with moody dramatic shadows').\n"
    "- DO NOT use quality tags like '8k, masterpiece' — append ', Ultra HD, 4K, "
    "cinematic composition' at the end instead if extra quality is desired.\n\n"
    "GOOD PROMPT EXAMPLE:\n"
    "\"A futuristic sports car parked under neon city lights, photorealistic style. "
    "Reflections shimmer on wet asphalt streets. Dramatic side lighting with deep shadows "
    "and vibrant highlights. \\\"Night Racer\\\" in metallic chrome text on the hood.\"\n"
    "negative_prompt: \"blurry, low quality, distorted, watermark, oversaturated\"\n\n"
    "BAD PROMPT EXAMPLE:\n"
    "\"beautiful car, amazing, neon, city, 8k, masterpiece, best quality\""
)

MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "black-forest-labs/FLUX.2-klein-4B": {
        "type": "distilled",
        "default_steps": 4,
        "default_guidance_scale": 1.0,
        "supports_negative_prompt": False,
        "default_resolution": (1024, 1024),
        "max_resolution_mp": 9.0,
        "resolution_multiple": 16,
        "estimated_vram_gb": 13,
        "supported_aspect_ratios": {
            "1:1": (1024, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
            "1:1_2k": (2048, 2048),
            "16:9_2k": (2560, 1440),
            "9:16_2k": (1440, 2560),
            "4:3_2k": (2048, 1536),
            "3:4_2k": (1536, 2048),
            "16:9_4k": (3840, 2160),
            "9:16_4k": (2160, 3840),
        },
        "prompt_guidance": _FLUX_KLEIN_PROMPT_GUIDANCE,
    },
    "black-forest-labs/FLUX.2-klein-9B": {
        "type": "distilled",
        "default_steps": 4,
        "default_guidance_scale": 1.0,
        "supports_negative_prompt": False,
        "default_resolution": (1024, 1024),
        "max_resolution_mp": 9.0,
        "resolution_multiple": 16,
        "estimated_vram_gb": 29,
        "supported_aspect_ratios": {
            "1:1": (1024, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
            "1:1_2k": (2048, 2048),
            "16:9_2k": (2560, 1440),
            "9:16_2k": (1440, 2560),
            "4:3_2k": (2048, 1536),
            "3:4_2k": (1536, 2048),
            "16:9_4k": (3840, 2160),
            "9:16_4k": (2160, 3840),
        },
        "prompt_guidance": _FLUX_KLEIN_PROMPT_GUIDANCE,
    },
    "Qwen/Qwen-Image-2512": {
        "type": "standard",
        "default_steps": 28,
        "default_guidance_scale": 5.0,
        "supports_negative_prompt": True,
        "default_resolution": (1024, 1024),
        "max_resolution_mp": 4.0,
        "resolution_multiple": 16,
        "estimated_vram_gb": 40,
        "supported_aspect_ratios": {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        },
        "prompt_guidance": _QWEN_IMAGE_PROMPT_GUIDANCE,
    },
    "Qwen/Qwen-Image": {
        "type": "standard",
        "default_steps": 50,
        "default_guidance_scale": 5.0,
        "supports_negative_prompt": True,
        "default_resolution": (1024, 1024),
        "max_resolution_mp": 4.0,
        "resolution_multiple": 16,
        "estimated_vram_gb": 40,
        "supported_aspect_ratios": {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
        },
        "prompt_guidance": _QWEN_IMAGE_PROMPT_GUIDANCE,
    },
    "Tongyi-MAI/Z-Image-Turbo": {
        "type": "distilled",
        "default_steps": 8,
        "default_guidance_scale": 1.0,
        "supports_negative_prompt": False,
        "default_resolution": (1024, 1024),
        "max_resolution_mp": 4.0,
        "resolution_multiple": 16,
        "estimated_vram_gb": 16,
        "supported_aspect_ratios": {
            "1:1": (1024, 1024),
        },
        "prompt_guidance": _FLUX_KLEIN_PROMPT_GUIDANCE,
    },
}


@dataclass
class ResolvedParameters:
    """Fully resolved parameters for an image generation request."""

    model: str
    prompt: str
    negative_prompt: str | None
    width: int
    height: int
    num_inference_steps: int
    guidance_scale: float
    seed: int | None
    notes: list[str] = field(default_factory=list)


class ModelRegistry:
    """Registry of known models and parameter resolution logic."""

    def __init__(self) -> None:
        self._profiles = dict(MODEL_PROFILES)
        self._cached_model_name: str | None = None

    @property
    def cached_model_name(self) -> str | None:
        return self._cached_model_name

    @cached_model_name.setter
    def cached_model_name(self, value: str) -> None:
        self._cached_model_name = value

    def get_profile(self, model_name: str) -> dict[str, Any]:
        """Get the profile for a model, falling back to defaults if unknown."""
        if model_name in self._profiles:
            return self._profiles[model_name]
        logger.warning(
            "Model '%s' is not in the known profiles. Using fallback defaults.",
            model_name,
        )
        return dict(FALLBACK_DEFAULTS)

    def is_known(self, model_name: str) -> bool:
        return model_name in self._profiles

    def resolve_parameters(
        self,
        model_name: str,
        prompt: str,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        aspect_ratio: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
    ) -> ResolvedParameters:
        """Resolve generation parameters using priority: user > model profile > fallback."""
        profile = self.get_profile(model_name)
        notes: list[str] = []

        if not self.is_known(model_name):
            notes.append(
                f"Model '{model_name}' is not in the known profiles. "
                "Using fallback defaults (steps=28, guidance=5.0). "
                "Generation may produce unexpected results."
            )

        # Resolve negative prompt
        resolved_negative = negative_prompt
        if negative_prompt and not profile["supports_negative_prompt"]:
            resolved_negative = None
            notes.append(
                f"Model '{model_name}' does not support negative prompts. "
                "The negative_prompt parameter was ignored."
            )

        # Resolve resolution
        resolved_width, resolved_height = self._resolve_resolution(
            profile, width, height, aspect_ratio, notes
        )

        # Resolve steps and guidance
        resolved_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else profile["default_steps"]
        )
        resolved_guidance = (
            guidance_scale
            if guidance_scale is not None
            else profile["default_guidance_scale"]
        )

        return ResolvedParameters(
            model=model_name,
            prompt=prompt,
            negative_prompt=resolved_negative,
            width=resolved_width,
            height=resolved_height,
            num_inference_steps=resolved_steps,
            guidance_scale=resolved_guidance,
            seed=seed,
            notes=notes,
        )

    def _resolve_resolution(
        self,
        profile: dict[str, Any],
        width: int | None,
        height: int | None,
        aspect_ratio: str | None,
        notes: list[str],
    ) -> tuple[int, int]:
        """Resolve width/height from aspect_ratio or explicit values."""
        multiple = profile["resolution_multiple"]
        max_mp = profile["max_resolution_mp"]

        if aspect_ratio:
            ratios = profile["supported_aspect_ratios"]
            if aspect_ratio in ratios:
                return ratios[aspect_ratio]
            notes.append(
                f"Aspect ratio '{aspect_ratio}' is not a preset for this model. "
                f"Available: {list(ratios.keys())}. Using default resolution."
            )
            return profile["default_resolution"]

        if width is not None or height is not None:
            w = width if width is not None else profile["default_resolution"][0]
            h = height if height is not None else profile["default_resolution"][1]
            w, h = self._snap_to_multiple(w, h, multiple, notes)
            w, h = self._clamp_to_max_mp(w, h, max_mp, multiple, notes)
            return w, h

        return profile["default_resolution"]

    def _snap_to_multiple(
        self,
        width: int,
        height: int,
        multiple: int,
        notes: list[str],
    ) -> tuple[int, int]:
        """Snap width/height to the nearest valid multiple."""
        snapped_w = round(width / multiple) * multiple
        snapped_h = round(height / multiple) * multiple
        snapped_w = max(multiple, snapped_w)
        snapped_h = max(multiple, snapped_h)
        if snapped_w != width or snapped_h != height:
            notes.append(
                f"Resolution {width}x{height} adjusted to {snapped_w}x{snapped_h} "
                f"(must be multiples of {multiple})."
            )
        return snapped_w, snapped_h

    def _clamp_to_max_mp(
        self,
        width: int,
        height: int,
        max_mp: float,
        multiple: int,
        notes: list[str],
    ) -> tuple[int, int]:
        """Clamp resolution to stay within max megapixel budget."""
        current_mp = (width * height) / 1_000_000
        if current_mp <= max_mp:
            return width, height

        scale = (max_mp / current_mp) ** 0.5
        new_w = round((width * scale) / multiple) * multiple
        new_h = round((height * scale) / multiple) * multiple
        new_w = max(multiple, new_w)
        new_h = max(multiple, new_h)
        notes.append(
            f"Resolution {width}x{height} ({current_mp:.1f}MP) exceeds max "
            f"{max_mp}MP. Clamped to {new_w}x{new_h}."
        )
        return new_w, new_h
