"""Tests for model registry and parameter resolution logic."""

from vllm_image_mcp.models import ModelRegistry, FALLBACK_DEFAULTS, MODEL_PROFILES


class TestModelRegistry:
    def setup_method(self):
        self.registry = ModelRegistry()

    def test_known_model_lookup(self):
        profile = self.registry.get_profile("black-forest-labs/FLUX.2-klein-4B")
        assert profile["type"] == "distilled"
        assert profile["default_steps"] == 4
        assert profile["default_guidance_scale"] == 1.0

    def test_unknown_model_returns_fallback(self):
        profile = self.registry.get_profile("unknown/model")
        assert profile["default_steps"] == FALLBACK_DEFAULTS["default_steps"]
        assert profile["default_guidance_scale"] == FALLBACK_DEFAULTS["default_guidance_scale"]

    def test_is_known(self):
        assert self.registry.is_known("black-forest-labs/FLUX.2-klein-4B")
        assert not self.registry.is_known("unknown/model")

    def test_cached_model_name(self):
        assert self.registry.cached_model_name is None
        self.registry.cached_model_name = "test/model"
        assert self.registry.cached_model_name == "test/model"


class TestParameterResolution:
    def setup_method(self):
        self.registry = ModelRegistry()
        self.model = "black-forest-labs/FLUX.2-klein-4B"

    def test_defaults_from_profile(self):
        params = self.registry.resolve_parameters(
            model_name=self.model,
            prompt="A cat",
        )
        assert params.width == 1024
        assert params.height == 1024
        assert params.num_inference_steps == 4
        assert params.guidance_scale == 1.0
        assert params.negative_prompt is None
        assert params.notes == []

    def test_user_overrides_take_priority(self):
        params = self.registry.resolve_parameters(
            model_name=self.model,
            prompt="A cat",
            width=512,
            height=512,
            num_inference_steps=8,
            guidance_scale=2.0,
            seed=42,
        )
        assert params.width == 512
        assert params.height == 512
        assert params.num_inference_steps == 8
        assert params.guidance_scale == 2.0
        assert params.seed == 42

    def test_aspect_ratio_resolution(self):
        params = self.registry.resolve_parameters(
            model_name=self.model,
            prompt="A cat",
            aspect_ratio="16:9",
        )
        assert params.width == 1024
        assert params.height == 576

    def test_invalid_aspect_ratio_falls_back_to_default(self):
        params = self.registry.resolve_parameters(
            model_name=self.model,
            prompt="A cat",
            aspect_ratio="21:9",
        )
        assert params.width == 1024
        assert params.height == 1024
        assert any("21:9" in n for n in params.notes)

    def test_negative_prompt_ignored_for_distilled(self):
        params = self.registry.resolve_parameters(
            model_name=self.model,
            prompt="A cat",
            negative_prompt="ugly, blurry",
        )
        assert params.negative_prompt is None
        assert any("negative prompt" in n.lower() for n in params.notes)

    def test_negative_prompt_kept_for_standard(self):
        params = self.registry.resolve_parameters(
            model_name="Qwen/Qwen-Image-2512",
            prompt="A cat",
            negative_prompt="ugly, blurry",
        )
        assert params.negative_prompt == "ugly, blurry"
        assert params.notes == []

    def test_unknown_model_adds_note(self):
        params = self.registry.resolve_parameters(
            model_name="unknown/model",
            prompt="A cat",
        )
        assert any("not in the known profiles" in n for n in params.notes)

    def test_resolution_snapped_to_multiple(self):
        params = self.registry.resolve_parameters(
            model_name=self.model,
            prompt="A cat",
            width=1000,
            height=1000,
        )
        assert params.width % 16 == 0
        assert params.height % 16 == 0
        assert params.width == 992  # round(1000/16)*16 = round(62.5)*16 = 62*16 (banker's rounding)
        assert params.height == 992

    def test_resolution_clamped_to_max_mp(self):
        params = self.registry.resolve_parameters(
            model_name=self.model,
            prompt="A cat",
            width=8192,
            height=8192,
        )
        mp = (params.width * params.height) / 1_000_000
        assert mp <= 9.5  # Allow small overshoot from rounding to 16-pixel multiples
        assert any("exceeds max" in n for n in params.notes)

    def test_aspect_ratio_overrides_width_height(self):
        """aspect_ratio should override width/height if both provided."""
        params = self.registry.resolve_parameters(
            model_name=self.model,
            prompt="A cat",
            width=512,
            height=512,
            aspect_ratio="16:9",
        )
        # aspect_ratio wins
        assert params.width == 1024
        assert params.height == 576

    def test_flux_klein_4k_aspect_ratio(self):
        params = self.registry.resolve_parameters(
            model_name=self.model,
            prompt="A landscape",
            aspect_ratio="16:9_4k",
        )
        assert params.width == 3840
        assert params.height == 2160

    def test_flux_klein_2k_aspect_ratio(self):
        params = self.registry.resolve_parameters(
            model_name=self.model,
            prompt="A portrait",
            aspect_ratio="16:9_2k",
        )
        assert params.width == 2560
        assert params.height == 1440

    def test_flux_klein_explicit_4k_not_clamped(self):
        params = self.registry.resolve_parameters(
            model_name=self.model,
            prompt="A city",
            width=3840,
            height=2160,
        )
        assert params.width == 3840
        assert params.height == 2160
        assert not any("exceeds max" in n for n in params.notes)

    def test_flux_klein_max_mp_is_9(self):
        profile = self.registry.get_profile("black-forest-labs/FLUX.2-klein-4B")
        assert profile["max_resolution_mp"] == 9.0

    def test_all_model_profiles_have_required_fields(self):
        required = {
            "type", "default_steps", "default_guidance_scale",
            "supports_negative_prompt", "default_resolution",
            "max_resolution_mp", "resolution_multiple",
            "estimated_vram_gb", "supported_aspect_ratios",
            "prompt_guidance",
        }
        for name, profile in MODEL_PROFILES.items():
            for field in required:
                assert field in profile, f"Model '{name}' missing field '{field}'"
