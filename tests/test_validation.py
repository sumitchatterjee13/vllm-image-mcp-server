"""Tests for resolution validation, input sanitization, and utility functions."""

import os
import tempfile
from pathlib import Path

from vllm_image_mcp.utils import (
    generate_filename,
    save_base64_image,
    validate_resolution,
    format_file_size,
)


class TestValidateResolution:
    def test_valid_resolution(self):
        valid, err = validate_resolution(1024, 1024)
        assert valid
        assert err is None

    def test_not_multiple_of_16_width(self):
        valid, err = validate_resolution(1000, 1024)
        assert not valid
        assert "multiple of 16" in err
        assert "992" in err  # nearest valid (round(1000/16)*16 = 62*16)

    def test_not_multiple_of_16_height(self):
        valid, err = validate_resolution(1024, 1000)
        assert not valid
        assert "multiple of 16" in err

    def test_exceeds_max_megapixels(self):
        valid, err = validate_resolution(4096, 4096, max_mp=4.0)
        assert not valid
        assert "exceeds" in err

    def test_zero_dimension(self):
        valid, err = validate_resolution(0, 1024)
        assert not valid
        assert "positive" in err

    def test_negative_dimension(self):
        valid, err = validate_resolution(-16, 1024)
        assert not valid
        assert "positive" in err

    def test_custom_multiple(self):
        valid, err = validate_resolution(100, 100, multiple=10)
        assert valid

    def test_custom_multiple_invalid(self):
        valid, err = validate_resolution(105, 100, multiple=10)
        assert not valid


class TestGenerateFilename:
    def test_format(self):
        fname = generate_filename("A cat sitting on a mat")
        assert fname.startswith("img_")
        assert fname.endswith(".png")
        parts = fname.replace(".png", "").split("_")
        # img_YYYYMMDD_HHMMSS_hash
        assert len(parts) == 4

    def test_deterministic_hash(self):
        fname1 = generate_filename("same prompt")
        fname2 = generate_filename("same prompt")
        # Hash part should be the same
        hash1 = fname1.split("_")[-1].replace(".png", "")
        hash2 = fname2.split("_")[-1].replace(".png", "")
        assert hash1 == hash2

    def test_different_prompts_different_hash(self):
        fname1 = generate_filename("prompt one")
        fname2 = generate_filename("prompt two")
        hash1 = fname1.split("_")[-1].replace(".png", "")
        hash2 = fname2.split("_")[-1].replace(".png", "")
        assert hash1 != hash2

    def test_custom_extension(self):
        fname = generate_filename("test", extension="jpg")
        assert fname.endswith(".jpg")

    def test_custom_extension_webp(self):
        fname = generate_filename("test", extension="webp")
        assert fname.endswith(".webp")

    def test_invalid_extension_defaults_to_png(self):
        fname = generate_filename("test", extension="bmp")
        assert fname.endswith(".png")


class TestSaveBase64Image:
    def test_save_creates_file(self):
        import base64

        # 1x1 red pixel PNG
        pixel_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        b64 = base64.b64encode(pixel_data).decode()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_base64_image(
                b64_data=b64,
                output_dir=tmpdir,
                filename="test_image.png",
                prompt="test",
            )
            assert path.exists()
            assert path.name == "test_image.png"
            assert path.read_bytes() == pixel_data

    def test_save_auto_filename(self):
        import base64

        b64 = base64.b64encode(b"fake png data").decode()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_base64_image(
                b64_data=b64,
                output_dir=tmpdir,
                prompt="auto name test",
            )
            assert path.exists()
            assert path.name.startswith("img_")
            assert path.name.endswith(".png")

    def test_save_creates_output_dir(self):
        import base64

        b64 = base64.b64encode(b"data").decode()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "sub", "dir")
            path = save_base64_image(
                b64_data=b64,
                output_dir=nested,
                filename="test.png",
            )
            assert path.exists()

    def test_adds_png_extension(self):
        import base64

        b64 = base64.b64encode(b"data").decode()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_base64_image(
                b64_data=b64,
                output_dir=tmpdir,
                filename="no_ext",
            )
            assert path.name == "no_ext.png"

    def test_save_as_jpg(self):
        import base64
        from io import BytesIO
        from PIL import Image

        img = Image.new("RGB", (2, 2), color=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_base64_image(
                b64_data=b64,
                output_dir=tmpdir,
                prompt="test jpg",
                format="jpg",
            )
            assert path.name.endswith(".jpg")
            # Verify JPEG magic bytes
            magic = path.read_bytes()[:2]
            assert magic == b"\xff\xd8"

    def test_save_as_webp(self):
        import base64
        from io import BytesIO
        from PIL import Image

        img = Image.new("RGB", (2, 2), color=(0, 255, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_base64_image(
                b64_data=b64,
                output_dir=tmpdir,
                prompt="test webp",
                format="webp",
            )
            assert path.name.endswith(".webp")
            # Verify WebP RIFF container magic
            magic = path.read_bytes()[:4]
            assert magic == b"RIFF"

    def test_save_rgba_as_jpg_composites_white(self):
        import base64
        from io import BytesIO
        from PIL import Image

        img = Image.new("RGBA", (2, 2), color=(255, 0, 0, 128))
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_base64_image(
                b64_data=b64,
                output_dir=tmpdir,
                prompt="test rgba jpg",
                format="jpg",
            )
            assert path.exists()
            result_img = Image.open(path)
            assert result_img.mode == "RGB"
            result_img.close()

    def test_invalid_format_defaults_to_png(self):
        import base64

        b64 = base64.b64encode(b"fake data").decode()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_base64_image(
                b64_data=b64,
                output_dir=tmpdir,
                prompt="test",
                format="bmp",
            )
            assert path.name.endswith(".png")


class TestFormatFileSize:
    def test_bytes(self):
        assert format_file_size(500) == "500.0 B"

    def test_kilobytes(self):
        assert format_file_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert format_file_size(1048576) == "1.0 MB"

    def test_gigabytes(self):
        assert format_file_size(1073741824) == "1.0 GB"
