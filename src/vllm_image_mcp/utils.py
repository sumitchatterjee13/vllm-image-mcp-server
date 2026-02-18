"""Utility functions for file saving, filename generation, and validation."""

from __future__ import annotations

import base64
import hashlib
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

SUPPORTED_FORMATS = {"png", "jpg", "webp"}

# Format-specific Pillow save options
FORMAT_SAVE_OPTIONS: dict[str, dict] = {
    "png": {},
    "jpg": {"quality": 95, "subsampling": 0},
    "webp": {"quality": 90, "method": 6},
}


def generate_filename(prompt: str, extension: str = "png") -> str:
    """Generate a filename from timestamp and prompt hash.

    Format: img_{YYYYMMDD}_{HHMMSS}_{short_hash}.{ext}
    """
    extension = extension.lower().lstrip(".")
    if extension not in SUPPORTED_FORMATS:
        extension = "png"
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.sha256(prompt.encode()).hexdigest()[:6]
    return f"img_{timestamp}_{short_hash}.{extension}"


def save_base64_image(
    b64_data: str,
    output_dir: str | Path,
    filename: str | None = None,
    prompt: str = "",
    format: str = "png",
) -> Path:
    """Decode a base64 image and save it to disk in the specified format.

    The vLLM API returns base64-encoded PNG data. If format is 'jpg' or 'webp',
    the PNG is decoded with Pillow and re-saved in the target format.

    Returns the full path to the saved file.
    """
    format = format.lower()
    if format not in SUPPORTED_FORMATS:
        format = "png"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = generate_filename(prompt, extension=format)

    # Ensure correct extension for the chosen format
    stem = Path(filename).stem
    filename = f"{stem}.{format}"

    file_path = output_dir / filename
    image_bytes = base64.b64decode(b64_data)

    if format == "png":
        # Fast path: write raw bytes directly, no conversion needed
        file_path.write_bytes(image_bytes)
    else:
        # Convert via Pillow
        img = Image.open(BytesIO(image_bytes))
        if format == "jpg":
            if img.mode == "RGBA":
                # JPEG doesn't support transparency; composite onto white
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
        pillow_format = "JPEG" if format == "jpg" else format.upper()
        save_kwargs = FORMAT_SAVE_OPTIONS.get(format, {})
        img.save(file_path, format=pillow_format, **save_kwargs)

    return file_path


def validate_resolution(
    width: int,
    height: int,
    multiple: int = 16,
    max_mp: float = 4.0,
) -> tuple[bool, str | None]:
    """Validate that a resolution meets constraints.

    Returns (is_valid, error_message).
    """
    if width <= 0 or height <= 0:
        return False, "Width and height must be positive integers."

    if width % multiple != 0:
        nearest = round(width / multiple) * multiple
        return False, (
            f"Width {width} is not a multiple of {multiple}. "
            f"Nearest valid width is {nearest}."
        )

    if height % multiple != 0:
        nearest = round(height / multiple) * multiple
        return False, (
            f"Height {height} is not a multiple of {multiple}. "
            f"Nearest valid height is {nearest}."
        )

    current_mp = (width * height) / 1_000_000
    if current_mp > max_mp:
        return False, (
            f"Resolution {width}x{height} ({current_mp:.1f}MP) exceeds "
            f"the maximum of {max_mp}MP."
        )

    return True, None


def format_file_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
