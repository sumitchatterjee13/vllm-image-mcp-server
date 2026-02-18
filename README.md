# vLLM Image MCP Server

An MCP (Model Context Protocol) server that provides AI-powered image generation tools via any vLLM-Omni compatible endpoint. The server is model-aware, GPU-aware, and supports both single and batch image generation with automatic parameter tuning based on the loaded model.

## Features

- **10 MCP tools** for image generation, batch processing, progress monitoring, and system status
- **Model-aware defaults** — auto-detects the loaded model and applies optimal parameters
- **5 built-in model profiles**: FLUX.2-klein 4B/9B, Qwen-Image variants, Z-Image-Turbo
- **Multi-format output** — save as PNG (lossless), JPG (quality=95), or WebP (smallest, ideal for web)
- **Up to 4K resolution** — FLUX Klein models support resolutions up to 3840x2160
- **Flexible output paths** — the AI model chooses where to save images per-call (e.g. directly into your project's `assets/` folder)
- **Batch generation** with automatic concurrency control based on estimated VRAM
- **Batch progress monitoring** — poll running batches to track completion and prevent timeouts
- **Resolution validation** — snaps to valid multiples, clamps to megapixel limits
- **Aspect ratio presets** — use shortcuts like `16:9`, `16:9_2k`, `16:9_4k` instead of raw pixels

## Tools

| Tool | Description |
|---|---|
| `generate_image` | Generate a single image from a text prompt |
| `batch_generate` | Generate multiple images concurrently (max 20) |
| `get_model_info` | Get current model info and recommended parameters |
| `list_presets` | List aspect ratio presets for the current model |
| `estimate_generation` | Estimate time and resource usage before generating |
| `gpu_status` | Check GPU/VRAM availability for capacity planning |
| `server_health` | Check vLLM server connectivity and status |
| `cancel_batch` | Cancel a running batch generation |
| `check_batch_progress` | Check progress of a running batch generation |
| `list_active_batches` | List all currently running batch jobs |

### Dynamic Output Directory

Both `generate_image` and `batch_generate` require an `output_dir` parameter. The AI model decides where to save images based on your project context — no hardcoded paths needed:

```python
generate_image(prompt="A hero banner", output_dir="./src/assets/images")
batch_generate(prompts=["cat", "dog"], output_dir="./public/img")
```

This means images land exactly where your project needs them.

### Output Formats

Both `generate_image` and `batch_generate` support a `format` parameter:

| Format | Use Case | Quality | File Size |
|---|---|---|---|
| `png` (default) | Lossless quality, transparency support | Lossless | Largest |
| `jpg` | General use, photographs | 95% quality | ~70% smaller |
| `webp` | Web projects, optimized delivery | 90% quality | ~80% smaller |

```python
generate_image(prompt="A hero banner", output_dir="./assets", format="webp")
batch_generate(prompts=["cat", "dog"], output_dir="./public", format="jpg")
```

The AI model chooses the format based on context (e.g. `webp` for web projects, `png` for graphic design).

### Batch Progress Monitoring

For long-running batch generations, use `check_batch_progress` to poll every ~50 seconds:

```python
# Start batch
batch_generate(prompts=[...], output_dir="./output")  # returns batch_id

# Poll progress
check_batch_progress(batch_id="batch_abc123def456")

# Discover running batches
list_active_batches()
```

## Prerequisites

- Python 3.11+
- A running [vLLM-Omni](https://github.com/vllm-project/vllm) server with an image generation model loaded
- `pip` or `uv` for package installation

## Installation

### From source

```bash
git clone https://github.com/your-username/vllm-image-mcp-server.git
cd vllm-image-mcp-server
pip install -e .
```

### Verify installation

```bash
vllm-image-mcp --help
```

Or run directly:

```bash
python -m vllm_image_mcp.server --help
```

## Usage

### Standalone

```bash
# Default: connects to http://localhost:6655
vllm-image-mcp

# Custom vLLM server URL
vllm-image-mcp --vllm-url http://192.168.1.100:6655

# With custom timeout
vllm-image-mcp --vllm-url http://localhost:6655 --timeout 600
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--vllm-url` | `http://localhost:6655` | Base URL of the vLLM-Omni server |
| `--max-concurrent` | auto | Max concurrent generations for batch |
| `--timeout` | `300` | Request timeout in seconds |

> **Note:** There is no `--output-dir` flag. The output path is provided by the AI model on every `generate_image` / `batch_generate` call, so images are saved wherever the project needs them.

---

## MCP Client Configuration

### Claude Code

**Option A — CLI command:**

```bash
claude mcp add vllm-image -- python -m vllm_image_mcp.server --vllm-url http://localhost:6655
```

**Option B — Project config (`.mcp.json` in project root):**

```json
{
  "mcpServers": {
    "vllm-image": {
      "command": "python",
      "args": [
        "-m", "vllm_image_mcp.server",
        "--vllm-url", "http://localhost:6655"
      ]
    }
  }
}
```

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vllm-image": {
      "command": "python",
      "args": [
        "-m", "vllm_image_mcp.server",
        "--vllm-url", "http://localhost:6655"
      ]
    }
  }
}
```

### Cursor

Create `.cursor/mcp.json` in your project root (or `~/.cursor/mcp.json` for global):

```json
{
  "mcpServers": {
    "vllm-image": {
      "command": "python",
      "args": [
        "-m", "vllm_image_mcp.server",
        "--vllm-url", "http://localhost:6655"
      ]
    }
  }
}
```

### Kilo Code

Create `.kilocode/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "vllm-image": {
      "command": "python",
      "args": [
        "-m", "vllm_image_mcp.server",
        "--vllm-url", "http://localhost:6655"
      ],
      "alwaysAllow": [],
      "disabled": false
    }
  }
}
```

### Windows Note

On native Windows (not WSL/Git Bash), if you get "Connection closed" errors, wrap the command with `cmd`:

```json
{
  "mcpServers": {
    "vllm-image": {
      "command": "cmd",
      "args": [
        "/c", "python", "-m", "vllm_image_mcp.server",
        "--vllm-url", "http://localhost:6655"
      ]
    }
  }
}
```

### Remote vLLM Server

To connect to a vLLM server on another machine, change the `--vllm-url` argument:

```json
"args": [
  "-m", "vllm_image_mcp.server",
  "--vllm-url", "http://192.168.1.100:6655"
]
```

---

## Supported Models

The server includes built-in profiles with optimal defaults for these models:

| Model | Type | Steps | Guidance | Neg. Prompt | Est. VRAM | Max Resolution |
|---|---|---|---|---|---|---|
| `black-forest-labs/FLUX.2-klein-4B` | Distilled | 4 | 1.0 | No | 13 GB | 4K (9.0 MP) |
| `black-forest-labs/FLUX.2-klein-9B` | Distilled | 4 | 1.0 | No | 29 GB | 4K (9.0 MP) |
| `Qwen/Qwen-Image-2512` | Standard | 28 | 5.0 | Yes | 40 GB | 2K (4.0 MP) |
| `Qwen/Qwen-Image` | Standard | 50 | 5.0 | Yes | 40 GB | 2K (4.0 MP) |
| `Tongyi-MAI/Z-Image-Turbo` | Distilled | 8 | 1.0 | No | 16 GB | 2K (4.0 MP) |

Unknown models automatically use fallback defaults (steps=28, guidance=5.0, 1024x1024).

### FLUX Klein Resolution Presets

FLUX Klein models support resolutions from standard to 4K:

| Preset | Resolution | Megapixels | Use Case |
|---|---|---|---|
| `1:1` | 1024x1024 | 1.0 MP | Default, fast |
| `16:9` | 1024x576 | 0.6 MP | Widescreen, fast |
| `9:16` | 576x1024 | 0.6 MP | Portrait, fast |
| `1:1_2k` | 2048x2048 | 4.2 MP | High detail square |
| `16:9_2k` | 2560x1440 | 3.7 MP | 2K widescreen |
| `4:3_2k` | 2048x1536 | 3.1 MP | 2K standard |
| `16:9_4k` | 3840x2160 | 8.3 MP | 4K, maximum detail |
| `9:16_4k` | 2160x3840 | 8.3 MP | 4K portrait |

4K resolutions take significantly longer and use more VRAM. Use only when the user needs high-resolution output (wallpapers, print-quality, posters).

---

## Prompt Writing Guide

The server includes **model-specific prompt guidance** that is returned by the `get_model_info` tool. The AI model should call `get_model_info` before its first generation to learn the loaded model's prompting rules. Below is a summary.

### General Rules (All Models)

- Write prompts as **natural language prose**, never comma-separated tags
- Put the **subject first** — the first 10-20 words carry the most weight
- **Always specify lighting** — it has the single greatest impact on quality
- Ideal length: **30-80 words**
- Do **NOT** use quality tags like `8k, masterpiece, best quality, ultra HD` — they waste tokens
- Do **NOT** describe sequential actions — images are a single moment
- Do **NOT** mix conflicting styles (e.g. "photorealistic oil painting")
- For photorealism, reference real cameras: `Shot on Sony A7IV, 85mm lens at f/2.0`
- For text in images, use quotation marks: `The sign reads "OPEN"`

### FLUX.2 Klein (Distilled Models)

- Negative prompts are **ignored** at CFG 1.0 — do not send them
- Be extra explicit and descriptive — no auto-enhancement available
- Every word must contribute visual information; filler text hurts quality
- Use emphasis phrases: "prominently featuring", "the focal point is"

**Good prompt:**

```text
Professional studio product shot on polished concrete. A minimalist ceramic
coffee mug with matte black finish, steam rising from hot coffee, centered
in frame. Ultra-realistic commercial photography. Three-point softbox
lighting, diffused highlights, no harsh shadows. Shot with 85mm lens at f/5.6.
```

**Bad prompt:**

```text
coffee mug, black, steam, studio, professional, 8k, masterpiece, best quality,
ultra detailed, sharp focus
```

### Qwen-Image / Z-Image (Standard Models)

- **Use negative prompts** — they improve results significantly
- Good default negative: `blurry, low quality, distorted, watermark, oversaturated, artificial`
- For portraits add: `extra fingers, deformed hands, unnatural proportions`
- The model interprets prompts **very literally** — be precise
- For text-heavy images, raise `guidance_scale` to 7.0
- Wrap desired text in double quotes and specify font style
- Use `photograph` instead of `photorealistic` or `3d render`

**Good prompt:**

```text
A futuristic sports car parked under neon city lights, photorealistic style.
Reflections shimmer on wet asphalt streets. Dramatic side lighting with deep
shadows and vibrant highlights. "Night Racer" in metallic chrome text on the hood.
```

**Negative prompt:**

```text
blurry, low quality, distorted, watermark, oversaturated
```

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

## License

MIT
