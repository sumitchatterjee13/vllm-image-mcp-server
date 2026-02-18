# MCP Server Project Plan: vLLM Image Generation

## Overview

Build an MCP (Model Context Protocol) server in Python (FastMCP) that provides AI-powered image generation tools via any vLLM-Omni compatible endpoint. The server should be model-aware, GPU-aware, and support both single and batch image generation with automatic parameter tuning based on the loaded model.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│  MCP Client (Claude Desktop / Claude Code / etc) │
└──────────────────────┬──────────────────────────┘
                       │ MCP Protocol (stdio)
┌──────────────────────▼──────────────────────────┐
│           MCP Server (this project)              │
│                                                  │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐ │
│  │ Tool:      │ │ Tool:      │ │ Tool:        │ │
│  │ generate   │ │ batch      │ │ get_model    │ │
│  │ _image     │ │ _generate  │ │ _info        │ │
│  └────────────┘ └────────────┘ └──────────────┘ │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐ │
│  │ Tool:      │ │ Tool:      │ │ Tool:        │ │
│  │ gpu_status │ │ list       │ │ server       │ │
│  │            │ │ _presets   │ │ _health      │ │
│  └────────────┘ └────────────┘ └──────────────┘ │
│  ┌────────────┐ ┌────────────┐                   │
│  │ Tool:      │ │ Tool:      │                   │
│  │ estimate   │ │ cancel     │                   │
│  │ _generation│ │ _batch     │                   │
│  └────────────┘ └────────────┘                   │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Core: ModelRegistry + ParameterResolver │    │
│  └──────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────┐    │
│  │  Core: vLLM API Client (configurable URL)│    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
                       │ HTTP (OpenAI-compatible API)
┌──────────────────────▼──────────────────────────┐
│  vLLM-Omni Server (local or remote)              │
│  e.g. http://localhost:6655                      │
│  or   http://192.168.1.100:6655                  │
└──────────────────────────────────────────────────┘
```

---

## Server Configuration

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--vllm-url` | string | `http://localhost:6655` | Base URL of the vLLM-Omni server |
| `--output-dir` | string | `./generated_images` | Directory to save generated images |
| `--max-concurrent` | int | `auto` | Max concurrent generations for batch (auto = determined by GPU check) |
| `--timeout` | int | `300` | Request timeout in seconds |

Example launch:
```bash
python mcp_server.py --vllm-url http://192.168.1.50:6655 --output-dir ./images
```

---

## Model Registry & Parameter Resolver

The server must maintain a registry of known models with their optimal default parameters. When a tool is called without explicit parameters, the server queries the vLLM endpoint for the loaded model name, looks it up in the registry, and applies the correct defaults.

### Known Model Profiles

```python
MODEL_PROFILES = {
    "black-forest-labs/FLUX.2-klein-4B": {
        "type": "distilled",
        "default_steps": 4,
        "default_guidance_scale": 1.0,
        "supports_negative_prompt": False,
        "default_resolution": (1024, 1024),
        "max_resolution_mp": 4.0,          # 4 megapixels max
        "resolution_multiple": 16,          # must be multiples of 16
        "estimated_vram_gb": 13,
        "supported_aspect_ratios": {
            "1:1": (1024, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
        },
    },
    "black-forest-labs/FLUX.2-klein-9B": {
        "type": "distilled",
        "default_steps": 4,
        "default_guidance_scale": 1.0,
        "supports_negative_prompt": False,
        "default_resolution": (1024, 1024),
        "max_resolution_mp": 4.0,
        "resolution_multiple": 16,
        "estimated_vram_gb": 29,
        "supported_aspect_ratios": {
            "1:1": (1024, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
        },
    },
    "Qwen/Qwen-Image-2512": {
        "type": "standard",
        "default_steps": 28,
        "default_guidance_scale": 5.0,
        "supports_negative_prompt": True,
        "default_resolution": (1024, 1024),
        "max_resolution_mp": 4.0,
        "resolution_multiple": 16,
        "estimated_vram_gb": 40,       # BF16, ~20 with FP8
        "supported_aspect_ratios": {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1104),
            "3:4": (1104, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        },
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
    },
    "Tongyi-MAI/Z-Image-Turbo": {
        "type": "standard",
        "default_steps": 50,
        "default_guidance_scale": 5.0,
        "supports_negative_prompt": True,
        "default_resolution": (1024, 1024),
        "max_resolution_mp": 4.0,
        "resolution_multiple": 16,
        "estimated_vram_gb": 40,
        "supported_aspect_ratios": {
            "1:1": (1024, 1024),
        },
    },
}
```

If the model is not in the registry, the server should use sensible fallback defaults (steps=28, guidance=5.0, 1024x1024) and log a warning.

---

## Tools Specification

### Tool 1: `generate_image`

**Purpose:** Generate a single image from a text prompt.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | yes | — | Text description of the image |
| `negative_prompt` | string | no | null | What to avoid (ignored if model doesn't support it) |
| `width` | int | no | model default | Image width in pixels |
| `height` | int | no | model default | Image height in pixels |
| `aspect_ratio` | string | no | null | Shortcut e.g. "16:9", "1:1" (overrides width/height) |
| `num_inference_steps` | int | no | model default | Number of diffusion steps |
| `guidance_scale` | float | no | model default | CFG scale |
| `seed` | int | no | random | Random seed for reproducibility |
| `filename` | string | no | auto-generated | Output filename (auto = timestamp + hash) |

**Behavior:**
1. Query model info if not cached
2. Resolve parameters from model profile + user overrides
3. If user provides `negative_prompt` but model doesn't support it, silently ignore it and include a note in the response
4. Validate resolution is within model limits and a multiple of `resolution_multiple`
5. Send request to vLLM-Omni `/v1/images/generations` endpoint
6. Decode base64 response, save PNG to `output_dir`
7. Return: file path, generation time, resolution, seed used, parameters used

**Response format:**
```json
{
    "status": "success",
    "file_path": "/path/to/generated_images/img_20260218_143022_a1b2c3.png",
    "generation_time_seconds": 1.23,
    "parameters_used": {
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "prompt": "A cat...",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "guidance_scale": 1.0,
        "seed": 42
    }
}
```

---

### Tool 2: `batch_generate`

**Purpose:** Generate multiple images from multiple prompts concurrently.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompts` | list[string] | yes | — | List of text prompts (max 20) |
| `width` | int | no | model default | Width for all images |
| `height` | int | no | model default | Height for all images |
| `aspect_ratio` | string | no | null | Aspect ratio shortcut for all images |
| `num_inference_steps` | int | no | model default | Steps for all images |
| `guidance_scale` | float | no | model default | CFG for all images |
| `seeds` | list[int] | no | random per image | List of seeds (must match prompts length if provided) |
| `max_concurrent` | int | no | auto | Override concurrent limit |

**Behavior:**
1. Query GPU status to determine concurrency if `max_concurrent` is auto
2. Use `asyncio.Semaphore` to limit concurrent requests
3. Fire all requests concurrently (up to the semaphore limit)
4. Collect results as they complete
5. Return summary with per-image results and total batch time

**Concurrency auto-detection logic:**
```
available_vram = total_gpu_vram - model_estimated_vram
estimated_per_image_overhead = 2 GB  (activation memory per concurrent generation)
max_concurrent = max(1, floor(available_vram / estimated_per_image_overhead))
cap at 4 to avoid OOM on consumer GPUs
```

**Response format:**
```json
{
    "status": "success",
    "total_time_seconds": 8.45,
    "images_generated": 5,
    "images_failed": 0,
    "results": [
        {
            "index": 0,
            "prompt": "A cat...",
            "file_path": "/path/to/img_001.png",
            "generation_time_seconds": 1.2,
            "seed": 42
        }
    ]
}
```

---

### Tool 3: `get_model_info`

**Purpose:** Get information about the currently loaded model and its recommended parameters.

**Parameters:** None

**Behavior:**
1. Call vLLM-Omni `/v1/models` endpoint to get loaded model name
2. Look up model in the registry
3. Return model name, type, recommended parameters, supported aspect ratios, VRAM estimate

**Response format:**
```json
{
    "model_name": "black-forest-labs/FLUX.2-klein-4B",
    "model_type": "distilled",
    "recommended_parameters": {
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "default_resolution": "1024x1024"
    },
    "supports_negative_prompt": false,
    "supported_aspect_ratios": {
        "1:1": "1024x1024",
        "16:9": "1024x576",
        "9:16": "576x1024"
    },
    "estimated_vram_gb": 13,
    "in_registry": true
}
```

---

### Tool 4: `gpu_status`

**Purpose:** Check GPU memory usage and availability for generation capacity planning.

**Parameters:** None

**Behavior:**
1. Call vLLM-Omni's metrics or health endpoint to get GPU stats
2. If that's not available, try parsing `/v1/models` response for any GPU info
3. Alternatively, call a custom endpoint or estimate from model profile
4. Calculate how many concurrent generations the GPU can support

**Response format:**
```json
{
    "gpu_name": "NVIDIA GeForce RTX 5090",
    "total_vram_gb": 32.0,
    "estimated_model_vram_gb": 13,
    "estimated_free_vram_gb": 19.0,
    "recommended_max_concurrent": 4,
    "server_url": "http://localhost:6655",
    "server_status": "healthy"
}
```

**Note:** Since vLLM-Omni may not expose detailed GPU metrics via API, this tool should do its best using the model registry's `estimated_vram_gb` and any available endpoint data. If GPU info can't be queried, return estimates based on the model profile and state that clearly.

---

### Tool 5: `list_presets`

**Purpose:** List available resolution presets and aspect ratios for the current model.

**Parameters:** None

**Behavior:**
1. Get current model name
2. Return all supported aspect ratios with their pixel dimensions
3. Include notes on max resolution and resolution constraints

**Response format:**
```json
{
    "model": "black-forest-labs/FLUX.2-klein-4B",
    "resolution_multiple": 16,
    "max_megapixels": 4.0,
    "presets": {
        "1:1": "1024x1024",
        "16:9": "1024x576",
        "9:16": "576x1024",
        "4:3": "1024x768",
        "3:4": "768x1024"
    }
}
```

---

### Tool 6: `server_health`

**Purpose:** Check if the vLLM-Omni server is running, ready, and responsive.

**Parameters:** None

**Behavior:**
1. Hit `{vllm_url}/health` endpoint
2. Hit `{vllm_url}/v1/models` to confirm model is loaded
3. Measure response latency
4. Report overall status

**Response format:**
```json
{
    "server_url": "http://localhost:6655",
    "status": "healthy",
    "model_loaded": "black-forest-labs/FLUX.2-klein-4B",
    "latency_ms": 12,
    "version": "0.14.0"
}
```

---

### Tool 7: `estimate_generation`

**Purpose:** Estimate generation time and resource usage without actually generating.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `num_images` | int | no | 1 | Number of images to estimate for |
| `width` | int | no | model default | Image width |
| `height` | int | no | model default | Image height |
| `num_inference_steps` | int | no | model default | Steps per image |

**Behavior:**
1. Calculate total pixels (width x height x num_images)
2. Estimate time based on model type (distilled ~1-2s per image, standard ~5-30s)
3. Estimate peak VRAM based on resolution and batch size
4. Recommend concurrency settings

**Response format:**
```json
{
    "estimated_time_per_image_seconds": 1.5,
    "estimated_total_time_seconds": 7.5,
    "estimated_peak_vram_gb": 15,
    "recommended_concurrent": 2,
    "resolution": "1024x1024",
    "total_megapixels_per_image": 1.05,
    "notes": "Distilled model, 4 steps. First generation may be slower due to torch.compile warmup."
}
```

---

### Tool 8: `cancel_batch`

**Purpose:** Cancel a running batch generation.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `batch_id` | string | yes | — | The batch ID returned by `batch_generate` |

**Behavior:**
1. Look up the batch in the in-memory tracking dict
2. Cancel pending futures via `asyncio.Task.cancel()`
3. Return number of completed vs cancelled images
4. Already-completed images are kept

**Response format:**
```json
{
    "batch_id": "batch_abc123",
    "status": "cancelled",
    "completed": 3,
    "cancelled": 7,
    "files_saved": ["/path/to/img_001.png", "/path/to/img_002.png", "/path/to/img_003.png"]
}
```

---

## Project Structure

```
vllm-image-mcp/
├── pyproject.toml              # Project metadata, dependencies, entry point
├── README.md                   # Setup and usage instructions
├── src/
│   └── vllm_image_mcp/
│       ├── __init__.py
│       ├── server.py           # MCP server entry point, tool registrations
│       ├── client.py           # vLLM API client (async httpx)
│       ├── models.py           # Model registry, parameter resolver, profiles
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── generate.py     # generate_image tool
│       │   ├── batch.py        # batch_generate + cancel_batch tools
│       │   ├── info.py         # get_model_info, list_presets, estimate_generation
│       │   └── system.py       # gpu_status, server_health
│       └── utils.py            # File saving, filename generation, validation
└── tests/
    ├── test_models.py          # Test parameter resolution logic
    ├── test_validation.py      # Test resolution validation, input sanitization
    └── test_client.py          # Test API client with mocked responses
```

---

## Dependencies

```toml
[project]
name = "vllm-image-mcp"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mcp[cli]>=1.0.0",       # FastMCP framework
    "httpx>=0.27.0",          # Async HTTP client for vLLM API
    "pydantic>=2.0.0",        # Input validation
    "Pillow>=10.0.0",         # Image handling (optional, for metadata)
]

[project.scripts]
vllm-image-mcp = "vllm_image_mcp.server:main"
```

---

## Key Implementation Notes

### 1. vLLM API Client

Use `httpx.AsyncClient` with configurable base URL:

```python
class VLLMClient:
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def generate_image(self, payload: dict) -> dict:
        response = await self.client.post(
            f"{self.base_url}/v1/images/generations",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def get_models(self) -> dict:
        response = await self.client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> bool:
        response = await self.client.get(f"{self.base_url}/health")
        return response.status_code == 200
```

### 2. Parameter Resolution Order

Parameters resolve in this priority (highest wins):
1. User-provided value in tool call
2. Model profile default from registry
3. Global fallback defaults

### 3. Error Handling

Every tool should return actionable error messages:
- **Server unreachable:** "Cannot connect to vLLM server at {url}. Is the container running? Check with `server_health` tool."
- **Model not in registry:** "Model '{name}' is not in the known profiles. Using fallback defaults (steps=28, guidance=5.0). Generation may produce unexpected results."
- **Resolution invalid:** "Width {w} is not a multiple of {multiple}. Nearest valid width is {nearest}."
- **OOM during generation:** "Server returned out-of-memory error. Try reducing resolution or use `gpu_status` to check available memory."

### 4. File Naming Convention

Auto-generated filenames: `img_{timestamp}_{short_hash}.png`
- timestamp: `YYYYMMDD_HHMMSS`
- short_hash: first 6 chars of SHA256 of prompt
- Example: `img_20260218_143022_a1b2c3.png`

### 5. Batch Tracking

Store active batches in an in-memory dict keyed by batch_id (UUID4):
```python
active_batches: dict[str, BatchState] = {}

class BatchState:
    batch_id: str
    tasks: list[asyncio.Task]
    results: list[dict]
    created_at: datetime
    cancelled: bool = False
```

### 6. Tool Annotations

All tools should include MCP annotations:
- `generate_image`: `readOnlyHint=False, destructiveHint=False, idempotentHint=True` (same seed = same result)
- `batch_generate`: `readOnlyHint=False, destructiveHint=False, idempotentHint=False`
- `get_model_info`: `readOnlyHint=True`
- `gpu_status`: `readOnlyHint=True`
- `list_presets`: `readOnlyHint=True`
- `server_health`: `readOnlyHint=True`
- `estimate_generation`: `readOnlyHint=True`
- `cancel_batch`: `readOnlyHint=False, destructiveHint=True`

---

## Claude Code Configuration

Add to `claude_desktop_config.json` or Claude Code MCP settings:

```json
{
    "mcpServers": {
        "vllm-image": {
            "command": "python",
            "args": [
                "-m", "vllm_image_mcp.server",
                "--vllm-url", "http://localhost:6655",
                "--output-dir", "./generated_images"
            ]
        }
    }
}
```

Or for a remote vLLM server on the network:

```json
{
    "mcpServers": {
        "vllm-image": {
            "command": "python",
            "args": [
                "-m", "vllm_image_mcp.server",
                "--vllm-url", "http://192.168.1.100:6655",
                "--output-dir", "D:/generated_images"
            ]
        }
    }
}
```

---

## Testing Plan

1. **Unit tests:** Model registry lookups, parameter resolution, resolution validation, filename generation
2. **Integration tests (mocked):** Mock httpx responses for all vLLM endpoints, test each tool end-to-end
3. **Manual smoke test:** Run against a real vLLM-Omni container with FLUX.2-klein-4B and verify image output
4. **Edge cases to test:**
   - Server unreachable
   - Unknown model not in registry
   - Invalid resolution (not multiple of 16)
   - Negative prompt sent to distilled model (should be silently ignored)
   - Batch with 0 prompts, 1 prompt, 20 prompts, 21 prompts (should reject >20)
   - Cancel batch that doesn't exist
   - Cancel batch that already completed
