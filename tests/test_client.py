"""Tests for the vLLM API client with mocked responses."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from vllm_image_mcp.client import VLLMClient
from vllm_image_mcp.models import ModelRegistry
from vllm_image_mcp.tools.generate import generate_image
from vllm_image_mcp.tools.info import get_model_info, list_presets, estimate_generation
from vllm_image_mcp.tools.system import server_health, gpu_status
from vllm_image_mcp.tools.batch import (
    batch_generate,
    cancel_batch,
    check_batch_progress,
    list_active_batches,
    active_batches,
    BatchState,
)


@pytest.fixture
def client():
    return VLLMClient(base_url="http://test:6655", timeout=30)


@pytest.fixture
def registry():
    reg = ModelRegistry()
    reg.cached_model_name = "black-forest-labs/FLUX.2-klein-4B"
    return reg


class TestVLLMClient:
    @pytest.mark.asyncio
    async def test_get_loaded_model_name(self, client):
        mock_response = httpx.Response(
            200,
            json={"data": [{"id": "black-forest-labs/FLUX.2-klein-4B"}]},
            request=httpx.Request("GET", "http://test:6655/v1/models"),
        )
        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            name = await client.get_loaded_model_name()
            assert name == "black-forest-labs/FLUX.2-klein-4B"

    @pytest.mark.asyncio
    async def test_get_loaded_model_name_empty(self, client):
        mock_response = httpx.Response(
            200,
            json={"data": []},
            request=httpx.Request("GET", "http://test:6655/v1/models"),
        )
        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            name = await client.get_loaded_model_name()
            assert name is None

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, client):
        mock_response = httpx.Response(
            200,
            text="OK",
            request=httpx.Request("GET", "http://test:6655/health"),
        )
        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_response
        ):
            healthy, latency = await client.health_check()
            assert healthy
            assert latency >= 0

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, client):
        with patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            healthy, latency = await client.health_check()
            assert not healthy


class TestGenerateImageTool:
    @pytest.mark.asyncio
    async def test_generate_image_success(self, client, registry, tmp_path):
        import base64

        fake_b64 = base64.b64encode(b"fake image data").decode()
        mock_response = httpx.Response(
            200,
            json={"data": [{"b64_json": fake_b64}]},
            request=httpx.Request("POST", "http://test:6655/v1/images/generations"),
        )
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await generate_image(
                client=client,
                registry=registry,
                output_dir=str(tmp_path),
                prompt="A beautiful sunset",
                seed=42,
            )
            assert result["status"] == "success"
            assert "file_path" in result
            assert result["parameters_used"]["seed"] == 42
            assert result["parameters_used"]["steps"] == 4
            assert result["parameters_used"]["model"] == "black-forest-labs/FLUX.2-klein-4B"

    @pytest.mark.asyncio
    async def test_generate_image_server_unreachable(self, client, registry, tmp_path):
        with patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            result = await generate_image(
                client=client,
                registry=registry,
                output_dir=str(tmp_path),
                prompt="test",
            )
            assert result["status"] == "error"
            assert "Cannot connect" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_image_no_model_cached(self, tmp_path):
        client = VLLMClient(base_url="http://test:6655")
        registry = ModelRegistry()
        # Model query returns empty
        mock_models = httpx.Response(
            200,
            json={"data": []},
            request=httpx.Request("GET", "http://test:6655/v1/models"),
        )
        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_models
        ):
            result = await generate_image(
                client=client,
                registry=registry,
                output_dir=str(tmp_path),
                prompt="test",
            )
            assert result["status"] == "error"
            assert "Cannot determine" in result["error"]


class TestInfoTools:
    @pytest.mark.asyncio
    async def test_get_model_info(self, client, registry):
        result = await get_model_info(client, registry)
        assert result["model_name"] == "black-forest-labs/FLUX.2-klein-4B"
        assert result["model_type"] == "distilled"
        assert result["in_registry"] is True
        assert result["supports_negative_prompt"] is False
        assert "prompt_guidance" in result
        assert "natural language" in result["prompt_guidance"].lower()

    @pytest.mark.asyncio
    async def test_list_presets(self, client, registry):
        result = await list_presets(client, registry)
        assert result["model"] == "black-forest-labs/FLUX.2-klein-4B"
        assert "1:1" in result["presets"]
        assert result["presets"]["16:9"] == "1024x576"

    @pytest.mark.asyncio
    async def test_estimate_generation(self, client, registry):
        result = await estimate_generation(client, registry, num_images=5)
        assert "estimated_time_per_image_seconds" in result
        assert "estimated_total_time_seconds" in result
        assert result["recommended_concurrent"] >= 1


class TestSystemTools:
    @pytest.mark.asyncio
    async def test_server_health_healthy(self, client):
        mock_health = httpx.Response(
            200,
            text="OK",
            request=httpx.Request("GET", "http://test:6655/health"),
        )
        mock_models = httpx.Response(
            200,
            json={"data": [{"id": "test-model"}]},
            request=httpx.Request("GET", "http://test:6655/v1/models"),
        )
        mock_version = httpx.Response(
            200,
            json={"version": "0.14.0"},
            request=httpx.Request("GET", "http://test:6655/version"),
        )

        async def mock_get(url, **kwargs):
            if "/health" in url:
                return mock_health
            if "/v1/models" in url:
                return mock_models
            if "/version" in url:
                return mock_version
            return mock_health

        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, side_effect=mock_get
        ):
            result = await server_health(client)
            assert result["status"] == "healthy"
            assert result["model_loaded"] == "test-model"

    @pytest.mark.asyncio
    async def test_server_health_unreachable(self, client):
        with patch.object(
            httpx.AsyncClient,
            "get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            result = await server_health(client)
            assert result["status"] == "unreachable"

    @pytest.mark.asyncio
    async def test_gpu_status(self, client, registry):
        mock_health = httpx.Response(
            200,
            text="OK",
            request=httpx.Request("GET", "http://test:6655/health"),
        )
        with patch.object(
            httpx.AsyncClient, "get", new_callable=AsyncMock, return_value=mock_health
        ):
            result = await gpu_status(client, registry)
            assert result["server_status"] == "healthy"
            assert result["estimated_model_vram_gb"] == 13
            assert result["recommended_max_concurrent"] >= 1


class TestBatchTools:
    @pytest.mark.asyncio
    async def test_batch_empty_prompts(self, client, registry, tmp_path):
        result = await batch_generate(
            client=client,
            registry=registry,
            output_dir=str(tmp_path),
            prompts=[],
        )
        assert result["status"] == "error"
        assert "No prompts" in result["error"]

    @pytest.mark.asyncio
    async def test_batch_too_many_prompts(self, client, registry, tmp_path):
        result = await batch_generate(
            client=client,
            registry=registry,
            output_dir=str(tmp_path),
            prompts=["p"] * 21,
        )
        assert result["status"] == "error"
        assert "21" in result["error"]

    @pytest.mark.asyncio
    async def test_batch_seeds_length_mismatch(self, client, registry, tmp_path):
        result = await batch_generate(
            client=client,
            registry=registry,
            output_dir=str(tmp_path),
            prompts=["a", "b", "c"],
            seeds=[1, 2],
        )
        assert result["status"] == "error"
        assert "Seeds list length" in result["error"]

    @pytest.mark.asyncio
    async def test_batch_generate_success(self, client, registry, tmp_path):
        import base64

        fake_b64 = base64.b64encode(b"fake image").decode()
        mock_response = httpx.Response(
            200,
            json={"data": [{"b64_json": fake_b64}]},
            request=httpx.Request("POST", "http://test:6655/v1/images/generations"),
        )
        with patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await batch_generate(
                client=client,
                registry=registry,
                output_dir=str(tmp_path),
                prompts=["cat", "dog"],
                seeds=[1, 2],
            )
            assert result["status"] == "success"
            assert result["images_generated"] == 2
            assert result["images_failed"] == 0
            assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_batch(self):
        result = await cancel_batch("batch_nonexistent")
        assert result["status"] == "error"
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_check_batch_progress_not_found(self):
        result = await check_batch_progress("batch_nonexistent")
        assert result["status"] == "not_found"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_list_active_batches_empty(self):
        active_batches.clear()
        result = await list_active_batches()
        assert result["status"] == "success"
        assert result["active_batches"] == []

    @pytest.mark.asyncio
    async def test_check_batch_progress_during_batch(self):
        import asyncio

        batch_state = BatchState(
            batch_id="batch_test123",
            total=3,
            prompts=["cat", "dog", "bird"],
        )
        batch_state.completed_count = 1
        batch_state.failed_count = 0

        async def mock_done():
            return {"status": "success", "file_path": "/tmp/test.png"}

        async def mock_pending():
            await asyncio.sleep(100)

        task_done = asyncio.create_task(mock_done())
        await asyncio.sleep(0)  # Let it complete
        task_pending1 = asyncio.create_task(mock_pending())
        task_pending2 = asyncio.create_task(mock_pending())

        batch_state.tasks = [task_done, task_pending1, task_pending2]
        active_batches["batch_test123"] = batch_state

        result = await check_batch_progress("batch_test123")
        assert result["status"] == "in_progress"
        assert result["total"] == 3
        assert result["completed"] == 1
        assert result["in_progress"] == 2
        assert result["elapsed_seconds"] >= 0
        assert len(result["images"]) == 3

        # Cleanup
        task_pending1.cancel()
        task_pending2.cancel()
        active_batches.pop("batch_test123", None)
        try:
            await task_pending1
        except asyncio.CancelledError:
            pass
        try:
            await task_pending2
        except asyncio.CancelledError:
            pass
