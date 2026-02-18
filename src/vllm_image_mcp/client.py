"""Async HTTP client for the vLLM-Omni image generation API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for communicating with a vLLM-Omni compatible server."""

    def __init__(self, base_url: str, timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0)
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def generate_image(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send an image generation request to /v1/images/generations.

        Args:
            payload: Request body matching the OpenAI images API format.

        Returns:
            The JSON response from the server.

        Raises:
            httpx.HTTPStatusError: If the server returns an error status.
            httpx.ConnectError: If the server is unreachable.
        """
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/v1/images/generations",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def get_models(self) -> dict[str, Any]:
        """Get the list of loaded models from /v1/models.

        Returns:
            The JSON response containing model information.
        """
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()

    async def get_loaded_model_name(self) -> str | None:
        """Get the name of the currently loaded model.

        Returns:
            The model ID string, or None if no model is found.
        """
        try:
            data = await self.get_models()
            models = data.get("data", [])
            if models:
                return models[0].get("id")
        except (httpx.HTTPError, KeyError, IndexError):
            logger.exception("Failed to get loaded model name")
        return None

    async def health_check(self) -> tuple[bool, float]:
        """Check server health via /health endpoint.

        Returns:
            Tuple of (is_healthy, latency_ms).
        """
        client = await self._get_client()
        try:
            import time

            start = time.monotonic()
            response = await client.get(f"{self.base_url}/health")
            latency_ms = (time.monotonic() - start) * 1000
            return response.status_code == 200, latency_ms
        except httpx.HTTPError:
            return False, 0.0

    async def get_server_version(self) -> str | None:
        """Try to get the server version from /version endpoint."""
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/version")
            if response.status_code == 200:
                data = response.json()
                return data.get("version")
        except (httpx.HTTPError, ValueError):
            pass
        return None
