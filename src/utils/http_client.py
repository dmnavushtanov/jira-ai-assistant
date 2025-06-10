"""Minimal HTTP client for testing API endpoints."""

from __future__ import annotations

from typing import Any, Dict
import logging
import requests

logger = logging.getLogger(__name__)


class SimpleHttpClient:
    """Thin wrapper around :mod:`requests` for basic API calls."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.session = requests.Session()
        logger.debug("SimpleHttpClient initialized with base_url=%s", self.base_url)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "SimpleHttpClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying :class:`requests.Session`."""
        self.session.close()

    def _build_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        """Send an HTTP request and return the :class:`Response`."""
        url = self._build_url(path)
        logger.debug("Performing %s request to %s with %s", method, url, kwargs)
        response = self.session.request(method, url, **kwargs)
        logger.info("%s %s -> %s", method, url, response.status_code)
        return response

    def get(self, path: str, params: Dict[str, Any] | None = None, **kwargs: Any) -> requests.Response:
        return self.request("GET", path, params=params, **kwargs)

    def post(self, path: str, data: Any | None = None, json: Any | None = None, **kwargs: Any) -> requests.Response:
        return self.request("POST", path, data=data, json=json, **kwargs)

    def put(self, path: str, data: Any | None = None, json: Any | None = None, **kwargs: Any) -> requests.Response:
        return self.request("PUT", path, data=data, json=json, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("DELETE", path, **kwargs)


__all__ = ["SimpleHttpClient"]
