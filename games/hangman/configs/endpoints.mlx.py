"""Endpoint aliases for a local OpenAI-compatible mlx_lm.server instance."""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit


def _normalize_base_url(raw_url: str) -> str:
    candidate = raw_url.strip()
    if not candidate:
        raise ValueError("LOCAL_MLX_BASE_URL must be non-empty")

    if "://" not in candidate:
        candidate = f"http://{candidate}"

    parts = urlsplit(candidate)
    if not parts.scheme or not parts.netloc:
        raise ValueError(
            "LOCAL_MLX_BASE_URL must be a host[:port] or a full http(s) URL"
        )

    path = parts.path.rstrip("/")
    if not path:
        path = "/v1"

    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def _build_local_mlx_endpoint() -> dict[str, dict[str, str]]:
    model = os.getenv("LOCAL_MLX_MODEL", "").strip()
    base_url = os.getenv("LOCAL_MLX_BASE_URL", "").strip()

    if not model or not base_url:
        return {}

    return {
        "local-mlx": {
            "model": model,
            "url": _normalize_base_url(base_url),
            "key": "LOCAL_LLM_API_KEY",
        }
    }


ENDPOINTS = _build_local_mlx_endpoint()
