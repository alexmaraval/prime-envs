from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "configs" / "endpoints.mlx.py"


def load_module():
    spec = importlib.util.spec_from_file_location("test_endpoints_mlx", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EndpointsMlxTest(unittest.TestCase):
    def test_skips_alias_when_required_env_vars_are_missing(self) -> None:
        with patch.dict(
            os.environ,
            {"LOCAL_MLX_MODEL": "", "LOCAL_MLX_BASE_URL": ""},
            clear=False,
        ):
            module = load_module()

        self.assertEqual(module.ENDPOINTS, {})

    def test_normalizes_bare_host_to_http_v1_url(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LOCAL_MLX_MODEL": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
                "LOCAL_MLX_BASE_URL": "127.0.0.1:8080",
            },
            clear=False,
        ):
            module = load_module()

        self.assertEqual(
            module.ENDPOINTS,
            {
                "local-mlx": {
                    "model": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
                    "url": "http://127.0.0.1:8080/v1",
                    "key": "LOCAL_LLM_API_KEY",
                }
            },
        )

    def test_preserves_existing_api_path(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LOCAL_MLX_MODEL": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
                "LOCAL_MLX_BASE_URL": "https://host.example/internal/openai/v1",
            },
            clear=False,
        ):
            module = load_module()

        self.assertEqual(
            module.ENDPOINTS["local-mlx"]["url"],
            "https://host.example/internal/openai/v1",
        )


if __name__ == "__main__":
    unittest.main()
