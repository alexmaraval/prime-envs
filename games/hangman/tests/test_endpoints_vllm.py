from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "endpoints.vllm.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("test_endpoints_vllm", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EndpointsVllmTest(unittest.TestCase):
    def test_skips_alias_when_required_env_vars_are_missing(self) -> None:
        with patch.dict(
            os.environ,
            {"LOCAL_VLLM_MODEL": "", "LOCAL_VLLM_BASE_URL": ""},
            clear=False,
        ):
            module = load_module()

        self.assertEqual(module.ENDPOINTS, {})

    def test_normalizes_bare_host_to_http_v1_url(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LOCAL_VLLM_MODEL": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                "LOCAL_VLLM_BASE_URL": "127.0.0.1:8000",
            },
            clear=False,
        ):
            module = load_module()

        self.assertEqual(
            module.ENDPOINTS,
            {
                "local-vllm": {
                    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                    "url": "http://127.0.0.1:8000/v1",
                    "key": "LOCAL_VLLM_API_KEY",
                }
            },
        )

    def test_preserves_existing_api_path(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LOCAL_VLLM_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
                "LOCAL_VLLM_BASE_URL": "https://host.example/internal/openai/v1",
            },
            clear=False,
        ):
            module = load_module()

        self.assertEqual(
            module.ENDPOINTS["local-vllm"]["url"],
            "https://host.example/internal/openai/v1",
        )


if __name__ == "__main__":
    unittest.main()
