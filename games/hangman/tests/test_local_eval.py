from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from hangman.local_eval import (
    DEFAULT_API_KEY_VAR,
    DEFAULT_FALLBACK_PORT,
    DEFAULT_STATE_COLUMNS,
    DEFAULT_STARTUP_TIMEOUT,
    _resolve_client_host,
    _resolve_port,
    build_prime_eval_command,
    build_server_command,
    merge_env_args,
    parse_args,
)


class LocalEvalTest(unittest.TestCase):
    def test_merge_env_args_overrides_difficulty(self) -> None:
        merged = merge_env_args({"difficulty": "hard", "seed": 7}, "easy", 6)
        self.assertEqual(merged, {"difficulty": "easy", "seed": 7, "num_examples": 6})

    def test_merge_env_args_preserves_explicit_num_examples(self) -> None:
        merged = merge_env_args({"seed": 7, "num_examples": 3}, "easy", 6)
        self.assertEqual(merged, {"difficulty": "easy", "seed": 7, "num_examples": 3})

    def test_build_mlx_server_command(self) -> None:
        with patch("hangman.local_eval._resolve_sibling_executable", return_value="/tmp/mlx_lm.server"):
            command = build_server_command(
                backend="mlx-lm",
                model="mlx-community/Qwen3-1.7B-4bit",
                host="127.0.0.1",
                port=8123,
                extra_args=["--trust-remote-code"],
            )

        self.assertEqual(
            command,
            [
                "/tmp/mlx_lm.server",
                "--model",
                "mlx-community/Qwen3-1.7B-4bit",
                "--host",
                "127.0.0.1",
                "--port",
                "8123",
                "--trust-remote-code",
            ],
        )

    def test_build_vllm_server_command(self) -> None:
        with patch("hangman.local_eval._resolve_sibling_executable", return_value="/tmp/vllm"):
            command = build_server_command(
                backend="vllm",
                model="Qwen/Qwen3-8B-Instruct",
                host="0.0.0.0",
                port=9000,
                extra_args=["--tensor-parallel-size", "2"],
            )

        self.assertEqual(
            command,
            [
                "/tmp/vllm",
                "serve",
                "Qwen/Qwen3-8B-Instruct",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--tensor-parallel-size",
                "2",
            ],
        )

    def test_build_prime_eval_command(self) -> None:
        command = build_prime_eval_command(
            prime_executable="/usr/local/bin/prime",
            env_id="hangman_agent",
            env_dir_path=Path("/workspace/environments"),
            model="mlx-community/Qwen3-1.7B-4bit",
            base_url="http://127.0.0.1:8123/v1",
            env_args={"difficulty": "easy", "seed": 3},
            num_examples=10,
            rollouts_per_example=4,
            max_concurrent=2,
            max_tokens=256,
            temperature=0.2,
            sampling_args={"top_p": 0.95},
            state_columns=["termination_reason", "total_reward"],
            save_results=True,
            skip_upload=True,
            debug=True,
            extra_prime_args=["--independent-scoring"],
        )

        self.assertEqual(
            command,
            [
                "/usr/local/bin/prime",
                "eval",
                "run",
                "hangman_agent",
                "--env-dir-path",
                "/workspace/environments",
                "--model",
                "mlx-community/Qwen3-1.7B-4bit",
                "--api-base-url",
                "http://127.0.0.1:8123/v1",
                "--api-key-var",
                DEFAULT_API_KEY_VAR,
                "--num-examples",
                "10",
                "--rollouts-per-example",
                "4",
                "--max-concurrent",
                "2",
                "--env-args",
                '{"difficulty":"easy","seed":3}',
                "--max-tokens",
                "256",
                "--temperature",
                "0.2",
                "--sampling-args",
                '{"top_p":0.95}',
                "--state-columns",
                "termination_reason,total_reward",
                "--save-results",
                "--skip-upload",
                "--debug",
                "--independent-scoring",
            ],
        )

    def test_resolve_client_host_prefers_loopback_for_wildcard_bind(self) -> None:
        self.assertEqual(_resolve_client_host("0.0.0.0"), "127.0.0.1")
        self.assertEqual(_resolve_client_host("127.0.0.1"), "127.0.0.1")

    def test_resolve_port_falls_back_when_socket_bind_is_unavailable(self) -> None:
        fake_socket = MagicMock()
        fake_socket.__enter__.return_value = fake_socket
        fake_socket.bind.side_effect = OSError("denied")
        with patch("hangman.local_eval.socket.socket", return_value=fake_socket):
            port = _resolve_port(None, "127.0.0.1")

        self.assertEqual(port, DEFAULT_FALLBACK_PORT)

    def test_parse_args_uses_long_startup_timeout_by_default(self) -> None:
        args = parse_args(["--model", "mlx-community/Qwen3.5-0.8B-MLX-4bit"])
        self.assertEqual(args.startup_timeout, DEFAULT_STARTUP_TIMEOUT)
        self.assertEqual(args.state_columns, ",".join(DEFAULT_STATE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
