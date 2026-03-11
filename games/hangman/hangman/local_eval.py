from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_ID = "hangman_agent"
DEFAULT_ENV_DIR = ROOT / "environments"
DEFAULT_STATE_COLUMNS = (
    "termination_reason",
    "last_outcome",
    "total_reward",
    "rollout_trace",
)
DEFAULT_API_KEY_VAR = "LOCAL_LLM_API_KEY"
DEFAULT_FALLBACK_PORT = 8080
DEFAULT_STARTUP_TIMEOUT = 900.0


class LocalEvalError(RuntimeError):
    """Raised when the local eval wrapper cannot complete its setup."""


@dataclass(frozen=True)
class ServerLaunch:
    backend: str
    command: list[str]
    base_url: str
    log_path: Path


def _json_dict(raw_value: str, *, flag_name: str) -> dict:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{flag_name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{flag_name} must decode to a JSON object")
    return parsed


def _resolve_port(port: int | None, host: str) -> int:
    if port is not None:
        return port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])
    except OSError:
        return DEFAULT_FALLBACK_PORT


def _resolve_client_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def _resolve_sibling_executable(name: str) -> str | None:
    candidate = Path(sys.executable).with_name(name)
    if candidate.exists():
        return str(candidate)
    return shutil.which(name)


def build_server_command(
    *,
    backend: str,
    model: str,
    host: str,
    port: int,
    extra_args: Sequence[str],
) -> list[str]:
    if backend == "mlx-lm":
        executable = _resolve_sibling_executable("mlx_lm.server")
        if executable is None:
            raise LocalEvalError(
                "The `mlx_lm.server` executable is not available in the current environment."
            )
        return [
            executable,
            "--model",
            model,
            "--host",
            host,
            "--port",
            str(port),
            *extra_args,
        ]

    if backend == "vllm":
        executable = _resolve_sibling_executable("vllm")
        if executable is None:
            raise LocalEvalError(
                "The `vllm` CLI is not installed. Install it, or use `--backend mlx-lm` in this workspace."
            )
        return [executable, "serve", model, "--host", host, "--port", str(port), *extra_args]

    raise LocalEvalError(f"Unsupported backend: {backend}")


def merge_env_args(
    raw_env_args: dict,
    difficulty: str | None,
    num_examples: int | None = None,
) -> dict:
    env_args = dict(raw_env_args)
    if difficulty is not None:
        env_args["difficulty"] = difficulty
    if num_examples is not None and "num_examples" not in env_args:
        env_args["num_examples"] = int(num_examples)
    return env_args


def build_prime_eval_command(
    *,
    prime_executable: str,
    env_id: str,
    env_dir_path: Path,
    model: str,
    base_url: str,
    env_args: dict,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    max_tokens: int | None,
    temperature: float | None,
    sampling_args: dict | None,
    state_columns: Sequence[str],
    save_results: bool,
    skip_upload: bool,
    debug: bool,
    extra_prime_args: Sequence[str],
) -> list[str]:
    command = [
        prime_executable,
        "eval",
        "run",
        env_id,
        "--env-dir-path",
        str(env_dir_path),
        "--model",
        model,
        "--api-base-url",
        base_url,
        "--api-key-var",
        DEFAULT_API_KEY_VAR,
        "--num-examples",
        str(num_examples),
        "--rollouts-per-example",
        str(rollouts_per_example),
        "--max-concurrent",
        str(max_concurrent),
        "--env-args",
        json.dumps(env_args, separators=(",", ":")),
    ]

    if max_tokens is not None:
        command.extend(["--max-tokens", str(max_tokens)])
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    if sampling_args:
        command.extend(
            ["--sampling-args", json.dumps(sampling_args, separators=(",", ":"))]
        )
    if state_columns:
        command.extend(["--state-columns", ",".join(state_columns)])
    if save_results:
        command.append("--save-results")
    if skip_upload:
        command.append("--skip-upload")
    if debug:
        command.append("--debug")
    command.extend(extra_prime_args)
    return command


def _healthcheck_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def server_is_ready(base_url: str, timeout: float = 2.0) -> bool:
    request = Request(_healthcheck_url(base_url), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(getattr(response, "status", 200)) == 200
    except HTTPError as exc:
        return exc.code == 200
    except URLError:
        return False
    except OSError:
        return False


def _tail_log(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    contents = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(contents[-lines:])


def wait_for_server(
    *,
    base_url: str,
    process: subprocess.Popen[str],
    log_path: Path,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server_is_ready(base_url):
            return
        if process.poll() is not None:
            tail = _tail_log(log_path)
            message = (
                f"The local {process.args[0]!r} process exited before the server became ready."
            )
            if tail:
                message += f"\n\nLast server log lines:\n{tail}"
            raise LocalEvalError(message)
        time.sleep(0.5)

    tail = _tail_log(log_path)
    message = f"Timed out waiting for the local server at {_healthcheck_url(base_url)}."
    if tail:
        message += f"\n\nLast server log lines:\n{tail}"
    raise LocalEvalError(message)


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def _default_log_path(backend: str) -> Path:
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return logs_dir / f"{backend}-server-{timestamp}.log"


def prepare_server_launch(args: argparse.Namespace) -> ServerLaunch:
    port = _resolve_port(args.port, args.host)
    base_url = f"http://{_resolve_client_host(args.host)}:{port}/v1"
    server_command = build_server_command(
        backend=args.backend,
        model=args.model,
        host=args.host,
        port=port,
        extra_args=args.server_arg,
    )
    log_path = Path(args.server_log_path) if args.server_log_path else _default_log_path(args.backend)
    return ServerLaunch(
        backend=args.backend,
        command=server_command,
        base_url=base_url,
        log_path=log_path,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start a local OpenAI-compatible inference server (`mlx-lm` or `vllm`) "
            "and run `prime eval run` against it."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("mlx-lm", "vllm"),
        default="mlx-lm",
        help="Local server backend to launch.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id or local path to serve through the selected backend.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local inference server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for the local inference server. Defaults to a free port.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT,
        help=(
            "Seconds to wait for the local server to expose /v1/models. "
            "The default is intentionally high because a first MLX launch may "
            "download model weights before the server becomes ready."
        ),
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="Leave the local model server running after `prime eval run` exits.",
    )
    parser.add_argument(
        "--server-log-path",
        help="Optional path for the background server log file.",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Repeatable extra flag passed directly to the backend server command.",
    )
    parser.add_argument(
        "--env-id",
        default=DEFAULT_ENV_ID,
        help="Environment id to evaluate.",
    )
    parser.add_argument(
        "--env-dir-path",
        default=str(DEFAULT_ENV_DIR),
        help="Path to the local environments directory.",
    )
    parser.add_argument(
        "--difficulty",
        default="easy",
        help="Convenience override for hangman difficulty. Use --env-args for more.",
    )
    parser.add_argument(
        "--env-args",
        type=lambda value: _json_dict(value, flag_name="--env-args"),
        default={},
        help="Additional environment args as a JSON object.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=6,
        help="Number of eval examples.",
    )
    parser.add_argument(
        "--rollouts-per-example",
        type=int,
        default=2,
        help="Number of rollouts per example.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum concurrent requests sent to the local server.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum generation tokens passed to `prime eval run`.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature passed to `prime eval run`.",
    )
    parser.add_argument(
        "--sampling-args",
        type=lambda value: _json_dict(value, flag_name="--sampling-args"),
        default=None,
        help="Sampling args JSON passed through to `prime eval run`.",
    )
    parser.add_argument(
        "--state-columns",
        default=",".join(DEFAULT_STATE_COLUMNS),
        help="Comma-separated list of state columns to save.",
    )
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save eval results locally.",
    )
    parser.add_argument(
        "--skip-upload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to skip uploading eval results to the platform.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass --debug to `prime eval run`.",
    )
    parser.add_argument(
        "--prime-arg",
        action="append",
        default=[],
        help="Repeatable extra flag passed directly to `prime eval run`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the server and eval commands without executing them.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    prime_executable = shutil.which("prime")
    if prime_executable is None:
        raise LocalEvalError("The `prime` CLI is not on PATH.")

    env_args = merge_env_args(args.env_args, args.difficulty, args.num_examples)
    launch = prepare_server_launch(args)
    state_columns = [value for value in args.state_columns.split(",") if value]
    prime_command = build_prime_eval_command(
        prime_executable=prime_executable,
        env_id=args.env_id,
        env_dir_path=Path(args.env_dir_path).resolve(),
        model=args.model,
        base_url=launch.base_url,
        env_args=env_args,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent=args.max_concurrent,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sampling_args=args.sampling_args,
        state_columns=state_columns,
        save_results=args.save_results,
        skip_upload=args.skip_upload,
        debug=args.debug,
        extra_prime_args=args.prime_arg,
    )

    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Base URL: {launch.base_url}")
    print(f"Server log: {launch.log_path}")
    print(f"Server command: {shlex.join(launch.command)}")
    print(f"Eval command:   {shlex.join(prime_command)}")

    if args.dry_run:
        return 0

    server_process: subprocess.Popen[str] | None = None
    reused_existing = server_is_ready(launch.base_url)

    if reused_existing:
        print("Using existing local inference server.")
    else:
        launch.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = launch.log_path.open("w", encoding="utf-8")
        try:
            server_process = subprocess.Popen(
                launch.command,
                cwd=str(ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        finally:
            log_handle.close()

        print("Starting local inference server...")
        print(
            f"Waiting up to {args.startup_timeout:.0f}s for {launch.base_url}/models ..."
        )
        try:
            wait_for_server(
                base_url=launch.base_url,
                process=server_process,
                log_path=launch.log_path,
                timeout_seconds=args.startup_timeout,
            )
        except Exception:
            if server_process is not None and not args.keep_server:
                print("Stopping local inference server after readiness failure...")
                terminate_process_group(server_process)
            raise
        print("Local inference server is ready.")

    try:
        result = subprocess.run(prime_command, cwd=str(ROOT), check=False)
        return int(result.returncode)
    finally:
        if server_process is not None and not args.keep_server:
            print("Stopping local inference server...")
            terminate_process_group(server_process)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LocalEvalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
