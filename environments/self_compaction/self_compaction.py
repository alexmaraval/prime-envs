from __future__ import annotations

import asyncio
import json
import logging
import pprint
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import tenacity as tc
import verifiers as vf
from datasets import Dataset, load_dataset
from prime_sandboxes import (
    CommandTimeoutError,
    SandboxImagePullError,
    SandboxOOMError,
    SandboxTimeoutError,
)
from requests.exceptions import ConnectionError as RequestsConnectionError
from verifiers.envs.experimental.sandbox_mixin import (
    is_retryable_sandbox_api_error,
    is_retryable_sandbox_read_error,
)

from utils.execution_log_parser import decolor_dict_keys, parse_log_fn
from utils.prompts import (
    ACTION_OBSERVATION_TEMPLATE,
    FORMAT_ERROR_TEMPLATE,
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    render_template,
)

logging.root.handlers.clear()

TOOLS_DIR = Path(__file__).resolve().parent / "tools"
EXECUTE_BASH = TOOLS_DIR / "execute_bash.py"
SEARCH_FILES = TOOLS_DIR / "search_files.py"
READ_FILE = TOOLS_DIR / "read_file.py"
STR_REPLACE = TOOLS_DIR / "str_replace.py"

PATH_SWEBENCH = (
    "PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:"
    "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
PATH_R2E = (
    "PATH=/testbed/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:"
    "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
ENV_VARS_SWEBENCH = (
    f"export {PATH_SWEBENCH} PAGER=cat MANPAGER=cat LESS=-R "
    "PIP_PROGRESS_BAR=off TQDM_DISABLE=1;"
)
ENV_VARS_R2E = (
    f"export {PATH_R2E} PAGER=cat MANPAGER=cat LESS=-R "
    "PIP_PROGRESS_BAR=off TQDM_DISABLE=1;"
)

DEFAULT_DATASET_NAME = "R2E-Gym/R2E-Gym-Subset"
DEFAULT_MAX_TURNS = 200
DEFAULT_MAX_SUMMARY_CHARS = 6000


def _make_test_spec(info: dict[str, Any], namespace: str = "swebench") -> Any:
    """Lazy SWE-bench test spec construction with retry for GitHub fetches."""
    from swebench.harness.test_spec.test_spec import make_test_spec

    @tc.retry(
        retry=tc.retry_if_exception_type(RequestsConnectionError),
        stop=tc.stop_after_attempt(5),
        wait=tc.wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def make_with_retry() -> Any:
        return make_test_spec(info, namespace=namespace)

    return make_with_retry()


def _get_logs_eval(test_spec: Any, content: str) -> tuple[dict[str, str], bool]:
    """Lazy import of SWE-bench log parsing to keep R2E imports lightweight."""
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL,
        MAP_REPO_VERSION_TO_SPECS,
        RESET_FAILED,
        TESTS_ERROR,
        TESTS_TIMEOUT,
    )
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

    repo = test_spec.repo
    version = test_spec.version
    log_parser = MAP_REPO_TO_PARSER[repo]
    test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    if isinstance(test_cmd, list):
        test_cmd = test_cmd[-1]

    bad_codes = [
        code
        for code in (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT)
        if code in content
    ]
    if bad_codes:
        return {}, False
    return log_parser(content.split(test_cmd)[-1], test_spec), True


def _process_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": PROMPT_TEMPLATE.format(
            problem_statement=example["problem_statement"]
        ),
        "info": {**example},
        "answer": "",
    }


def _passes_simple_filter(example: dict[str, Any]) -> bool:
    file_fields = (
        "num_non_test_files",
        "non_test_files_count",
        "num_modified_non_test_files",
    )
    line_fields = (
        "num_non_test_lines",
        "non_test_lines_count",
        "num_modified_non_test_lines",
    )
    for field in file_fields:
        if field in example and example[field] is not None:
            try:
                if int(example[field]) > 1:
                    return False
            except (TypeError, ValueError):
                pass
    for field in line_fields:
        if field in example and example[field] is not None:
            try:
                if int(example[field]) > 80:
                    return False
            except (TypeError, ValueError):
                pass
    return True


def _default_split(dataset_name: str) -> str:
    lowered = dataset_name.lower()
    if lowered.startswith("r2e-gym/r2e-gym"):
        return "train"
    return "test" if "bench" in lowered else "train"


def _harness_for_dataset(dataset_name: str) -> str:
    return "r2e" if dataset_name.lower().startswith("r2e-gym/") else "swebench"


def _json_tool_message(tool_call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": json.dumps(payload, ensure_ascii=True, sort_keys=True),
        "tool_call_id": tool_call_id,
    }


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=True, sort_keys=True)


def _plain_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return dict(message)


def _parse_tool_call(tool_call: Any) -> tuple[str, dict[str, Any], str]:
    if isinstance(tool_call, dict):
        function_payload = tool_call.get("function") or {}
        name = function_payload.get("name") or tool_call.get("name") or ""
        raw_arguments = function_payload.get("arguments", tool_call.get("arguments", {}))
        call_id = str(tool_call.get("id") or "")
    else:
        name = str(getattr(tool_call, "name", ""))
        raw_arguments = getattr(tool_call, "arguments", "{}")
        call_id = str(getattr(tool_call, "id", "") or "")

    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        arguments = json.loads(str(raw_arguments or "{}"))
    if not isinstance(arguments, dict):
        raise ValueError(
            f"Expected tool arguments to be an object, got {type(arguments).__name__}"
        )
    return str(name), arguments, call_id


def _tool_call_names(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for raw_message in messages:
        message = _plain_message(raw_message)
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            try:
                name, _arguments, _call_id = _parse_tool_call(tool_call)
            except Exception:
                continue
            if name:
                names.append(name)
    return names


def _concat_messages(parts: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for part in parts:
        messages.extend(_plain_message(message) for message in part)
    return messages


class SelfCompactionMonitorRubric(vf.Rubric):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_metric(self.submitted)
        self.add_metric(self.submitted_after_compact)
        self.add_metric(self.compaction_count)
        self.add_metric(self.turns_before_first_compact)
        self.add_metric(self.command_timeout_count)
        self.add_metric(self.rollout_duration_seconds)
        self.add_metric(self.sandbox_oom)
        self.add_metric(self.sandbox_timeout)
        self.add_metric(self.sandbox_image_pull_error)
        self.add_metric(self.patch_broke_tests)

    async def submitted(self, state: vf.State) -> int:
        return int(bool(state.get("submitted")))

    async def submitted_after_compact(self, state: vf.State) -> int:
        return int(bool(state.get("submitted_after_compact")))

    async def compaction_count(self, state: vf.State) -> int:
        return int(state.get("compaction_count", 0))

    async def turns_before_first_compact(self, state: vf.State) -> int:
        value = state.get("turns_before_first_compact")
        return -1 if value is None else int(value)

    async def command_timeout_count(self, state: vf.State) -> int:
        return int(state.get("command_timeout_count", 0))

    async def rollout_duration_seconds(self, state: vf.State) -> float:
        return float(time.time() - state["timing"]["start_time"])

    async def sandbox_oom(self, state: vf.State) -> int:
        return int(bool(state.get("sandbox_oom")))

    async def sandbox_timeout(self, state: vf.State) -> int:
        return int(bool(state.get("sandbox_timeout")))

    async def sandbox_image_pull_error(self, state: vf.State) -> int:
        return int(bool(state.get("sandbox_image_pull_error")))

    async def patch_broke_tests(self, state: vf.State) -> int:
        return int(bool(state.get("patch_broke_tests")))


class SelfCompactionToolMonitorRubric(vf.Rubric):
    TOOL_NAMES = (
        "execute_bash",
        "search_files",
        "read",
        "edit_via_str_replace",
        "compact",
        "submit",
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_metric(self.total_tool_calls)
        for tool_name in self.TOOL_NAMES:
            self.add_metric(self._tool_count_metric(tool_name))

    def _names(self, state: vf.State) -> list[str]:
        messages = state.get("full_rollout_messages")
        if not isinstance(messages, list):
            messages = state.get("completion", [])
        return _tool_call_names(messages if isinstance(messages, list) else [])

    async def total_tool_calls(self, state: vf.State) -> int:
        return len(self._names(state))

    def _tool_count_metric(self, tool_name: str) -> Any:
        async def count_tool(state: vf.State) -> int:
            return self._names(state).count(tool_name)

        count_tool.__name__ = f"{tool_name}_calls"
        return count_tool


class SelfCompactionSandboxEnv(vf.SandboxEnv):
    def __init__(
        self,
        dataset: Any,
        system_prompt: str,
        parser: vf.Parser,
        rubric: vf.Rubric,
        max_turns: int = DEFAULT_MAX_TURNS,
        sandbox_command_timeout: int = 90,
        test_timeout: int = 900,
        total_timeout_minutes: int = 360,
        harness: str = "r2e",
        cpu_cores: int = 4,
        memory_gb: int = 4,
        disk_size_gb: int = 2,
        labels: list[str] | None = None,
        sandbox_client_max_workers: int = 10,
        max_retries: int = 3,
        rollout_timeout_seconds: float = 5400.0,
        max_command_timeouts: int = 5,
        allow_git: bool = False,
        skip_swebench_install: bool = True,
        min_compactions: int = 1,
        max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
        logger: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset=dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            sandbox_name="self-compaction-sandbox",
            start_command="tail -f /dev/null",
            timeout_minutes=total_timeout_minutes,
            max_turns=max_turns,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            sandbox_client_max_workers=sandbox_client_max_workers,
            **kwargs,
        )
        if logger is not None:
            self.logger = logger

        self.sandbox_command_timeout = sandbox_command_timeout
        self.test_timeout = test_timeout
        self.repo_path = "/testbed"
        self.alt_path = "/root"
        self.harness = harness
        self.labels = labels or ["self-compaction"]
        self.max_retries = max_retries
        self.rollout_timeout_seconds = rollout_timeout_seconds
        self.max_command_timeouts = max_command_timeouts
        self.allow_git = allow_git
        self.skip_swebench_install = skip_swebench_install
        self.min_compactions = min_compactions
        self.max_summary_chars = max_summary_chars

        self._remove_inherited_tool_monitor_rubric()
        self.add_rubric(SelfCompactionMonitorRubric())
        self.add_rubric(SelfCompactionToolMonitorRubric())

        self.with_retry_on_connection_errors = tc.AsyncRetrying(
            retry=tc.retry_if_exception(is_retryable_sandbox_api_error),
            stop=tc.stop_after_attempt(max_retries),
            wait=tc.wait_exponential_jitter(initial=1, max=30),
            before_sleep=self._log_retry_with_sandbox_id,
            reraise=True,
        ).wraps
        self.with_retry_on_read_errors = tc.AsyncRetrying(
            retry=tc.retry_if_exception(is_retryable_sandbox_read_error),
            stop=tc.stop_after_attempt(max_retries),
            wait=tc.wait_exponential_jitter(initial=1, max=30),
            before_sleep=self._log_retry_with_sandbox_id,
            reraise=True,
        ).wraps

        self.remove_tool(self.bash)
        self.add_tool(self.execute_bash)
        self.add_tool(self.search_files)
        self.add_tool(self.read)
        self._relax_tool_schema("read", required=["path", "start_line", "end_line"])
        self.add_tool(self.edit_via_str_replace)
        self.add_tool(self.compact)
        self.add_tool(self.submit)

    async def no_tools_called(self, state: vf.State) -> bool:
        return False

    def _remove_inherited_tool_monitor_rubric(self) -> None:
        monitor = getattr(self, "tool_monitor_rubric", None)
        if monitor is None:
            return
        if isinstance(self.rubric, vf.RubricGroup):
            self.rubric.rubrics = [
                rubric for rubric in self.rubric.rubrics if rubric is not monitor
            ]
        elif self.rubric is monitor:
            self.rubric = vf.Rubric()

    def _relax_tool_schema(self, tool_name: str, required: list[str]) -> None:
        for tool in self.tool_defs:
            if getattr(tool, "name", None) != tool_name:
                continue
            parameters = getattr(tool, "parameters", None)
            if isinstance(parameters, dict):
                parameters["required"] = list(required)
            if hasattr(tool, "strict"):
                tool.strict = False
            return

    def _log_retry_with_sandbox_id(self, retry_state: Any) -> None:
        sandbox_id = "unknown"
        if "sandbox_id" in retry_state.kwargs:
            sandbox_id = retry_state.kwargs["sandbox_id"]
        elif retry_state.args:
            sandbox_id = retry_state.args[0]
        sleep_seconds = getattr(retry_state.next_action, "sleep", None)
        exception = retry_state.outcome.exception() if retry_state.outcome else None
        self.logger.warning(
            f"sandbox_id={sandbox_id} retrying sandbox API call "
            f"(attempt={retry_state.attempt_number}, sleep={sleep_seconds}s): "
            f"{repr(exception)}"
        )

    def _raise_sandbox_error(
        self, state: vf.State, command: str, error: Exception
    ) -> None:
        error_map = {
            SandboxOOMError: ("sandbox_oom", "Sandbox OOM", "Sandbox OOM killed"),
            SandboxTimeoutError: (
                "sandbox_timeout",
                "Sandbox timeout",
                "Sandbox timeout",
            ),
        }
        sandbox_id = state.get("sandbox_id", "unknown")
        for exc_type, (state_key, log_prefix, error_message) in error_map.items():
            if isinstance(error, exc_type):
                state[state_key] = True
                self.logger.warning(
                    f"sandbox_id={sandbox_id} {log_prefix} during {command=}"
                )
                raise vf.SandboxError(f"{error_message} (sandbox_id={sandbox_id})")
        raise error

    def _raise_retry_exhausted_sandbox_error(
        self, sandbox_id: str, action: str, error: Exception
    ) -> None:
        raise vf.SandboxError(
            f"{action} failed after {self.max_retries} attempts: {repr(error)} "
            f"(sandbox_id={sandbox_id})"
        ) from error

    def _env_vars(self) -> str:
        base = ENV_VARS_SWEBENCH if self.harness == "swebench" else ENV_VARS_R2E
        return f"export ALLOW_GIT=1; {base}" if self.allow_git else base

    async def _execute_command(
        self,
        state: vf.State,
        command: str,
        timeout: int = 90,
        working_dir: str | None = None,
    ) -> tuple[int, str]:
        sandbox_id = state.get("sandbox_id", "unknown")
        start = time.time()
        try:
            results = await self.with_retry_on_connection_errors(
                self.sandbox_client.execute_command
            )(
                state["sandbox_id"],
                command,
                timeout=timeout,
                working_dir=working_dir,
            )
        except (SandboxOOMError, SandboxTimeoutError) as exc:
            self._raise_sandbox_error(state, command, exc)
        except CommandTimeoutError:
            state["command_timeout_count"] = state.get("command_timeout_count", 0) + 1
            self.logger.warning(
                f"sandbox_id={sandbox_id} {command=} timed out after {timeout}s "
                f"(count={state['command_timeout_count']})"
            )
            state["sandbox_state"]["command_execution_times"].append(
                time.time() - start
            )
            return (
                -1,
                f"The last command <command>{command}</command> timed out and "
                "has been killed.\nPlease try another command and avoid "
                "interactive input.",
            )
        except Exception as exc:
            self.logger.error(
                f"sandbox_id={sandbox_id} {command=} failed: {repr(exc)}"
            )
            raise vf.SandboxError(
                f"{command=} failed: {repr(exc)} (sandbox_id={sandbox_id})"
            )

        stdout = results.stdout.strip()
        stderr = (results.stderr or "").strip()
        output_parts = [stdout] if stdout else []
        if stderr:
            output_parts.append(f"stderr:\n{stderr}")
        state["sandbox_state"]["command_execution_times"].append(time.time() - start)
        return results.exit_code, "\n".join(output_parts) if output_parts else "(no output)"

    async def execute_command_raise_on_exit_code(
        self,
        state: vf.State,
        command: str,
        working_dir: str | None = None,
        timeout: int = 90,
    ) -> Any:
        sandbox_id = state.get("sandbox_id", "unknown")
        try:
            results = await self.with_retry_on_connection_errors(
                self.sandbox_client.execute_command
            )(
                state["sandbox_id"],
                command,
                working_dir=working_dir,
                timeout=timeout,
            )
        except (SandboxOOMError, SandboxTimeoutError) as exc:
            self._raise_sandbox_error(state, command, exc)
        except CommandTimeoutError:
            state["command_timeout_count"] = state.get("command_timeout_count", 0) + 1
            raise vf.SandboxError(f"Command timeout (sandbox_id={sandbox_id})")
        except Exception as exc:
            raise vf.SandboxError(
                f"{command=} failed: {repr(exc)} (sandbox_id={sandbox_id})"
            )
        if results.exit_code != 0:
            raise RuntimeError(
                f"Error executing command: {command} {results.exit_code=} "
                f"{results.stdout=} {results.stderr=}"
            )
        return results

    async def execute_bash(self, command: str) -> str:
        """Execute a bash command in the repository sandbox.

        Args:
            command: The command and optional arguments to execute.
        """
        return "Internal error: execute_bash is dispatched by the environment."

    async def _execute_bash_impl(
        self,
        command: str | None = None,
        state: str | None = None,
        sandbox_command_timeout: int = 90,
        working_dir: str | None = None,
    ) -> str:
        args = ["-h"] if not command else ["--cmd", command]
        return await self.run_tool_script(
            EXECUTE_BASH.name,
            args,
            state=state,
            sandbox_command_timeout=sandbox_command_timeout,
            working_dir=working_dir,
        )

    async def search_files(self, pattern: str) -> str:
        """Search repository text files for a Python regex or literal pattern.

        Args:
            pattern: Pattern to search for. Python regex syntax is supported;
                invalid regexes are treated as literal text.
        """
        return "Internal error: search_files is dispatched by the environment."

    async def _search_files_impl(
        self,
        pattern: str | None = None,
        state: str | None = None,
        sandbox_command_timeout: int = 90,
        working_dir: str | None = None,
    ) -> str:
        args = ["-h"] if not pattern else ["--pattern", pattern]
        return await self.run_tool_script(
            SEARCH_FILES.name,
            args,
            state=state,
            sandbox_command_timeout=sandbox_command_timeout,
            working_dir=working_dir,
        )

    async def read(self, path: str, start_line: int, end_line: int) -> str:
        """Read a bounded slice of a repository text file.

        Args:
            path: Path to a text file under the repository root.
            start_line: 1-indexed line number where reading starts.
            end_line: 1-indexed inclusive line number where reading ends.
        """
        return "Internal error: read is dispatched by the environment."

    async def _read_impl(
        self,
        path: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        state: str | None = None,
        sandbox_command_timeout: int = 90,
        working_dir: str | None = None,
    ) -> str:
        if not path or start_line is None or end_line is None:
            return "Error: read requires path, start_line, and end_line."
        args = [path, "--start-line", str(start_line), "--end-line", str(end_line)]
        return await self.run_tool_script(
            READ_FILE.name,
            args,
            state=state,
            sandbox_command_timeout=sandbox_command_timeout,
            working_dir=working_dir,
        )

    async def edit_via_str_replace(self, path: str, old_str: str, new_str: str) -> str:
        """Replace exactly one occurrence of text in a file.

        Args:
            path: Path to the text file.
            old_str: Old string to replace.
            new_str: New string, or an empty string to delete.
        """
        return "Internal error: edit_via_str_replace is dispatched by the environment."

    async def _edit_via_str_replace_impl(
        self,
        path: str,
        old_str: str,
        new_str: str,
        context_lines: int = 3,
        encoding: str = "utf-8",
        backup_suffix: str = "",
        dry_run: bool = False,
        expand_tabs: bool = False,
        tabsize: int = 8,
        state: str | None = None,
        sandbox_command_timeout: int = 90,
        working_dir: str | None = None,
    ) -> str:
        """Replace exactly one occurrence of text in a file.

        Args:
            path: Path to the text file.
            old_str: Old string to replace.
            new_str: New string, or an empty string to delete.
            context_lines: Lines of context to show after a successful edit.
            encoding: File encoding.
            backup_suffix: Optional backup suffix.
            dry_run: Preview without modifying the file.
            expand_tabs: Expand tabs before matching.
            tabsize: Tab width when expanding tabs.
        """
        args = [str(path), old_str, new_str]
        if context_lines != 3:
            args.extend(["--context-lines", str(context_lines)])
        if encoding != "utf-8":
            args.extend(["--encoding", encoding])
        if backup_suffix:
            args.extend(["--backup-suffix", backup_suffix])
        if dry_run:
            args.append("--dry-run")
        if expand_tabs:
            args.append("--expand-tabs")
            if tabsize != 8:
                args.extend(["--tabsize", str(tabsize)])
        return await self.run_tool_script(
            STR_REPLACE.name,
            args,
            state=state,
            sandbox_command_timeout=sandbox_command_timeout,
            working_dir=working_dir,
        )

    async def compact(self, summary: str) -> str:
        """Replace the visible conversation history with this summary.

        Args:
            summary: A complete summary of all task-relevant facts needed to
                continue, including files inspected, edits made, tests run,
                failures, and next steps.
        """
        return "Internal error: compact is dispatched by the environment."

    async def _compact_impl(self, summary: str, state: str | None = None) -> str:
        assert isinstance(state, dict)
        cleaned = (summary or "").strip()
        if not cleaned:
            return "Error: summary must not be empty."
        if len(cleaned) > self.max_summary_chars:
            return (
                f"Error: summary is {len(cleaned)} characters, exceeding the "
                f"{self.max_summary_chars} character limit. Retry with a shorter "
                "summary."
            )
        if state.get("last_successful_tool_name") == "compact":
            return (
                "Error: the previous successful tool call was already compact. "
                "Continue working with another tool, or call submit if the patch "
                "is ready."
            )
        state["compaction_count"] = int(state.get("compaction_count", 0)) + 1
        state["last_compaction_summary"] = cleaned
        state["pending_compaction_summary"] = cleaned
        if state.get("turns_before_first_compact") is None:
            state["turns_before_first_compact"] = max(
                0, len(state.get("trajectory", [])) - 1
            )
        state.setdefault("compaction_summaries", []).append(
            {
                "turn": len(state.get("trajectory", [])),
                "summary": cleaned,
                "summary_chars": len(cleaned),
            }
        )
        return (
            "Context compacted. On the next turn, the original task prompt, this "
            "compact tool call/result, your summary, and future messages will "
            "remain visible."
        )

    async def submit(self) -> str:
        """Submit the current repository changes for hidden-test scoring."""
        return "Internal error: submit is dispatched by the environment."

    async def _submit_impl(self, state: str | None = None) -> str:
        assert isinstance(state, dict)
        compactions = int(state.get("compaction_count", 0))
        if compactions < self.min_compactions:
            return (
                f"Error: call compact at least {self.min_compactions} time(s) "
                "before submitting."
            )
        state["submitted"] = True
        state["submitted_after_compact"] = True
        state["agent_signaled_done"] = True
        return "Submission accepted. The environment will run hidden tests now."

    def _compatible_tool_args(
        self, tool_name: str, tool_args: dict[str, Any]
    ) -> dict[str, Any]:
        if tool_name == "edit_via_str_replace":
            return tool_args
        updated = dict(tool_args)
        aliases = {
            "execute_bash": {"cmd": "command"},
            "search_files": {"query": "pattern"},
            "read": {
                "file": "path",
                "filepath": "path",
                "filename": "path",
                "start": "start_line",
                "end": "end_line",
                "stop_line": "end_line",
            },
        }
        for source, target in aliases.get(tool_name, {}).items():
            if target not in updated and source in updated:
                updated[target] = updated[source]
        allowed = {
            "execute_bash": {"command"},
            "search_files": {"pattern"},
            "read": {"path", "start_line", "end_line"},
            "compact": {"summary"},
            "submit": set(),
        }.get(tool_name)
        if allowed is None:
            return updated
        return {key: value for key, value in updated.items() if key in allowed}

    def _internal_tool_args(
        self, tool_name: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        state = kwargs.get("state")
        if tool_name in {
            "execute_bash",
            "search_files",
            "read",
            "edit_via_str_replace",
        }:
            return {
                "state": state,
                "sandbox_command_timeout": self.sandbox_command_timeout,
                "working_dir": self.repo_path,
            }
        if tool_name in {"compact", "submit"}:
            return {"state": state}
        return {}

    async def call_tool(
        self, tool_name: str, tool_args: dict[str, Any], tool_call_id: str, **kwargs: Any
    ) -> vf.Message:
        dispatch = {
            "execute_bash": self._execute_bash_impl,
            "search_files": self._search_files_impl,
            "read": self._read_impl,
            "edit_via_str_replace": self._edit_via_str_replace_impl,
            "compact": self._compact_impl,
            "submit": self._submit_impl,
        }
        if tool_name not in dispatch:
            return await super().call_tool(tool_name, tool_args, tool_call_id, **kwargs)
        call_args = self._compatible_tool_args(tool_name, tool_args)
        call_args.update(self._internal_tool_args(tool_name, kwargs))
        result = await dispatch[tool_name](**call_args)
        return vf.ToolMessage(role="tool", content=str(result), tool_call_id=tool_call_id)

    async def run_tool_script(
        self,
        tool_name: str,
        args: list[str],
        state: vf.State,
        sandbox_command_timeout: int = 90,
        working_dir: str | None = None,
    ) -> str:
        cmd_parts = ["python", f"/sandbox-workspace/tools/{tool_name}", *args]
        quoted_parts = [shlex.quote(str(part)) for part in cmd_parts]
        command = f"{self._env_vars()} {' '.join(quoted_parts)}"
        exit_code, output = await self._execute_command(
            state, command, sandbox_command_timeout, working_dir=working_dir
        )
        if exit_code == -1:
            return output
        return render_template(
            ACTION_OBSERVATION_TEMPLATE, exit_code=exit_code, output=output
        )

    async def upload_tools(self, state: vf.State) -> None:
        upload = self.with_retry_on_read_errors(self.sandbox_client.upload_file)
        tasks = [
            upload(
                state["sandbox_id"],
                f"/sandbox-workspace/tools/{tool_path.name}",
                str(tool_path),
            )
            for tool_path in (EXECUTE_BASH, SEARCH_FILES, READ_FILE, STR_REPLACE)
        ]
        try:
            await asyncio.gather(*tasks)
        except Exception as exc:
            self._raise_retry_exhausted_sandbox_error(
                state.get("sandbox_id", "unknown"), "Tool upload", exc
            )

    async def setup_repo(self, state: vf.State) -> None:
        if self.harness == "swebench":
            await self.setup_repo_swebench(state)
        else:
            await self.setup_repo_r2e(state)

    async def setup_repo_swebench(self, state: vf.State) -> None:
        self.alt_path = "/"
        await self.execute_command_raise_on_exit_code(
            state, "ln -sfn /opt/miniconda3/envs/testbed /root/.venv"
        )

    async def download_and_remove_r2e_tests(
        self, state: vf.State, timeout: int = 300
    ) -> None:
        sandbox_id = state.get("sandbox_id", "unknown")
        remote_archive = "/tmp/r2e_tests.tar.gz"
        local_archive_path = str(Path("/tmp") / f"r2e_tests_{sandbox_id}.tar.gz")
        download = self.with_retry_on_read_errors(self.sandbox_client.download_file)
        await self.execute_command_raise_on_exit_code(
            state, f"tar -C / -czf {remote_archive} r2e_tests", timeout=timeout
        )
        try:
            await download(
                sandbox_id=sandbox_id,
                file_path=remote_archive,
                local_file_path=local_archive_path,
                timeout=timeout,
            )
        except Exception as exc:
            self._raise_retry_exhausted_sandbox_error(
                sandbox_id, "r2e_tests download", exc
            )
        state["r2e_tests_archive_local_path"] = local_archive_path
        await self.execute_command_raise_on_exit_code(
            state, "rm -rf /r2e_tests", timeout=timeout
        )
        await self.execute_command_raise_on_exit_code(
            state, f"rm -f {remote_archive}", timeout=timeout
        )

    async def setup_repo_r2e(self, state: vf.State) -> None:
        link_commands = [
            f"ln -sfn {self.repo_path}/.venv {self.alt_path}/.venv",
            f"ln -sfn {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python",
            f"ln -sfn {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3",
            f"find {self.repo_path}/.venv/bin -type f -executable -exec ln -sfn {{}} {self.alt_path}/.local/bin/ \\;",
        ]
        for command in link_commands:
            await self.execute_command_raise_on_exit_code(state, command)
        cleanup_commands = [
            (
                "timeout 30 bash -c 'shopt -s globstar; rm -rf **/*.pyc **/__pycache__' "
                "2>/dev/null || timeout 30 find . -name '*.pyc' -delete || true",
                self.repo_path,
            ),
            (
                "timeout 30 bash -c 'shopt -s globstar; rm -rf /r2e_tests/**/*.pyc "
                "/r2e_tests/**/__pycache__' 2>/dev/null || timeout 30 find /r2e_tests "
                "-name '*.pyc' -delete || true",
                None,
            ),
        ]
        for command, working_dir in cleanup_commands:
            try:
                await self.execute_command_raise_on_exit_code(
                    state, command, working_dir=working_dir
                )
            except Exception as exc:
                sandbox_id = state.get("sandbox_id", "unknown")
                self.logger.warning(
                    f"sandbox_id={sandbox_id} continuing after pycache cleanup "
                    f"failure: {repr(exc)}"
                )
        await self.download_and_remove_r2e_tests(state, timeout=300)

    def get_sandbox_request(self, state: vf.State) -> Any:
        if self.harness == "swebench":
            test_spec = _make_test_spec(state["info"], namespace="swebench")
            state["info"]["docker_image"] = test_spec.instance_image_key
        return self.sandbox_request.model_copy(
            update={
                "docker_image": (
                    "us-central1-docker.pkg.dev/prime-intellect-platform/"
                    f"prod-sandbox/{state['info']['docker_image']}"
                ),
                "labels": self.labels,
            }
        )

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state["submitted"] = False
        state["submitted_after_compact"] = False
        state["compaction_count"] = 0
        state["turns_before_first_compact"] = None
        state["compaction_summaries"] = []
        state["full_rollout_messages"] = [
            _plain_message(message) for message in state["prompt"]
        ]
        state["agent_visible_messages"] = [
            _plain_message(message) for message in state["prompt"]
        ]
        state["last_successful_tool_name"] = None

        request = self.get_sandbox_request(state)
        try:
            sandbox = await self.with_retry(self.sandbox_client.create)(request)
        except Exception as exc:
            raise vf.SandboxError(f"Sandbox creation failed: {repr(exc)}")

        self.active_sandboxes.add(sandbox.id)
        state["sandbox_id"] = sandbox.id
        state["sandbox_state"] = {
            "ready": False,
            "ready_wait_time": 0.0,
            "command_execution_times": [],
        }
        try:
            await self._wait_for_sandbox_ready(
                state["sandbox_state"], state["sandbox_id"]
            )
        except SandboxImagePullError as exc:
            state["sandbox_image_pull_error"] = True
            docker_image = state["info"].get("docker_image", "unknown")
            raise vf.SandboxError(
                f"Failed to pull sandbox image {docker_image=}: {repr(exc)} "
                f"(sandbox_id={state['sandbox_id']})"
            )

        try:
            await self.setup_repo(state)
            await self.upload_tools(state)
        except Exception as exc:
            docker_image = state["info"].get("docker_image", "unknown")
            sandbox_id = state.get("sandbox_id", "unknown")
            raise vf.SandboxError(
                f"Setup failed for {docker_image=}: {repr(exc)} "
                f"(sandbox_id={sandbox_id})"
            )
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        updated_args = dict(tool_args)
        if tool_name in {"execute_bash", "search_files", "edit_via_str_replace"}:
            updated_args["state"] = state
            updated_args["sandbox_command_timeout"] = self.sandbox_command_timeout
            updated_args["working_dir"] = self.repo_path
            if tool_name == "edit_via_str_replace":
                updated_args.setdefault("context_lines", 3)
                updated_args.setdefault("encoding", "utf-8")
                updated_args.setdefault("backup_suffix", "")
                updated_args.setdefault("dry_run", False)
                updated_args.setdefault("expand_tabs", False)
                updated_args.setdefault("tabsize", 8)
        elif tool_name in {"compact", "submit"}:
            updated_args["state"] = state
        return updated_args

    def _compacted_prompt(
        self,
        state: vf.State,
        summary: str,
        assistant_message: dict[str, Any] | None = None,
        env_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        original_prompt = [_plain_message(message) for message in state["prompt"]]
        compact_turn: list[dict[str, Any]] = []
        if assistant_message is not None:
            compact_turn.append(_plain_message(assistant_message))
        if env_messages:
            compact_turn.extend(_plain_message(message) for message in env_messages)
        summary_message = {
            "role": "user",
            "content": (
                "<compacted_context>\n"
                "The prior interaction history has been replaced by this summary. "
                "The original task prompt remains visible above; use this summary "
                "for everything learned or changed after the task began.\n\n"
                f"{summary}\n"
                "</compacted_context>"
            ),
        }
        return [*original_prompt, *compact_turn, summary_message]

    def _tool_message_is_error(self, message: dict[str, Any]) -> bool:
        content = _message_content(message).lstrip()
        if content.startswith("Error:"):
            return True
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict) and payload.get("status") == "error"

    def _append_audit_messages(
        self,
        state: vf.State,
        assistant_message: dict[str, Any],
        env_messages: list[dict[str, Any]],
    ) -> None:
        full = list(state.get("full_rollout_messages") or state["prompt"])
        full.append(_plain_message(assistant_message))
        full.extend(_plain_message(message) for message in env_messages)
        state["full_rollout_messages"] = full

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        if len(state["trajectory"]) == 0:
            prompt = [_plain_message(message) for message in state["prompt"]]
            state["agent_visible_messages"] = prompt
            state["full_rollout_messages"] = list(
                state.get("full_rollout_messages") or prompt
            )
            return prompt

        previous_prompt = [
            _plain_message(message) for message in state["trajectory"][-1]["prompt"]
        ]
        previous_completion = [
            _plain_message(message) for message in state["trajectory"][-1]["completion"]
        ]
        messages = _concat_messages([previous_prompt, previous_completion])
        env_messages = await self.env_response(messages, state)
        compacted_prompt = state.pop("pending_visible_prompt", None)
        if compacted_prompt is not None:
            state["agent_visible_messages"] = list(compacted_prompt)
            return compacted_prompt
        prompt = _concat_messages([messages, env_messages])
        state["agent_visible_messages"] = prompt
        return prompt

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        assert isinstance(messages, list)
        sandbox_id = state.get("sandbox_id", "unknown")
        assistant_message = _plain_message(messages[-1])
        tool_calls = assistant_message.get("tool_calls") or []
        env_messages: list[dict[str, Any]] = []

        if not tool_calls:
            env_messages.append(
                {
                    "role": "user",
                    "content": render_template(FORMAT_ERROR_TEMPLATE, actions=[]),
                }
            )
            self._append_audit_messages(state, assistant_message, env_messages)
            return env_messages

        if len(tool_calls) != 1:
            env_messages.append(
                {
                    "role": "user",
                    "content": render_template(FORMAT_ERROR_TEMPLATE, actions=tool_calls),
                }
            )
            self._append_audit_messages(state, assistant_message, env_messages)
            return env_messages

        tool_call = tool_calls[0]
        try:
            tool_name, tool_args, tool_call_id = _parse_tool_call(tool_call)
        except Exception as exc:
            tool_call_id = (
                str(tool_call.get("id", "invalid"))
                if isinstance(tool_call, dict)
                else "invalid"
            )
            env_messages.append(
                _json_tool_message(
                    tool_call_id,
                    {
                        "status": "error",
                        "message": f"Invalid tool call arguments: {exc}",
                    },
                )
            )
            self._append_audit_messages(state, assistant_message, env_messages)
            return env_messages

        try:
            tool_args = self.update_tool_args(
                tool_name, tool_args, messages, state, **kwargs
            )
            tool_message = await self.call_tool(
                tool_name, tool_args, tool_call_id, state=state
            )
        except vf.Error:
            raise
        except Exception as exc:
            tool_message = _json_tool_message(
                tool_call_id,
                {
                    "status": "error",
                    "message": f"Error executing tool '{tool_name}': {repr(exc)}",
                },
            )
            self.logger.warning(
                f"sandbox_id={sandbox_id} tool {tool_name!r} failed: {repr(exc)}"
            )

        tool_message_dict = _plain_message(tool_message)
        env_messages.append(tool_message_dict)

        if tool_name == "compact" and state.get("pending_compaction_summary"):
            summary = state.pop("pending_compaction_summary")
            state["pending_visible_prompt"] = self._compacted_prompt(
                state,
                summary,
                assistant_message=assistant_message,
                env_messages=env_messages,
            )

        if tool_name == "submit" and state.get("submitted"):
            state["final_env_response"] = env_messages

        if not self._tool_message_is_error(tool_message_dict):
            state["last_successful_tool_name"] = tool_name

        self._append_audit_messages(state, assistant_message, env_messages)
        trunc = pprint.pformat(env_messages)
        if len(trunc) > 2000:
            trunc = trunc[:1000] + "\n...\n" + trunc[-1000:]
        self.logger.debug(f"sandbox_id={sandbox_id} env response:\n{trunc}")
        return env_messages

    async def run_background_job(
        self,
        state: vf.State,
        command: str,
        timeout: int,
        working_dir: str | None = None,
        poll_interval: int = 3,
    ) -> Any:
        sandbox_id = state["sandbox_id"]
        start_job = self.with_retry_on_connection_errors(
            self.sandbox_client.start_background_job
        )
        get_job = self.with_retry_on_read_errors(self.sandbox_client.get_background_job)
        try:
            job = await start_job(
                sandbox_id=sandbox_id, command=command, working_dir=working_dir
            )
        except SandboxOOMError as exc:
            state["sandbox_oom"] = True
            raise vf.SandboxError(
                f"Sandbox OOM during background job: {repr(exc)} "
                f"(sandbox_id={sandbox_id})"
            )
        except SandboxTimeoutError as exc:
            state["sandbox_timeout"] = True
            raise vf.SandboxError(
                f"Sandbox timeout during background job: {repr(exc)} "
                f"(sandbox_id={sandbox_id})"
            )
        except (CommandTimeoutError, httpx.ReadTimeout) as exc:
            raise vf.SandboxError(
                f"Failed to start background job: {repr(exc)} "
                f"(sandbox_id={sandbox_id})"
            )
        except Exception as exc:
            raise vf.SandboxError(
                f"Failed to start background job after retries: {repr(exc)} "
                f"(sandbox_id={sandbox_id})"
            ) from exc

        try:
            for _elapsed in range(0, timeout + poll_interval, poll_interval):
                results = await get_job(sandbox_id, job)
                if results.completed:
                    return results
                await asyncio.sleep(poll_interval)
        except SandboxOOMError as exc:
            state["sandbox_oom"] = True
            raise vf.SandboxError(
                f"Sandbox OOM during polling: {repr(exc)} (sandbox_id={sandbox_id})"
            )
        except SandboxTimeoutError as exc:
            state["sandbox_timeout"] = True
            raise vf.SandboxError(
                f"Sandbox timeout during polling: {repr(exc)} "
                f"(sandbox_id={sandbox_id})"
            )
        except (CommandTimeoutError, httpx.ReadTimeout) as exc:
            raise vf.SandboxError(
                f"Failed to poll background job due to timeout: {repr(exc)} "
                f"(sandbox_id={sandbox_id})"
            )
        except Exception as exc:
            raise vf.SandboxError(
                f"Failed to poll background job after retries: {repr(exc)} "
                f"(sandbox_id={sandbox_id})"
            ) from exc
        raise CommandTimeoutError(sandbox_id=sandbox_id, command=command, timeout=timeout)

    async def _read_test_output_tail(
        self, state: vf.State, path: str, max_chars: int = 4000
    ) -> str:
        try:
            results = await self.sandbox_client.execute_command(
                state["sandbox_id"],
                f"tail -c {int(max_chars)} {shlex.quote(path)} 2>/dev/null",
                timeout=self.sandbox_command_timeout,
            )
            return (results.stdout or "")[-max_chars:]
        except Exception:
            return ""

    _PATCH_BROKE_TESTS_EXIT_CODES = {
        2: "collection error",
        4: "usage error",
        5: "no tests collected",
    }

    def _mark_patch_broke_tests(
        self, state: vf.State, exit_code: int, tail: str, path: str
    ) -> None:
        kind = self._PATCH_BROKE_TESTS_EXIT_CODES[exit_code]
        state["patch_broke_tests"] = True
        state["patch_broke_tests_reason"] = (
            f"pytest exit_code={exit_code} ({kind}) from {path}"
        )
        state["patch_broke_tests_exit_code"] = exit_code
        state["patch_broke_tests_tail"] = tail

    async def run_tests_swebench(self, state: vf.State, test_timeout: int) -> str:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

        test_spec = _make_test_spec(state["info"], namespace="swebench")
        eval_script = test_spec.eval_script
        if self.skip_swebench_install:
            try:
                diff_cmd = "cd /testbed && git -c core.fileMode=false diff --name-only HEAD"
                diff_res = await self.execute_command_raise_on_exit_code(state, diff_cmd)
                changed = [
                    path.strip()
                    for path in diff_res.stdout.splitlines()
                    if path.strip()
                ]
                compiled_exts = {
                    ".pyx",
                    ".pxd",
                    ".pxi",
                    ".c",
                    ".cc",
                    ".cpp",
                    ".h",
                    ".hpp",
                    ".f",
                    ".f90",
                }
                build_files = {
                    "setup.py",
                    "setup.cfg",
                    "pyproject.toml",
                    "meson.build",
                    "CMakeLists.txt",
                }
                needs_rebuild = any(
                    Path(path).name in build_files
                    or "/_build_utils/" in path
                    or "/__check_build/" in path
                    or Path(path).suffix in compiled_exts
                    for path in changed
                )
                if not needs_rebuild:
                    specs = MAP_REPO_VERSION_TO_SPECS[test_spec.repo][test_spec.version]
                    install_cmd = specs.get("install")
                    if install_cmd:
                        eval_lines = [
                            line
                            for line in test_spec.eval_script_list
                            if line.strip() != install_cmd.strip()
                        ]
                        eval_script = (
                            "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_lines)
                            + "\n"
                        )
            except Exception as exc:
                self.logger.warning(
                    f"Could not decide whether to skip SWE-bench install: {exc!r}"
                )
        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w") as eval_file:
            eval_file.write(eval_script)
            eval_file.flush()
            upload = self.with_retry_on_read_errors(self.sandbox_client.upload_file)
            try:
                await upload(state["sandbox_id"], "/eval.sh", eval_file.name)
            except Exception as exc:
                self._raise_retry_exhausted_sandbox_error(
                    state.get("sandbox_id", "unknown"), "eval script upload", exc
                )
        await self.execute_command_raise_on_exit_code(state, "chmod +x /eval.sh")
        command = f"{self._env_vars()} /eval.sh > /test_output.txt 2>&1"
        results = await self.run_background_job(state, command, test_timeout)
        if results.exit_code in self._PATCH_BROKE_TESTS_EXIT_CODES:
            tail = await self._read_test_output_tail(state, "/test_output.txt")
            self._mark_patch_broke_tests(state, results.exit_code, tail, "/test_output.txt")
            return ""
        if results.exit_code > 1:
            tail = await self._read_test_output_tail(state, "/test_output.txt")
            raise RuntimeError(
                f"Error running tests: {results.exit_code=} {results.stdout=} "
                f"{results.stderr=} test_output_tail={tail!r}"
            )
        results = await self.sandbox_client.execute_command(
            state["sandbox_id"], "cat /test_output.txt"
        )
        return results.stdout

    async def upload_r2e_tests(self, state: vf.State, timeout: int = 300) -> None:
        sandbox_id = state.get("sandbox_id", "unknown")
        remote_archive = "/tmp/r2e_tests_roundtrip.tar.gz"
        upload = self.with_retry_on_read_errors(self.sandbox_client.upload_file)
        local_archive_path = state.get("r2e_tests_archive_local_path")
        if not local_archive_path or not Path(local_archive_path).exists():
            raise vf.SandboxError(
                f"Missing cached r2e_tests archive path (sandbox_id={sandbox_id})"
            )
        try:
            await upload(
                sandbox_id=sandbox_id,
                file_path=remote_archive,
                local_file_path=local_archive_path,
                timeout=timeout,
            )
        except Exception as exc:
            raise vf.SandboxError(
                f"Failed to upload r2e_tests archive: {repr(exc)} "
                f"(sandbox_id={sandbox_id})"
            )
        await self.execute_command_raise_on_exit_code(
            state, f"tar -C {self.repo_path} -xzf {remote_archive}", timeout=timeout
        )
        Path(local_archive_path).unlink(missing_ok=True)
        del state["r2e_tests_archive_local_path"]

    async def run_tests_r2e(self, state: vf.State, test_timeout: int) -> str:
        await self.upload_r2e_tests(state, timeout=300)
        command = f"{self._env_vars()} /bin/bash run_tests.sh > test_output.txt 2>&1"
        results = await self.run_background_job(
            state, command, test_timeout, working_dir="/testbed"
        )
        if results.exit_code in self._PATCH_BROKE_TESTS_EXIT_CODES:
            tail = await self._read_test_output_tail(state, "/testbed/test_output.txt")
            self._mark_patch_broke_tests(
                state, results.exit_code, tail, "/testbed/test_output.txt"
            )
            return ""
        if results.exit_code > 1:
            tail = await self._read_test_output_tail(state, "/testbed/test_output.txt")
            raise RuntimeError(
                f"Error running tests: {results.exit_code=} {results.stdout=} "
                f"{results.stderr=} test_output_tail={tail!r}"
            )
        results = await self.sandbox_client.execute_command(
            state["sandbox_id"], "cat /testbed/test_output.txt", timeout=test_timeout
        )
        return results.stdout

    async def run_tests(self, state: vf.State, test_timeout: int) -> str:
        if self.harness == "swebench":
            return await self.run_tests_swebench(state, test_timeout)
        return await self.run_tests_r2e(state, test_timeout)

    async def post_rollout(self, state: vf.State) -> None:
        sandbox_id = state.get("sandbox_id", "unknown")
        if not state.get("submitted_after_compact"):
            state["test_output"] = ""
            return
        if isinstance(state.get("error"), vf.InfraError):
            state["test_output"] = ""
            return
        try:
            state["test_output"] = await self.run_tests(
                state, test_timeout=self.test_timeout
            )
        except Exception as exc:
            state["test_output"] = ""
            state["error"] = vf.SandboxError(
                f"Error running tests: {repr(exc)} (sandbox_id={sandbox_id})"
            )

    @vf.stop(priority=80)
    async def agent_submitted(self, state: vf.State) -> bool:
        return bool(state.get("agent_signaled_done"))

    @vf.stop
    async def max_command_timeouts_reached(self, state: vf.State) -> bool:
        timeout_count = int(state.get("command_timeout_count", 0))
        if timeout_count >= self.max_command_timeouts:
            state["max_command_timeouts_reached"] = True
            return True
        return False

    @vf.stop
    async def rollout_timeout_reached(self, state: vf.State) -> bool:
        elapsed = time.time() - state["timing"]["start_time"]
        if elapsed > self.rollout_timeout_seconds:
            state["error"] = vf.InfraError(f"Rollout timeout after {elapsed:.0f}s")
            return True
        return False

    async def render_completion(self, state: vf.State) -> None:
        full = state.get("full_rollout_messages")
        visible = state.get("agent_visible_messages")
        if full is None:
            if not state.get("trajectory"):
                full = list(state.get("prompt", []))
            else:
                last_prompt = state["trajectory"][-1]["prompt"]
                last_completion = state["trajectory"][-1]["completion"]
                full = _concat_messages([list(last_prompt), list(last_completion)])
                if state.get("final_env_response"):
                    full = _concat_messages([full, list(state["final_env_response"])])
        if visible is None and state.get("trajectory"):
            last_prompt = state["trajectory"][-1]["prompt"]
            last_completion = state["trajectory"][-1]["completion"]
            visible = _concat_messages([list(last_prompt), list(last_completion)])
            if state.get("final_env_response"):
                visible = _concat_messages([visible, list(state["final_env_response"])])
        state["full_rollout_messages"] = list(full or [])
        state["agent_visible_messages"] = list(visible or [])
        initial_len = len(state.get("prompt", []))
        state["completion"] = list(state["full_rollout_messages"])[initial_len:]


class SelfCompactionRubric(vf.Rubric):
    def __init__(self, harness: str = "r2e", **kwargs: Any):
        super().__init__(**kwargs)
        self.harness = harness
        self.add_reward_func(self.solved, weight=1.0)
        self.add_metric(self.solved_metric)

    def _calculate_reward_swebench(self, state: vf.State, info: dict[str, Any]) -> int:
        from swebench.harness.constants import (
            FAIL_ONLY_REPOS,
            FAIL_TO_PASS,
            KEY_INSTANCE_ID,
            PASS_TO_PASS,
            EvalType,
            ResolvedStatus,
        )
        from swebench.harness.grading import (
            get_eval_tests_report,
            get_resolution_status,
        )

        output = state.get("test_output", "")
        test_spec = _make_test_spec(info, namespace="swebench")
        eval_status_map, _found = _get_logs_eval(test_spec, output)
        eval_ref = {
            KEY_INSTANCE_ID: test_spec.instance_id,
            FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: test_spec.PASS_TO_PASS,
        }
        eval_type = (
            EvalType.FAIL_ONLY
            if test_spec.repo in FAIL_ONLY_REPOS
            else EvalType.PASS_AND_FAIL
        )
        report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=eval_type)
        return int(get_resolution_status(report) == ResolvedStatus.FULL.value)

    def _calculate_reward_r2e(self, state: vf.State, info: dict[str, Any]) -> int:
        output = state.get("test_output", "")
        parsed = decolor_dict_keys(parse_log_fn(info["repo_name"])(output))
        expected = decolor_dict_keys(json.loads(info["expected_output_json"]))
        parsed = {key.split(" - ")[0]: parsed[key] for key in sorted(parsed)}
        expected = {key.split(" - ")[0]: expected[key] for key in sorted(expected)}
        if len(parsed) != len(expected):
            return 0
        for key, value in parsed.items():
            if not key:
                continue
            if key not in expected or expected[key] != value:
                return 0
        return 1

    def solved(self, state: vf.State, info: dict[str, Any], **kwargs: Any) -> int:
        if not state.get("submitted_after_compact"):
            return 0
        if isinstance(state.get("error"), vf.InfraError):
            return 0
        if state.get("max_command_timeouts_reached"):
            return 0
        if state.get("patch_broke_tests"):
            return 0
        if self.harness == "swebench":
            return self._calculate_reward_swebench(state, info)
        return self._calculate_reward_r2e(state, info)

    async def solved_metric(self, state: vf.State, info: dict[str, Any], **kwargs: Any) -> int:
        return int(self.solved(state, info, **kwargs))


def load_environment(
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str | None = None,
    simple_only: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
    sandbox_command_timeout: int = 90,
    total_timeout_minutes: int = 360,
    test_timeout: int = 900,
    cpu_cores: int = 4,
    memory_gb: int = 4,
    disk_size_gb: int = 2,
    labels: list[str] | None = None,
    sandbox_client_max_workers: int = 64,
    rollout_timeout_seconds: float = 5400.0,
    max_command_timeouts: int = 5,
    allow_git: bool = False,
    filter_repos: list[str] | None = None,
    skip_swebench_install: bool = True,
    min_compactions: int = 1,
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
    num_examples: int | None = None,
    seed: int = 0,
    logger: Any = None,
) -> vf.Environment:
    resolved_split = split or _default_split(dataset_name)
    harness = _harness_for_dataset(dataset_name)

    def build_dataset() -> Dataset:
        dataset = load_dataset(dataset_name, split=resolved_split)
        if filter_repos:
            filter_set = set(filter_repos)
            dataset = dataset.filter(
                lambda row: filter_set.isdisjoint(
                    (row.get("repo"), row.get("repo_name"))
                )
            )
        if simple_only:
            dataset = dataset.filter(_passes_simple_filter)
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        if num_examples is not None:
            dataset = dataset.select(range(min(int(num_examples), len(dataset))))
        return dataset.map(_process_example, remove_columns=dataset.column_names)

    parser = vf.Parser()
    rubric = SelfCompactionRubric(harness=harness, parser=parser)
    return SelfCompactionSandboxEnv(
        dataset=build_dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        sandbox_command_timeout=sandbox_command_timeout,
        max_turns=max_turns,
        test_timeout=test_timeout,
        total_timeout_minutes=total_timeout_minutes,
        harness=harness,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        labels=labels or ["self-compaction"],
        sandbox_client_max_workers=sandbox_client_max_workers,
        rollout_timeout_seconds=rollout_timeout_seconds,
        max_command_timeouts=max_command_timeouts,
        allow_git=allow_git,
        skip_swebench_install=skip_swebench_install,
        min_compactions=min_compactions,
        max_summary_chars=max_summary_chars,
        logger=logger,
        env_id="self-compaction",
        env_args={
            "dataset_name": dataset_name,
            "split": resolved_split,
            "simple_only": simple_only,
            "min_compactions": min_compactions,
            "max_summary_chars": max_summary_chars,
        },
        map_kwargs={"load_from_cache_file": False, "keep_in_memory": True},
    )


if __name__ == "__main__":
    load_environment()
