from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from datasets import Dataset
import verifiers as vf

ENV_ROOT = Path(__file__).resolve().parents[1]
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from self_compaction import (  # noqa: E402
    SANDBOX_IMAGE_HINT_LABELS,
    SYSTEM_PROMPT,
    SelfCompactionRubric,
    SelfCompactionSandboxEnv,
    SelfCompactionToolMonitorRubric,
    _annotate_difficulty,
    _coerce_difficulty_mix,
    _empirical_difficulty_from_solve_rate,
    _select_mixed_difficulty_dataset,
    _static_difficulty,
    load_environment,
)


def make_env() -> SelfCompactionSandboxEnv:
    parser = vf.Parser()
    return SelfCompactionSandboxEnv(
        dataset=Dataset.from_list([{"question": "Fix it", "answer": ""}]),
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=SelfCompactionRubric(parser=parser),
        max_turns=8,
        min_compactions=1,
    )


def tool_call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def assistant_tool(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [tool_call(name, arguments, call_id)],
    }


def state_with_prompt(prompt: list[dict]) -> vf.State:
    state = vf.State(input={"prompt": prompt, "answer": "", "info": {}})
    state["trajectory"] = []
    state["full_rollout_messages"] = list(prompt)
    state["agent_visible_messages"] = list(prompt)
    state["compaction_count"] = 0
    state["turns_before_first_compact"] = None
    state["compaction_summaries"] = []
    state["submitted"] = False
    state["submitted_after_compact"] = False
    state["final_env_response"] = None
    state["last_successful_tool_name"] = None
    state["timing"] = {"start_time": 0.0}
    return state


class SelfCompactionEnvTests(unittest.IsolatedAsyncioTestCase):
    def test_static_difficulty_classifies_metadata(self) -> None:
        self.assertEqual(
            _static_difficulty(
                {
                    "num_non_test_files": 1,
                    "num_non_test_lines": 10,
                    "num_non_test_func_methods": 1,
                }
            ),
            "easy",
        )
        self.assertEqual(
            _static_difficulty(
                {
                    "num_non_test_files": 1,
                    "num_non_test_lines": 40,
                    "num_non_test_func_methods": 2,
                }
            ),
            "medium",
        )
        self.assertEqual(
            _static_difficulty(
                {
                    "num_non_test_files": 2,
                    "num_non_test_lines": 20,
                    "num_non_test_func_methods": 1,
                }
            ),
            "hard",
        )

    def test_empirical_difficulty_overrides_static_tier(self) -> None:
        example = {
            "repo_name": "repo",
            "commit_hash": "abc",
            "num_non_test_files": 1,
            "num_non_test_lines": 5,
            "num_non_test_func_methods": 1,
        }
        key = ("R2E-Gym/R2E-Gym-Subset", "train", "repo", "abc")

        annotated = _annotate_difficulty(
            example,
            dataset_name="R2E-Gym/R2E-Gym-Subset",
            split="train",
            difficulty_map={key: {"solve_rate": 0.5, "num_rollouts": 8}},
        )

        self.assertEqual(annotated["static_difficulty"], "easy")
        self.assertEqual(annotated["empirical_difficulty"], "medium")
        self.assertEqual(annotated["difficulty"], "medium")
        self.assertEqual(annotated["empirical_num_rollouts"], 8)

    def test_empirical_difficulty_map_columns_are_stable(self) -> None:
        rows = Dataset.from_list(
            [
                {
                    "repo_name": "repo",
                    "commit_hash": "abc",
                    "num_non_test_files": 1,
                    "num_non_test_lines": 5,
                    "num_non_test_func_methods": 1,
                },
                {
                    "repo_name": "repo",
                    "commit_hash": "def",
                    "num_non_test_files": 1,
                    "num_non_test_lines": 5,
                    "num_non_test_func_methods": 1,
                },
            ]
        )
        key = ("R2E-Gym/R2E-Gym-Subset", "train", "repo", "abc")

        mapped = rows.map(
            lambda row: _annotate_difficulty(
                row,
                dataset_name="R2E-Gym/R2E-Gym-Subset",
                split="train",
                difficulty_map={key: {"solve_rate": 0.5, "num_rollouts": 8}},
            )
        )

        self.assertEqual(mapped[0]["empirical_difficulty"], "medium")
        self.assertIsNone(mapped[1]["empirical_difficulty"])
        self.assertIn("empirical_solve_rate", mapped.column_names)

    def test_empirical_difficulty_thresholds(self) -> None:
        self.assertEqual(_empirical_difficulty_from_solve_rate(0.9), "easy")
        self.assertEqual(_empirical_difficulty_from_solve_rate(0.7), "easy")
        self.assertEqual(_empirical_difficulty_from_solve_rate(0.2), "medium")
        self.assertEqual(_empirical_difficulty_from_solve_rate(0.0), "hard")

    def test_difficulty_mix_samples_exact_counts(self) -> None:
        dataset = Dataset.from_list(
            [
                {"difficulty": difficulty, "id": f"{difficulty}-{index}"}
                for difficulty in ("easy", "medium", "hard")
                for index in range(8)
            ]
        )
        mix = _coerce_difficulty_mix("[0.25, 0.5, 0.25]")
        assert mix is not None

        selected = _select_mixed_difficulty_dataset(
            dataset,
            mix,
            num_examples=8,
            seed=7,
        )

        self.assertEqual(len(selected), 8)
        self.assertEqual(Counter(selected["difficulty"]), {"easy": 2, "medium": 4, "hard": 2})

    def test_load_environment_rejects_difficulty_and_mix_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "either difficulty or difficulty_mix"):
            load_environment(difficulty="easy", difficulty_mix=[0.3, 0.4, 0.3])

    async def test_load_environment_exposes_expected_tools(self) -> None:
        env = load_environment(num_examples=1)
        tool_names = [tool.__name__ for tool in env.tools]
        self.assertEqual(
            tool_names,
            [
                "execute_bash",
                "search_files",
                "read",
                "edit_via_str_replace",
                "compact",
                "submit",
            ],
        )
        self.assertNotIn("bash", tool_names)
        for label in SANDBOX_IMAGE_HINT_LABELS:
            self.assertIn(label, env.labels)

    async def test_tool_schema_is_task_focused(self) -> None:
        env = load_environment(num_examples=1)
        tool_defs = [
            tool.model_dump() if hasattr(tool, "model_dump") else tool
            for tool in env.tool_defs
        ]
        defs = {tool["name"]: tool for tool in tool_defs}

        self.assertEqual(
            defs["edit_via_str_replace"]["parameters"]["required"],
            ["path", "old_str", "new_str"],
        )
        self.assertEqual(defs["search_files"]["parameters"]["required"], ["pattern"])
        self.assertEqual(
            defs["read"]["parameters"]["required"],
            ["path"],
        )
        self.assertNotIn("state", defs["execute_bash"]["parameters"]["properties"])

    def test_read_file_outputs_plain_bounded_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sample.py").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ENV_ROOT / "tools" / "read_file.py"),
                    "sample.py",
                    "--start-line",
                    "2",
                    "--end-line",
                    "3",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "sample.py lines 2-3 of 4:\n<content>\ntwo\nthree\n</content>\n",
        )

    def test_read_file_defaults_to_first_200_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sample.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ENV_ROOT / "tools" / "read_file.py"),
                    "sample.py",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "sample.py lines 1-3 of 3:\n<content>\none\ntwo\nthree\n</content>\n",
        )

    def test_search_files_accepts_path_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pkg").mkdir()
            (root / "pkg" / "target.py").write_text("needle\n", encoding="utf-8")
            (root / "other.py").write_text("needle\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ENV_ROOT / "tools" / "search_files.py"),
                    "--pattern",
                    "needle",
                    "--path",
                    "pkg",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "pkg/target.py:1: needle\n")

    async def test_tool_metrics_count_dict_transcripts(self) -> None:
        rubric = SelfCompactionToolMonitorRubric()
        state = state_with_prompt([{"role": "user", "content": "task"}])
        state["full_rollout_messages"] = [
            assistant_tool("search_files", {"pattern": "needle"}, "c1"),
            assistant_tool("read", {"path": "README.md"}, "c_read"),
            assistant_tool("compact", {"summary": "summary"}, "c2"),
            assistant_tool("submit", {}, "c3"),
            assistant_tool("shell", {"command": "pwd"}, "c4"),
        ]
        metrics = {func.__name__: await func(state) for func in rubric.funcs}

        self.assertEqual(metrics["total_tool_calls"], 5)
        self.assertEqual(metrics["tool_alias_calls"], 1)
        self.assertEqual(metrics["search_files_calls"], 1)
        self.assertEqual(metrics["read_calls"], 1)
        self.assertEqual(metrics["execute_bash_calls"], 1)
        self.assertEqual(metrics["shell_alias_calls"], 1)
        self.assertEqual(metrics["compact_calls"], 1)
        self.assertEqual(metrics["submit_calls"], 1)

    async def test_non_edit_tools_ignore_benign_extra_args(self) -> None:
        env = make_env()
        calls = []

        async def fake_run_tool_script(
            tool_name: str,
            args: list[str],
            state: vf.State,
            sandbox_command_timeout: int = 90,
            working_dir: str | None = None,
        ) -> str:
            calls.append(
                {
                    "tool_name": tool_name,
                    "args": args,
                    "state": state,
                    "timeout": sandbox_command_timeout,
                    "working_dir": working_dir,
                }
            )
            return "ok"

        env.run_tool_script = fake_run_tool_script  # type: ignore[method-assign]
        prompt = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        state = state_with_prompt(prompt)

        await env.env_response(
            prompt
            + [
                assistant_tool(
                    "execute_bash",
                    {
                        "cmd": "pwd",
                        "description": "inspect current directory",
                        "state": "model-supplied state is ignored",
                        "sandbox_command_timeout": 999,
                        "working_dir": "/tmp",
                    },
                    "c1",
                )
            ],
            state,
        )
        await env.env_response(
            prompt
            + [
                assistant_tool(
                    "search_files",
                    {"query": "needle", "path": "src", "working_dir": "/tmp"},
                    "c2",
                )
            ],
            state,
        )
        await env.env_response(
            prompt
            + [
                assistant_tool(
                    "read",
                    {
                        "path": "self_compaction.py",
                        "start_line": 3,
                        "end_line": 7,
                        "title": "inspect file",
                    },
                    "c3",
                )
            ],
            state,
        )

        self.assertEqual(calls[0]["tool_name"], "execute_bash.py")
        self.assertEqual(calls[0]["args"], ["--cmd", "pwd"])
        self.assertEqual(calls[1]["tool_name"], "search_files.py")
        self.assertEqual(calls[1]["args"], ["--pattern", "needle", "--path", "src"])
        self.assertEqual(calls[2]["tool_name"], "read_file.py")
        self.assertEqual(
            calls[2]["args"],
            ["self_compaction.py", "--start-line", "3", "--end-line", "7"],
        )
        for call in calls:
            self.assertIs(call["state"], state)
            self.assertEqual(call["timeout"], env.sandbox_command_timeout)
            self.assertEqual(call["working_dir"], env.repo_path)

    async def test_hidden_tool_aliases_dispatch_to_canonical_tools(self) -> None:
        env = make_env()
        calls = []

        async def fake_run_tool_script(
            tool_name: str,
            args: list[str],
            state: vf.State,
            sandbox_command_timeout: int = 90,
            working_dir: str | None = None,
        ) -> str:
            calls.append({"tool_name": tool_name, "args": args})
            return "ok"

        env.run_tool_script = fake_run_tool_script  # type: ignore[method-assign]
        prompt = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        state = state_with_prompt(prompt)

        await env.env_response(
            prompt + [assistant_tool("shell", {"command": "pwd", "title": "inspect"}, "c1")],
            state,
        )
        await env.env_response(
            prompt
            + [
                assistant_tool(
                    "edit",
                    {"path": "a.py", "old": "old", "new": "new", "description": "fix"},
                    "c2",
                )
            ],
            state,
        )
        await env.env_response(
            prompt + [assistant_tool("read_file", {"path": "a.py"}, "c3")],
            state,
        )
        await env.env_response(
            prompt + [assistant_tool("grep", {"pattern": "needle", "path": "src"}, "c4")],
            state,
        )

        self.assertEqual(calls[0], {"tool_name": "execute_bash.py", "args": ["--cmd", "pwd"]})
        self.assertEqual(calls[1]["tool_name"], "str_replace.py")
        self.assertEqual(calls[1]["args"][:3], ["a.py", "old", "new"])
        self.assertEqual(
            calls[2],
            {
                "tool_name": "read_file.py",
                "args": ["a.py", "--start-line", "1", "--end-line", "200"],
            },
        )
        self.assertEqual(
            calls[3],
            {"tool_name": "search_files.py", "args": ["--pattern", "needle", "--path", "src"]},
        )
        self.assertEqual(state["tool_alias_counts"], {"shell": 1, "edit": 1, "read_file": 1, "grep": 1})
        self.assertEqual(state["last_successful_tool_name"], "search_files")

    async def test_sandbox_tool_status_reports_missing_ripgrep_without_failing(
        self,
    ) -> None:
        env = make_env()
        commands = []

        async def fake_execute_command(
            state: vf.State,
            command: str,
            timeout: int = 90,
            working_dir: str | None = None,
        ) -> tuple[int, str]:
            commands.append(
                {"command": command, "timeout": timeout, "working_dir": working_dir}
            )
            return 0, "rg=missing"

        env._execute_command = fake_execute_command  # type: ignore[method-assign]
        state = state_with_prompt(
            [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        )

        await env.check_sandbox_tool_availability(state)
        env._append_sandbox_tool_status_prompt(state)

        self.assertEqual(len(commands), 1)
        self.assertIn("command -v rg", commands[0]["command"])
        self.assertEqual(commands[0]["timeout"], 30)
        self.assertEqual(commands[0]["working_dir"], env.repo_path)
        self.assertFalse(state["rg_available"])
        self.assertEqual(state["sandbox_tool_status"]["rg"], "missing")
        self.assertIn("`rg` is not available", state["prompt"][-1]["content"])
        self.assertIn("Do not run `rg`", state["prompt"][-1]["content"])
        self.assertIn("search_files", state["prompt"][-1]["content"])
        self.assertEqual(state["full_rollout_messages"], state["prompt"])
        self.assertEqual(state["agent_visible_messages"], state["prompt"])

    async def test_sandbox_tool_status_reports_available_ripgrep(self) -> None:
        env = make_env()

        async def fake_execute_command(
            state: vf.State,
            command: str,
            timeout: int = 90,
            working_dir: str | None = None,
        ) -> tuple[int, str]:
            return 0, "rg=available"

        env._execute_command = fake_execute_command  # type: ignore[method-assign]
        state = state_with_prompt(
            [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        )

        await env.check_sandbox_tool_availability(state)
        env._append_sandbox_tool_status_prompt(state)

        self.assertTrue(state["rg_available"])
        self.assertEqual(state["sandbox_tool_status"]["rg"], "available")
        self.assertIn("`rg` is available", state["prompt"][-1]["content"])

    async def test_submit_ignores_extra_args(self) -> None:
        env = make_env()
        prompt = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        state = state_with_prompt(prompt)
        state["compaction_count"] = 1

        response = await env.env_response(
            prompt
            + [
                assistant_tool(
                    "submit",
                    {"path": "README.md", "summary": "ready", "message": "done"},
                )
            ],
            state,
        )

        self.assertTrue(state["submitted"])
        self.assertIn("Submission accepted", response[0]["content"])

    async def test_edit_tool_missing_required_args_still_errors(self) -> None:
        env = make_env()
        prompt = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        state = state_with_prompt(prompt)

        response = await env.env_response(
            prompt
            + [
                assistant_tool(
                    "edit_via_str_replace",
                    {"old_str": "old", "new_str": "new"},
                )
            ],
            state,
        )

        payload = json.loads(response[0]["content"])
        self.assertEqual(payload["status"], "error")
        self.assertIn("Error executing tool 'edit_via_str_replace'", payload["message"])

    async def test_submit_before_compact_is_rejected(self) -> None:
        env = make_env()
        prompt = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        state = state_with_prompt(prompt)
        response = await env.env_response(prompt + [assistant_tool("submit", {})], state)

        self.assertFalse(state["submitted"])
        self.assertFalse(state.get("agent_signaled_done", False))
        self.assertIsNone(state["final_env_response"])
        self.assertIn("call compact", response[0]["content"])

    async def test_compact_replaces_next_prompt(self) -> None:
        env = make_env()
        prompt = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "SENTINEL original task"},
        ]
        state = state_with_prompt(prompt)
        prompt_seen_during_compaction = [
            *prompt,
            {"role": "assistant", "content": "OLD OBSERVATION from earlier turn"},
            {"role": "tool", "content": "OLD TOOL RESULT", "tool_call_id": "old"},
        ]
        state["trajectory"].append(
            {
                "prompt": prompt_seen_during_compaction,
                "completion": [
                    assistant_tool("compact", {"summary": "Only keep this plan."})
                ],
            }
        )

        next_prompt = await env.get_prompt_messages(state)
        joined = "\n".join(message.get("content", "") for message in next_prompt)
        full_joined = "\n".join(
            message.get("content", "") for message in state["full_rollout_messages"]
        )

        self.assertIn("Only keep this plan.", joined)
        self.assertIn("SENTINEL", joined)
        self.assertIn("compact", json.dumps(next_prompt))
        self.assertIn("Context compacted", joined)
        self.assertNotIn("OLD OBSERVATION", joined)
        self.assertNotIn("OLD TOOL RESULT", joined)
        self.assertIn("SENTINEL", full_joined)
        self.assertEqual(state["compaction_count"], 1)
        self.assertEqual(state["turns_before_first_compact"], 0)

    async def test_multiple_compactions_replace_visible_context_only(self) -> None:
        env = make_env()
        prompt = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first task context"},
        ]
        state = state_with_prompt(prompt)
        state["trajectory"].append(
            {
                "prompt": prompt,
                "completion": [assistant_tool("compact", {"summary": "summary one"})],
            }
        )
        first_visible = await env.get_prompt_messages(state)
        state["last_successful_tool_name"] = "execute_bash"
        state["trajectory"].append(
            {
                "prompt": first_visible,
                "completion": [assistant_tool("compact", {"summary": "summary two"})],
            }
        )

        second_visible = await env.get_prompt_messages(state)
        second_text = "\n".join(message.get("content", "") for message in second_visible)
        full_text = json.dumps(state["full_rollout_messages"])

        self.assertIn("summary two", second_text)
        self.assertNotIn("summary one", second_text)
        self.assertIn("first task context", second_text)
        self.assertIn("summary one", full_text)
        self.assertIn("first task context", full_text)
        self.assertEqual(state["compaction_count"], 2)

    async def test_consecutive_compactions_are_rejected(self) -> None:
        env = make_env()
        prompt = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]
        state = state_with_prompt(prompt)
        state["trajectory"].append(
            {
                "prompt": prompt,
                "completion": [assistant_tool("compact", {"summary": "summary one"})],
            }
        )
        first_visible = await env.get_prompt_messages(state)
        state["trajectory"].append(
            {
                "prompt": first_visible,
                "completion": [assistant_tool("compact", {"summary": "summary two"})],
            }
        )

        second_visible = await env.get_prompt_messages(state)
        second_text = "\n".join(
            message.get("content", "") for message in second_visible
        )

        self.assertIn("previous successful tool call was already compact", second_text)
        self.assertEqual(state["compaction_count"], 1)

    async def test_render_completion_keeps_full_and_visible_views(self) -> None:
        env = make_env()
        prompt = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        state = state_with_prompt(prompt)
        state["full_rollout_messages"] = prompt + [
            assistant_tool("compact", {"summary": "summary"}),
            {"role": "tool", "content": "Context compacted.", "tool_call_id": "call_1"},
        ]
        state["agent_visible_messages"] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "user", "content": "summary"},
        ]

        await env.render_completion(state)

        self.assertEqual(state["completion"], state["full_rollout_messages"][len(prompt) :])
        self.assertEqual(len(state["agent_visible_messages"]), 3)
        self.assertIn("summary", state["agent_visible_messages"][2]["content"])


if __name__ == "__main__":
    unittest.main()
