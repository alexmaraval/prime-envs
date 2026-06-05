from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

from datasets import Dataset
import verifiers as vf

ENV_ROOT = Path(__file__).resolve().parents[1]
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from self_compaction import (  # noqa: E402
    SYSTEM_PROMPT,
    SelfCompactionRubric,
    SelfCompactionSandboxEnv,
    SelfCompactionToolMonitorRubric,
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
        self.assertEqual(defs["read"]["parameters"]["required"], ["path"])
        self.assertNotIn("state", defs["execute_bash"]["parameters"]["properties"])

    async def test_tool_metrics_count_dict_transcripts(self) -> None:
        rubric = SelfCompactionToolMonitorRubric()
        state = state_with_prompt([{"role": "user", "content": "task"}])
        state["full_rollout_messages"] = [
            assistant_tool("search_files", {"pattern": "needle"}, "c1"),
            assistant_tool("read", {"path": "README.md"}, "c_read"),
            assistant_tool("compact", {"summary": "summary"}, "c2"),
            assistant_tool("submit", {}, "c3"),
        ]
        metrics = {func.__name__: await func(state) for func in rubric.funcs}

        self.assertEqual(metrics["total_tool_calls"], 4)
        self.assertEqual(metrics["search_files_calls"], 1)
        self.assertEqual(metrics["read_calls"], 1)
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
                    {"cmd": "pwd", "description": "inspect current directory"},
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
                    {"query": "needle", "path": "src"},
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
                        "limit": 5,
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
        self.assertEqual(calls[1]["args"], ["--pattern", "needle"])
        self.assertEqual(calls[2]["tool_name"], "read_file.py")
        self.assertEqual(
            calls[2]["args"],
            ["self_compaction.py", "--start-line", "3", "--limit", "5"],
        )

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
