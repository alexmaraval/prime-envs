from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "visualize_rollouts.py"
SPEC = importlib.util.spec_from_file_location("visualize_rollouts", SCRIPT_PATH)
assert SPEC is not None
viewer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(viewer)


def tool_call(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class VisualizeRolloutsTests(unittest.TestCase):
    def test_visualizer_builds_full_and_visible_pairs(self) -> None:
        record = {
            "example_id": 3,
            "prompt": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "SENTINEL original task"},
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        tool_call("compact", {"summary": "Summary after compaction"}, "c1")
                    ],
                },
                {"role": "tool", "content": "Context compacted.", "tool_call_id": "c1"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call("execute_bash", {"command": "pytest -q"}, "c_after")],
                },
                {"role": "tool", "content": "POST COMPACTION OUTPUT", "tool_call_id": "c_after"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call("submit", {}, "c2")],
                },
                {"role": "tool", "content": "Submission accepted.", "tool_call_id": "c2"},
            ],
            "reward": 1,
            "metrics": {"compaction_count": 1, "submitted_after_compact": 1},
        }

        normalized = viewer.normalize_record(record, 1)
        visible_text = "\n".join(
            message.get("content", "") for message in normalized["visibleMessages"]
        )
        full_text = "\n".join(
            message.get("content", "") for message in normalized["fullMessages"]
        )

        self.assertIn("SENTINEL", full_text)
        self.assertIn("SENTINEL", visible_text)
        self.assertIn("Summary after compaction", visible_text)
        self.assertIn("Context compacted", visible_text)
        self.assertIn("POST COMPACTION OUTPUT", visible_text)

    def test_visualizer_reconstructs_when_explicit_visible_state_is_stale(self) -> None:
        record = {
            "example_id": 4,
            "prompt": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Original task"},
            ],
            "agent_visible_messages": [
                {"role": "system", "content": "Use tools."},
                {"role": "user", "content": "Summary only"},
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        tool_call("compact", {"summary": "Summary after compaction"}, "c1")
                    ],
                },
                {"role": "tool", "content": "Context compacted.", "tool_call_id": "c1"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call("execute_bash", {"command": "pwd"}, "c2")],
                },
                {"role": "tool", "content": "/testbed", "tool_call_id": "c2"},
            ],
            "reward": 0,
        }

        normalized = viewer.normalize_record(record, 1)
        visible_text = "\n".join(
            message.get("content", "") for message in normalized["visibleMessages"]
        )

        self.assertIn("Original task", visible_text)
        self.assertIn("Summary after compaction", visible_text)
        self.assertIn("/testbed", visible_text)

    def test_dashboard_reads_jsonl_and_escapes_html(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "example_id": "</script>",
                        "prompt": [{"role": "user", "content": "<b>task</b>"}],
                        "completion": [],
                        "reward": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            data = viewer.build_dashboard_data(path)
            rendered = viewer.build_html(data)

        self.assertEqual(data["summary"]["rollouts"], 1)
        self.assertIn("&lt;/script&gt;", rendered)
        self.assertIn("Full Rollout", rendered)
        self.assertIn("Agent-Visible Rollout", rendered)
        self.assertIn("class=\"full-rollout\"", rendered)
        self.assertIn("<details class='message collapsible-message", rendered)
        self.assertIn("Expand All", rendered)
        self.assertIn("Collapse All", rendered)


if __name__ == "__main__":
    unittest.main()
