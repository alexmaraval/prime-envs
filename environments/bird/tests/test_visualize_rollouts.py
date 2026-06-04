from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "visualize_rollouts.py"
SPEC = importlib.util.spec_from_file_location("visualize_rollouts", SCRIPT_PATH)
assert SPEC is not None
viewer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(viewer)


def tool_call(name: str, arguments: dict, call_id: str) -> str:
    return json.dumps(
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        }
    )


def sample_record(question: str = "Which rows match?") -> dict:
    return {
        "example_id": 7,
        "prompt": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "BIRD-SQL task\nQuestion: " + question},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": "I should inspect users before submitting SQL.",
                "reasoning_content": "Need to understand the available user rows.",
                "tool_calls": [
                    tool_call("describe", {"targets": ["users"]}, "call_1"),
                    tool_call("explore", {"sql": "SELECT * FROM users"}, "call_2"),
                ],
            },
            {
                "role": "tool",
                "content": json.dumps(
                    {"status": "error", "message": "Call exactly one tool per turn."}
                ),
            },
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "status": "ok",
                        "columns": ["id"],
                        "rows": [],
                        "row_count": 0,
                        "truncated": False,
                    }
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    tool_call("submit", {"sql": "SELECT 1"}, "call_3"),
                ],
            },
            {
                "role": "tool",
                "content": json.dumps(
                    {"status": "ok", "message": "Final SQL submitted."}
                ),
            },
        ],
        "answer": "SELECT 2",
        "task": "bird",
        "info": {
            "question_id": 99,
            "db_id": "toy_db",
            "difficulty": "simple",
            "question": question,
            "evidence": "toy evidence",
            "gold_sql": "SELECT 2",
            "gold_sqls": ["SELECT 2"],
            "table_names": ["users"],
        },
        "reward": 0.05,
        "metrics": {
            "_execution_match_metric": 0.0,
            "_sql_executable_metric": 1.0,
            "_invalid_tool_calls_metric": 1.0,
            "_turn_count_metric": 3.0,
        },
    }


class VisualizeRolloutsTests(unittest.TestCase):
    def test_normalize_record_extracts_sql_tools_and_flags(self) -> None:
        record = viewer.normalize_record(sample_record(), 1)

        self.assertEqual(record["exampleId"], 7)
        self.assertEqual(record["dbId"], "toy_db")
        self.assertEqual(record["finalSql"], "SELECT 1")
        self.assertEqual(record["goldSql"], "SELECT 2")
        self.assertEqual(record["toolSequence"], ["describe", "explore", "submit"])
        self.assertIn("missed match", record["flags"])
        self.assertIn("invalid tool call", record["flags"])
        self.assertIn("tool error", record["flags"])
        self.assertIn("multi-tool turn", record["flags"])
        self.assertIn("no terminate", record["flags"])
        self.assertNotIn("no explore", record["flags"])
        self.assertEqual(record["toolPattern"], "describe -> explore -> submit")
        self.assertEqual(len(record["exploreCalls"]), 1)
        self.assertEqual(record["exploreCalls"][0]["rowCount"], 0.0)
        self.assertEqual(
            record["messages"][2]["content"],
            "I should inspect users before submitting SQL.",
        )
        self.assertEqual(
            record["messages"][2]["thinkingBlocks"],
            [
                {
                    "label": "Reasoning content",
                    "text": "Need to understand the available user rows.",
                }
            ],
        )
        self.assertIn("empty explore result", record["failureTags"])
        self.assertIn("submitted SQL mismatch", record["failureTags"])
        self.assertFalse(record["sqlComparison"]["normalizedMatch"])
        self.assertEqual(record["rewardBreakdown"][-1]["value"], 0.05)

    def test_sql_comparison_extracts_tables_and_identifiers(self) -> None:
        comparison = viewer.compare_sql(
            "SELECT name FROM users WHERE city = 'Paris'",
            "SELECT name FROM customers WHERE country = 'France'",
        )

        self.assertEqual(comparison["finalTables"], ["users"])
        self.assertEqual(comparison["goldTables"], ["customers"])
        self.assertIn("customers", comparison["missingTables"])
        self.assertIn("users", comparison["extraTables"])

    def test_execution_comparison_previews_submitted_and_gold_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "toy.sqlite"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            conn.executemany(
                "INSERT INTO users VALUES (?, ?)",
                [(1, "Ada"), (2, "Grace")],
            )
            conn.commit()
            conn.close()

            comparison = viewer.build_execution_comparison(
                db_path=str(db_path),
                submitted_sql="SELECT name FROM users WHERE id = 1",
                gold_sql="SELECT name FROM users WHERE id = 2",
                cache_dir=str(Path(temp_dir) / "sql-cache"),
            )
            cached_comparison = viewer.build_execution_comparison(
                db_path=str(db_path),
                submitted_sql="SELECT name FROM users WHERE id = 1",
                gold_sql="SELECT name FROM users WHERE id = 2",
                cache_dir=str(Path(temp_dir) / "sql-cache"),
            )

        self.assertEqual(comparison["submitted"]["status"], "ok")
        self.assertEqual(comparison["gold"]["status"], "ok")
        self.assertEqual(comparison["submitted"]["rows"], [["Ada"]])
        self.assertEqual(comparison["gold"]["rows"], [["Grace"]])
        self.assertFalse(comparison["previewMatch"])
        self.assertFalse(comparison["gold"]["cacheHit"])
        self.assertTrue(cached_comparison["gold"]["cacheHit"])
        self.assertIsNotNone(cached_comparison["gold"]["elapsedMs"])

    def test_build_dashboard_data_reads_jsonl_metadata_and_script_safe_html(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            results_path = run_dir / "results.jsonl"
            results_path.write_text(
                json.dumps(sample_record("</script><b>question</b>")) + "\n",
                encoding="utf-8",
            )
            (run_dir / "metadata.json").write_text(
                json.dumps({"model": "gpt-test", "env_id": "bird"}),
                encoding="utf-8",
            )

            data = viewer.build_dashboard_data(results_path)
            html = viewer.build_html(data)

        self.assertEqual(data["summary"]["rollouts"], 1)
        self.assertEqual(data["metadata"]["model"], "gpt-test")
        self.assertIn("<\\/script>", html)
        self.assertIn("Assistant text with tool call", html)
        self.assertIn("Need to understand the available user rows.", html)
        self.assertIn("BIRD Trajectory Dashboard", html)


if __name__ == "__main__":
    unittest.main()
