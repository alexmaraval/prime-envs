from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verifiers as vf

from bird import (
    BIRDSQLEnv,
    _execute_sql,
    _load_column_descriptions,
    _rows_match_official,
    _rows_match_strict,
    load_environment,
    resolve_bird_data_dir,
)


DATA_DIR = Path(__file__).resolve().parents[1] / ".data"


def assistant_tool_call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


class BirdEnvTests(unittest.TestCase):
    def make_env(self, **kwargs) -> BIRDSQLEnv:
        env = load_environment(
            split="test",
            eval_split="test",
            bird_data_dir=DATA_DIR,
            **kwargs,
        )
        self.assertIsInstance(env, BIRDSQLEnv)
        return env

    def make_state(self, env: BIRDSQLEnv) -> vf.State:
        row = env.get_eval_dataset(n=1)[0]
        state = vf.State(
            input={
                "prompt": row["prompt"],
                "answer": row["answer"],
                "info": row["info"],
                "task": "bird",
                "example_id": 0,
            }
        )
        state["trajectory"] = []
        state["final_env_response"] = None
        return asyncio.run(env.setup_state(state))

    def run_tool(
        self, env: BIRDSQLEnv, state: vf.State, name: str, arguments: dict
    ) -> list[dict]:
        message = assistant_tool_call(name, arguments)
        state["trajectory"].append(
            {
                "prompt": [],
                "completion": [message],
                "response": None,
                "tokens": None,
                "reward": None,
                "advantage": None,
                "is_truncated": False,
                "trajectory_id": "test",
                "extras": {},
            }
        )
        return asyncio.run(env.env_response([message], state))

    def test_load_environment_exposes_expected_tools(self) -> None:
        env = self.make_env()
        self.assertEqual(
            [tool["function"]["name"] for tool in env.oai_tools],
            ["describe", "explore", "suggest", "submit", "terminate"],
        )

    def test_evidence_is_enabled_by_default_and_can_be_disabled(self) -> None:
        default_env = self.make_env()
        default_prompt = default_env.get_eval_dataset(n=1)[0]["prompt"][-1]["content"]
        self.assertIn("Evidence:", default_prompt)
        self.assertIn("Database schema:", default_prompt)
        self.assertIn("CREATE TABLE", default_prompt)

        no_evidence_env = self.make_env(include_evidence=False)
        no_evidence_prompt = no_evidence_env.get_eval_dataset(n=1)[0]["prompt"][-1][
            "content"
        ]
        self.assertNotIn("Evidence:", no_evidence_prompt)

    def test_system_prompt_describes_tool_protocol(self) -> None:
        env = self.make_env()

        self.assertIn("Use exactly one tool call per turn", env.system_prompt)
        self.assertIn("Example: `suggest", env.system_prompt)
        self.assertIn("Example: `submit", env.system_prompt)
        self.assertIn("Quote identifiers with spaces", env.system_prompt)

    def test_initial_schema_modes_can_include_ddl_and_descriptions(self) -> None:
        tables_env = self.make_env(initial_schema_mode="tables")
        tables_prompt = tables_env.get_eval_dataset(n=1)[0]["prompt"][-1]["content"]
        self.assertIn("Available tables:", tables_prompt)
        self.assertNotIn("Database schema:", tables_prompt)

        ddl_env = self.make_env(initial_schema_mode="ddl")
        ddl_prompt = ddl_env.get_eval_dataset(n=1)[0]["prompt"][-1]["content"]
        self.assertIn("Database schema:", ddl_prompt)
        self.assertIn("CREATE TABLE", ddl_prompt)

        described_env = self.make_env(initial_schema_mode="ddl_with_descriptions")
        described_prompt = described_env.get_eval_dataset(n=1)[0]["prompt"][-1][
            "content"
        ]
        self.assertIn("Column descriptions:", described_prompt)

        with self.assertRaises(ValueError):
            self.make_env(initial_schema_mode="everything")

    def test_describe_returns_table_and_column_metadata(self) -> None:
        env = self.make_env()
        state = self.make_state(env)
        payload = env._describe_targets("frpm, frpm.CDSCode", state)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["results"][0]["type"], "table")
        self.assertEqual(payload["results"][0]["table"], "frpm")
        self.assertIn("CREATE TABLE frpm", payload["results"][0]["sqlite_schema"])
        self.assertEqual(payload["results"][1]["type"], "column")
        self.assertEqual(payload["results"][1]["column"]["name"], "CDSCode")

    def test_describe_can_return_capped_sample_rows(self) -> None:
        env = self.make_env(max_describe_sample_rows=1)
        state = self.make_state(env)
        payload = env._describe_targets("frpm, frpm.CDSCode", state, sample_rows=5)

        self.assertEqual(payload["status"], "ok")
        self.assertIn("capped", payload["warnings"][0])
        self.assertEqual(payload["results"][0]["sample"]["row_count"], 1)
        self.assertEqual(payload["results"][1]["sample"]["row_count"], 1)

    def test_column_description_reader_handles_cp1252_bullets(self) -> None:
        descriptions = _load_column_descriptions(
            str(DATA_DIR), "test", "student_club", "budget"
        )

        self.assertIn("event_status", descriptions)
        self.assertIn("Closed", descriptions["event_status"]["value_description"])

    def test_explore_enforces_read_only_sql_and_output_limit(self) -> None:
        env = self.make_env(max_explore_rows=1)
        state = self.make_state(env)

        rejected = env._explore_sql("DROP TABLE frpm", state)
        self.assertEqual(rejected["status"], "error")

        result = env._explore_sql("SELECT * FROM frpm", state)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["row_count"], 1)
        self.assertTrue(result["truncated"])

    def test_suggest_dry_runs_candidate_without_submitting(self) -> None:
        env = self.make_env()
        state = self.make_state(env)

        response = self.run_tool(env, state, "suggest", {"sql": "SELECT 1 AS ok"})
        payload = json.loads(response[0]["content"])

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["columns"], ["ok"])
        self.assertEqual(payload["rows"], [[1]])
        self.assertIn("executed successfully", payload["message"])
        self.assertTrue(payload["candidate_recorded"])
        self.assertEqual(state["last_suggested_sql"], "SELECT 1 AS ok")
        self.assertIsNone(state["final_sql"])
        self.assertFalse(state["submitted"])
        self.assertAlmostEqual(state["final_reward"], 0.0)

        response = self.run_tool(
            env, state, "suggest", {"sql": "SELECT * FROM definitely_missing_table"}
        )
        payload = json.loads(response[0]["content"])
        self.assertEqual(payload["status"], "error")
        self.assertIn("failed to execute", payload["message"])
        self.assertIn("no such table", payload["message"])
        self.assertEqual(payload["error"], "no such table: definitely_missing_table")
        self.assertFalse(payload["candidate_recorded"])
        self.assertFalse(state["submitted"])

    def test_final_gold_sql_gets_match_reward_plus_execute_bonus(self) -> None:
        env = self.make_env()
        state = self.make_state(env)
        gold_sql = state["info"]["gold_sql"]

        submit_response = self.run_tool(env, state, "submit", {"sql": gold_sql})
        submit_payload = json.loads(submit_response[0]["content"])
        self.assertEqual(submit_payload["status"], "ok")
        self.assertIn("executed successfully", submit_payload["message"])
        self.assertIn("Call `terminate` to finish", submit_payload["message"])
        self.assertIn("columns", submit_payload)
        self.assertIn("rows", submit_payload)
        self.assertTrue(submit_payload["submitted"])
        self.assertTrue(state["submitted"])
        self.assertEqual(state["final_sql"], gold_sql)

        self.run_tool(env, state, "terminate", {})

        self.assertTrue(state["terminated_after_submit"])
        self.assertTrue(state["sql_executable"])
        self.assertTrue(state["execution_match"])
        self.assertTrue(state["official_execution_match"])
        self.assertTrue(state["strict_execution_match"])
        self.assertAlmostEqual(state["final_reward"], 1.05)

    def test_executable_wrong_sql_gets_small_bonus_only(self) -> None:
        env = self.make_env()
        state = self.make_state(env)

        self.run_tool(
            env,
            state,
            "submit",
            {"sql": "SELECT 'definitely not the correct bird answer'"},
        )
        self.run_tool(env, state, "terminate", {})

        self.assertTrue(state["sql_executable"])
        self.assertFalse(state["execution_match"])
        self.assertAlmostEqual(state["final_reward"], 0.05)

    def test_official_scoring_uses_bird_set_equality_by_default(self) -> None:
        env = self.make_env()
        state = self.make_state(env)
        state["info"]["gold_sqls"] = ["SELECT 1"]
        state["final_sql"] = "SELECT 1 UNION ALL SELECT 1"
        state["terminated_after_submit"] = True

        env._score_final_answer(state)

        self.assertEqual(env.scoring_mode, "official")
        self.assertTrue(state["official_execution_match"])
        self.assertFalse(state["strict_execution_match"])
        self.assertTrue(state["execution_match"])
        self.assertAlmostEqual(state["final_reward"], 1.05)

    def test_gold_sql_results_are_cached_across_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = self.make_env(sql_result_cache_dir=temp_dir)
            state = self.make_state(env)
            state["info"]["gold_sqls"] = ["SELECT 1 AS answer"]
            state["final_sql"] = "SELECT 1 AS answer"
            state["terminated_after_submit"] = True

            env._score_final_answer(state)

            self.assertTrue(state["execution_match"])
            self.assertEqual(state["gold_sql_cache_hits"], 0)
            self.assertTrue(list(Path(temp_dir).glob("**/*.json")))
            self.assertTrue(state["gold_sql_execution_ms"])

            second_state = self.make_state(env)
            second_state["info"]["gold_sqls"] = ["SELECT 1 AS answer"]
            second_state["final_sql"] = "SELECT 1 AS answer"
            second_state["terminated_after_submit"] = True

            env._score_final_answer(second_state)

            self.assertTrue(second_state["execution_match"])
            self.assertEqual(second_state["gold_sql_cache_hits"], 1)
            self.assertTrue(second_state["gold_sql_execution_ms"])

    def test_strict_scoring_preserves_duplicate_sensitive_diagnostic(self) -> None:
        self.assertTrue(_rows_match_official([(1,), (1,)], [(1,)]))
        self.assertFalse(_rows_match_strict([(1,), (1,)], [(1,)], ordered=False))

        env = self.make_env(scoring_mode="strict")
        state = self.make_state(env)
        state["info"]["gold_sqls"] = ["SELECT 1"]
        state["final_sql"] = "SELECT 1 UNION ALL SELECT 1"
        state["terminated_after_submit"] = True

        env._score_final_answer(state)

        self.assertTrue(state["official_execution_match"])
        self.assertFalse(state["strict_execution_match"])
        self.assertFalse(state["execution_match"])
        self.assertAlmostEqual(state["final_reward"], 0.05)

    def test_invalid_sql_and_missing_terminate_score_zero(self) -> None:
        env = self.make_env()
        state = self.make_state(env)

        submit_response = self.run_tool(
            env, state, "submit", {"sql": "DROP TABLE frpm"}
        )
        submit_payload = json.loads(submit_response[0]["content"])
        self.assertEqual(submit_payload["status"], "error")
        self.assertIn("Revise", submit_payload["message"])
        self.assertIn("Only read-only SELECT or WITH queries are allowed", submit_payload["message"])
        self.assertIn("Only read-only SELECT or WITH queries are allowed", submit_payload["error"])
        self.assertFalse(state["submitted"])
        self.assertIsNone(state["final_sql"])

        terminate_response = self.run_tool(env, state, "terminate", {})
        self.assertIn(
            "Call `submit` before `terminate`", terminate_response[0]["content"]
        )
        self.assertFalse(state["sql_executable"])
        self.assertAlmostEqual(state["final_reward"], 0.0)

        state = self.make_state(env)
        self.run_tool(env, state, "submit", {"sql": state["info"]["gold_sql"]})
        self.assertFalse(state["terminated_after_submit"])
        self.assertAlmostEqual(state["final_reward"], 0.0)

    def test_terminate_before_submit_is_rejected(self) -> None:
        env = self.make_env()
        state = self.make_state(env)

        response = self.run_tool(env, state, "terminate", {})

        self.assertFalse(state["terminated_after_submit"])
        self.assertIsNone(state["final_env_response"])
        self.assertIn("Call `submit` before `terminate`", response[0]["content"])

        state = self.make_state(env)
        self.run_tool(env, state, "suggest", {"sql": "SELECT 1"})
        response = self.run_tool(env, state, "terminate", {})
        self.assertFalse(state["terminated_after_submit"])
        self.assertIn("Call `submit` before `terminate`", response[0]["content"])

    def test_data_dir_resolution(self) -> None:
        self.assertEqual(resolve_bird_data_dir(DATA_DIR), DATA_DIR.resolve())
        with patch.dict(os.environ, {"BIRD_DATA_DIR": str(DATA_DIR)}):
            self.assertEqual(resolve_bird_data_dir(), DATA_DIR.resolve())
        with self.assertRaises(FileNotFoundError):
            resolve_bird_data_dir(DATA_DIR / "does-not-exist")

    def test_execute_sql_rejects_multiple_statements(self) -> None:
        env = self.make_env()
        state = self.make_state(env)
        result = _execute_sql(
            db_path=state["db_path"],
            sql="SELECT 1; SELECT 2",
            max_seconds=1.0,
        )
        self.assertFalse(result["ok"])
        self.assertIn("Only one SQL statement", result["error"])


if __name__ == "__main__":
    unittest.main()
