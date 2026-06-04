from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
import tempfile
import unittest

import verifiers as vf

from bird import BIRDSQLEnv, load_environment
from bird_descriptors import (
    descriptor_paths,
    describe_profile_target,
    ensure_profile,
    profile_database,
    prompt_payload_for_column,
    prompt_payload_for_database,
    prompt_payload_for_table,
    write_json,
)


class FakeDescriptionClient:
    model = "fake-model"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def describe(self, *, scope: str, payload: dict) -> dict:
        target = str(
            payload.get("column", {}).get("name")
            or payload.get("table", {}).get("name")
            or payload.get("database", {}).get("db_id")
        )
        self.calls.append((scope, target))
        return {"description": f"{scope} description for {target}"}


class FailingDescriptionClient:
    model = "failing-model"

    def describe(self, *, scope: str, payload: dict) -> dict:
        raise RuntimeError(f"{scope} exploded")


def create_fixture_sqlite(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE employees (
                "employee id" INTEGER PRIMARY KEY,
                department_id INTEGER NOT NULL,
                salary REAL,
                hired_on TEXT,
                status TEXT,
                notes TEXT,
                FOREIGN KEY (department_id) REFERENCES departments(id)
            );
            CREATE INDEX employees_status_idx ON employees(status);
            INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Finance');
            INSERT INTO employees VALUES
              (1, 1, 100.5, '2020-01-15', 'active', 'first hire'),
              (2, 1, 120.0, '2021-02-10', 'active', NULL),
              (3, 2, NULL, '2022-03-01', 'inactive', 'part time');
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_bird_data_dir(root: Path) -> Path:
    data_dir = root / "bird_data"
    db_path = data_dir / "dev_20240627" / "dev_databases" / "toy" / "toy.sqlite"
    create_fixture_sqlite(db_path)
    write_json(
        data_dir / "dev_20240627" / "dev.json",
        [
            {
                "question_id": 0,
                "db_id": "toy",
                "question": "How many employees are active?",
                "evidence": "",
                "SQL": "SELECT COUNT(*) FROM employees WHERE status = 'active'",
                "difficulty": "simple",
            }
        ],
    )
    write_json(
        data_dir / "dev_20240627" / "dev_tables.json",
        [
            {
                "db_id": "toy",
                "table_names_original": ["departments", "employees"],
                "table_names": ["departments", "employees"],
                "column_names_original": [
                    [-1, "*"],
                    [0, "id"],
                    [0, "name"],
                    [1, "employee id"],
                    [1, "department_id"],
                    [1, "salary"],
                    [1, "hired_on"],
                    [1, "status"],
                    [1, "notes"],
                ],
                "column_names": [
                    [-1, "*"],
                    [0, "id"],
                    [0, "name"],
                    [1, "employee id"],
                    [1, "department id"],
                    [1, "salary"],
                    [1, "hired on"],
                    [1, "status"],
                    [1, "notes"],
                ],
                "column_types": [
                    "text",
                    "integer",
                    "text",
                    "integer",
                    "integer",
                    "real",
                    "date",
                    "text",
                    "text",
                ],
                "primary_keys": [1, 3],
                "foreign_keys": [[4, 1]],
            }
        ],
    )
    return data_dir


def make_state(env: BIRDSQLEnv) -> vf.State:
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


class BirdDescriptorTests(unittest.TestCase):
    def test_profiler_collects_schema_relationships_and_column_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "toy.sqlite"
            create_fixture_sqlite(db_path)

            profile = profile_database(db_path=db_path, db_id="toy", split_group="dev")
            employees = next(
                table for table in profile["profile"]["tables"] if table["name"] == "employees"
            )
            status = next(column for column in employees["columns"] if column["name"] == "status")
            salary = next(column for column in employees["columns"] if column["name"] == "salary")
            hired_on = next(
                column for column in employees["columns"] if column["name"] == "hired_on"
            )

            self.assertEqual(profile["generation"]["profile_status"], "complete")
            self.assertEqual(employees["row_count"], 3)
            self.assertEqual(employees["primary_keys"], ["employee id"])
            self.assertEqual(employees["foreign_keys"][0]["to_table"], "departments")
            self.assertEqual(status["distinct_count"], 2)
            self.assertEqual(status["top_values"][0], {"value": "active", "count": 2})
            self.assertEqual(salary["null_count"], 1)
            self.assertAlmostEqual(salary["numeric_stats"]["mean"], 110.25)
            self.assertEqual(hired_on["inferred_type"], "date_like")
            self.assertEqual(hired_on["date_like_stats"]["min"], "2020-01-15")

    def test_profile_cache_is_written_once_and_prompt_slices_describe_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            db_path = data_dir / "dev_20240627" / "dev_databases" / "toy" / "toy.sqlite"
            create_fixture_sqlite(db_path)
            profile, generated, profile_path = ensure_profile(
                data_dir=data_dir,
                split="dev",
                db_id="toy",
                db_path=db_path,
            )
            self.assertTrue(generated)
            self.assertTrue(profile_path.exists())
            self.assertEqual(
                profile_path,
                descriptor_paths(data_dir, "dev", "toy").profile,
            )

            cached_profile, generated_again, _ = ensure_profile(
                data_dir=data_dir,
                split="dev",
                db_id="toy",
                db_path=db_path,
            )
            self.assertFalse(generated_again)
            self.assertEqual(cached_profile["generated_at"], profile["generated_at"])

            client = FakeDescriptionClient()
            database_desc = describe_profile_target(
                client=client,
                scope="database",
                payload=prompt_payload_for_database(profile),
            )
            table_desc = describe_profile_target(
                client=client,
                scope="table",
                payload=prompt_payload_for_table(profile, "employees"),
            )
            column_desc = describe_profile_target(
                client=client,
                scope="column",
                payload=prompt_payload_for_column(profile, "employees", "status"),
            )
            self.assertEqual(database_desc["description"], "database description for toy")
            self.assertEqual(table_desc["description"], "table description for employees")
            self.assertEqual(column_desc["description"], "column description for status")

    def test_describe_lazily_profiles_and_returns_runtime_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = create_bird_data_dir(Path(temp_dir))
            env = load_environment(
                split="test",
                eval_split="test",
                bird_data_dir=data_dir,
                sql_result_cache_dir=Path(temp_dir) / "cache",
            )
            self.assertIsInstance(env, BIRDSQLEnv)
            state = make_state(env)
            env.describe_model = "fake-model"
            env._description_client = FakeDescriptionClient()
            profile_path = descriptor_paths(data_dir, "test", "toy").profile
            self.assertFalse(profile_path.exists())
            enriched = env._describe_targets(
                "database, employees, employees.status",
                state,
            )

            self.assertTrue(profile_path.exists())
            database_result = enriched["results"][0]
            table_result = enriched["results"][1]
            column_result = enriched["results"][2]
            status_column = next(
                column for column in table_result["columns"] if column["name"] == "status"
            )
            self.assertEqual(database_result["description"], "database description for toy")
            self.assertEqual(table_result["description"], "table description for employees")
            self.assertEqual(table_result["profile"]["row_count"], 3)
            self.assertEqual(status_column["profile"]["distinct_count"], 2)
            self.assertEqual(
                column_result["description"],
                "column description for status",
            )

    def test_describe_keeps_profile_context_when_runtime_description_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = create_bird_data_dir(Path(temp_dir))
            env = load_environment(
                split="test",
                eval_split="test",
                bird_data_dir=data_dir,
                sql_result_cache_dir=Path(temp_dir) / "cache",
            )
            self.assertIsInstance(env, BIRDSQLEnv)
            state = make_state(env)
            env.describe_model = "failing-model"
            env._description_client = FailingDescriptionClient()

            payload = env._describe_targets("employees.status", state)

            self.assertEqual(payload["status"], "ok")
            self.assertIn("profile", payload["results"][0]["column"])
            self.assertNotIn("description", payload["results"][0])
            self.assertIn("description unavailable for column", payload["warnings"][0])


if __name__ == "__main__":
    unittest.main()
