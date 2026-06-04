from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_official_predictions.py"
SPEC = importlib.util.spec_from_file_location("export_official_predictions", SCRIPT_PATH)
assert SPEC is not None
export_official_predictions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(export_official_predictions)


class ExportOfficialPredictionsTests(unittest.TestCase):
    def test_extract_final_sql_prefers_submit_and_falls_back_to_suggest(self) -> None:
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "suggest",
                "arguments": json.dumps({"sql": " SELECT 1 "}),
            },
        }
        submit_call = {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "submit",
                "arguments": json.dumps({"sql": " SELECT 2 "}),
            },
        }
        record = {
            "completion": [
                {
                    "role": "assistant",
                    "tool_calls": [json.dumps(tool_call), json.dumps(submit_call)],
                }
            ]
        }

        self.assertEqual(export_official_predictions.extract_final_sql(record), "SELECT 2")
        self.assertEqual(
            export_official_predictions.extract_suggest_sql(
                {"completion": [{"role": "assistant", "tool_calls": [json.dumps(tool_call)]}]}
            ),
            "SELECT 1",
        )

    def test_build_predictions_uses_official_bird_delimiter(self) -> None:
        records = [
            {
                "example_id": 7,
                "final_sql": "SELECT 1",
                "info": {"db_id": "toy_db"},
            }
        ]

        predictions = export_official_predictions.build_predictions(records)

        self.assertEqual(
            predictions,
            {"7": "SELECT 1\t----- bird -----\ttoy_db"},
        )


if __name__ == "__main__":
    unittest.main()
