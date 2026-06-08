from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_difficulty_map.py"
SPEC = importlib.util.spec_from_file_location("build_difficulty_map", SCRIPT_PATH)
assert SPEC is not None
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


class BuildDifficultyMapTests(unittest.TestCase):
    def test_builds_empirical_records_from_rollouts(self) -> None:
        info = {
            "repo_name": "repo",
            "commit_hash": "abc",
            "num_non_test_files": 1,
            "num_non_test_lines": 30,
            "num_non_test_func_methods": 1,
        }
        records = [
            {"info": info, "reward": 1, "num_turns": 12, "decode_len": 1000},
            {"info": info, "reward": 0, "num_turns": 18, "decode_len": 2000},
        ]

        output = builder.build_difficulty_records(
            records,
            dataset_name="R2E-Gym/R2E-Gym-Subset",
            split="train",
            min_rollouts=1,
        )

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["repo_name"], "repo")
        self.assertEqual(output[0]["commit_hash"], "abc")
        self.assertEqual(output[0]["num_rollouts"], 2)
        self.assertEqual(output[0]["solve_rate"], 0.5)
        self.assertEqual(output[0]["static_difficulty"], "medium")
        self.assertEqual(output[0]["empirical_difficulty"], "medium")
        self.assertEqual(output[0]["mean_turns"], 15)
        self.assertEqual(output[0]["mean_decode_length"], 1500)

    def test_write_jsonl_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "map.jsonl"
            builder.write_jsonl([{"repo_name": "repo", "commit_hash": "abc"}], path)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["repo_name"], "repo")


if __name__ == "__main__":
    unittest.main()
