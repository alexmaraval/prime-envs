from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ENV_ROOT = Path(__file__).resolve().parents[1]
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from self_compaction import (  # noqa: E402
    DEFAULT_DATASET_NAME,
    _default_split,
    _empirical_difficulty_from_solve_rate,
    _static_difficulty,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a self-compaction empirical difficulty JSONL map."
    )
    parser.add_argument("results_jsonl", type=Path, help="Saved eval results.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSONL path. Defaults to <results stem>_difficulty_map.jsonl.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name to record in the map.",
    )
    parser.add_argument(
        "--split",
        help="Dataset split to record in the map. Defaults from dataset name.",
    )
    parser.add_argument(
        "--min-rollouts",
        type=int,
        default=1,
        help="Drop tasks with fewer than this many rollouts.",
    )
    return parser.parse_args()


def parse_json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def load_results(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            records.append(payload)
    return records


def nested_dict(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = parse_json_value(record.get(key))
    return value if isinstance(value, dict) else {}


def extract_info(record: dict[str, Any]) -> dict[str, Any]:
    info = nested_dict(record, "info")
    if info:
        return info
    input_payload = nested_dict(record, "input")
    input_info = parse_json_value(input_payload.get("info"))
    if isinstance(input_info, dict):
        return input_info
    state = nested_dict(record, "state")
    state_info = parse_json_value(state.get("info"))
    if isinstance(state_info, dict):
        return state_info
    if record.get("repo_name") or record.get("commit_hash"):
        return record
    return {}


def numeric_value(value: Any) -> float | None:
    value = parse_json_value(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def nested_numeric(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    containers = [record, nested_dict(record, "metrics"), nested_dict(record, "state")]
    for container in containers:
        for key in keys:
            if key in container:
                value = numeric_value(container[key])
                if value is not None:
                    return value
    return None


def rollout_solved(record: dict[str, Any]) -> int:
    value = nested_numeric(record, ("reward", "solved_metric"))
    return int(value is not None and value > 0.0)


def task_key(info: dict[str, Any]) -> tuple[str, str]:
    repo = str(info.get("repo_name") or info.get("repo") or "")
    commit = str(
        info.get("commit_hash")
        or info.get("base_commit")
        or info.get("instance_id")
        or ""
    )
    if not repo or not commit:
        raise ValueError(
            "Each rollout needs task info with repo_name/repo and "
            "commit_hash/base_commit/instance_id. Save the info column in eval "
            "results if it is missing."
        )
    return repo, commit


def build_difficulty_records(
    records: list[dict[str, Any]],
    *,
    dataset_name: str,
    split: str,
    min_rollouts: int = 1,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "rollouts": 0,
            "solved": 0,
            "turns": [],
            "decode_lengths": [],
            "info": {},
        }
    )
    for record in records:
        info = extract_info(record)
        key = task_key(info)
        group = grouped[key]
        group["info"] = info
        group["rollouts"] += 1
        group["solved"] += rollout_solved(record)
        turns = nested_numeric(record, ("num_turns", "turns"))
        if turns is not None:
            group["turns"].append(turns)
        decode_length = nested_numeric(record, ("decode_len", "decode_length"))
        if decode_length is not None:
            group["decode_lengths"].append(decode_length)

    output: list[dict[str, Any]] = []
    for (repo_name, commit_hash), group in sorted(grouped.items()):
        num_rollouts = int(group["rollouts"])
        if num_rollouts < min_rollouts:
            continue
        solve_rate = float(group["solved"]) / num_rollouts
        info = dict(group["info"])
        record: dict[str, Any] = {
            "dataset_name": dataset_name,
            "split": split,
            "repo_name": repo_name,
            "commit_hash": commit_hash,
            "num_rollouts": num_rollouts,
            "solve_rate": solve_rate,
            "static_difficulty": _static_difficulty(info),
            "empirical_difficulty": _empirical_difficulty_from_solve_rate(
                solve_rate
            ),
        }
        if group["turns"]:
            record["mean_turns"] = sum(group["turns"]) / len(group["turns"])
        if group["decode_lengths"]:
            record["mean_decode_length"] = sum(group["decode_lengths"]) / len(
                group["decode_lengths"]
            )
        output.append(record)
    return output


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    split = args.split or _default_split(args.dataset_name)
    output = args.output or args.results_jsonl.with_name(
        f"{args.results_jsonl.stem}_difficulty_map.jsonl"
    )
    records = build_difficulty_records(
        load_results(args.results_jsonl),
        dataset_name=args.dataset_name,
        split=split,
        min_rollouts=args.min_rollouts,
    )
    write_jsonl(records, output)
    print(f"Wrote {len(records)} task difficulty records to {output}")


if __name__ == "__main__":
    main()
