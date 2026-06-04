from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BIRD_DELIMITER = "\t----- bird -----\t"


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object on line {line_number}")
            records.append(record)
    return records


def _decode_tool_call(raw_tool_call: Any) -> dict[str, Any] | None:
    if isinstance(raw_tool_call, str):
        try:
            raw_tool_call = json.loads(raw_tool_call)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw_tool_call, dict):
        return None
    return raw_tool_call


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        name = function_payload.get("name")
        if isinstance(name, str):
            return name
    name = tool_call.get("name")
    return name if isinstance(name, str) else ""


def _tool_call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    function_payload = tool_call.get("function")
    raw_args = {}
    if isinstance(function_payload, dict):
        raw_args = function_payload.get("arguments", {})
    elif "arguments" in tool_call:
        raw_args = tool_call.get("arguments", {})

    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
    return raw_args if isinstance(raw_args, dict) else {}


def extract_final_sql(record: dict[str, Any]) -> str:
    final_sql = record.get("final_sql")
    if isinstance(final_sql, str) and final_sql.strip():
        return final_sql.strip()

    submitted_sql = ""
    suggested_sql = ""
    for message in record.get("completion") or []:
        if not isinstance(message, dict):
            continue
        for raw_tool_call in message.get("tool_calls") or []:
            tool_call = _decode_tool_call(raw_tool_call)
            if not tool_call:
                continue
            args = _tool_call_arguments(tool_call)
            candidate = args.get("sql")
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            if _tool_call_name(tool_call) == "submit":
                submitted_sql = candidate.strip()
            elif _tool_call_name(tool_call) == "suggest":
                suggested_sql = candidate.strip()
    return submitted_sql or suggested_sql


def extract_suggest_sql(record: dict[str, Any]) -> str:
    """Backward-compatible alias for older callers."""
    return extract_final_sql(record)


def db_id_for_record(record: dict[str, Any]) -> str:
    info = record.get("info")
    if isinstance(info, dict):
        db_id = info.get("db_id")
        if isinstance(db_id, str) and db_id.strip():
            return db_id.strip()
    return "financial"


def prediction_value(sql: str, db_id: str) -> str:
    clean_sql = sql.strip() if sql.strip() else " "
    return f"{clean_sql}{BIRD_DELIMITER}{db_id}"


def choose_records(records: list[dict[str, Any]], dedupe: str) -> list[dict[str, Any]]:
    if dedupe == "none":
        return records

    chosen: dict[str, dict[str, Any]] = {}
    for offset, record in enumerate(records):
        key = str(record.get("example_id", offset))
        if key not in chosen or dedupe == "last":
            chosen[key] = record
            continue
        if dedupe == "best":
            current = float(chosen[key].get("reward") or chosen[key].get("_reward_total") or 0.0)
            candidate = float(record.get("reward") or record.get("_reward_total") or 0.0)
            if candidate > current:
                chosen[key] = record
    return [chosen[key] for key in sorted(chosen, key=lambda value: int(value))]


def build_predictions(records: list[dict[str, Any]]) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for offset, record in enumerate(records):
        key = str(record.get("example_id", offset))
        predictions[key] = prediction_value(extract_final_sql(record), db_id_for_record(record))
    return predictions


def export_predictions(
    *,
    results_jsonl: Path,
    output_dir: Path,
    data_mode: str,
    dedupe: str,
) -> Path:
    records = choose_records(iter_jsonl(results_jsonl), dedupe)
    predictions = build_predictions(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predict_{data_mode}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=4)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Prime BIRD eval results to the official BIRD predict_dev.json format."
    )
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-mode", default="dev")
    parser.add_argument(
        "--dedupe",
        choices=["first", "last", "best", "none"],
        default="first",
        help="How to reduce multiple rollouts per example. Official-style export should usually use first.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_jsonl = args.results_jsonl.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_jsonl.parent
    )
    output_path = export_predictions(
        results_jsonl=results_jsonl,
        output_dir=output_dir,
        data_mode=str(args.data_mode),
        dedupe=str(args.dedupe),
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
