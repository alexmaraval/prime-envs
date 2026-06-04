from __future__ import annotations

import csv
import json
import os
import random
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence, cast
from urllib.parse import quote

from datasets import Dataset, load_dataset
from openai.types.chat import ChatCompletionAssistantMessageParam
import verifiers as vf

from bird_descriptors import (
    DEFAULT_LARGE_TABLE_SAMPLE_ROWS,
    DEFAULT_SAMPLE_VALUES_LIMIT,
    DEFAULT_TOP_VALUES_LIMIT,
    OpenAIDescriptionClient,
    describe_profile_target,
    ensure_profile,
    find_profile_column,
    find_profile_table,
    prompt_payload_for_column,
    prompt_payload_for_database,
    prompt_payload_for_table,
)
from bird_sql_cache import execute_sql_with_cache, resolve_sql_result_cache_dir


ENV_ID = "bird"
TRAIN_DATASET = "birdsql/bird23-train-filtered"
VAL_DATASET = "birdsql/bird_mini_dev"
DEFAULT_MAX_TURNS = 12
SQL_EXECUTE_BONUS = 0.05
SQL_MATCH_REWARD = 1.0
DEFAULT_INITIAL_SCHEMA_MODE = "ddl"
DEFAULT_SCORING_MODE = "official"
INITIAL_SCHEMA_MODES = {"tables", "ddl", "ddl_with_descriptions"}
SCORING_MODES = {"official", "strict"}

SYSTEM_PROMPT = """You are solving BIRD-SQL tasks by producing one final valid SQLite query.

Use exactly one tool call per turn. Do not answer in plain text. Do not wrap SQL in Markdown.

Workflow:
1. Inspect the schema with `describe`.
2. Use `explore` for ad hoc database checks.
3. Use `suggest` to dry-run a candidate final SQL and inspect its limited output.
4. Use `submit` when you are ready to record the final SQL answer. It executes the SQL, returns limited output, and records it only if execution succeeds.
5. Call `terminate` immediately after `submit`.

Tools:
- `describe(targets, sample_rows)`: inspect table or column metadata. `targets` can be a table name, `table.column`, a comma-separated string, or a list. Use `sample_rows: 0` unless you want a few sample rows.
  Example: `describe({"targets": ["orders", "customers.customer_id"], "sample_rows": 0})`

- `explore(sql)`: run one read-only SQLite `SELECT` or `WITH` query for investigation. Use it to inspect values, test joins, check date formats, count rows, and sample categorical values.
  Example: `explore({"sql": "SELECT status, COUNT(*) FROM orders GROUP BY status LIMIT 10"})`

- `suggest(sql)`: dry-run a candidate final answer SQL. It executes the SQL and returns limited rows or an error, but it does not reveal correctness and does not submit the final answer.
  Example: `suggest({"sql": "SELECT name FROM customers ORDER BY created_at DESC LIMIT 1"})`

- `submit(sql)`: execute and submit the final SQL for scoring. It returns limited rows or an error. Submit SQL only, with no explanation.
  Example: `submit({"sql": "SELECT name FROM customers ORDER BY created_at DESC LIMIT 1"})`

- `terminate()`: finish the rollout after `submit`.
  Example: `terminate({})`

SQL guidance:
- Use valid SQLite syntax.
- Quote identifiers with spaces or special characters using double quotes.
- Match the question's requested output columns and aggregation exactly.
- Use the evidence when provided, but verify table and column meanings with tools.
- Inspect representative values before assuming date formats, enum values, units, or join keys.
- Prefer explicit joins and aliases for multi-table queries.
- You might want to `describe` and `explore` the database first, before `suggest` and finally `submit` once you're sure of your solution.
- After a successful `submit`, call `terminate`. If `submit` returns an error, revise the SQL before submitting again.

The final answer is scored by executing your submitted SQL and comparing its result to the gold result.
Explain your reasoning when you think it's necessary.
"""


def describe(targets: str | list[str], sample_rows: int | None = None) -> str:
    """Describe the database, or one or more tables or columns in it.

    Args:
        targets: `database`, a table name, a column as `table.column`, a
            comma-separated string, or a list of such targets.
        sample_rows: Optional tiny sample count to return for each target. Values are
            capped by the environment.
    """
    raise RuntimeError("The environment handles this tool call.")


def explore(sql: str) -> str:
    """Execute one read-only SQL query against the active SQLite database.

    Args:
        sql: A single SELECT or WITH query used to inspect the database.
    """
    raise RuntimeError("The environment handles this tool call.")


def suggest(sql: str) -> str:
    """Dry-run a candidate final SQL query and inspect its limited output.

    Args:
        sql: A candidate SQL answer for the question.
    """
    raise RuntimeError("The environment handles this tool call.")


def submit(sql: str) -> str:
    """Execute and submit the final SQL query for scoring.

    Args:
        sql: The final SQL answer for the question.
    """
    raise RuntimeError("The environment handles this tool call.")


def terminate() -> str:
    """Finish the rollout after calling `submit`."""
    raise RuntimeError("The environment handles this tool call.")


def _last_assistant_message(
    messages: vf.Messages,
) -> ChatCompletionAssistantMessageParam:
    if not isinstance(messages, list):
        raise TypeError(f"expected chat messages, got {type(messages).__name__}")
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return cast(ChatCompletionAssistantMessageParam, message)
    raise ValueError("expected an assistant message in the conversation")


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        name = function_payload.get("name")
        if isinstance(name, str):
            return name
    name = tool_call.get("name")
    return str(name) if isinstance(name, str) else ""


def _tool_call_arguments(tool_call: dict[str, Any]) -> Any:
    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict) and "arguments" in function_payload:
        return function_payload.get("arguments")
    return tool_call.get("arguments", "{}")


def _tool_message(tool_call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": json.dumps(payload, ensure_ascii=True, sort_keys=True),
        "tool_call_id": tool_call_id,
    }


def _package_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def resolve_bird_data_dir(bird_data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the BIRD data root without packaging the large SQLite assets."""
    if bird_data_dir is not None:
        path = Path(bird_data_dir).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path
        raise FileNotFoundError(
            f"bird_data_dir does not exist or is not a directory: {path}"
        )

    env_value = os.environ.get("BIRD_DATA_DIR")
    if env_value:
        path = Path(env_value).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path
        raise FileNotFoundError(
            f"BIRD_DATA_DIR does not exist or is not a directory: {path}"
        )

    package_root = Path(__file__).resolve().parent
    for package_data in [package_root / ".data", package_root / "data"]:
        if package_data.exists() and package_data.is_dir():
            return package_data

    raise FileNotFoundError(
        "BIRD data directory not found. Provide `bird_data_dir`, set BIRD_DATA_DIR, "
        "or run from a workspace that has environments/bird/.data."
    )


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {path}")
    return [cast(dict[str, Any], row) for row in data]


def _split_kind(split: str) -> str:
    split = split.lower()
    if split in {"train", "training"}:
        return "train"
    if split in {"val", "valid", "validation", "mini-dev", "mini_dev"}:
        return "val"
    if split in {"test", "dev", "full-dev", "full_dev"}:
        return "test"
    raise ValueError("split must be one of: train, val, test")


def _db_layout(split: str) -> tuple[str, str, str]:
    kind = _split_kind(split)
    if kind == "train":
        return "train", "train/train_tables.json", "train/train_databases"
    return "dev", "dev_20240627/dev_tables.json", "dev_20240627/dev_databases"


def _sqlite_path(data_dir: Path, split: str, db_id: str) -> Path:
    _, _, db_root = _db_layout(split)
    return data_dir / db_root / db_id / f"{db_id}.sqlite"


@lru_cache(maxsize=16)
def _load_tables(data_dir_str: str, split: str) -> dict[str, dict[str, Any]]:
    data_dir = Path(data_dir_str)
    _, tables_rel, _ = _db_layout(split)
    tables_path = data_dir / tables_rel
    tables = _read_json_list(tables_path)
    return {str(table["db_id"]): table for table in tables}


@lru_cache(maxsize=256)
def _load_column_descriptions(
    data_dir_str: str, split: str, db_id: str, table_name: str
) -> dict[str, dict[str, str]]:
    data_dir = Path(data_dir_str)
    _, _, db_root = _db_layout(split)
    csv_path = data_dir / db_root / db_id / "database_description" / f"{table_name}.csv"
    if not csv_path.exists():
        return {}

    descriptions: dict[str, dict[str, str]] = {}
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    normalized = {
                        str(k).strip(): (v or "").strip() for k, v in row.items()
                    }
                    original = normalized.get("original_column_name", "").strip()
                    if not original:
                        continue
                    descriptions[original.lower()] = normalized
            return descriptions
        except UnicodeDecodeError as exc:
            descriptions.clear()
            last_error = exc
    if last_error is not None:
        raise last_error
    return descriptions


def _local_metadata_path(data_dir: Path, split: str) -> Path | None:
    kind = _split_kind(split)
    if kind == "train":
        metadata = data_dir / "metadata" / "train_filtered.json"
        return metadata if metadata.exists() else data_dir / "train" / "train.json"
    if kind == "val":
        metadata = data_dir / "metadata" / "bird_mini_dev_sqlite.json"
        return metadata if metadata.exists() else None
    return data_dir / "dev_20240627" / "dev.json"


def _load_hf_rows(dataset_name: str) -> list[dict[str, Any]]:
    attempts: list[tuple[tuple[Any, ...], str]] = []
    if dataset_name == VAL_DATASET:
        attempts.extend(
            [
                ((dataset_name,), "mini_dev_sqlite"),
                ((dataset_name, "sqlite"), "train"),
                ((dataset_name, "SQLite"), "train"),
            ]
        )
    else:
        attempts.append(((dataset_name,), "train"))

    last_error: Exception | None = None
    for args, split_name in attempts:
        try:
            dataset = load_dataset(*args, split=split_name)
            return [dict(row) for row in dataset]
        except Exception as exc:  # pragma: no cover - network/cache dependent
            last_error = exc
    assert last_error is not None
    raise last_error


def _load_rows(data_dir: Path, split: str) -> tuple[list[dict[str, Any]], str]:
    kind = _split_kind(split)
    local_path = _local_metadata_path(data_dir, split)
    if local_path is not None and local_path.exists():
        return _read_json_list(local_path), str(local_path)

    if kind == "train":
        return _load_hf_rows(TRAIN_DATASET), TRAIN_DATASET

    if kind == "val":
        try:
            return _load_hf_rows(VAL_DATASET), VAL_DATASET
        except Exception:
            fallback = data_dir / "dev_20240627" / "dev.json"
            if fallback.exists():
                return _read_json_list(fallback), f"{fallback} (full-dev fallback)"
            raise

    raise FileNotFoundError(
        f"Could not find metadata for split={split!r} under {data_dir}"
    )


@lru_cache(maxsize=8)
def _load_tied_sqls(data_dir_str: str) -> dict[int, list[str]]:
    tied_path = Path(data_dir_str) / "dev_20240627" / "dev_tied_append.json"
    if not tied_path.exists():
        return {}
    tied: dict[int, list[str]] = {}
    for row in _read_json_list(tied_path):
        try:
            question_id = int(row["question_id"])
        except Exception:
            continue
        sql = _row_sql(row)
        if sql:
            tied.setdefault(question_id, []).append(sql)
    return tied


def _row_sql(row: dict[str, Any]) -> str:
    for key in ("SQL", "sql", "query", "gold_sql"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _row_question(row: dict[str, Any]) -> str:
    for key in ("question", "utterance"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"row is missing a question field: {row}")


def _validate_initial_schema_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in INITIAL_SCHEMA_MODES:
        raise ValueError(
            f"initial_schema_mode must be one of {sorted(INITIAL_SCHEMA_MODES)}, got {mode!r}"
        )
    return normalized


def _validate_scoring_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in SCORING_MODES:
        raise ValueError(
            f"scoring_mode must be one of {sorted(SCORING_MODES)}, got {mode!r}"
        )
    return normalized


def _shorten_text(value: Any, max_chars: int = 300) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _column_note(column: dict[str, Any]) -> str:
    parts = [str(column["name"])]
    column_type = str(column.get("type") or "").strip()
    natural_name = str(column.get("natural_name") or "").strip()
    flags = []
    if column_type:
        flags.append(column_type)
    if natural_name and natural_name.lower() != str(column["name"]).lower():
        flags.append(f"natural: {natural_name}")
    if column.get("primary_key"):
        flags.append("primary key")
    if flags:
        parts.append(f"({'; '.join(flags)})")

    description = _shorten_text(column.get("description"))
    data_format = _shorten_text(column.get("data_format"), 120)
    value_description = _shorten_text(column.get("value_description"))
    details = []
    if description:
        details.append(description)
    if data_format:
        details.append(f"format: {data_format}")
    if value_description:
        details.append(f"values: {value_description}")
    if details:
        parts.append("- " + " | ".join(details))
    return " ".join(parts)


def _columns_with_descriptions(
    *,
    data_dir: Path,
    split: str,
    db_id: str,
    table_metadata: dict[str, Any],
    table_idx: int,
    table_name: str,
) -> list[dict[str, Any]]:
    columns = _column_records(table_metadata, table_idx)
    descriptions = _load_column_descriptions(str(data_dir), split, db_id, table_name)
    described_columns = []
    for column in columns:
        enriched = dict(column)
        description = descriptions.get(str(enriched["name"]).lower(), {})
        enriched["description"] = description.get("column_description", "")
        enriched["data_format"] = description.get("data_format", "")
        enriched["value_description"] = description.get("value_description", "")
        described_columns.append(enriched)
    return described_columns


def _format_schema_context(
    *,
    info: dict[str, Any],
    data_dir: Path,
    split: str,
    initial_schema_mode: str,
) -> list[str]:
    table_names = [str(name) for name in info.get("table_names", [])]
    if initial_schema_mode == "tables":
        lines = ["Available tables:"]
        lines.extend(f"- {table_name}" for table_name in table_names)
        return lines

    lines = ["Database schema:"]
    db_id = str(info["db_id"])
    db_path = Path(str(info["db_path"]))
    table_metadata = _load_tables(str(data_dir), split).get(db_id, {})
    table_lookup = _table_lookup(table_metadata)

    for table_name in table_names:
        lines.extend(["", f"Table `{table_name}`:", "```sql"])
        ddl = _sqlite_schema_for_table(db_path, table_name)
        lines.append(ddl.rstrip() if ddl else f"-- schema unavailable for {table_name}")
        lines.append("```")

        if initial_schema_mode != "ddl_with_descriptions":
            continue

        table_idx = table_lookup.get(table_name.lower())
        if table_idx is None:
            continue
        columns = _columns_with_descriptions(
            data_dir=data_dir,
            split=split,
            db_id=db_id,
            table_metadata=table_metadata,
            table_idx=table_idx,
            table_name=table_name,
        )
        if columns:
            lines.append("Column descriptions:")
            lines.extend(f"- {_column_note(column)}" for column in columns)

    return lines


def _format_user_prompt(
    info: dict[str, Any],
    *,
    include_evidence: bool,
    data_dir: Path,
    split: str,
    initial_schema_mode: str,
) -> str:
    lines = [
        "BIRD-SQL task",
        f"Database: {info['db_id']}",
        "",
        f"Question: {info['question']}",
    ]
    evidence = str(info.get("evidence") or "").strip()
    if include_evidence and evidence:
        lines.extend(["", f"Evidence: {evidence}"])

    lines.append("")
    lines.extend(
        _format_schema_context(
            info=info,
            data_dir=data_dir,
            split=split,
            initial_schema_mode=initial_schema_mode,
        )
    )

    lines.extend(
        [
            "",
            "Use valid SQLite. Inspect schema and data as needed. Use `suggest` to test a candidate, submit the final SQL with `submit`, then call `terminate`.",
        ]
    )
    return "\n".join(lines)


def _gold_sqls_for_row(
    data_dir: Path, split: str, row: dict[str, Any], primary_sql: str
) -> list[str]:
    sqls = [primary_sql]
    if _split_kind(split) == "test" and row.get("question_id") is not None:
        try:
            question_id = int(row["question_id"])
        except Exception:
            question_id = -1
        sqls.extend(_load_tied_sqls(str(data_dir)).get(question_id, []))

    deduped: list[str] = []
    for sql in sqls:
        if sql and sql not in deduped:
            deduped.append(sql)
    return deduped


def _make_dataset(
    *,
    data_dir: Path,
    split: str,
    include_evidence: bool,
    initial_schema_mode: str,
    num_examples: int | None,
    seed: int,
) -> Dataset:
    rows, source = _load_rows(data_dir, split)
    if num_examples is not None:
        rows = list(rows)
        random.Random(seed).shuffle(rows)
        rows = rows[: int(num_examples)]

    table_metadata = _load_tables(str(data_dir), split)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        db_id = str(row["db_id"])
        gold_sql = _row_sql(row)
        if not gold_sql:
            raise ValueError(f"row {index} in split={split} is missing gold SQL")
        db_table_metadata = table_metadata.get(db_id, {})
        table_names = list(db_table_metadata.get("table_names_original") or [])
        question_id = row.get("question_id", index)

        info: dict[str, Any] = {
            "question_id": question_id,
            "db_id": db_id,
            "question": _row_question(row),
            "evidence": str(row.get("evidence") or ""),
            "difficulty": row.get("difficulty"),
            "gold_sql": gold_sql,
            "gold_sqls": _gold_sqls_for_row(data_dir, split, row, gold_sql),
            "table_names": table_names,
            "db_path": str(_sqlite_path(data_dir, split, db_id)),
            "split": _split_kind(split),
            "metadata_source": source,
        }
        records.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": _format_user_prompt(
                            info,
                            include_evidence=include_evidence,
                            data_dir=data_dir,
                            split=split,
                            initial_schema_mode=initial_schema_mode,
                        ),
                    }
                ],
                "answer": gold_sql,
                "info": info,
                "task": ENV_ID,
            }
        )

    return Dataset.from_list(records)


def _make_dataset_builder(
    *,
    data_dir: Path,
    split: str,
    include_evidence: bool,
    initial_schema_mode: str,
    num_examples: int | None,
    seed: int,
):
    def build() -> Dataset:
        return _make_dataset(
            data_dir=data_dir,
            split=split,
            include_evidence=include_evidence,
            initial_schema_mode=initial_schema_mode,
            num_examples=num_examples,
            seed=seed,
        )

    return build


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _db_uri(path: Path) -> str:
    return "file:" + quote(str(path.resolve()), safe="/") + "?mode=ro"


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    conn = sqlite3.connect(_db_uri(db_path), uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _strip_leading_sql_comments(sql: str) -> str:
    text = sql.strip()
    while True:
        if text.startswith("--"):
            newline = text.find("\n")
            if newline == -1:
                return ""
            text = text[newline + 1 :].lstrip()
            continue
        if text.startswith("/*"):
            end = text.find("*/")
            if end == -1:
                return ""
            text = text[end + 2 :].lstrip()
            continue
        return text


def _validate_readonly_sql(sql: str) -> str:
    stripped = _strip_leading_sql_comments(sql)
    if not stripped:
        raise ValueError("SQL query is empty.")
    single = stripped.rstrip().rstrip(";").strip()
    if ";" in single:
        raise ValueError("Only one SQL statement is allowed.")
    first_token = single.split(None, 1)[0].lower()
    if first_token not in {"select", "with"}:
        raise ValueError("Only read-only SELECT or WITH queries are allowed.")
    return single


def _sqlite_authorizer(action: int, *_args: Any) -> int:
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
    if hasattr(sqlite3, "SQLITE_RECURSIVE"):
        allowed.add(sqlite3.SQLITE_RECURSIVE)
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _normalize_sql_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _normalize_rows(rows: Sequence[Sequence[Any]]) -> list[tuple[Any, ...]]:
    return [tuple(_normalize_sql_value(value) for value in row) for row in rows]


def _is_ordered_query(sql: str) -> bool:
    return "order by" in sql.lower()


def _rows_match_strict(
    predicted: list[tuple[Any, ...]], gold: list[tuple[Any, ...]], *, ordered: bool
) -> bool:
    if ordered:
        return predicted == gold
    return sorted(predicted, key=repr) == sorted(gold, key=repr)


def _rows_match_official(
    predicted: list[tuple[Any, ...]], gold: list[tuple[Any, ...]]
) -> bool:
    return set(predicted) == set(gold)


def _rows_as_dicts(
    columns: list[str], rows: list[tuple[Any, ...]]
) -> list[dict[str, Any]]:
    return [dict(zip(columns, row, strict=False)) for row in rows]


def _sample_table_rows(
    db_path: Path, table_name: str, sample_rows: int
) -> dict[str, Any]:
    result = _execute_sql(
        db_path=db_path,
        sql=f"SELECT * FROM {_quote_identifier(table_name)} LIMIT {sample_rows}",
        max_seconds=2.0,
        max_rows=sample_rows,
    )
    if not result["ok"]:
        return {"status": "error", "message": result["error"]}
    columns = cast(list[str], result["columns"])
    rows = cast(list[tuple[Any, ...]], result["rows"])
    return {
        "status": "ok",
        "columns": columns,
        "rows": _rows_as_dicts(columns, rows),
        "row_count": len(rows),
        "truncated": result["truncated"],
    }


def _sample_column_values(
    db_path: Path, table_name: str, column_name: str, sample_rows: int
) -> dict[str, Any]:
    table = _quote_identifier(table_name)
    column = _quote_identifier(column_name)
    result = _execute_sql(
        db_path=db_path,
        sql=f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT {sample_rows}",
        max_seconds=2.0,
        max_rows=sample_rows,
    )
    if not result["ok"]:
        return {"status": "error", "message": result["error"]}
    rows = cast(list[tuple[Any, ...]], result["rows"])
    return {
        "status": "ok",
        "values": [row[0] for row in rows],
        "row_count": len(rows),
        "truncated": result["truncated"],
    }


def _execute_sql(
    *,
    db_path: Path,
    sql: str,
    max_seconds: float,
    max_rows: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    conn: sqlite3.Connection | None = None
    try:
        query = _validate_readonly_sql(sql)
        conn = _connect_readonly(db_path)
        conn.set_authorizer(_sqlite_authorizer)
        deadline = started + max_seconds

        def progress_handler() -> int:
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(progress_handler, 1000)
        cursor = conn.execute(query)
        columns = [description[0] for description in (cursor.description or [])]
        if max_rows is None:
            rows = cursor.fetchall()
            truncated = False
        else:
            rows = cursor.fetchmany(max_rows + 1)
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
        elapsed = time.monotonic() - started
        return {
            "ok": True,
            "columns": columns,
            "rows": _normalize_rows(rows),
            "truncated": truncated,
            "elapsed_ms": round(elapsed * 1000.0, 3),
            "error": None,
        }
    except sqlite3.OperationalError as exc:
        elapsed = time.monotonic() - started
        timed_out = "interrupted" in str(exc).lower() and elapsed >= max_seconds
        return {
            "ok": False,
            "columns": [],
            "rows": [],
            "truncated": False,
            "elapsed_ms": round(elapsed * 1000.0, 3),
            "error": "query timed out" if timed_out else str(exc),
        }
    except Exception as exc:
        elapsed = time.monotonic() - started
        return {
            "ok": False,
            "columns": [],
            "rows": [],
            "truncated": False,
            "elapsed_ms": round(elapsed * 1000.0, 3),
            "error": str(exc),
        }
    finally:
        if conn is not None:
            conn.close()


def _table_lookup(table_metadata: dict[str, Any]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    originals = table_metadata.get("table_names_original") or []
    natural = table_metadata.get("table_names") or []
    for idx, name in enumerate(originals):
        lookup[str(name).lower()] = idx
    for idx, name in enumerate(natural):
        lookup[str(name).lower()] = idx
    return lookup


def _column_records(
    table_metadata: dict[str, Any], table_idx: int
) -> list[dict[str, Any]]:
    original_columns = table_metadata.get("column_names_original") or []
    natural_columns = table_metadata.get("column_names") or []
    column_types = table_metadata.get("column_types") or []
    primary_keys = table_metadata.get("primary_keys") or []
    flat_primary_keys: set[int] = set()
    for primary_key in primary_keys:
        if isinstance(primary_key, list):
            flat_primary_keys.update(int(value) for value in primary_key)
        else:
            flat_primary_keys.add(int(primary_key))

    records = []
    for column_idx, pair in enumerate(original_columns):
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] != table_idx:
            continue
        natural_name = ""
        if column_idx < len(natural_columns) and isinstance(
            natural_columns[column_idx], list
        ):
            natural_name = str(natural_columns[column_idx][1])
        records.append(
            {
                "column_index": column_idx,
                "name": str(pair[1]),
                "natural_name": natural_name,
                "type": str(column_types[column_idx])
                if column_idx < len(column_types)
                else "",
                "primary_key": column_idx in flat_primary_keys,
            }
        )
    return records


def _foreign_key_descriptions(
    table_metadata: dict[str, Any], table_idx: int
) -> list[str]:
    original_columns = table_metadata.get("column_names_original") or []
    table_names = table_metadata.get("table_names_original") or []
    by_index: dict[int, tuple[int, str]] = {}
    for column_idx, pair in enumerate(original_columns):
        if isinstance(pair, list) and len(pair) == 2:
            by_index[column_idx] = (int(pair[0]), str(pair[1]))

    descriptions = []
    for source_idx, target_idx in table_metadata.get("foreign_keys") or []:
        if source_idx not in by_index or target_idx not in by_index:
            continue
        source_table_idx, source_column = by_index[source_idx]
        target_table_idx, target_column = by_index[target_idx]
        if source_table_idx != table_idx and target_table_idx != table_idx:
            continue
        source_table = str(table_names[source_table_idx])
        target_table = str(table_names[target_table_idx])
        descriptions.append(
            f"{source_table}.{source_column} -> {target_table}.{target_column}"
        )
    return descriptions


@lru_cache(maxsize=4096)
def _sqlite_schema_for_table(db_path: Path, table_name: str) -> str:
    conn = _connect_readonly(db_path)
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
            (table_name,),
        ).fetchone()
        return str(row[0]) if row and row[0] else ""
    finally:
        conn.close()


def _column_descriptor_profile(column_profile: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "declared_type",
        "declared_type_family",
        "inferred_type",
        "not_null",
        "primary_key_position",
        "table_row_count",
        "profile_rows",
        "profile_is_sampled",
        "null_count",
        "non_null_count",
        "null_fraction",
        "distinct_count",
        "distinct_count_is_exact",
        "distinct_fraction",
        "storage_types",
        "min_value",
        "max_value",
        "sample_values",
        "top_values",
        "numeric_stats",
        "text_length_stats",
        "date_like_stats",
    ]
    return {key: column_profile[key] for key in keys if key in column_profile}


def _table_descriptor_profile(table_profile: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "row_count",
        "profile_rows",
        "profile_is_sampled",
        "column_count",
        "primary_keys",
        "foreign_keys",
        "indexes",
    ]
    return {key: table_profile[key] for key in keys if key in table_profile}


def _enrich_column_from_profile(
    column: dict[str, Any],
    *,
    table_name: str,
    profile_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    if not profile_artifact:
        return column
    enriched = dict(column)
    column_name = str(enriched["name"])
    table_profile = find_profile_table(profile_artifact, table_name)
    column_profile = find_profile_column(table_profile, column_name)
    if column_profile:
        enriched["profile"] = _column_descriptor_profile(column_profile)
        enriched["profile_version"] = profile_artifact.get("schema_version")
    return enriched


def _enrich_table_result_from_profile(
    result: dict[str, Any],
    *,
    profile_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    if not profile_artifact:
        return result
    table_name = str(result["table"])
    table_profile = find_profile_table(profile_artifact, table_name)
    if table_profile:
        result["profile"] = _table_descriptor_profile(table_profile)
        result["profile_version"] = profile_artifact.get("schema_version")
    return result


class BIRDSQLEnv(vf.ToolEnv):
    def __init__(
        self,
        *,
        data_dir: Path,
        split: str,
        max_explore_rows: int,
        max_explore_seconds: float,
        max_score_seconds: float,
        scoring_mode: str,
        sql_result_cache_dir: Path,
        max_describe_sample_rows: int,
        describe_model: str | None,
        describe_base_url: str | None,
        describe_timeout_seconds: float | None,
        describe_top_values_limit: int,
        describe_sample_values_limit: int,
        describe_large_table_sample_rows: int,
        max_turns: int = DEFAULT_MAX_TURNS,
        **kwargs: Any,
    ):
        self.data_dir = data_dir
        self.split = split
        self.max_explore_rows = max_explore_rows
        self.max_explore_seconds = max_explore_seconds
        self.max_score_seconds = max_score_seconds
        self.scoring_mode = _validate_scoring_mode(scoring_mode)
        self.sql_result_cache_dir = sql_result_cache_dir
        self.max_describe_sample_rows = max(0, int(max_describe_sample_rows))
        self.describe_model = str(describe_model).strip() if describe_model else None
        self.describe_base_url = str(describe_base_url).strip() if describe_base_url else None
        self.describe_timeout_seconds = describe_timeout_seconds
        self.describe_top_values_limit = max(0, int(describe_top_values_limit))
        self.describe_sample_values_limit = max(0, int(describe_sample_values_limit))
        self.describe_large_table_sample_rows = max(0, int(describe_large_table_sample_rows))
        self._description_client: OpenAIDescriptionClient | None = None
        super().__init__(
            tools=[describe, explore, suggest, submit, terminate],
            max_turns=max_turns,
            **kwargs,
        )

    async def no_tools_called(self, state: vf.State) -> bool:
        return False

    async def setup_state(self, state: vf.State) -> vf.State:
        info = cast(dict[str, Any], state["info"])
        db_path = Path(str(info["db_path"]))
        info["sql_result_cache_dir"] = str(self.sql_result_cache_dir)
        state["db_path"] = db_path
        state["sql_result_cache_dir"] = self.sql_result_cache_dir
        state["profile_artifact"] = None
        state["profile_generated_on_describe"] = False
        state["table_metadata"] = _load_tables(
            str(self.data_dir), str(info.get("split", self.split))
        ).get(str(info["db_id"]), {})
        state["turn_count"] = 0
        state["invalid_tool_calls"] = 0
        state["final_sql"] = None
        state["last_suggested_sql"] = None
        state["submitted"] = False
        state["terminated_after_submit"] = False
        state["terminated_after_suggest"] = False
        state["sql_executable"] = False
        state["execution_match"] = False
        state["official_execution_match"] = False
        state["strict_execution_match"] = False
        state["scoring_mode"] = self.scoring_mode
        state["sql_runtime_error"] = None
        state["gold_runtime_error"] = None
        state["gold_sql_cache_hits"] = 0
        state["gold_sql_execution_ms"] = []
        state["final_reward"] = 0.0
        return await super().setup_state(state)

    def _parse_single_tool_call(
        self, assistant_message: ChatCompletionAssistantMessageParam, state: vf.State
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
        tool_calls = list(assistant_message.get("tool_calls") or [])
        if not tool_calls:
            state["invalid_tool_calls"] = int(state.get("invalid_tool_calls", 0)) + 1
            return (
                None,
                [],
                "Call exactly one tool: describe, explore, suggest, submit, or terminate.",
            )
        if len(tool_calls) != 1:
            state["invalid_tool_calls"] = int(state.get("invalid_tool_calls", 0)) + 1
            messages = [
                _tool_message(
                    str(tool_call.get("id", "")),
                    {
                        "status": "error",
                        "message": "Call exactly one tool per turn.",
                    },
                )
                for tool_call in tool_calls
                if tool_call.get("id")
            ]
            return None, messages, "Call exactly one tool per turn."

        tool_call = tool_calls[0]
        tool_name = _tool_call_name(tool_call)
        tool_call_id = str(tool_call.get("id", ""))
        if tool_name not in {"describe", "explore", "suggest", "submit", "terminate"}:
            state["invalid_tool_calls"] = int(state.get("invalid_tool_calls", 0)) + 1
            return (
                None,
                [
                    _tool_message(
                        tool_call_id,
                        {
                            "status": "error",
                            "message": f"Unknown tool `{tool_name}`.",
                        },
                    )
                ],
                "Use one of: describe, explore, suggest, submit, terminate.",
            )

        raw_args = _tool_call_arguments(tool_call)
        try:
            parsed_args = (
                raw_args
                if isinstance(raw_args, dict)
                else json.loads(str(raw_args or "{}"))
            )
        except json.JSONDecodeError:
            state["invalid_tool_calls"] = int(state.get("invalid_tool_calls", 0)) + 1
            return (
                None,
                [
                    _tool_message(
                        tool_call_id,
                        {
                            "status": "error",
                            "message": "Tool arguments must be valid JSON.",
                        },
                    )
                ],
                "Retry with valid JSON tool arguments.",
            )
        if not isinstance(parsed_args, dict):
            state["invalid_tool_calls"] = int(state.get("invalid_tool_calls", 0)) + 1
            return (
                None,
                [
                    _tool_message(
                        tool_call_id,
                        {
                            "status": "error",
                            "message": "Tool arguments must be a JSON object.",
                        },
                    )
                ],
                "Retry with a JSON object.",
            )
        return (
            {
                "id": tool_call_id,
                "name": tool_name,
                "arguments": parsed_args,
            },
            [],
            "",
        )

    def _describe_sample_limit(self, sample_rows: Any) -> tuple[int, list[str]]:
        if sample_rows in (None, ""):
            return 0, []
        try:
            requested = int(sample_rows)
        except (TypeError, ValueError):
            raise ValueError("`sample_rows` must be an integer.")
        if requested < 0:
            raise ValueError("`sample_rows` must be non-negative.")
        sample_limit = min(requested, self.max_describe_sample_rows)
        warnings = []
        if requested > self.max_describe_sample_rows:
            warnings.append(
                f"`sample_rows` capped at {self.max_describe_sample_rows} for describe."
            )
        return sample_limit, warnings

    def _description_client_or_none(self) -> OpenAIDescriptionClient | None:
        if not self.describe_model:
            return None
        if self._description_client is None:
            self._description_client = OpenAIDescriptionClient(
                model=self.describe_model,
                base_url=self.describe_base_url,
                timeout=self.describe_timeout_seconds,
            )
        return self._description_client

    def _profile_artifact_for_describe(
        self,
        *,
        state: vf.State,
        split: str,
        db_id: str,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        cached = state.get("profile_artifact")
        if isinstance(cached, dict):
            return cached, []
        try:
            profile_artifact, generated, _path = ensure_profile(
                data_dir=self.data_dir,
                split=split,
                db_id=db_id,
                db_path=cast(Path, state["db_path"]),
                top_values_limit=self.describe_top_values_limit,
                sample_values_limit=self.describe_sample_values_limit,
                large_table_sample_rows=self.describe_large_table_sample_rows,
            )
        except Exception as exc:
            return None, [f"profile unavailable: {exc}"]
        state["profile_artifact"] = profile_artifact
        state["profile_generated_on_describe"] = generated
        return profile_artifact, []

    def _describe_with_model(
        self,
        *,
        scope: str,
        payload: dict[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        client = self._description_client_or_none()
        if client is None or payload is None:
            return None, None
        try:
            description = describe_profile_target(
                client=client,
                scope=scope,
                payload=payload,
            )
        except Exception as exc:
            return None, f"description unavailable for {scope}: {exc}"
        text = str(description.get("description") or "").strip()
        return (text or None), None

    def _describe_targets(
        self, targets: Any, state: vf.State, sample_rows: Any = None
    ) -> dict[str, Any]:
        if isinstance(targets, str):
            target_values = [
                part.strip() for part in targets.split(",") if part.strip()
            ]
        elif isinstance(targets, list):
            target_values = [str(part).strip() for part in targets if str(part).strip()]
        else:
            return {
                "status": "error",
                "message": "`targets` must be a string or list of strings.",
            }
        if not target_values:
            return {
                "status": "error",
                "message": "Provide `database`, or at least one table or column target.",
            }
        try:
            sample_limit, warnings = self._describe_sample_limit(sample_rows)
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        table_metadata = cast(dict[str, Any], state["table_metadata"])
        table_names = list(table_metadata.get("table_names_original") or [])
        table_lookup = _table_lookup(table_metadata)
        db_path = cast(Path, state["db_path"])
        split = str(cast(dict[str, Any], state["info"]).get("split", self.split))
        db_id = str(cast(dict[str, Any], state["info"])["db_id"])
        profile_artifact, profile_warnings = self._profile_artifact_for_describe(
            state=state,
            split=split,
            db_id=db_id,
        )
        warnings.extend(profile_warnings)
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for target in target_values:
            if target.lower() == "database":
                result = {
                    "type": "database",
                    "database": db_id,
                }
                if profile_artifact:
                    result["profile"] = cast(
                        dict[str, Any],
                        cast(dict[str, Any], profile_artifact.get("profile") or {}).get(
                            "database"
                        )
                        or {},
                    )
                    result["profile_version"] = profile_artifact.get("schema_version")
                    description, warning = self._describe_with_model(
                        scope="database",
                        payload=prompt_payload_for_database(profile_artifact),
                    )
                    if description:
                        result["description"] = description
                    if warning:
                        warnings.append(warning)
                results.append(result)
                continue

            table_part, _, column_part = target.partition(".")
            table_idx = table_lookup.get(table_part.lower())
            if table_idx is None:
                errors.append(f"Unknown table `{table_part}`.")
                continue
            table_name = str(table_names[table_idx])
            columns = _columns_with_descriptions(
                data_dir=self.data_dir,
                split=split,
                db_id=db_id,
                table_metadata=table_metadata,
                table_idx=table_idx,
                table_name=table_name,
            )
            columns = [
                _enrich_column_from_profile(
                    column,
                    table_name=table_name,
                    profile_artifact=profile_artifact,
                )
                for column in columns
            ]

            if column_part:
                matching = [
                    column
                    for column in columns
                    if column_part.lower()
                    in {
                        str(column["name"]).lower(),
                        str(column["natural_name"]).lower(),
                    }
                ]
                if not matching:
                    errors.append(
                        f"Unknown column `{column_part}` in table `{table_name}`."
                    )
                    continue
                result = {"type": "column", "table": table_name, "column": matching[0]}
                if sample_limit:
                    result["sample"] = _sample_column_values(
                        db_path, table_name, str(matching[0]["name"]), sample_limit
                    )
                if profile_artifact:
                    result["profile_version"] = profile_artifact.get("schema_version")
                    description, warning = self._describe_with_model(
                        scope="column",
                        payload=prompt_payload_for_column(
                            profile_artifact,
                            table_name,
                            str(matching[0]["name"]),
                        ),
                    )
                    if description:
                        result["description"] = description
                    if warning:
                        warnings.append(warning)
                results.append(result)
            else:
                result = {
                    "type": "table",
                    "table": table_name,
                    "sqlite_schema": _sqlite_schema_for_table(db_path, table_name),
                    "columns": columns,
                    "foreign_keys": _foreign_key_descriptions(
                        table_metadata, table_idx
                    ),
                }
                if sample_limit:
                    result["sample"] = _sample_table_rows(
                        db_path, table_name, sample_limit
                    )
                _enrich_table_result_from_profile(
                    result,
                    profile_artifact=profile_artifact,
                )
                if profile_artifact:
                    description, warning = self._describe_with_model(
                        scope="table",
                        payload=prompt_payload_for_table(profile_artifact, table_name),
                    )
                    if description:
                        result["description"] = description
                    if warning:
                        warnings.append(warning)
                results.append(result)

        return {
            "status": "ok" if results else "error",
            "db_id": db_id,
            "results": results,
            "errors": errors,
            "warnings": warnings,
        }

    def _explore_sql(self, sql: Any, state: vf.State) -> dict[str, Any]:
        if not isinstance(sql, str):
            return {"status": "error", "message": "`sql` must be a string."}
        result = _execute_sql(
            db_path=cast(Path, state["db_path"]),
            sql=sql,
            max_seconds=self.max_explore_seconds,
            max_rows=self.max_explore_rows,
        )
        if not result["ok"]:
            return {
                "status": "error",
                "message": result["error"],
                "elapsed_ms": result["elapsed_ms"],
            }
        return {
            "status": "ok",
            "columns": result["columns"],
            "rows": result["rows"],
            "row_count": len(result["rows"]),
            "truncated": result["truncated"],
            "elapsed_ms": result["elapsed_ms"],
        }

    def _suggest_sql(self, sql: Any, state: vf.State) -> dict[str, Any]:
        if not isinstance(sql, str) or not sql.strip():
            return {"status": "error", "message": "`sql` must be a non-empty string."}
        state["last_suggested_sql"] = sql.strip()
        payload = self._explore_sql(sql, state)
        payload["candidate_recorded"] = payload.get("status") == "ok"
        if payload.get("status") == "ok":
            payload["message"] = (
                "Candidate SQL executed successfully. Inspect the limited output, "
                "then revise or call `submit`."
            )
        else:
            error = str(payload.get("message") or payload.get("error") or "unknown error")
            payload["error"] = error
            payload["message"] = (
                f"Candidate SQL failed to execute: {error}. "
                "Revise it before calling `submit`."
            )
        return payload

    def _submit_sql(self, sql: Any, state: vf.State) -> dict[str, Any]:
        if not isinstance(sql, str) or not sql.strip():
            state["invalid_tool_calls"] = int(state.get("invalid_tool_calls", 0)) + 1
            return {"status": "error", "message": "`sql` is required."}

        result = _execute_sql(
            db_path=cast(Path, state["db_path"]),
            sql=sql,
            max_seconds=self.max_score_seconds,
            max_rows=self.max_explore_rows,
        )
        if not result["ok"]:
            state["sql_executable"] = False
            state["sql_runtime_error"] = result["error"]
            error = str(result["error"] or "unknown error")
            if state.get("submitted"):
                message = (
                    f"Submission SQL failed to execute: {error}. Previous submitted SQL "
                    "is still recorded; revise and submit again, or call `terminate` to keep it."
                )
            else:
                message = (
                    f"Submission SQL failed to execute: {error}. "
                    "Revise it before calling `submit` again."
                )
            return {
                "status": "error",
                "message": message,
                "elapsed_ms": result["elapsed_ms"],
                "error": error,
                "submitted": bool(state.get("submitted")),
            }

        state["final_sql"] = sql.strip()
        state["submitted"] = True
        state["sql_executable"] = True
        state["sql_runtime_error"] = None
        return {
            "status": "ok",
            "message": "Final SQL submitted and executed successfully. Call `terminate` to finish.",
            "columns": result["columns"],
            "rows": result["rows"],
            "row_count": len(result["rows"]),
            "truncated": result["truncated"],
            "elapsed_ms": result["elapsed_ms"],
            "submitted": True,
        }

    def _score_final_answer(self, state: vf.State) -> None:
        info = cast(dict[str, Any], state["info"])
        final_sql = state.get("final_sql")
        if not state.get("terminated_after_submit") or not isinstance(final_sql, str):
            state["final_reward"] = 0.0
            return

        pred_result = _execute_sql(
            db_path=cast(Path, state["db_path"]),
            sql=final_sql,
            max_seconds=self.max_score_seconds,
            max_rows=None,
        )
        state["sql_executable"] = bool(pred_result["ok"])
        if not pred_result["ok"]:
            state["sql_runtime_error"] = pred_result["error"]
            state["final_reward"] = 0.0
            return

        predicted_rows = cast(list[tuple[Any, ...]], pred_result["rows"])
        official_match = False
        strict_match = False
        for gold_sql in info.get("gold_sqls") or [info.get("gold_sql", "")]:
            gold_result = execute_sql_with_cache(
                db_path=cast(Path, state["db_path"]),
                sql=str(gold_sql),
                max_seconds=self.max_score_seconds,
                execute_fn=_execute_sql,
                cache_dir=cast(Path, state["sql_result_cache_dir"]),
            )
            if gold_result.get("cache_hit"):
                state["gold_sql_cache_hits"] = (
                    int(state.get("gold_sql_cache_hits", 0)) + 1
                )
            gold_execution_ms = state.get("gold_sql_execution_ms")
            if isinstance(gold_execution_ms, list):
                gold_execution_ms.append(gold_result.get("elapsed_ms"))
            if not gold_result["ok"]:
                state["gold_runtime_error"] = gold_result["error"]
                continue
            ordered = _is_ordered_query(final_sql) or _is_ordered_query(str(gold_sql))
            gold_rows = cast(list[tuple[Any, ...]], gold_result["rows"])
            official_match = official_match or _rows_match_official(
                predicted_rows, gold_rows
            )
            strict_match = strict_match or _rows_match_strict(
                predicted_rows,
                gold_rows,
                ordered=ordered,
            )
            if (self.scoring_mode == "official" and official_match) or (
                self.scoring_mode == "strict" and strict_match
            ):
                break

        state["official_execution_match"] = official_match
        state["strict_execution_match"] = strict_match
        state["execution_match"] = (
            official_match if self.scoring_mode == "official" else strict_match
        )
        state["final_reward"] = SQL_EXECUTE_BONUS + (
            SQL_MATCH_REWARD if state.get("execution_match") else 0.0
        )

    def _handle_tool_call(
        self, tool_call: dict[str, Any], state: vf.State
    ) -> tuple[dict[str, Any], bool]:
        tool_name = str(tool_call["name"])
        tool_call_id = str(tool_call["id"])
        args = cast(dict[str, Any], tool_call["arguments"])

        if tool_name == "describe":
            payload = self._describe_targets(
                args.get("targets"), state, args.get("sample_rows")
            )
            return _tool_message(tool_call_id, payload), False

        if tool_name == "explore":
            payload = self._explore_sql(args.get("sql"), state)
            return _tool_message(tool_call_id, payload), False

        if tool_name == "suggest":
            payload = self._suggest_sql(args.get("sql"), state)
            return _tool_message(tool_call_id, payload), False

        if tool_name == "submit":
            payload = self._submit_sql(args.get("sql"), state)
            return _tool_message(tool_call_id, payload), False

        if tool_name == "terminate":
            if not state.get("submitted"):
                state["invalid_tool_calls"] = (
                    int(state.get("invalid_tool_calls", 0)) + 1
                )
                return (
                    _tool_message(
                        tool_call_id,
                        {
                            "status": "error",
                            "message": "Call `submit` before `terminate`.",
                        },
                    ),
                    False,
                )
            state["terminated_after_submit"] = True
            state["terminated_after_suggest"] = True
            self._score_final_answer(state)
            return (
                _tool_message(
                    tool_call_id,
                    {
                        "status": "ok",
                        "message": "Rollout terminated.",
                    },
                ),
                True,
            )

        state["invalid_tool_calls"] = int(state.get("invalid_tool_calls", 0)) + 1
        return _tool_message(
            tool_call_id, {"status": "error", "message": "Unknown tool."}
        ), False

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        assistant_message = _last_assistant_message(messages)
        state["turn_count"] = len(state["trajectory"])
        tool_call, pre_messages, retry_message = self._parse_single_tool_call(
            assistant_message, state
        )
        if tool_call is None:
            return [
                *pre_messages,
                {
                    "role": "user",
                    "content": retry_message or "Retry with exactly one tool call.",
                },
            ]

        tool_message, terminated = self._handle_tool_call(tool_call, state)
        response_messages: list[dict[str, Any]] = [tool_message]
        if terminated:
            state["final_env_response"] = response_messages
            return response_messages

        response_messages.append(
            {
                "role": "user",
                "content": "Continue. Use exactly one tool call; use `suggest` to test candidates, then `submit` and `terminate`.",
            }
        )
        return response_messages

    @vf.stop(priority=80)
    async def final_answer_terminated(self, state: vf.State) -> bool:
        return bool(state.get("terminated_after_submit"))


def _reward_total(state: vf.State) -> float:
    return float(state.get("final_reward", 0.0))


def _execution_match_metric(state: vf.State) -> float:
    return 1.0 if state.get("execution_match") else 0.0


def _official_execution_match_metric(state: vf.State) -> float:
    return 1.0 if state.get("official_execution_match") else 0.0


def _strict_execution_match_metric(state: vf.State) -> float:
    return 1.0 if state.get("strict_execution_match") else 0.0


def _sql_executable_metric(state: vf.State) -> float:
    return 1.0 if state.get("sql_executable") else 0.0


def _sql_runtime_error_metric(state: vf.State) -> float:
    return 1.0 if state.get("sql_runtime_error") else 0.0


def _terminated_after_submit_metric(state: vf.State) -> float:
    return 1.0 if state.get("terminated_after_submit") else 0.0


def _terminated_after_suggest_metric(state: vf.State) -> float:
    return 1.0 if state.get("terminated_after_submit") else 0.0


def _turn_count_metric(state: vf.State) -> float:
    return float(state.get("turn_count", 0.0))


def _invalid_tool_calls_metric(state: vf.State) -> float:
    return float(state.get("invalid_tool_calls", 0.0))


def _gold_sql_cache_hits_metric(state: vf.State) -> float:
    return float(state.get("gold_sql_cache_hits", 0.0))


def _gold_sql_execution_ms_metric(state: vf.State) -> float:
    values = state.get("gold_sql_execution_ms")
    if not isinstance(values, list):
        return 0.0
    return sum(float(value) for value in values if isinstance(value, int | float))


def build_rubric() -> vf.Rubric:
    rubric = vf.Rubric()
    rubric.add_reward_func(_reward_total, weight=1.0)
    rubric.add_metric(_execution_match_metric)
    rubric.add_metric(_official_execution_match_metric)
    rubric.add_metric(_strict_execution_match_metric)
    rubric.add_metric(_sql_executable_metric)
    rubric.add_metric(_sql_runtime_error_metric)
    rubric.add_metric(_terminated_after_submit_metric)
    rubric.add_metric(_terminated_after_suggest_metric)
    rubric.add_metric(_turn_count_metric)
    rubric.add_metric(_invalid_tool_calls_metric)
    rubric.add_metric(_gold_sql_cache_hits_metric)
    rubric.add_metric(_gold_sql_execution_ms_metric)
    return rubric


def load_environment(
    split: str = "train",
    eval_split: str = "val",
    bird_data_dir: str | os.PathLike[str] | None = None,
    include_evidence: bool = True,
    initial_schema_mode: str = DEFAULT_INITIAL_SCHEMA_MODE,
    scoring_mode: str = DEFAULT_SCORING_MODE,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_describe_sample_rows: int = 3,
    describe_model: str | None = None,
    describe_base_url: str | None = None,
    describe_timeout_seconds: float | None = 30.0,
    describe_top_values_limit: int = DEFAULT_TOP_VALUES_LIMIT,
    describe_sample_values_limit: int = DEFAULT_SAMPLE_VALUES_LIMIT,
    describe_large_table_sample_rows: int = DEFAULT_LARGE_TABLE_SAMPLE_ROWS,
    max_explore_rows: int = 20,
    max_explore_seconds: float = 5.0,
    max_score_seconds: float = 30.0,
    sql_result_cache_dir: str | os.PathLike[str] | None = None,
    num_examples: int | None = None,
    seed: int = 0,
) -> vf.Environment:
    data_dir = resolve_bird_data_dir(bird_data_dir)
    schema_mode = _validate_initial_schema_mode(initial_schema_mode)
    scoring_mode = _validate_scoring_mode(scoring_mode)
    resolved_sql_result_cache_dir = resolve_sql_result_cache_dir(
        data_dir=data_dir,
        cache_dir=sql_result_cache_dir,
    )
    return BIRDSQLEnv(
        data_dir=data_dir,
        split=split,
        dataset=_make_dataset_builder(
            data_dir=data_dir,
            split=split,
            include_evidence=include_evidence,
            initial_schema_mode=schema_mode,
            num_examples=num_examples,
            seed=seed,
        ),
        eval_dataset=_make_dataset_builder(
            data_dir=data_dir,
            split=eval_split,
            include_evidence=include_evidence,
            initial_schema_mode=schema_mode,
            num_examples=num_examples,
            seed=seed + 10_000,
        ),
        system_prompt=SYSTEM_PROMPT,
        rubric=build_rubric(),
        env_id=ENV_ID,
        map_kwargs={"load_from_cache_file": False, "keep_in_memory": True},
        max_turns=max_turns,
        max_describe_sample_rows=max_describe_sample_rows,
        describe_model=describe_model,
        describe_base_url=describe_base_url,
        describe_timeout_seconds=describe_timeout_seconds,
        describe_top_values_limit=describe_top_values_limit,
        describe_sample_values_limit=describe_sample_values_limit,
        describe_large_table_sample_rows=describe_large_table_sample_rows,
        max_explore_rows=max_explore_rows,
        max_explore_seconds=max_explore_seconds,
        max_score_seconds=max_score_seconds,
        scoring_mode=scoring_mode,
        sql_result_cache_dir=resolved_sql_result_cache_dir,
    )
