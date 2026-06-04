from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any
from urllib.parse import quote

ENV_ROOT = Path(__file__).resolve().parents[1]
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from bird_sql_cache import execute_sql_with_cache, resolve_sql_result_cache_dir  # noqa: E402


SQL_TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|`[^`]+`|\"[^\"]+\"|'[^']*'|<=|>=|<>|!=|[(),.=<>*+-/]|\d+(?:\.\d+)?"
)
SQL_KEYWORDS = {
    "add",
    "all",
    "and",
    "as",
    "asc",
    "avg",
    "between",
    "by",
    "case",
    "cast",
    "count",
    "desc",
    "distinct",
    "else",
    "end",
    "except",
    "exists",
    "from",
    "group",
    "having",
    "in",
    "inner",
    "intersect",
    "is",
    "join",
    "left",
    "like",
    "limit",
    "max",
    "min",
    "not",
    "null",
    "on",
    "or",
    "order",
    "outer",
    "right",
    "select",
    "sum",
    "then",
    "union",
    "when",
    "where",
    "with",
}
EXECUTION_PREVIEW_ROWS = 20
EXECUTION_PREVIEW_SECONDS = 2.0
GOLD_EXECUTION_CACHE_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a static HTML dashboard for BIRD-SQL eval trajectories."
    )
    parser.add_argument("results_jsonl", type=Path, help="Path to a results.jsonl file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path. Defaults to <results stem>_dashboard.html next to the input.",
    )
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}")
            records.append(payload)
    return records


def load_metadata(results_path: Path) -> dict[str, Any]:
    metadata_path = results_path.with_name("metadata.json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def parse_json_value(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def parse_json_object(raw: Any) -> dict[str, Any]:
    payload = parse_json_value(raw)
    return payload if isinstance(payload, dict) else {}


def pretty_json(value: Any) -> str:
    if isinstance(value, str):
        parsed = parse_json_value(value)
        if parsed is not value:
            value = parsed
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True)


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def readable_text(value: Any) -> str:
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                for key in ("text", "content", "reasoning", "summary", "output_text"):
                    text = text_value(item.get(key)).strip()
                    if text:
                        parts.append(text)
                        break
                else:
                    text = text_value(item).strip()
                    if text:
                        parts.append(text)
            else:
                text = text_value(item).strip()
                if text:
                    parts.append(text)
        return "\n\n".join(parts)
    return text_value(value)


THINKING_FIELDS = [
    ("reasoning_content", "Reasoning content"),
    ("thinking", "Thinking"),
    ("reasoning", "Reasoning"),
    ("thought", "Thought"),
    ("analysis", "Analysis"),
]


def extract_thinking_blocks(message: dict[str, Any]) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for field, label in THINKING_FIELDS:
        text = readable_text(message.get(field)).strip()
        if text:
            blocks.append({"label": label, "text": text})
    return blocks


def parse_tool_call(raw_tool_call: Any) -> dict[str, Any]:
    payload = parse_json_object(raw_tool_call)
    function_payload = payload.get("function")
    if not isinstance(function_payload, dict):
        function_payload = payload

    name = function_payload.get("name")
    arguments = parse_json_value(
        function_payload.get("arguments", payload.get("arguments", {}))
    )
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}

    return {
        "id": text_value(payload.get("id")),
        "name": text_value(name or "unknown_tool"),
        "arguments": arguments,
        "argumentsText": pretty_json(arguments),
    }


def extract_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for section_name in ("prompt", "completion"):
        section = record.get(section_name)
        if not isinstance(section, list):
            continue
        for message in section:
            if isinstance(message, dict):
                copied = dict(message)
                copied["_section"] = section_name
                messages.append(copied)
    return messages


def normalize_message(message: dict[str, Any], index: int) -> dict[str, Any]:
    role = text_value(message.get("role") or "unknown")
    content = readable_text(message.get("content"))
    tool_calls = []
    raw_tool_calls = message.get("tool_calls")
    if isinstance(raw_tool_calls, list):
        tool_calls = [parse_tool_call(tool_call) for tool_call in raw_tool_calls]

    tool_payload: Any = None
    if role == "tool":
        tool_payload = parse_json_value(content)

    return {
        "index": index,
        "section": text_value(message.get("_section")),
        "role": role,
        "content": content,
        "thinkingBlocks": extract_thinking_blocks(message),
        "toolCalls": tool_calls,
        "toolPayload": tool_payload,
    }


def metric_value(record: dict[str, Any], *names: str) -> Any:
    metrics = record.get("metrics")
    for name in names:
        if name in record:
            return record[name]
        if isinstance(metrics, dict) and name in metrics:
            return metrics[name]
    return None


def number_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_metric(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    numeric = number_value(value)
    if numeric is None:
        return None
    return numeric > 0


def tool_error_messages(messages: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for message in messages:
        if message["role"] != "tool":
            continue
        payload = message.get("toolPayload")
        if not isinstance(payload, dict):
            continue
        if payload.get("status") == "error" or payload.get("error"):
            errors.append(
                text_value(
                    payload.get("message") or payload.get("error") or "tool error"
                )
            )
    return errors


def final_submission(tool_calls: list[dict[str, Any]]) -> str:
    submissions = [
        text_value(tool_call.get("arguments", {}).get("sql"))
        for tool_call in tool_calls
        if tool_call.get("name") == "submit"
    ]
    submissions = [sql for sql in submissions if sql.strip()]
    if submissions:
        return submissions[-1]

    suggestions = [
        text_value(tool_call.get("arguments", {}).get("sql"))
        for tool_call in tool_calls
        if tool_call.get("name") == "suggest"
    ]
    suggestions = [sql for sql in suggestions if sql.strip()]
    return suggestions[-1] if suggestions else ""


def final_suggestion(tool_calls: list[dict[str, Any]]) -> str:
    return final_submission(tool_calls)


def normalize_sql_text(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").split()).lower()


def sql_tokens(sql: str) -> list[str]:
    return [match.group(0) for match in SQL_TOKEN_PATTERN.finditer(sql)]


def clean_identifier(token: str) -> str:
    token = token.strip().strip('`"[]').lower()
    return token


def sql_identifier_tokens(sql: str) -> set[str]:
    identifiers: set[str] = set()
    for raw_token in sql_tokens(sql):
        token = clean_identifier(raw_token)
        if not token or token in SQL_KEYWORDS:
            continue
        if re.fullmatch(r"t\d+", token) or re.fullmatch(r"\d+(?:\.\d+)?", token):
            continue
        if raw_token.startswith("'") and raw_token.endswith("'"):
            continue
        if token in {",", ".", "(", ")", "*", "=", "<", ">", "+", "-", "/"}:
            continue
        identifiers.add(token)
    return identifiers


def sql_table_tokens(sql: str) -> set[str]:
    tokens = [clean_identifier(token) for token in sql_tokens(sql)]
    tables: set[str] = set()
    markers = {"from", "join", "update", "into"}
    for index, token in enumerate(tokens[:-1]):
        if token not in markers:
            continue
        cursor = index + 1
        while cursor < len(tokens) and tokens[cursor] in {"(", ","}:
            cursor += 1
        if cursor >= len(tokens):
            continue
        candidate = tokens[cursor]
        if (
            candidate
            and candidate not in SQL_KEYWORDS
            and re.match(r"[a-z_]", candidate)
        ):
            tables.add(candidate)
    return tables


def compare_sql(final_sql: str, gold_sql: str) -> dict[str, Any]:
    normalized_final = normalize_sql_text(final_sql)
    normalized_gold = normalize_sql_text(gold_sql)
    final_tables = sorted(sql_table_tokens(final_sql))
    gold_tables = sorted(sql_table_tokens(gold_sql))
    final_identifiers = sorted(sql_identifier_tokens(final_sql))
    gold_identifiers = sorted(sql_identifier_tokens(gold_sql))

    return {
        "normalizedMatch": bool(
            normalized_final and normalized_final == normalized_gold
        ),
        "finalTables": final_tables,
        "goldTables": gold_tables,
        "missingTables": sorted(set(gold_tables) - set(final_tables)),
        "extraTables": sorted(set(final_tables) - set(gold_tables)),
        "missingIdentifiers": sorted(set(gold_identifiers) - set(final_identifiers)),
        "extraIdentifiers": sorted(set(final_identifiers) - set(gold_identifiers)),
    }


def _db_uri(path: Path) -> str:
    return "file:" + quote(str(path.resolve()), safe="/") + "?mode=ro"


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


def execute_sql_preview(
    *,
    db_path: str,
    sql: str,
    max_rows: int = EXECUTION_PREVIEW_ROWS,
    max_seconds: float = EXECUTION_PREVIEW_SECONDS,
) -> dict[str, Any]:
    started = time.monotonic()
    if not sql.strip():
        return {
            "status": "missing",
            "error": "No SQL submitted.",
            "columns": [],
            "rows": [],
            "rowCount": 0,
            "truncated": False,
            "elapsedMs": None,
        }

    path = Path(db_path).expanduser() if db_path else Path()
    if not db_path or not path.exists():
        return {
            "status": "missing",
            "error": "Database file is not available locally.",
            "columns": [],
            "rows": [],
            "rowCount": 0,
            "truncated": False,
            "elapsedMs": None,
        }

    conn: sqlite3.Connection | None = None
    try:
        query = _validate_readonly_sql(sql)
        conn = sqlite3.connect(_db_uri(path), uri=True)
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_sqlite_authorizer)
        deadline = started + max_seconds

        def progress_handler() -> int:
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(progress_handler, 1000)
        cursor = conn.execute(query)
        columns = [description[0] for description in (cursor.description or [])]
        rows = cursor.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        normalized_rows = [
            [_normalize_sql_value(value) for value in row] for row in rows
        ]
        return {
            "status": "ok",
            "error": "",
            "columns": columns,
            "rows": normalized_rows,
            "rowCount": len(normalized_rows),
            "truncated": truncated,
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 3),
        }
    except sqlite3.OperationalError as exc:
        elapsed = time.monotonic() - started
        timed_out = "interrupted" in str(exc).lower() and elapsed >= max_seconds
        return {
            "status": "error",
            "error": "query timed out" if timed_out else str(exc),
            "columns": [],
            "rows": [],
            "rowCount": 0,
            "truncated": False,
            "elapsedMs": round(elapsed * 1000.0, 3),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "columns": [],
            "rows": [],
            "rowCount": 0,
            "truncated": False,
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 3),
        }
    finally:
        if conn is not None:
            conn.close()


def execute_sql_full(
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
        conn = sqlite3.connect(_db_uri(db_path), uri=True)
        conn.execute("PRAGMA query_only = ON")
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
        normalized_rows = [
            tuple(_normalize_sql_value(value) for value in row) for row in rows
        ]
        return {
            "ok": True,
            "columns": columns,
            "rows": normalized_rows,
            "truncated": truncated,
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
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
        return {
            "ok": False,
            "columns": [],
            "rows": [],
            "truncated": False,
            "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
            "error": str(exc),
        }
    finally:
        if conn is not None:
            conn.close()


def preview_from_full_result(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("ok"):
        return {
            "status": "error",
            "error": text_value(result.get("error")),
            "columns": [],
            "rows": [],
            "rowCount": 0,
            "totalRowCount": 0,
            "truncated": False,
            "elapsedMs": result.get("elapsed_ms"),
            "cacheHit": bool(result.get("cache_hit")),
        }

    rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    preview_rows = rows[:EXECUTION_PREVIEW_ROWS]
    return {
        "status": "ok",
        "error": "",
        "columns": list(result.get("columns") or []),
        "rows": [
            list(row) if isinstance(row, (list, tuple)) else [row]
            for row in preview_rows
        ],
        "rowCount": len(preview_rows),
        "totalRowCount": len(rows),
        "truncated": len(rows) > EXECUTION_PREVIEW_ROWS,
        "elapsedMs": result.get("elapsed_ms"),
        "cacheHit": bool(result.get("cache_hit")),
    }


def execute_gold_sql_preview(
    *, db_path: str, sql: str, cache_dir: str = ""
) -> dict[str, Any]:
    if not sql.strip():
        return {
            "status": "missing",
            "error": "No gold SQL available.",
            "columns": [],
            "rows": [],
            "rowCount": 0,
            "totalRowCount": 0,
            "truncated": False,
            "elapsedMs": None,
            "cacheHit": False,
        }

    path = Path(db_path).expanduser() if db_path else Path()
    if not db_path or not path.exists():
        return {
            "status": "missing",
            "error": "Database file is not available locally.",
            "columns": [],
            "rows": [],
            "rowCount": 0,
            "totalRowCount": 0,
            "truncated": False,
            "elapsedMs": None,
            "cacheHit": False,
        }

    resolved_cache_dir = resolve_sql_result_cache_dir(
        db_path=path,
        cache_dir=cache_dir or None,
    )
    result = execute_sql_with_cache(
        db_path=path,
        sql=sql,
        max_seconds=GOLD_EXECUTION_CACHE_SECONDS,
        execute_fn=execute_sql_full,
        cache_dir=resolved_cache_dir,
    )
    return preview_from_full_result(result)


def build_execution_comparison(
    *, db_path: str, submitted_sql: str, gold_sql: str, cache_dir: str = ""
) -> dict[str, Any]:
    submitted = execute_sql_preview(db_path=db_path, sql=submitted_sql)
    gold = execute_gold_sql_preview(db_path=db_path, sql=gold_sql, cache_dir=cache_dir)
    preview_match: bool | None = None
    if submitted["status"] == "ok" and gold["status"] == "ok":
        preview_match = (
            submitted["columns"] == gold["columns"]
            and submitted["rows"] == gold["rows"]
        )
    return {
        "submitted": submitted,
        "gold": gold,
        "previewMatch": preview_match,
        "previewRows": EXECUTION_PREVIEW_ROWS,
    }


def extract_explore_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending_calls: list[dict[str, Any]] = []
    explores: list[dict[str, Any]] = []
    for message in messages:
        for tool_call in message.get("toolCalls", []):
            if not isinstance(tool_call, dict):
                continue
            pending_calls.append(
                {
                    "messageIndex": message["index"],
                    "name": tool_call.get("name"),
                    "arguments": tool_call.get("arguments", {}),
                }
            )

        if message.get("role") != "tool":
            continue

        pending_call = pending_calls.pop(0) if pending_calls else {}
        if pending_call.get("name") != "explore":
            continue

        arguments = pending_call.get("arguments", {})
        payload = message.get("toolPayload")
        payload = payload if isinstance(payload, dict) else {}
        rows = payload.get("rows")
        row_count = number_value(payload.get("row_count"))
        if row_count is None and isinstance(rows, list):
            row_count = float(len(rows))
        explores.append(
            {
                "messageIndex": pending_call.get("messageIndex"),
                "responseIndex": message["index"],
                "sql": text_value(
                    arguments.get("sql") if isinstance(arguments, dict) else ""
                ),
                "status": text_value(payload.get("status") or "unknown"),
                "error": text_value(payload.get("message") or payload.get("error")),
                "columns": payload.get("columns")
                if isinstance(payload.get("columns"), list)
                else [],
                "rows": rows if isinstance(rows, list) else [],
                "rowCount": row_count,
                "truncated": bool(payload.get("truncated")),
                "elapsedMs": number_value(payload.get("elapsed_ms")),
            }
        )
    return explores


def reward_breakdown(
    *,
    reward: float | None,
    sql_executable: bool | None,
    execution_match: bool | None,
    terminated_after_submit: bool | None,
) -> list[dict[str, Any]]:
    return [
        {
            "label": "Executable SQL bonus",
            "value": 0.05
            if sql_executable is True and terminated_after_submit is not False
            else 0.0,
            "earned": sql_executable is True and terminated_after_submit is not False,
        },
        {
            "label": "Execution match reward",
            "value": 1.0 if execution_match is True else 0.0,
            "earned": execution_match is True,
        },
        {
            "label": "Recorded total",
            "value": reward,
            "earned": reward is not None and reward > 0,
        },
    ]


def first_tool_index(tool_names: list[str], name: str) -> int | None:
    try:
        return tool_names.index(name)
    except ValueError:
        return None


def build_failure_tags(
    *,
    execution_match: bool | None,
    sql_executable: bool | None,
    tool_names: list[str],
    tool_errors: list[str],
    explore_calls: list[dict[str, Any]],
    sql_comparison: dict[str, Any],
    multi_tool_turns: int,
    final_sql: str,
    terminated_after_submit: bool | None,
) -> list[str]:
    tags: list[str] = []
    if execution_match is True:
        return tags

    submit_index = first_tool_index(tool_names, "submit")
    explore_index = first_tool_index(tool_names, "explore")

    if not final_sql:
        tags.append("no final SQL")
    if sql_executable is False:
        tags.append("SQL runtime error")
    if tool_errors:
        tags.append("tool protocol error")
    if multi_tool_turns:
        tags.append("multiple tools in one turn")
    if explore_index is None:
        tags.append("submitted without explore")
    elif submit_index is not None and submit_index < explore_index:
        tags.append("submitted before explore")
    if terminated_after_submit is False:
        tags.append("did not terminate after submit")

    if any(call.get("rowCount") == 0 for call in explore_calls):
        tags.append("empty explore result")
    if any(call.get("status") == "error" for call in explore_calls):
        tags.append("explore error")
    if any(call.get("truncated") for call in explore_calls):
        tags.append("truncated explore result")

    missing_tables = sql_comparison.get("missingTables") or []
    extra_tables = sql_comparison.get("extraTables") or []
    missing_identifiers = sql_comparison.get("missingIdentifiers") or []
    if missing_tables:
        tags.append("missing gold table")
    if extra_tables and missing_tables:
        tags.append("wrong table")
    if missing_identifiers and not missing_tables:
        tags.append("missing gold column/filter")
    if sql_comparison.get("normalizedMatch") is False:
        tags.append("submitted SQL mismatch")

    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def build_flags(
    *,
    execution_match: bool | None,
    sql_executable: bool | None,
    invalid_tool_calls: float,
    tool_errors: list[str],
    tool_counts: Counter[str],
    multi_tool_turns: int,
    terminated_after_submit: bool | None,
    final_sql: str,
) -> list[str]:
    flags: list[str] = []
    if execution_match is False:
        flags.append("missed match")
    if sql_executable is False:
        flags.append("sql error")
    if invalid_tool_calls > 0:
        flags.append("invalid tool call")
    if tool_errors:
        flags.append("tool error")
    if tool_counts.get("explore", 0) == 0:
        flags.append("no explore")
    if tool_counts.get("submit", 0) == 0 and not final_sql:
        flags.append("no submit")
    if tool_counts.get("terminate", 0) == 0:
        flags.append("no terminate")
    if terminated_after_submit is False:
        flags.append("not terminated after submit")
    if multi_tool_turns:
        flags.append("multi-tool turn")
    return flags


def normalize_record(record: dict[str, Any], ordinal: int) -> dict[str, Any]:
    info = record.get("info") if isinstance(record.get("info"), dict) else {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    messages = [
        normalize_message(message, index)
        for index, message in enumerate(extract_messages(record), start=1)
    ]
    tool_calls = [
        tool_call
        for message in messages
        for tool_call in message.get("toolCalls", [])
        if isinstance(tool_call, dict)
    ]
    tool_names = [text_value(tool_call.get("name")) for tool_call in tool_calls]
    tool_counts = Counter(tool_names)
    multi_tool_turns = sum(
        1 for message in messages if len(message.get("toolCalls", [])) > 1
    )

    reward = number_value(record.get("reward", metric_value(record, "_reward_total")))
    execution_match = bool_metric(
        metric_value(record, "_execution_match_metric", "execution_match")
    )
    sql_executable = bool_metric(
        metric_value(record, "_sql_executable_metric", "sql_executable")
    )
    terminated_after_submit = bool_metric(
        metric_value(
            record,
            "_terminated_after_submit_metric",
            "terminated_after_submit",
            "_terminated_after_suggest_metric",
            "terminated_after_suggest",
        )
    )
    invalid_tool_calls = number_value(
        metric_value(record, "_invalid_tool_calls_metric", "invalid_tool_calls")
    )
    tool_errors = tool_error_messages(messages)
    if invalid_tool_calls is None:
        invalid_tool_calls = float(multi_tool_turns + len(tool_errors))

    final_sql = final_submission(tool_calls)
    gold_sql = text_value(info.get("gold_sql") or record.get("answer"))
    db_path = text_value(info.get("db_path"))
    sql_result_cache_dir = text_value(info.get("sql_result_cache_dir"))
    final_reward = (
        reward
        if reward is not None
        else number_value(metric_value(record, "final_reward"))
    )
    turn_count = number_value(
        metric_value(record, "num_turns", "_turn_count_metric", "turn_count")
    )
    explore_calls = extract_explore_calls(messages)
    sql_comparison = compare_sql(final_sql, gold_sql)
    execution_comparison = build_execution_comparison(
        db_path=db_path,
        submitted_sql=final_sql,
        gold_sql=gold_sql,
        cache_dir=sql_result_cache_dir,
    )
    tool_pattern = " -> ".join(tool_names)

    flags = build_flags(
        execution_match=execution_match,
        sql_executable=sql_executable,
        invalid_tool_calls=invalid_tool_calls,
        tool_errors=tool_errors,
        tool_counts=tool_counts,
        multi_tool_turns=multi_tool_turns,
        terminated_after_submit=terminated_after_submit,
        final_sql=final_sql,
    )
    failure_tags = build_failure_tags(
        execution_match=execution_match,
        sql_executable=sql_executable,
        tool_names=tool_names,
        tool_errors=tool_errors,
        explore_calls=explore_calls,
        sql_comparison=sql_comparison,
        multi_tool_turns=multi_tool_turns,
        final_sql=final_sql,
        terminated_after_submit=terminated_after_submit,
    )

    question = text_value(info.get("question"))
    if not question:
        for message in messages:
            if message.get("role") == "user":
                question = text_value(message.get("content")).splitlines()[0]
                break

    return {
        "ordinal": ordinal,
        "exampleId": record.get("example_id", ordinal),
        "questionId": info.get("question_id"),
        "task": text_value(record.get("task") or "bird"),
        "dbId": text_value(info.get("db_id")),
        "difficulty": text_value(info.get("difficulty") or "unknown"),
        "question": question,
        "evidence": text_value(info.get("evidence")),
        "tableNames": info.get("table_names")
        if isinstance(info.get("table_names"), list)
        else [],
        "metadataSource": text_value(info.get("metadata_source")),
        "dbPath": db_path,
        "sqlResultCacheDir": sql_result_cache_dir,
        "reward": final_reward,
        "executionMatch": execution_match,
        "sqlExecutable": sql_executable,
        "terminatedAfterSubmit": terminated_after_submit,
        "terminatedAfterSuggest": terminated_after_submit,
        "turnCount": turn_count,
        "stopCondition": text_value(
            record.get("stop_condition") or record.get("termination_reason")
        ),
        "error": text_value(record.get("error")),
        "tokenUsage": record.get("token_usage")
        if isinstance(record.get("token_usage"), dict)
        else {},
        "timing": record.get("timing")
        if isinstance(record.get("timing"), dict)
        else {},
        "metrics": metrics,
        "toolCounts": dict(tool_counts),
        "toolSequence": tool_names,
        "toolPattern": tool_pattern,
        "invalidToolCalls": invalid_tool_calls,
        "toolErrors": tool_errors,
        "multiToolTurns": multi_tool_turns,
        "finalSql": final_sql,
        "goldSql": gold_sql,
        "goldSqls": info.get("gold_sqls")
        if isinstance(info.get("gold_sqls"), list)
        else [],
        "sqlComparison": sql_comparison,
        "executionComparison": execution_comparison,
        "exploreCalls": explore_calls,
        "rewardBreakdown": reward_breakdown(
            reward=final_reward,
            sql_executable=sql_executable,
            execution_match=execution_match,
            terminated_after_submit=terminated_after_submit,
        ),
        "failureTags": failure_tags,
        "flags": flags,
        "messages": messages,
    }


def average(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [record["reward"] for record in records if record["reward"] is not None]
    turns = [
        record["turnCount"] for record in records if record["turnCount"] is not None
    ]
    solved = sum(1 for record in records if record["executionMatch"] is True)
    executable = sum(1 for record in records if record["sqlExecutable"] is True)
    invalid_tool_calls = sum(
        float(record["invalidToolCalls"] or 0) for record in records
    )
    tool_totals: Counter[str] = Counter()
    failure_tags: Counter[str] = Counter()
    tool_patterns: Counter[str] = Counter()
    for record in records:
        tool_totals.update(record["toolCounts"])
        failure_tags.update(record["failureTags"])
        if record.get("toolPattern"):
            tool_patterns.update([record["toolPattern"]])

    return {
        "rollouts": len(records),
        "solved": solved,
        "unsolved": len(records) - solved,
        "executable": executable,
        "avgReward": average(rewards),
        "avgTurns": average(turns),
        "invalidToolCalls": invalid_tool_calls,
        "toolTotals": dict(tool_totals),
        "failureTagTotals": dict(failure_tags),
        "toolPatterns": dict(tool_patterns),
    }


def build_dashboard_data(results_path: Path) -> dict[str, Any]:
    source_path = results_path.resolve()
    raw_records = load_results(source_path)
    records = [
        normalize_record(record, ordinal)
        for ordinal, record in enumerate(raw_records, start=1)
    ]
    metadata = load_metadata(source_path)
    return {
        "sourcePath": str(source_path),
        "metadata": metadata,
        "summary": build_summary(records),
        "records": records,
    }


def script_safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True).replace("</", "<\\/")


def build_html(data: dict[str, Any]) -> str:
    return PAGE_TEMPLATE.replace("__DASHBOARD_DATA__", script_safe_json(data))


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BIRD Trajectory Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f3;
      --surface: #ffffff;
      --surface-alt: #f0f4f2;
      --ink: #17211f;
      --muted: #64716c;
      --border: #d8ded9;
      --accent: #2f6f68;
      --accent-soft: #dceee9;
      --warn: #9a5b13;
      --warn-soft: #fff0d8;
      --bad: #9d3731;
      --bad-soft: #f9dfdc;
      --good: #276841;
      --good-soft: #dff0e4;
      --code: #182320;
      --code-bg: #eef2ef;
      --shadow: 0 14px 34px rgba(20, 31, 28, 0.08);
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button,
    input,
    select {
      font: inherit;
    }
    button {
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--ink);
      border-radius: 8px;
      cursor: pointer;
    }
    button:hover,
    .rollout-button.active {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .app-shell {
      display: grid;
      grid-template-rows: auto auto 1fr;
      min-height: 100vh;
    }
    .topbar {
      padding: 18px 24px 14px;
      border-bottom: 1px solid var(--border);
      background: var(--surface);
    }
    .title-row {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 6px;
    }
    h1 {
      margin: 0;
      font-size: 1.35rem;
      font-weight: 760;
      letter-spacing: 0;
    }
    .source {
      min-width: 0;
      color: var(--muted);
      font-size: 0.9rem;
      word-break: break-all;
    }
    .run-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .summary-strip {
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 1px;
      border-bottom: 1px solid var(--border);
      background: var(--border);
    }
    .stat {
      padding: 13px 16px;
      background: var(--surface);
    }
    .stat-label {
      display: block;
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }
    .stat-value {
      font-size: 1.2rem;
      font-weight: 760;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(280px, 390px) minmax(0, 1fr);
      min-height: 0;
    }
    .sidebar {
      border-right: 1px solid var(--border);
      background: #fbfcfa;
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .filters {
      display: grid;
      gap: 10px;
      padding: 14px;
      border-bottom: 1px solid var(--border);
    }
    .filter-row {
      display: grid;
      grid-template-columns: 1fr 138px;
      gap: 8px;
    }
    .search-input,
    .status-select {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 9px 10px;
      background: var(--surface);
      color: var(--ink);
    }
    .toggles {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 8px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      color: var(--muted);
      font-size: 0.82rem;
    }
    .rollout-list {
      overflow: auto;
      padding: 8px;
    }
    .rollout-button {
      display: grid;
      gap: 5px;
      width: 100%;
      margin-bottom: 8px;
      padding: 11px;
      text-align: left;
    }
    .rollout-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-weight: 730;
    }
    .rollout-question {
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.35;
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }
    .mini-line {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
    }
    .detail {
      overflow: auto;
      min-width: 0;
      padding: 22px;
    }
    .detail-empty {
      color: var(--muted);
      padding: 30px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
    }
    .detail-head {
      display: grid;
      gap: 12px;
      margin-bottom: 16px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }
    .detail-title {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 18px;
    }
    h2 {
      margin: 0;
      max-width: 960px;
      font-size: 1.3rem;
      line-height: 1.28;
      letter-spacing: 0;
    }
    h3 {
      margin: 0 0 10px;
      font-size: 0.96rem;
      letter-spacing: 0;
    }
    .badge-row {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 25px;
      padding: 4px 8px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--surface-alt);
      color: var(--ink);
      font-size: 0.78rem;
      font-weight: 660;
      white-space: nowrap;
    }
    .badge.good {
      background: var(--good-soft);
      color: var(--good);
      border-color: #b7d7c0;
    }
    .badge.bad {
      background: var(--bad-soft);
      color: var(--bad);
      border-color: #ecc0bd;
    }
    .badge.warn {
      background: var(--warn-soft);
      color: var(--warn);
      border-color: #ebc88e;
    }
    .section-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 16px;
    }
    .panel {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .detail > .panel {
      margin-bottom: 16px;
    }
    .panel.full {
      grid-column: 1 / -1;
    }
    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 760;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .panel-body {
      padding: 14px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 1px;
      background: var(--border);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 16px;
    }
    .metric {
      background: var(--surface);
      padding: 12px;
    }
    .tool-sequence {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .tool-pill {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 8px;
      border-radius: 8px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.82rem;
      font-weight: 760;
    }
    .postmortem-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }
    .postmortem-section {
      display: grid;
      gap: 14px;
      margin-top: 16px;
      padding: 16px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
    }
    .postmortem-title {
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 760;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .postmortem-block {
      min-width: 0;
    }
    .postmortem-block.full {
      grid-column: 1 / -1;
    }
    .reward-list {
      display: grid;
      gap: 8px;
    }
    .reward-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 0;
      border-bottom: 1px solid var(--border);
    }
    .reward-row:last-child {
      border-bottom: 0;
    }
    .execution-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .execution-card {
      display: grid;
      gap: 10px;
      min-width: 0;
      padding: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fbfcfa;
    }
    .execution-title {
      font-weight: 800;
    }
    .explore-list {
      display: grid;
      gap: 12px;
    }
    .explore-card {
      display: grid;
      gap: 10px;
      padding: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fbfcfa;
    }
    .timeline {
      display: grid;
      gap: 10px;
    }
    .message {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      overflow: hidden;
    }
    .message-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: var(--surface-alt);
    }
    .role {
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .message-index {
      color: var(--muted);
      font-size: 0.82rem;
    }
    .message-body {
      padding: 12px;
      display: grid;
      gap: 10px;
    }
    .message-note {
      display: grid;
      gap: 6px;
    }
    .message-note-title {
      color: var(--muted);
      font-size: 0.74rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .message-note.thinking pre {
      border-color: #d8d1bb;
      background: #fff9e8;
    }
    .message-note.assistant-text pre {
      border-color: #c9d8d1;
      background: #f2f8f5;
    }
    pre {
      margin: 0;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--code);
      background: var(--code-bg);
      border: 1px solid #dce3de;
      border-radius: 8px;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.84rem;
      line-height: 1.45;
    }
    .tool-call {
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .tool-call-name {
      padding: 8px 10px;
      font-weight: 800;
      color: var(--accent);
      background: var(--accent-soft);
      border-bottom: 1px solid var(--border);
    }
    .data-table-wrap {
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
    }
    th,
    td {
      padding: 8px 9px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }
    th {
      background: var(--surface-alt);
      color: var(--muted);
      font-weight: 760;
    }
    tr:last-child td {
      border-bottom: 0;
    }
    .muted {
      color: var(--muted);
    }
    .compact-list {
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.5;
    }
    .hidden {
      display: none;
    }
    @media (max-width: 1120px) {
      .summary-strip {
        grid-template-columns: repeat(4, minmax(0, 1fr));
      }
      .workspace {
        grid-template-columns: 320px minmax(0, 1fr);
      }
      .section-grid {
        grid-template-columns: 1fr;
      }
      .execution-grid {
        grid-template-columns: 1fr;
      }
      .postmortem-grid {
        grid-template-columns: 1fr;
      }
      .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 760px) {
      .topbar,
      .detail {
        padding: 16px;
      }
      .summary-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .workspace {
        grid-template-columns: 1fr;
      }
      .sidebar {
        border-right: 0;
        border-bottom: 1px solid var(--border);
        max-height: 52vh;
      }
      .filter-row {
        grid-template-columns: 1fr;
      }
      .detail-title {
        display: grid;
      }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="topbar">
      <div class="title-row">
        <h1>BIRD Trajectory Dashboard</h1>
        <div id="record-count" class="muted"></div>
      </div>
      <div id="source-path" class="source"></div>
      <div id="run-meta" class="run-meta"></div>
    </header>
    <section id="summary-strip" class="summary-strip" aria-label="Run summary"></section>
    <main class="workspace">
      <aside class="sidebar">
        <div class="filters">
          <div class="filter-row">
            <input id="search" class="search-input" type="search" placeholder="Search question, SQL, database">
            <select id="status-filter" class="status-select">
              <option value="all">All trajectories</option>
              <option value="matched">Execution match</option>
              <option value="missed">Missed match</option>
              <option value="executable">SQL executable</option>
              <option value="sql-error">SQL error</option>
            </select>
          </div>
          <input id="sequence-filter" class="search-input" type="search" placeholder="Tool pattern: describe -> explore -> submit">
          <div class="toggles">
            <label class="toggle"><input id="invalid-only" type="checkbox"> Invalid tools</label>
            <label class="toggle"><input id="no-explore-only" type="checkbox"> No explore</label>
            <label class="toggle"><input id="tool-error-only" type="checkbox"> Tool errors</label>
          </div>
        </div>
        <div id="rollout-list" class="rollout-list"></div>
      </aside>
      <section id="detail" class="detail"></section>
    </main>
  </div>
  <script type="application/json" id="dashboard-data">__DASHBOARD_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById("dashboard-data").textContent);
    const state = {
      selectedOrdinal: null,
      filtered: data.records.slice()
    };

    const els = {
      sourcePath: document.getElementById("source-path"),
      recordCount: document.getElementById("record-count"),
      runMeta: document.getElementById("run-meta"),
      summaryStrip: document.getElementById("summary-strip"),
      search: document.getElementById("search"),
      statusFilter: document.getElementById("status-filter"),
      sequenceFilter: document.getElementById("sequence-filter"),
      invalidOnly: document.getElementById("invalid-only"),
      noExploreOnly: document.getElementById("no-explore-only"),
      toolErrorOnly: document.getElementById("tool-error-only"),
      rolloutList: document.getElementById("rollout-list"),
      detail: document.getElementById("detail")
    };

    function fmtNumber(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
      return Number(value).toFixed(digits).replace(/\\.00$/, "");
    }

    function shortText(text, max = 170) {
      const value = String(text || "");
      return value.length > max ? value.slice(0, max - 1) + "..." : value;
    }

    function normalizePattern(text) {
      return String(text || "")
        .toLowerCase()
        .replace(/\\s*->\\s*/g, " -> ")
        .replace(/\\s+/g, " ")
        .trim();
    }

    function el(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined && text !== null) node.textContent = text;
      return node;
    }

    function badge(text, kind) {
      return el("span", "badge" + (kind ? " " + kind : ""), text);
    }

    function pre(text) {
      return el("pre", "", text || "");
    }

    function jsonText(value) {
      if (typeof value === "string") return value;
      return JSON.stringify(value, null, 2);
    }

    function clear(node) {
      while (node.firstChild) node.removeChild(node.firstChild);
    }

    function metricNode(label, value) {
      const node = el("div", "stat");
      node.appendChild(el("span", "stat-label", label));
      node.appendChild(el("span", "stat-value", value));
      return node;
    }

    function smallMetric(label, value) {
      const node = el("div", "metric");
      node.appendChild(el("span", "stat-label", label));
      node.appendChild(el("span", "stat-value", value));
      return node;
    }

    function renderTop() {
      els.sourcePath.textContent = data.sourcePath;
      els.recordCount.textContent = data.summary.rollouts + " rollouts";
      clear(els.runMeta);
      const metadata = data.metadata || {};
      const metaItems = [
        ["model", metadata.model],
        ["env", metadata.env_id],
        ["avg reward", metadata.avg_reward],
        ["examples", metadata.num_examples],
        ["rollouts/example", metadata.rollouts_per_example]
      ];
      metaItems.forEach(([label, value]) => {
        if (value !== undefined && value !== null && value !== "") {
          els.runMeta.appendChild(badge(label + ": " + value));
        }
      });

      clear(els.summaryStrip);
      const summary = data.summary;
      const matchRate = summary.rollouts ? (summary.solved / summary.rollouts) * 100 : 0;
      const executableRate = summary.rollouts ? (summary.executable / summary.rollouts) * 100 : 0;
      els.summaryStrip.appendChild(metricNode("Rollouts", summary.rollouts));
      els.summaryStrip.appendChild(metricNode("Match", summary.solved + " (" + fmtNumber(matchRate, 1) + "%)"));
      els.summaryStrip.appendChild(metricNode("Executable", summary.executable + " (" + fmtNumber(executableRate, 1) + "%)"));
      els.summaryStrip.appendChild(metricNode("Avg Reward", fmtNumber(summary.avgReward, 3)));
      els.summaryStrip.appendChild(metricNode("Avg Turns", fmtNumber(summary.avgTurns, 1)));
      els.summaryStrip.appendChild(metricNode("Invalid Tools", fmtNumber(summary.invalidToolCalls, 0)));
      const tools = summary.toolTotals || {};
      els.summaryStrip.appendChild(metricNode("Tool Calls", Object.values(tools).reduce((a, b) => a + b, 0)));
    }

    function recordMatches(record) {
      const query = els.search.value.trim().toLowerCase();
      if (query) {
        const haystack = [
          record.exampleId,
          record.questionId,
          record.dbId,
          record.difficulty,
          record.question,
          record.evidence,
          record.finalSql,
          record.goldSql,
          record.toolPattern,
          (record.toolSequence || []).join(" "),
          (record.flags || []).join(" "),
          (record.failureTags || []).join(" "),
          (record.exploreCalls || []).map(call => call.sql).join(" ")
        ].join(" ").toLowerCase();
        if (!haystack.includes(query)) return false;
      }

      const patternQuery = normalizePattern(els.sequenceFilter.value);
      if (patternQuery && !normalizePattern(record.toolPattern).includes(patternQuery)) {
        return false;
      }

      const status = els.statusFilter.value;
      if (status === "matched" && record.executionMatch !== true) return false;
      if (status === "missed" && record.executionMatch !== false) return false;
      if (status === "executable" && record.sqlExecutable !== true) return false;
      if (status === "sql-error" && record.sqlExecutable !== false) return false;
      if (els.invalidOnly.checked && !(record.invalidToolCalls > 0)) return false;
      if (els.noExploreOnly.checked && !record.flags.includes("no explore")) return false;
      if (els.toolErrorOnly.checked && !(record.toolErrors || []).length) return false;
      return true;
    }

    function statusBadge(record) {
      if (record.executionMatch === true) return badge("match", "good");
      if (record.sqlExecutable === false) return badge("sql error", "bad");
      if (record.executionMatch === false) return badge("miss", "bad");
      return badge("unknown", "warn");
    }

    function renderList() {
      state.filtered = data.records.filter(recordMatches);
      clear(els.rolloutList);
      if (!state.filtered.length) {
        els.rolloutList.appendChild(el("div", "detail-empty", "No matching trajectories."));
        clear(els.detail);
        els.detail.appendChild(el("div", "detail-empty", "No trajectory selected."));
        return;
      }
      if (!state.filtered.some(record => record.ordinal === state.selectedOrdinal)) {
        state.selectedOrdinal = state.filtered[0].ordinal;
      }
      state.filtered.forEach(record => {
        const button = el("button", "rollout-button" + (record.ordinal === state.selectedOrdinal ? " active" : ""));
        button.type = "button";
        button.addEventListener("click", () => {
          state.selectedOrdinal = record.ordinal;
          renderList();
          renderDetail(record);
        });

        const title = el("div", "rollout-title");
        title.appendChild(el("span", "", "Example " + record.exampleId));
        title.appendChild(statusBadge(record));
        button.appendChild(title);
        button.appendChild(el("div", "rollout-question", shortText(record.question, 150)));
        const line = el("div", "mini-line");
        line.appendChild(badge(record.dbId || "db n/a"));
        line.appendChild(badge("r=" + fmtNumber(record.reward, 2)));
        line.appendChild(badge("turns=" + fmtNumber(record.turnCount, 0)));
        if (record.invalidToolCalls > 0) line.appendChild(badge("invalid=" + fmtNumber(record.invalidToolCalls, 0), "warn"));
        button.appendChild(line);
        els.rolloutList.appendChild(button);
      });
      const selected = state.filtered.find(record => record.ordinal === state.selectedOrdinal);
      if (selected) renderDetail(selected);
    }

    function renderPanel(title, bodyNode, extraClass) {
      const panel = el("section", "panel" + (extraClass ? " " + extraClass : ""));
      panel.appendChild(el("div", "panel-head", title));
      const body = el("div", "panel-body");
      body.appendChild(bodyNode);
      panel.appendChild(body);
      return panel;
    }

    function renderToolSequence(record) {
      const wrap = el("div", "tool-sequence");
      if (!record.toolSequence.length) {
        wrap.appendChild(el("span", "muted", "No tool calls found."));
        return wrap;
      }
      record.toolSequence.forEach(name => wrap.appendChild(el("span", "tool-pill", name)));
      return wrap;
    }

    function renderTagList(tags, emptyText) {
      const wrap = el("div", "badge-row");
      if (!tags || !tags.length) {
        wrap.appendChild(badge(emptyText || "none", "good"));
        return wrap;
      }
      tags.forEach(tag => {
        const severe = String(tag).toLowerCase().includes("error") || String(tag).toLowerCase().includes("wrong") || String(tag).toLowerCase().includes("missing");
        wrap.appendChild(badge(tag, severe ? "bad" : "warn"));
      });
      return wrap;
    }

    function renderRewardBreakdown(record) {
      const wrap = el("div", "reward-list");
      (record.rewardBreakdown || []).forEach(item => {
        const row = el("div", "reward-row");
        row.appendChild(el("span", "", item.label));
        row.appendChild(badge(fmtNumber(item.value, 3), item.earned ? "good" : ""));
        wrap.appendChild(row);
      });
      return wrap;
    }

    function renderSqlHeuristics(record) {
      const wrap = el("div", "reward-list");
      const comparison = record.sqlComparison || {};
      const rows = [
        ["Final tables", (comparison.finalTables || []).join(", ") || "none"],
        ["Gold tables", (comparison.goldTables || []).join(", ") || "none"],
        ["Missing identifiers", (comparison.missingIdentifiers || []).slice(0, 12).join(", ") || "none"]
      ];
      rows.forEach(([label, value]) => {
        const row = el("div", "reward-row");
        row.appendChild(el("span", "", label));
        row.appendChild(el("span", "muted", value));
        wrap.appendChild(row);
      });
      return wrap;
    }

    function outputStatusBadge(output) {
      if (!output) return badge("missing", "warn");
      if (output.status === "ok") return badge("ok", "good");
      if (output.status === "missing") return badge("missing", "warn");
      return badge("error", "bad");
    }

    function renderExecutionOutput(title, output) {
      const card = el("div", "execution-card");
      card.appendChild(el("div", "execution-title", title));
      const line = el("div", "badge-row");
      line.appendChild(outputStatusBadge(output));
      line.appendChild(badge("rows shown=" + fmtNumber(output && output.rowCount, 0)));
      if (output && output.totalRowCount !== undefined) {
        line.appendChild(badge("total rows=" + fmtNumber(output.totalRowCount, 0)));
      }
      if (output && output.cacheHit) line.appendChild(badge("cache hit", "good"));
      if (output && output.truncated) line.appendChild(badge("truncated", "warn"));
      if (output && output.elapsedMs !== null && output.elapsedMs !== undefined) {
        line.appendChild(badge(fmtNumber(output.elapsedMs, 1) + " ms"));
      }
      card.appendChild(line);
      if (output && output.error) card.appendChild(badge(output.error, output.status === "error" ? "bad" : "warn"));
      if (output && output.columns && output.columns.length && output.rows && output.rows.length) {
        card.appendChild(renderTable(output.columns, output.rows));
      } else if (output && output.status === "ok") {
        card.appendChild(el("span", "muted", "Query returned no preview rows."));
      }
      return card;
    }

    function renderExecutionComparison(record) {
      const comparison = record.executionComparison || {};
      const wrap = el("div", "");
      const badges = el("div", "badge-row");
      if (comparison.previewMatch === true) {
        badges.appendChild(badge("preview rows match", "good"));
      } else if (comparison.previewMatch === false) {
        badges.appendChild(badge("preview rows differ", "bad"));
      } else {
        badges.appendChild(badge("preview unavailable", "warn"));
      }
      badges.appendChild(badge("limit=" + (comparison.previewRows || "n/a") + " rows"));
      wrap.appendChild(badges);
      const grid = el("div", "execution-grid");
      grid.style.marginTop = "10px";
      grid.appendChild(renderExecutionOutput("Submitted output", comparison.submitted || {}));
      grid.appendChild(renderExecutionOutput("Gold output", comparison.gold || {}));
      wrap.appendChild(grid);
      return wrap;
    }

    function renderTable(columns, rows) {
      const wrap = el("div", "data-table-wrap");
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      columns.forEach(column => headerRow.appendChild(el("th", "", String(column))));
      thead.appendChild(headerRow);
      table.appendChild(thead);
      const tbody = document.createElement("tbody");
      rows.forEach(row => {
        const tr = document.createElement("tr");
        const values = Array.isArray(row) ? row : columns.map(column => row[column]);
        values.forEach(value => tr.appendChild(el("td", "", jsonText(value))));
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      return wrap;
    }

    function renderExploreSummary(record) {
      const wrap = el("div", "explore-list");
      const calls = record.exploreCalls || [];
      if (!calls.length) {
        wrap.appendChild(el("span", "muted", "No explore calls."));
        return wrap;
      }
      calls.forEach((call, index) => {
        const card = el("div", "explore-card");
        const line = el("div", "badge-row");
        line.appendChild(badge("explore " + (index + 1)));
        line.appendChild(badge(call.status || "unknown", call.status === "ok" ? "good" : "bad"));
        line.appendChild(badge("rows=" + fmtNumber(call.rowCount, 0)));
        if (call.truncated) line.appendChild(badge("truncated", "warn"));
        if (call.elapsedMs !== null && call.elapsedMs !== undefined) line.appendChild(badge(fmtNumber(call.elapsedMs, 1) + " ms"));
        card.appendChild(line);
        card.appendChild(pre(call.sql || ""));
        if (call.error) card.appendChild(badge(call.error, "bad"));
        if (call.columns && call.columns.length && call.rows && call.rows.length) {
          card.appendChild(renderTable(call.columns, call.rows));
        }
        card.appendChild(el("span", "muted", "tool #" + call.messageIndex + " -> response #" + call.responseIndex));
        wrap.appendChild(card);
      });
      return wrap;
    }

    function renderToolPayload(payload) {
      const wrap = el("div", "");
      if (payload && typeof payload === "object" && !Array.isArray(payload)) {
        if (payload.status === "error" || payload.error) {
          wrap.appendChild(badge(payload.message || payload.error || "tool error", "bad"));
        }
        if (Array.isArray(payload.columns) && Array.isArray(payload.rows)) {
          wrap.appendChild(renderTable(payload.columns, payload.rows));
          if (payload.truncated) wrap.appendChild(badge("truncated", "warn"));
          return wrap;
        }
        if (Array.isArray(payload.results)) {
          payload.results.slice(0, 6).forEach(result => {
            const title = result.table || result.type || "result";
            wrap.appendChild(el("h3", "", title));
            if (Array.isArray(result.columns)) {
              wrap.appendChild(renderTable(
                ["name", "type", "primary_key", "description"],
                result.columns.map(column => ({
                  name: column.name,
                  type: column.type,
                  primary_key: column.primary_key,
                  description: column.description
                }))
              ));
            }
            if (result.sqlite_schema) wrap.appendChild(pre(result.sqlite_schema));
          });
          if (payload.results.length > 6) {
            wrap.appendChild(badge("showing first 6 describe results", "warn"));
          }
          return wrap;
        }
      }
      wrap.appendChild(pre(jsonText(payload)));
      return wrap;
    }

    function renderMessageNote(title, content, className) {
      const note = el("div", "message-note" + (className ? " " + className : ""));
      note.appendChild(el("div", "message-note-title", title));
      note.appendChild(pre(content || ""));
      return note;
    }

    function renderMessage(message) {
      const node = el("article", "message");
      const head = el("div", "message-head");
      head.appendChild(el("span", "role", message.role));
      head.appendChild(el("span", "message-index", "#" + message.index + " " + message.section));
      node.appendChild(head);
      const body = el("div", "message-body");
      if (message.thinkingBlocks && message.thinkingBlocks.length) {
        message.thinkingBlocks.forEach(block => {
          body.appendChild(renderMessageNote(block.label || "Thinking", block.text || "", "thinking"));
        });
      }
      if (message.content && message.role !== "tool") {
        if (message.role === "assistant" && message.toolCalls && message.toolCalls.length) {
          body.appendChild(renderMessageNote("Assistant text with tool call", message.content, "assistant-text"));
        } else {
          body.appendChild(pre(message.content));
        }
      }
      if (message.toolCalls && message.toolCalls.length) {
        message.toolCalls.forEach(call => {
          const callNode = el("div", "tool-call");
          callNode.appendChild(el("div", "tool-call-name", call.name));
          const callBody = el("div", "message-body");
          callBody.appendChild(pre(call.argumentsText || jsonText(call.arguments)));
          callNode.appendChild(callBody);
          body.appendChild(callNode);
        });
      }
      if (message.role === "tool") {
        body.appendChild(renderToolPayload(message.toolPayload));
      }
      if (!body.childNodes.length) {
        body.appendChild(el("span", "muted", "Empty message."));
      }
      node.appendChild(body);
      return node;
    }

    function renderPostmortem(record) {
      const panel = el("section", "postmortem-section");
      panel.appendChild(el("div", "postmortem-title", "Trajectory postmortem"));
      const grid = el("div", "postmortem-grid");
      [
        ["Reward breakdown", renderRewardBreakdown(record)],
        ["Failure tags", renderTagList(record.failureTags, "no failure tags")],
        ["SQL heuristics", renderSqlHeuristics(record)]
      ].forEach(([title, content]) => {
        const block = el("div", "postmortem-block");
        block.appendChild(el("h3", "", title));
        block.appendChild(content);
        grid.appendChild(block);
      });
      panel.appendChild(grid);

      const sqlGrid = el("div", "section-grid");
      sqlGrid.style.marginTop = "14px";
      [
        ["Final submission", pre(record.finalSql || "No final SQL found.")],
        ["Gold SQL", pre(record.goldSql || "No gold SQL found.")]
      ].forEach(([title, content]) => {
        const block = el("div", "postmortem-block");
        block.appendChild(el("h3", "", title));
        block.appendChild(content);
        sqlGrid.appendChild(block);
      });
      panel.appendChild(sqlGrid);

      return panel;
    }

    function renderDetail(record) {
      clear(els.detail);
      const head = el("div", "detail-head");
      const title = el("div", "detail-title");
      title.appendChild(el("h2", "", record.question || "Untitled trajectory"));
      title.appendChild(statusBadge(record));
      head.appendChild(title);
      const badges = el("div", "badge-row");
      badges.appendChild(badge("example " + record.exampleId));
      if (record.questionId !== undefined && record.questionId !== null) badges.appendChild(badge("question " + record.questionId));
      badges.appendChild(badge(record.dbId || "db n/a"));
      badges.appendChild(badge(record.difficulty || "unknown"));
      badges.appendChild(badge("stop: " + (record.stopCondition || "n/a")));
      (record.flags || []).forEach(flag => badges.appendChild(badge(flag, flag.includes("miss") || flag.includes("error") ? "bad" : "warn")));
      head.appendChild(badges);
      els.detail.appendChild(head);

      const metrics = el("div", "metric-grid");
      metrics.appendChild(smallMetric("Reward", fmtNumber(record.reward, 3)));
      metrics.appendChild(smallMetric("Turns", fmtNumber(record.turnCount, 0)));
      metrics.appendChild(smallMetric("Invalid Tools", fmtNumber(record.invalidToolCalls, 0)));
      metrics.appendChild(smallMetric("Total Tools", record.toolSequence.length));
      els.detail.appendChild(metrics);

      els.detail.appendChild(renderPanel("Executed outputs", renderExecutionComparison(record), "full"));

      const grid = el("div", "section-grid");
      if (record.evidence) grid.appendChild(renderPanel("Evidence", pre(record.evidence)));
      if (record.tableNames && record.tableNames.length) {
        const list = el("ul", "compact-list");
        record.tableNames.forEach(name => list.appendChild(el("li", "", name)));
        grid.appendChild(renderPanel("Available tables", list));
      }
      grid.appendChild(renderPanel("Tool sequence", renderToolSequence(record), "full"));
      grid.appendChild(renderPanel("Final submission", pre(record.finalSql || "No final SQL found.")));
      grid.appendChild(renderPanel("Gold SQL", pre(record.goldSql || "No gold SQL found.")));
      grid.appendChild(renderPanel("Explore summary", renderExploreSummary(record), "full"));
      els.detail.appendChild(grid);

      const timelinePanel = el("section", "panel full");
      timelinePanel.appendChild(el("div", "panel-head", "Timeline"));
      const timelineBody = el("div", "panel-body");
      const timeline = el("div", "timeline");
      record.messages.forEach(message => timeline.appendChild(renderMessage(message)));
      timelineBody.appendChild(timeline);
      timelinePanel.appendChild(timelineBody);
      els.detail.appendChild(timelinePanel);
      els.detail.appendChild(renderPostmortem(record));
    }

    function bindFilters() {
      [els.search, els.statusFilter, els.sequenceFilter, els.invalidOnly, els.noExploreOnly, els.toolErrorOnly].forEach(input => {
        input.addEventListener("input", renderList);
        input.addEventListener("change", renderList);
      });
    }

    renderTop();
    bindFilters();
    renderList();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data = build_dashboard_data(args.results_jsonl)
    output_path = args.output or args.results_jsonl.with_name(
        f"{args.results_jsonl.stem}_dashboard.html"
    )
    output_path.write_text(build_html(data), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
