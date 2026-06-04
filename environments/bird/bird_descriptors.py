from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Protocol
from urllib.parse import quote


DESCRIPTOR_SCHEMA_VERSION = 1
DEFAULT_TOP_VALUES_LIMIT = 10
DEFAULT_SAMPLE_VALUES_LIMIT = 5
DEFAULT_LARGE_TABLE_SAMPLE_ROWS = 100_000
DEFAULT_MAX_LLM_RETRIES = 2


class DescriptionClient(Protocol):
    model: str

    def describe(self, *, scope: str, payload: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class DescriptorPaths:
    profile: Path
    manifest: Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def descriptor_split_group(split: str) -> str:
    normalized = split.strip().lower()
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"dev", "test", "val", "valid", "validation", "mini-dev", "mini_dev", "full-dev", "full_dev"}:
        return "dev"
    raise ValueError("split must be one of: train, dev, val, or test")


def descriptor_root(data_dir: str | os.PathLike[str]) -> Path:
    return Path(data_dir).expanduser().resolve() / "metadata" / "descriptors"


def descriptor_paths(data_dir: str | os.PathLike[str], split: str, db_id: str) -> DescriptorPaths:
    root = descriptor_root(data_dir)
    split_group = descriptor_split_group(split)
    return DescriptorPaths(
        profile=root / split_group / f"{db_id}.profile.json",
        manifest=root / "manifest.json",
    )


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(output_path)
    return output_path


def load_profile(
    data_dir: str | os.PathLike[str], split: str, db_id: str
) -> dict[str, Any] | None:
    path = descriptor_paths(data_dir, split, db_id).profile
    if not path.exists():
        return None
    try:
        profile = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if profile.get("schema_version") != DESCRIPTOR_SCHEMA_VERSION:
        return None
    if profile.get("artifact_type") != "profile":
        return None
    if profile.get("db_id") != db_id:
        return None
    return profile


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _db_uri(path: Path) -> str:
    return "file:" + quote(str(path.resolve()), safe="/") + "?mode=ro"


def connect_readonly(db_path: str | os.PathLike[str]) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    conn = sqlite3.connect(_db_uri(path), uri=True)
    conn.execute("PRAGMA query_only = ON")
    conn.row_factory = sqlite3.Row
    return conn


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 6)
    return value


def _rows_as_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [{key: _json_value(row[key]) for key in row.keys()} for row in rows]


def _sqlite_tables(conn: sqlite3.Connection) -> list[dict[str, str]]:
    rows = conn.execute(
        """
        SELECT name, type, sql
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()
    return [
        {
            "name": str(row["name"]),
            "type": str(row["type"]),
            "ddl": str(row["sql"] or ""),
        }
        for row in rows
    ]


def _table_columns(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA table_xinfo({_quote_identifier(table_name)})").fetchall()
    columns = []
    for row in rows:
        hidden = int(row["hidden"]) if "hidden" in row.keys() else 0
        columns.append(
            {
                "cid": int(row["cid"]),
                "name": str(row["name"]),
                "declared_type": str(row["type"] or ""),
                "not_null": bool(row["notnull"]),
                "default_value": row["dflt_value"],
                "primary_key_position": int(row["pk"] or 0),
                "hidden": hidden,
            }
        )
    return columns


def _table_foreign_keys(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA foreign_key_list({_quote_identifier(table_name)})").fetchall()
    keys = []
    for row in rows:
        keys.append(
            {
                "id": int(row["id"]),
                "seq": int(row["seq"]),
                "from": str(row["from"]),
                "to_table": str(row["table"]),
                "to": str(row["to"]),
                "on_update": str(row["on_update"]),
                "on_delete": str(row["on_delete"]),
                "match": str(row["match"]),
            }
        )
    return keys


def _table_indexes(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA index_list({_quote_identifier(table_name)})").fetchall()
    indexes = []
    for row in rows:
        index_name = str(row["name"])
        index_info = conn.execute(f"PRAGMA index_info({_quote_identifier(index_name)})").fetchall()
        indexes.append(
            {
                "name": index_name,
                "unique": bool(row["unique"]),
                "origin": str(row["origin"]),
                "partial": bool(row["partial"]),
                "columns": [str(info["name"]) for info in index_info],
            }
        )
    return indexes


def _count_rows(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {_quote_identifier(table_name)}").fetchone()
    return int(row["count"] or 0)


def _profile_source(table_name: str, row_count: int, large_table_sample_rows: int) -> tuple[str, int, bool]:
    table = _quote_identifier(table_name)
    if large_table_sample_rows <= 0 or row_count <= large_table_sample_rows:
        return table, row_count, False
    return f"(SELECT * FROM {table} LIMIT {int(large_table_sample_rows)})", int(large_table_sample_rows), True


def _declared_type_family(declared_type: str) -> str:
    value = declared_type.strip().upper()
    if not value:
        return "unknown"
    if "INT" in value:
        return "integer"
    if any(marker in value for marker in ("REAL", "FLOA", "DOUB")):
        return "real"
    if any(marker in value for marker in ("NUM", "DEC", "BOOL")):
        return "numeric"
    if any(marker in value for marker in ("DATE", "TIME")):
        return "date_like"
    if any(marker in value for marker in ("CHAR", "CLOB", "TEXT", "VARCHAR")):
        return "text"
    if "BLOB" in value:
        return "blob"
    return "unknown"


_DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$"),
    re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"),
    re.compile(r"^\d{6}$"),
    re.compile(r"^\d{8}$"),
]


def _date_like_ratio(values: list[Any]) -> float:
    strings = [str(value).strip() for value in values if value is not None and str(value).strip()]
    if not strings:
        return 0.0
    matches = 0
    for value in strings:
        if any(pattern.match(value) for pattern in _DATE_PATTERNS):
            matches += 1
    return matches / len(strings)


def _infer_column_type(
    *,
    declared_type: str,
    storage_types: dict[str, int],
    sample_values: list[Any],
) -> str:
    declared_family = _declared_type_family(declared_type)
    non_null_storage = {key: value for key, value in storage_types.items() if key != "null" and value}
    if not non_null_storage:
        return declared_family if declared_family != "unknown" else "null"
    if set(non_null_storage).issubset({"integer"}):
        return "integer"
    if set(non_null_storage).issubset({"integer", "real"}):
        return "numeric"
    if declared_family == "date_like" or _date_like_ratio(sample_values) >= 0.8:
        return "date_like"
    if set(non_null_storage).issubset({"text"}):
        return "text"
    if set(non_null_storage).issubset({"blob"}):
        return "blob"
    if declared_family != "unknown":
        return f"mixed_{declared_family}"
    return "mixed"


def _safe_scalar_query(
    conn: sqlite3.Connection, sql: str, default: Any = None
) -> sqlite3.Row | None:
    try:
        return conn.execute(sql).fetchone()
    except sqlite3.Error:
        return default


def _type_distribution(
    conn: sqlite3.Connection, source_sql: str, column_name: str
) -> dict[str, int]:
    column = _quote_identifier(column_name)
    rows = conn.execute(
        f"""
        SELECT typeof({column}) AS storage_type, COUNT(*) AS count
        FROM {source_sql}
        GROUP BY storage_type
        ORDER BY count DESC, storage_type
        """
    ).fetchall()
    return {str(row["storage_type"]): int(row["count"] or 0) for row in rows}


def _sample_values(
    conn: sqlite3.Connection, source_sql: str, column_name: str, limit: int
) -> list[Any]:
    if limit <= 0:
        return []
    column = _quote_identifier(column_name)
    rows = conn.execute(
        f"""
        SELECT {column} AS value
        FROM {source_sql}
        WHERE {column} IS NOT NULL
        LIMIT {int(limit)}
        """
    ).fetchall()
    return [_json_value(row["value"]) for row in rows]


def _top_values(
    conn: sqlite3.Connection, source_sql: str, column_name: str, limit: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    column = _quote_identifier(column_name)
    rows = conn.execute(
        f"""
        SELECT {column} AS value, COUNT(*) AS count
        FROM {source_sql}
        WHERE {column} IS NOT NULL
        GROUP BY {column}
        ORDER BY count DESC, CAST({column} AS TEXT) ASC
        LIMIT {int(limit)}
        """
    ).fetchall()
    return [{"value": _json_value(row["value"]), "count": int(row["count"] or 0)} for row in rows]


def _numeric_stats(
    conn: sqlite3.Connection, source_sql: str, column_name: str
) -> dict[str, Any] | None:
    column = _quote_identifier(column_name)
    row = _safe_scalar_query(
        conn,
        f"""
        SELECT
          COUNT(*) AS count,
          MIN(CAST({column} AS REAL)) AS min,
          MAX(CAST({column} AS REAL)) AS max,
          AVG(CAST({column} AS REAL)) AS mean,
          AVG(CAST({column} AS REAL) * CAST({column} AS REAL)) AS mean_square,
          SUM(CAST({column} AS REAL)) AS sum,
          SUM(CASE WHEN CAST({column} AS REAL) = 0 THEN 1 ELSE 0 END) AS zero_count,
          SUM(CASE WHEN CAST({column} AS REAL) < 0 THEN 1 ELSE 0 END) AS negative_count
        FROM {source_sql}
        WHERE {column} IS NOT NULL AND typeof({column}) IN ('integer', 'real')
        """,
    )
    if row is None or int(row["count"] or 0) == 0:
        return None
    mean = float(row["mean"] or 0.0)
    mean_square = float(row["mean_square"] or 0.0)
    variance = max(0.0, mean_square - mean * mean)
    return {
        "count": int(row["count"] or 0),
        "min": _json_value(row["min"]),
        "max": _json_value(row["max"]),
        "mean": _json_value(mean),
        "sum": _json_value(row["sum"]),
        "variance": _json_value(variance),
        "stddev": _json_value(math.sqrt(variance)),
        "zero_count": int(row["zero_count"] or 0),
        "negative_count": int(row["negative_count"] or 0),
    }


def _text_length_stats(
    conn: sqlite3.Connection, source_sql: str, column_name: str
) -> dict[str, Any] | None:
    column = _quote_identifier(column_name)
    row = _safe_scalar_query(
        conn,
        f"""
        SELECT
          COUNT(*) AS count,
          MIN(LENGTH(CAST({column} AS TEXT))) AS min_length,
          MAX(LENGTH(CAST({column} AS TEXT))) AS max_length,
          AVG(LENGTH(CAST({column} AS TEXT))) AS avg_length
        FROM {source_sql}
        WHERE {column} IS NOT NULL
        """,
    )
    if row is None or int(row["count"] or 0) == 0:
        return None
    return {
        "count": int(row["count"] or 0),
        "min_length": int(row["min_length"] or 0),
        "max_length": int(row["max_length"] or 0),
        "avg_length": _json_value(row["avg_length"]),
    }


def _date_like_stats(
    conn: sqlite3.Connection, source_sql: str, column_name: str
) -> dict[str, Any] | None:
    column = _quote_identifier(column_name)
    row = _safe_scalar_query(
        conn,
        f"""
        SELECT MIN(CAST({column} AS TEXT)) AS min, MAX(CAST({column} AS TEXT)) AS max
        FROM {source_sql}
        WHERE {column} IS NOT NULL
        """,
    )
    if row is None or row["min"] is None:
        return None
    return {"min": str(row["min"]), "max": str(row["max"])}


def _profile_column(
    conn: sqlite3.Connection,
    *,
    source_sql: str,
    table_name: str,
    column: dict[str, Any],
    table_row_count: int,
    profile_rows: int,
    is_sampled: bool,
    top_values_limit: int,
    sample_values_limit: int,
) -> dict[str, Any]:
    column_name = str(column["name"])
    quoted_column = _quote_identifier(column_name)
    row = conn.execute(
        f"""
        SELECT
          COUNT(*) AS profile_rows,
          SUM(CASE WHEN {quoted_column} IS NULL THEN 1 ELSE 0 END) AS null_count,
          COUNT({quoted_column}) AS non_null_count,
          COUNT(DISTINCT {quoted_column}) AS distinct_count,
          MIN({quoted_column}) AS min_value,
          MAX({quoted_column}) AS max_value
        FROM {source_sql}
        """
    ).fetchone()
    actual_profile_rows = int(row["profile_rows"] or 0)
    null_count = int(row["null_count"] or 0)
    non_null_count = int(row["non_null_count"] or 0)
    sample_values = _sample_values(conn, source_sql, column_name, sample_values_limit)
    storage_types = _type_distribution(conn, source_sql, column_name)
    inferred_type = _infer_column_type(
        declared_type=str(column.get("declared_type") or ""),
        storage_types=storage_types,
        sample_values=sample_values,
    )
    profile: dict[str, Any] = {
        "name": column_name,
        "table": table_name,
        "cid": column["cid"],
        "declared_type": column["declared_type"],
        "declared_type_family": _declared_type_family(str(column.get("declared_type") or "")),
        "inferred_type": inferred_type,
        "not_null": column["not_null"],
        "default_value": column["default_value"],
        "primary_key_position": column["primary_key_position"],
        "hidden": column["hidden"],
        "table_row_count": table_row_count,
        "profile_rows": actual_profile_rows,
        "profile_is_sampled": is_sampled,
        "null_count": null_count,
        "non_null_count": non_null_count,
        "null_fraction": round(null_count / actual_profile_rows, 6) if actual_profile_rows else 0.0,
        "distinct_count": int(row["distinct_count"] or 0),
        "distinct_count_is_exact": not is_sampled,
        "distinct_fraction": round(int(row["distinct_count"] or 0) / non_null_count, 6)
        if non_null_count
        else 0.0,
        "storage_types": storage_types,
        "min_value": _json_value(row["min_value"]),
        "max_value": _json_value(row["max_value"]),
        "sample_values": sample_values,
        "top_values": _top_values(conn, source_sql, column_name, top_values_limit),
    }
    numeric_stats = _numeric_stats(conn, source_sql, column_name)
    if numeric_stats is not None:
        profile["numeric_stats"] = numeric_stats
    text_stats = _text_length_stats(conn, source_sql, column_name)
    if text_stats is not None and inferred_type in {"text", "date_like", "mixed", "mixed_text"}:
        profile["text_length_stats"] = text_stats
    if inferred_type == "date_like":
        date_stats = _date_like_stats(conn, source_sql, column_name)
        if date_stats is not None:
            profile["date_like_stats"] = date_stats
    return profile


def profile_database(
    *,
    db_path: str | os.PathLike[str],
    db_id: str,
    split_group: str,
    top_values_limit: int = DEFAULT_TOP_VALUES_LIMIT,
    sample_values_limit: int = DEFAULT_SAMPLE_VALUES_LIMIT,
    large_table_sample_rows: int = DEFAULT_LARGE_TABLE_SAMPLE_ROWS,
) -> dict[str, Any]:
    path = Path(db_path).expanduser().resolve()
    conn = connect_readonly(path)
    tables: list[dict[str, Any]] = []
    try:
        for table in _sqlite_tables(conn):
            table_name = table["name"]
            row_count = _count_rows(conn, table_name) if table["type"] == "table" else 0
            source_sql, profile_rows, is_sampled = _profile_source(
                table_name, row_count, large_table_sample_rows
            )
            columns = _table_columns(conn, table_name)
            column_profiles = [
                _profile_column(
                    conn,
                    source_sql=source_sql,
                    table_name=table_name,
                    column=column,
                    table_row_count=row_count,
                    profile_rows=profile_rows,
                    is_sampled=is_sampled,
                    top_values_limit=top_values_limit,
                    sample_values_limit=sample_values_limit,
                )
                for column in columns
                if not column.get("hidden")
            ]
            primary_keys = [
                column["name"]
                for column in sorted(columns, key=lambda item: int(item["primary_key_position"] or 0))
                if int(column["primary_key_position"] or 0) > 0
            ]
            foreign_keys = _table_foreign_keys(conn, table_name)
            tables.append(
                {
                    "name": table_name,
                    "type": table["type"],
                    "ddl": table["ddl"],
                    "row_count": row_count,
                    "profile_rows": profile_rows,
                    "profile_is_sampled": is_sampled,
                    "column_count": len(column_profiles),
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys,
                    "indexes": _table_indexes(conn, table_name),
                    "columns": column_profiles,
                }
            )
    finally:
        conn.close()

    relationship_count = sum(len(table["foreign_keys"]) for table in tables)
    profile = {
        "database": {
            "db_id": db_id,
            "split_group": split_group,
            "sqlite_path": str(path),
            "table_count": len(tables),
            "table_names": [table["name"] for table in tables],
            "total_rows": sum(int(table["row_count"]) for table in tables),
            "relationship_count": relationship_count,
        },
        "tables": tables,
    }
    return {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "artifact_type": "profile",
        "generated_at": utc_now_iso(),
        "db_id": db_id,
        "split_group": split_group,
        "settings": {
            "top_values_limit": int(top_values_limit),
            "sample_values_limit": int(sample_values_limit),
            "large_table_sample_rows": int(large_table_sample_rows),
        },
        "profile": profile,
        "generation": {
            "profile_status": "complete",
            "llm_status": "not_started",
        },
    }


def ensure_profile(
    *,
    data_dir: str | os.PathLike[str],
    split: str,
    db_id: str,
    db_path: str | os.PathLike[str],
    top_values_limit: int = DEFAULT_TOP_VALUES_LIMIT,
    sample_values_limit: int = DEFAULT_SAMPLE_VALUES_LIMIT,
    large_table_sample_rows: int = DEFAULT_LARGE_TABLE_SAMPLE_ROWS,
    force: bool = False,
) -> tuple[dict[str, Any], bool, Path]:
    paths = descriptor_paths(data_dir, split, db_id)
    if not force:
        cached = load_profile(data_dir, split, db_id)
        if cached is not None:
            return cached, False, paths.profile
    split_group = descriptor_split_group(split)
    profile = profile_database(
        db_path=db_path,
        db_id=db_id,
        split_group=split_group,
        top_values_limit=top_values_limit,
        sample_values_limit=sample_values_limit,
        large_table_sample_rows=large_table_sample_rows,
    )
    write_json(paths.profile, profile)
    update_manifest(data_dir=data_dir, profile_artifact=profile, profile_path=paths.profile)
    return profile, True, paths.profile


def _compact_column_for_prompt(column: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "name",
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
    return {key: column[key] for key in keep if key in column}


def _compact_table_for_prompt(table: dict[str, Any], include_columns: bool = True) -> dict[str, Any]:
    payload = {
        "name": table.get("name"),
        "row_count": table.get("row_count"),
        "profile_rows": table.get("profile_rows"),
        "profile_is_sampled": table.get("profile_is_sampled"),
        "column_count": table.get("column_count"),
        "primary_keys": table.get("primary_keys"),
        "foreign_keys": table.get("foreign_keys"),
        "indexes": table.get("indexes"),
        "ddl": table.get("ddl"),
    }
    if include_columns:
        payload["columns"] = [
            _compact_column_for_prompt(column) for column in table.get("columns") or []
        ]
    return payload


def prompt_payload_for_database(profile_artifact: dict[str, Any]) -> dict[str, Any]:
    profile = profile_artifact["profile"]
    return {
        "database": profile["database"],
        "tables": [
            {
                "name": table.get("name"),
                "row_count": table.get("row_count"),
                "column_count": table.get("column_count"),
                "primary_keys": table.get("primary_keys"),
                "foreign_keys": table.get("foreign_keys"),
            }
            for table in profile.get("tables") or []
        ],
    }


def prompt_payload_for_table(
    profile_artifact: dict[str, Any], table_name: str
) -> dict[str, Any] | None:
    table = find_profile_table(profile_artifact, table_name)
    if table is None:
        return None
    return {
        "database": profile_artifact["profile"]["database"],
        "table": _compact_table_for_prompt(table, include_columns=True),
    }


def prompt_payload_for_column(
    profile_artifact: dict[str, Any], table_name: str, column_name: str
) -> dict[str, Any] | None:
    table = find_profile_table(profile_artifact, table_name)
    column = find_profile_column(table, column_name)
    if table is None or column is None:
        return None
    return {
        "database": profile_artifact["profile"]["database"],
        "table": _compact_table_for_prompt(table, include_columns=False),
        "column": _compact_column_for_prompt(column),
    }


def _validate_description(payload: dict[str, Any], *, scope: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError(f"{scope} description response must be a JSON object")
    description = payload.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{scope} response is missing description")
    return {"description": description.strip()}


def _call_with_retries(callable_obj: Any, payload: dict[str, Any], *, scope: str, max_retries: int) -> dict[str, str]:
    last_error: Exception | None = None
    for _attempt in range(max_retries + 1):
        try:
            return _validate_description(callable_obj(payload), scope=scope)
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def describe_profile_target(
    *,
    client: DescriptionClient,
    scope: str,
    payload: dict[str, Any],
    max_retries: int = DEFAULT_MAX_LLM_RETRIES,
) -> dict[str, str]:
    return _call_with_retries(
        lambda prompt_payload: client.describe(scope=scope, payload=prompt_payload),
        payload,
        scope=scope,
        max_retries=max_retries,
    )


def _extract_json_object(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(content[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


class OpenAIDescriptionClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._client = OpenAI(**kwargs)
        self.model = model

    def describe(self, *, scope: str, payload: dict[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "You write accurate data dictionary descriptions from SQLite database profiles. "
            "Use only the provided schema and statistical profile. Do not invent source metadata. "
            "Return one JSON object with exactly one string key: description."
        )
        user_prompt = (
            f"Write a useful description for this {scope}.\n\n"
            "Keep it concise but substantive. Explain likely meaning, units or formats, "
            "nullability, important values, relationships, and SQL query hints only when the "
            "profile supports them.\n\n"
            f"Profile JSON:\n{json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        return _extract_json_object(content)


def iter_sqlite_databases(
    data_dir: str | os.PathLike[str], split: str
) -> list[tuple[str, Path]]:
    root = Path(data_dir).expanduser().resolve()
    split_group = descriptor_split_group(split)
    if split_group == "train":
        db_root = root / "train" / "train_databases"
    else:
        db_root = root / "dev_20240627" / "dev_databases"
    if not db_root.exists():
        raise FileNotFoundError(f"SQLite database root not found: {db_root}")
    pairs: list[tuple[str, Path]] = []
    for path in sorted(db_root.glob("*/*.sqlite")):
        pairs.append((path.stem, path))
    return pairs


def update_manifest(
    *,
    data_dir: str | os.PathLike[str],
    profile_artifact: dict[str, Any],
    profile_path: str | os.PathLike[str],
) -> dict[str, Any]:
    root = descriptor_root(data_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            manifest = {}
    else:
        manifest = {}
    manifest.setdefault("schema_version", DESCRIPTOR_SCHEMA_VERSION)
    manifest.setdefault("updated_at", utc_now_iso())
    manifest.setdefault("databases", {})
    key = f"{profile_artifact['split_group']}/{profile_artifact['db_id']}"
    entry = {
        "db_id": profile_artifact["db_id"],
        "split_group": profile_artifact["split_group"],
        "profile_path": str(Path(profile_path).resolve()),
        "profile_status": (profile_artifact.get("generation") or {}).get("profile_status"),
        "updated_at": utc_now_iso(),
    }
    manifest["databases"][key] = entry
    manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, manifest)
    return manifest


def _table_matches(table: dict[str, Any], table_name: str) -> bool:
    return str(table.get("name", "")).lower() == table_name.lower()


def find_profile_table(profile_artifact: dict[str, Any] | None, table_name: str) -> dict[str, Any] | None:
    if not profile_artifact:
        return None
    profile = profile_artifact.get("profile")
    if not isinstance(profile, dict):
        return None
    for table in profile.get("tables") or []:
        if isinstance(table, dict) and _table_matches(table, table_name):
            return table
    return None


def find_profile_column(
    table_profile: dict[str, Any] | None, column_name: str
) -> dict[str, Any] | None:
    if not table_profile:
        return None
    for column in table_profile.get("columns") or []:
        if isinstance(column, dict) and str(column.get("name", "")).lower() == column_name.lower():
            return column
    return None

