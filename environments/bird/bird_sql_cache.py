from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable


CACHE_SCHEMA_VERSION = 1
DEFAULT_SQL_RESULT_CACHE_RELATIVE = Path(".cache") / "sql_results"
SQL_RESULT_CACHE_ENV_VAR = "BIRD_SQL_RESULT_CACHE_DIR"


ExecuteSQL = Callable[..., dict[str, Any]]


def infer_bird_data_dir_from_db_path(db_path: str | os.PathLike[str]) -> Path | None:
    path = Path(db_path).expanduser().resolve()
    parts = path.parts
    for marker in ("dev_20240627", "train"):
        if marker not in parts:
            continue
        index = parts.index(marker)
        if index > 0:
            return Path(*parts[:index])
    return None


def resolve_sql_result_cache_dir(
    *,
    data_dir: str | os.PathLike[str] | None = None,
    db_path: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()

    env_value = os.environ.get(SQL_RESULT_CACHE_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser().resolve()

    if data_dir:
        return Path(data_dir).expanduser().resolve() / DEFAULT_SQL_RESULT_CACHE_RELATIVE

    if db_path:
        inferred_data_dir = infer_bird_data_dir_from_db_path(db_path)
        if inferred_data_dir is not None:
            return inferred_data_dir / DEFAULT_SQL_RESULT_CACHE_RELATIVE

    return Path.cwd() / DEFAULT_SQL_RESULT_CACHE_RELATIVE


def _db_fingerprint(db_path: Path) -> dict[str, Any]:
    stat = db_path.stat()
    return {
        "path": str(db_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def sql_result_cache_key(*, db_path: str | os.PathLike[str], sql: str) -> str:
    path = Path(db_path).expanduser().resolve()
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "db": _db_fingerprint(path),
        "sql": " ".join(sql.strip().rstrip(";").split()),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sql_result_cache_path(
    *,
    cache_dir: str | os.PathLike[str],
    db_path: str | os.PathLike[str],
    sql: str,
) -> Path:
    key = sql_result_cache_key(db_path=db_path, sql=sql)
    return Path(cache_dir).expanduser().resolve() / key[:2] / f"{key}.json"


def _json_rows(rows: Any) -> list[list[Any]]:
    if not isinstance(rows, list):
        return []
    return [list(row) if isinstance(row, (list, tuple)) else [row] for row in rows]


def _tuple_rows(rows: Any) -> list[tuple[Any, ...]]:
    if not isinstance(rows, list):
        return []
    tupled: list[tuple[Any, ...]] = []
    for row in rows:
        if isinstance(row, list):
            tupled.append(tuple(row))
        elif isinstance(row, tuple):
            tupled.append(row)
        else:
            tupled.append((row,))
    return tupled


def _cache_payload(
    *,
    db_path: str | os.PathLike[str],
    sql: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "db": _db_fingerprint(Path(db_path).expanduser().resolve()),
        "sql": sql,
        "ok": bool(result.get("ok")),
        "columns": list(result.get("columns") or []),
        "rows": _json_rows(result.get("rows")),
        "truncated": bool(result.get("truncated")),
        "elapsed_ms": result.get("elapsed_ms"),
        "error": result.get("error"),
    }


def _result_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    return {
        "ok": bool(payload.get("ok")),
        "columns": list(payload.get("columns") or []),
        "rows": _tuple_rows(payload.get("rows")),
        "truncated": bool(payload.get("truncated")),
        "elapsed_ms": payload.get("elapsed_ms"),
        "error": payload.get("error"),
        "cache_hit": True,
        "cached_at_unix": payload.get("created_at_unix"),
    }


def read_cached_sql_result(
    *,
    cache_dir: str | os.PathLike[str],
    db_path: str | os.PathLike[str],
    sql: str,
) -> dict[str, Any] | None:
    path = sql_result_cache_path(cache_dir=cache_dir, db_path=db_path, sql=sql)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return _result_from_payload(payload)


def write_cached_sql_result(
    *,
    cache_dir: str | os.PathLike[str],
    db_path: str | os.PathLike[str],
    sql: str,
    result: dict[str, Any],
) -> Path | None:
    if not result.get("ok") or result.get("truncated"):
        return None

    path = sql_result_cache_path(cache_dir=cache_dir, db_path=db_path, sql=sql)
    payload = _cache_payload(db_path=db_path, sql=sql, result=result)
    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
        tmp_path.replace(path)
    except (OSError, TypeError, ValueError):
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return None
    return path


def execute_sql_with_cache(
    *,
    db_path: str | os.PathLike[str],
    sql: str,
    max_seconds: float,
    execute_fn: ExecuteSQL,
    cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    resolved_cache_dir = resolve_sql_result_cache_dir(
        db_path=db_path, cache_dir=cache_dir
    )
    cached = read_cached_sql_result(
        cache_dir=resolved_cache_dir,
        db_path=db_path,
        sql=sql,
    )
    if cached is not None:
        return cached

    result = execute_fn(
        db_path=Path(db_path).expanduser().resolve(),
        sql=sql,
        max_seconds=max_seconds,
        max_rows=None,
    )
    result["cache_hit"] = False
    cache_path = write_cached_sql_result(
        cache_dir=resolved_cache_dir,
        db_path=db_path,
        sql=sql,
        result=result,
    )
    if cache_path is not None:
        result["cache_path"] = str(cache_path)
    return result
