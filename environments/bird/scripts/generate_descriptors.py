from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
ENV_ROOT = Path(__file__).resolve().parents[1]
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from bird_descriptors import (  # noqa: E402
    DEFAULT_LARGE_TABLE_SAMPLE_ROWS,
    DEFAULT_SAMPLE_VALUES_LIMIT,
    DEFAULT_TOP_VALUES_LIMIT,
    descriptor_split_group,
    ensure_profile,
    iter_sqlite_databases,
)


def resolve_data_dir(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    env_value = os.environ.get("BIRD_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    hidden_data = ENV_ROOT / ".data"
    if hidden_data.exists():
        return hidden_data.resolve()
    return (ENV_ROOT / "data").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cached deterministic BIRD database profiles."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="BIRD data root. Defaults to BIRD_DATA_DIR or package-local .data.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev"],
        help="Splits to process. val and test map to the shared dev databases.",
    )
    parser.add_argument(
        "--db-id",
        action="append",
        default=[],
        help="Database id to process. Can be passed multiple times.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing profile artifacts.",
    )
    parser.add_argument(
        "--top-values-limit",
        type=int,
        default=DEFAULT_TOP_VALUES_LIMIT,
        help="Maximum frequent values stored per column.",
    )
    parser.add_argument(
        "--sample-values-limit",
        type=int,
        default=DEFAULT_SAMPLE_VALUES_LIMIT,
        help="Maximum example values stored per column.",
    )
    parser.add_argument(
        "--large-table-sample-rows",
        type=int,
        default=DEFAULT_LARGE_TABLE_SAMPLE_ROWS,
        help="Use first N rows for per-column stats on larger tables. Set 0 for full scans.",
    )
    return parser.parse_args()


def _dedupe_splits(raw_splits: list[str]) -> list[str]:
    seen: set[str] = set()
    splits: list[str] = []
    for split in raw_splits:
        group = descriptor_split_group(split)
        if group in seen:
            continue
        seen.add(group)
        splits.append(group)
    return splits


def _selected_databases(
    *,
    data_dir: Path,
    split_group: str,
    requested_db_ids: set[str],
) -> list[tuple[str, Path]]:
    databases = iter_sqlite_databases(data_dir, split_group)
    if not requested_db_ids:
        return databases
    return [(db_id, path) for db_id, path in databases if db_id in requested_db_ids]


def main() -> int:
    args = parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    requested_db_ids = {str(value).strip() for value in args.db_id if str(value).strip()}

    processed = 0
    for split_group in _dedupe_splits(args.splits):
        databases = _selected_databases(
            data_dir=data_dir,
            split_group=split_group,
            requested_db_ids=requested_db_ids,
        )
        if requested_db_ids and not databases:
            print(f"No matching databases for split {split_group}: {sorted(requested_db_ids)}")
            continue

        for db_id, db_path in databases:
            _profile, profile_written, profile_path = ensure_profile(
                data_dir=data_dir,
                split=split_group,
                db_id=db_id,
                db_path=db_path,
                top_values_limit=args.top_values_limit,
                sample_values_limit=args.sample_values_limit,
                large_table_sample_rows=args.large_table_sample_rows,
                force=args.force,
            )
            if profile_written:
                print(f"profile wrote {profile_path}")
            else:
                print(f"profile reused {profile_path}")
            processed += 1

    print(f"Processed {processed} database(s) under {data_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
