from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REQUIRED_PATHS = [
    "train/train.json",
    "train/train_tables.json",
    "train/train_databases",
    "dev_20240627/dev.json",
    "dev_20240627/dev_tables.json",
    "dev_20240627/dev_databases",
]


def resolve_data_dir(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    env_value = os.environ.get("BIRD_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    env_root = Path(__file__).resolve().parents[1]
    hidden_data = env_root / ".data"
    if hidden_data.exists():
        return hidden_data.resolve()
    return (env_root / "data").resolve()


def check_layout(data_dir: Path) -> list[str]:
    missing = []
    for rel_path in REQUIRED_PATHS:
        path = data_dir / rel_path
        if not path.exists():
            missing.append(rel_path)
    return missing


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def cache_metadata(data_dir: Path) -> None:
    from datasets import load_dataset

    train = [dict(row) for row in load_dataset("birdsql/bird23-train-filtered", split="train")]
    write_json(data_dir / "metadata" / "train_filtered.json", train)

    last_error: Exception | None = None
    attempts = [
        (("birdsql/bird_mini_dev",), "mini_dev_sqlite"),
        (("birdsql/bird_mini_dev", "sqlite"), "train"),
        (("birdsql/bird_mini_dev", "SQLite"), "train"),
    ]
    for args, split_name in attempts:
        try:
            val = [dict(row) for row in load_dataset(*args, split=split_name)]
            write_json(data_dir / "metadata" / "bird_mini_dev_sqlite.json", val)
            return
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or cache small BIRD metadata.")
    parser.add_argument("--data-dir", default=None, help="BIRD data root. Defaults to BIRD_DATA_DIR or ./data.")
    parser.add_argument(
        "--cache-metadata",
        action="store_true",
        help="Download small Hugging Face metadata files. This does not download database assets.",
    )
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    missing = check_layout(data_dir)
    if missing:
        print(f"BIRD data layout is incomplete under {data_dir}:")
        for rel_path in missing:
            print(f"- missing {rel_path}")
        return 1

    print(f"BIRD data layout looks good under {data_dir}.")
    if args.cache_metadata:
        cache_metadata(data_dir)
        print(f"Cached metadata under {data_dir / 'metadata'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
