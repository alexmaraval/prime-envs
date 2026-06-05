#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


MAX_LIMIT = 1000


def resolve_inside_root(root: Path, raw_path: str) -> Path:
    base = root.resolve(strict=True)
    requested = Path(raw_path)
    candidate = requested if requested.is_absolute() else base / requested
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {raw_path}") from exc
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Read a bounded slice of a file.")
    parser.add_argument("path", help="Path to a text file under the repository root")
    parser.add_argument(
        "--start-line",
        type=int,
        default=1,
        help="1-indexed line number where reading starts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help=f"Maximum number of lines to return, capped at {MAX_LIMIT}",
    )
    args = parser.parse_args()

    root = Path.cwd()
    try:
        path = resolve_inside_root(root, args.path)
    except FileNotFoundError:
        sys.stderr.write(f"Path does not exist: {args.path}\n")
        return 1
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    if not path.is_file():
        sys.stderr.write(f"Path is not a file: {args.path}\n")
        return 1

    start_line = max(1, args.start_line)
    limit = min(MAX_LIMIT, max(1, args.limit))
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        sys.stderr.write(f"Failed to read {args.path}: {exc}\n")
        return 1

    rel_path = path.relative_to(root.resolve(strict=True))
    start_idx = start_line - 1
    end_idx = min(len(lines), start_idx + limit)
    if start_idx >= len(lines):
        sys.stdout.write(f"{rel_path}: no lines at or after {start_line}\n")
        return 0

    width = len(str(end_idx))
    for line_number in range(start_line, end_idx + 1):
        sys.stdout.write(f"{rel_path}:{line_number:>{width}}: {lines[line_number - 1]}\n")
    if end_idx < len(lines):
        remaining = len(lines) - end_idx
        sys.stdout.write(f"\nStopped after {limit} lines. {remaining} lines remain.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
