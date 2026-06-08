#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


MAX_LINES = 200


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
        "--end-line",
        type=int,
        default=None,
        help="1-indexed inclusive line number where reading ends",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_LINES,
        help=f"Maximum lines to read when --end-line is omitted (default: {MAX_LINES})",
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

    start_line = args.start_line
    end_line = args.end_line
    if start_line < 1:
        sys.stderr.write("--start-line must be >= 1\n")
        return 1
    if end_line is None:
        if args.limit < 1:
            sys.stderr.write("--limit must be >= 1\n")
            return 1
        end_line = start_line + args.limit - 1
    if end_line < start_line:
        sys.stderr.write("--end-line must be greater than or equal to --start-line\n")
        return 1
    if end_line - start_line + 1 > MAX_LINES:
        sys.stderr.write(
            f"Requested {end_line - start_line + 1} lines; read at most {MAX_LINES} "
            "lines at a time with a narrower range.\n"
        )
        return 1
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        sys.stderr.write(f"Failed to read {args.path}: {exc}\n")
        return 1

    rel_path = path.relative_to(root.resolve(strict=True))
    start_idx = start_line - 1
    end_idx = min(len(lines), end_line)
    if start_idx >= len(lines):
        sys.stdout.write(
            f"{rel_path}: no lines at or after {start_line} "
            f"(file has {len(lines)} lines)\n"
        )
        return 0

    sys.stdout.write(f"{rel_path} lines {start_line}-{end_idx} of {len(lines)}:\n")
    sys.stdout.write("<content>\n")
    sys.stdout.write("\n".join(lines[start_idx:end_idx]))
    sys.stdout.write("\n</content>\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
