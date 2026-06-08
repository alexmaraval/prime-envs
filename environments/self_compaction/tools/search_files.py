#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".go",
    ".h",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".pyi",
    ".rst",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

TEXT_NAMES = {
    "Dockerfile",
    "LICENSE",
    "Makefile",
    "README",
    "setup.cfg",
    "setup.py",
}

MAX_FILE_BYTES = 2_000_000
MAX_MATCHES = 100


def compile_pattern(pattern: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern)
    except re.error:
        return re.compile(re.escape(pattern))


def should_search(path: Path) -> bool:
    if path.name in TEXT_NAMES:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


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


def iter_files(root: Path, search_root: Path):
    paths = [search_root] if search_root.is_file() else sorted(search_root.rglob("*"))
    for path in paths:
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file() or not should_search(path):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search repository text files for a regex or literal pattern."
    )
    parser.add_argument("--pattern", required=True, help="Regex or literal text to find")
    parser.add_argument(
        "--path",
        default=".",
        help="Optional file or directory path under the repository root",
    )
    args = parser.parse_args()

    root = Path.cwd()
    try:
        search_root = resolve_inside_root(root, args.path)
    except FileNotFoundError:
        sys.stderr.write(f"Path does not exist: {args.path}\n")
        return 1
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    matcher = compile_pattern(args.pattern)
    matches = 0
    searched = 0

    for path in iter_files(root, search_root):
        searched += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "\x00" in text[:4096]:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not matcher.search(line):
                continue
            rel_path = path.relative_to(root)
            sys.stdout.write(f"{rel_path}:{line_number}: {line[:500]}\n")
            matches += 1
            if matches >= MAX_MATCHES:
                sys.stdout.write(
                    f"\nStopped after {MAX_MATCHES} matches. Use a narrower pattern "
                    "or inspect a specific file next.\n"
                )
                return 0

    if matches == 0:
        sys.stdout.write(
            f"No matches for {args.pattern!r} in {searched} searchable files.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
