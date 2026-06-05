from __future__ import annotations

import re


def parse_log_pytest(log: str | None) -> dict[str, str]:
    if log is None or "short test summary info" not in log:
        return {}
    status_map: dict[str, str] = {}
    summary = log.split("short test summary info", 1)[1].strip().splitlines()
    for line in summary:
        if "PASSED" in line:
            test_name = ".".join(line.split("::")[1:])
            status_map[test_name] = "PASSED"
        elif "FAILED" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
            status_map[test_name] = "FAILED"
        elif "ERROR" in line:
            test_name = ".".join(line.split("::")[1:]).split(" - ")[0] or line
            status_map[test_name] = "ERROR"
    return status_map


def parse_log_fn(repo_name: str):
    return parse_log_pytest


def decolor_dict_keys(value: dict[str, str]) -> dict[str, str]:
    return {re.sub(r"\u001b\[\d+m", "", key): item for key, item in value.items()}
