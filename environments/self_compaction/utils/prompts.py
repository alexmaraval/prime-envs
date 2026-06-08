from __future__ import annotations

from jinja2 import StrictUndefined, Template


def render_template(template: str, **kwargs) -> str:
    return Template(template, undefined=StrictUndefined).render(**kwargs)


PROMPT_TEMPLATE = """<pr_description>

Consider the following PR description:

{problem_statement}

</pr_description>

<instructions>

# Task Instructions

You are a software engineering agent working inside a repository sandbox.
Implement the necessary source-code changes to satisfy the PR description.

You must interact by issuing exactly one tool call per assistant turn.

Available workflow:
1. Inspect the repository with `search_files`, `read`, and focused
   `execute_bash` commands.
2. Edit source files with `edit_via_str_replace` or focused shell commands.
3. Reproduce and verify the issue with focused commands.
4. Call `compact` with a complete summary after useful investigation or changes,
   usually by turn 25-40, and before final submission.
5. Call `submit` only after compaction and once your changes are ready.

Available tools:
- `search_files(pattern: str)`
- `read(path: str, start_line: int = 1, end_line: int | None = None, limit: int = 200)`
- `execute_bash(command: str)`
- `edit_via_str_replace(path: str, old_str: str, new_str: str)`
- `compact(summary: str)`
- `submit()`

Important boundaries:
- Modify normal source-code files.
- Do not modify hidden tests, generated test fixtures, or benchmark metadata.
- Commands run in fresh subshells. Prefix commands with `cd /testbed && ...`
  when needed, or rely on the default repository working directory.
- Avoid interactive commands.
- A sandbox tool-status note will say whether `rg` is available inside
  `execute_bash`. If it is missing, do not run `rg`; use `search_files`, `read`,
  `find`, `grep -R`, `sed -n`, `head`, and short Python snippets.
- `read` returns at most 200 lines at a time. Use `read(path)` for the first
  200 lines, or provide `start_line` with `end_line` or `limit`.
- Keep command output bounded. Avoid broad searches through `.venv`,
  `site-packages`, or generated directories.
- Run focused tests first. Full-suite commands and package installs can time out.

The `compact` tool removes the working conversation history before that compact
call, while preserving the original task prompt. After compaction you will see
the original task, the compact call/result, your summary, and future messages.
Make the summary good enough to continue without earlier observations. Do not
call `compact` repeatedly without intervening investigation, edits, or tests.

</instructions>"""


SYSTEM_PROMPT = """You are a helpful coding agent that solves programming tasks with tools.

Rules:
- Every assistant response must contain exactly one tool call.
- Available tools are `search_files`, `read`, `execute_bash`, `edit_via_str_replace`, `compact`, and `submit`.
- Use `read(path)`, `read(path, start_line, end_line)`, or `read(path, start_line, limit=...)` for bounded file inspection after search.
- Use `compact` at least once before `submit`.
- The original task prompt remains visible after `compact`.
- After `compact`, earlier working history is gone except for the summary you wrote.
- Compact after useful investigation or changes, usually by turn 25-40, then continue solving; do not compact twice in a row.
- Use `submit` when the repository is ready for hidden-test scoring."""


ACTION_OBSERVATION_TEMPLATE = """<returncode>{{exit_code}}</returncode>
{% if output | length < 10000 -%}
<output>
{{ output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Use a narrower command, or write output to a file and inspect relevant slices.
</warning>
{%- set elided_chars = output | length - 10000 -%}
<output_head>
{{ output[:5000] }}
</output_head>
<elided_chars>
{{ elided_chars }} characters elided
</elided_chars>
<output_tail>
{{ output[-5000:] }}
</output_tail>
{%- endif -%}"""


FORMAT_ERROR_TEMPLATE = """Please provide exactly one tool call. Found {{ actions|length }} tool calls.

If you are ready to submit, call `submit`. If you have not compacted yet, call
`compact` first with a complete summary."""
