from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a side-by-side self-compaction rollout dashboard."
    )
    parser.add_argument("results_jsonl", type=Path, help="Path to results.jsonl")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path. Defaults to <results stem>_self_compaction.html.",
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
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            records.append(payload)
    return records


def parse_json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def message_text(message: dict[str, Any]) -> str:
    return text_value(message.get("content"))


def short_text(value: str, max_chars: int = 140) -> str:
    squashed = " ".join(value.split())
    if len(squashed) <= max_chars:
        return squashed
    return squashed[: max_chars - 1].rstrip() + "..."


def parse_tool_call(raw_tool_call: Any) -> dict[str, Any]:
    payload = parse_json_value(raw_tool_call)
    if not isinstance(payload, dict):
        return {"id": "", "name": "unknown", "arguments": {}}
    function_payload = payload.get("function")
    if not isinstance(function_payload, dict):
        function_payload = payload
    raw_args = function_payload.get("arguments", payload.get("arguments", {}))
    args = parse_json_value(raw_args)
    if not isinstance(args, dict):
        args = {"value": args}
    return {
        "id": text_value(payload.get("id")),
        "name": text_value(function_payload.get("name") or "unknown"),
        "arguments": args,
    }


def tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []
    return [parse_tool_call(tool_call) for tool_call in raw_tool_calls]


def is_error_tool_message(message: dict[str, Any]) -> bool:
    content = message_text(message).lstrip()
    if content.startswith("Error:"):
        return True
    parsed = parse_json_value(content)
    return isinstance(parsed, dict) and parsed.get("status") == "error"


def compacted_summary_message(summary: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "<compacted_context>\n"
            "The prior interaction history has been replaced by this summary. "
            "The original task prompt remains visible above; use this summary "
            "for everything learned or changed after the task began.\n\n"
            f"{summary}\n"
            "</compacted_context>"
        ),
    }


def full_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = record.get("full_rollout_messages")
    if isinstance(explicit, list):
        return [message for message in explicit if isinstance(message, dict)]
    messages: list[dict[str, Any]] = []
    for section in ("prompt", "completion"):
        value = record.get(section)
        if isinstance(value, list):
            messages.extend(message for message in value if isinstance(message, dict))
    return messages


def derive_visible_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = record.get("agent_visible_messages")
    prompt = record.get("prompt")
    if not isinstance(prompt, list):
        prompt = []
    visible = [dict(message) for message in prompt if isinstance(message, dict)]
    skipped_tool_call_ids: set[str] = set()
    completion = record.get("completion")
    if not isinstance(completion, list):
        if isinstance(explicit, list):
            return [message for message in explicit if isinstance(message, dict)]
        return visible

    pending_compact: dict[str, Any] | None = None
    for raw_message in completion:
        if not isinstance(raw_message, dict):
            continue
        message = dict(raw_message)
        calls = tool_calls(message)
        compact_calls = [call for call in calls if call["name"] == "compact"]
        if compact_calls:
            summary = text_value(compact_calls[-1]["arguments"].get("summary")).strip()
            pending_compact = {
                "assistant": message,
                "summary": summary,
                "ids": {call["id"] for call in compact_calls if call.get("id")},
            }
            continue
        if pending_compact is not None:
            is_matching_tool = (
                message.get("role") == "tool"
                and text_value(message.get("tool_call_id")) in pending_compact["ids"]
            )
            if is_matching_tool:
                if is_error_tool_message(message):
                    visible.append(pending_compact["assistant"])
                    visible.append(message)
                else:
                    visible = [
                        *[dict(message) for message in prompt if isinstance(message, dict)],
                        pending_compact["assistant"],
                        message,
                        compacted_summary_message(pending_compact["summary"]),
                    ]
                    skipped_tool_call_ids.update(pending_compact["ids"])
                pending_compact = None
                continue
            visible.append(pending_compact["assistant"])
            pending_compact = None
        if (
            message.get("role") == "tool"
            and text_value(message.get("tool_call_id")) in skipped_tool_call_ids
        ):
            continue
        visible.append(message)
    if pending_compact is not None:
        visible.append(pending_compact["assistant"])
    return visible


def normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    full = full_messages(record)
    visible = derive_visible_messages(record)
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    compact_calls = [
        call
        for message in full
        for call in tool_calls(message)
        if call["name"] == "compact"
    ]
    return {
        "index": index,
        "exampleId": record.get("example_id", index),
        "reward": record.get("reward", 0),
        "stopCondition": record.get("stop_condition", ""),
        "compactionCount": record.get(
            "compaction_count", metrics.get("compaction_count", len(compact_calls))
        ),
        "submittedAfterCompact": record.get(
            "submitted_after_compact", metrics.get("submitted_after_compact", 0)
        ),
        "fullMessages": full,
        "visibleMessages": visible,
    }


def build_dashboard_data(results_path: Path) -> dict[str, Any]:
    records = [
        normalize_record(record, index)
        for index, record in enumerate(load_results(results_path), start=1)
    ]
    solved = sum(1 for record in records if float(record["reward"] or 0) > 0)
    return {
        "summary": {
            "rollouts": len(records),
            "solved": solved,
            "compactions": sum(int(record["compactionCount"] or 0) for record in records),
        },
        "records": records,
    }


def message_summary(message: dict[str, Any], index: int | None = None) -> str:
    role = text_value(message.get("role") or "unknown")
    calls = tool_calls(message)
    labels = [role]
    if index is not None:
        labels.insert(0, f"{index:03d}")
    if calls:
        labels.append(", ".join(call["name"] for call in calls))
    elif role == "tool" and message.get("tool_call_id"):
        labels.append(f"tool result {short_text(text_value(message.get('tool_call_id')), 24)}")
    preview = short_text(message_text(message))
    if not preview and calls:
        preview = "tool call"
    if not preview:
        preview = "(empty)"
    return (
        f"<span class='message-label'>{html.escape(' | '.join(labels))}</span>"
        f"<span class='message-preview'>{html.escape(preview)}</span>"
    )


def render_message(
    message: dict[str, Any], collapsed: bool = False, index: int | None = None
) -> str:
    role = html.escape(text_value(message.get("role") or "unknown"))
    content = html.escape(message_text(message))
    calls = tool_calls(message)
    call_html = ""
    if calls:
        rendered = []
        for call in calls:
            args = html.escape(json.dumps(call["arguments"], ensure_ascii=True, indent=2))
            rendered.append(
                f"<div class='tool-call'><strong>{html.escape(call['name'])}</strong>"
                f"<pre>{args}</pre></div>"
            )
        call_html = "".join(rendered)
    body = f"<pre>{content}</pre>{call_html}"
    if collapsed:
        summary = message_summary(message, index=index)
        return (
            f"<details class='message collapsible-message {role}'>"
            f"<summary>{summary}<span class='toggle-label'></span></summary>"
            f"<div class='message-body'>{body}</div>"
            "</details>"
        )
    return f"<article class='message {role}'><h4>{role}</h4><pre>{content}</pre>{call_html}</article>"


def build_html(data: dict[str, Any]) -> str:
    summary = data["summary"]
    cards = []
    for record in data["records"]:
        full = "\n".join(
            render_message(message, collapsed=True, index=index)
            for index, message in enumerate(record["fullMessages"], start=1)
        )
        visible = "\n".join(
            render_message(message) for message in record["visibleMessages"]
        )
        cards.append(
            f"""
<section class="rollout">
  <header>
    <h2>Example {html.escape(text_value(record['exampleId']))}</h2>
    <span>reward {html.escape(text_value(record['reward']))}</span>
    <span>compactions {html.escape(text_value(record['compactionCount']))}</span>
    <span>submitted after compact {html.escape(text_value(record['submittedAfterCompact']))}</span>
  </header>
  <div class="pair">
    <div class="full-rollout">
      <div class="pane-heading">
        <h3>Full Rollout</h3>
        <div class="pane-actions">
          <button type="button" onclick="this.closest('.full-rollout').querySelectorAll('details').forEach(d => d.open = true)">Expand All</button>
          <button type="button" onclick="this.closest('.full-rollout').querySelectorAll('details').forEach(d => d.open = false)">Collapse All</button>
        </div>
      </div>
      {full}
    </div>
    <div class="visible-rollout">
      <h3>Agent-Visible Rollout</h3>
      {visible}
    </div>
  </div>
</section>
"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Self-Compaction Rollouts</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }}
    body {{ margin: 0; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .summary {{ display: flex; gap: 12px; margin: 16px 0 24px; flex-wrap: wrap; }}
    .summary span, .rollout header span {{
      border: 1px solid #ccd3dd;
      background: #fff;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 13px;
    }}
    .rollout {{ border-top: 1px solid #cbd5df; padding: 20px 0 28px; }}
    .rollout header {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .rollout h2 {{ margin: 0 8px 0 0; font-size: 20px; }}
    .pair {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; margin-top: 14px; }}
    .pair > div {{ min-width: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .pane-heading {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 8px; }}
    .pane-heading h3 {{ margin: 0; }}
    .pane-actions {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .pane-actions button {{
      border: 1px solid #b8c2cf;
      background: #fff;
      border-radius: 6px;
      color: #263442;
      cursor: pointer;
      font-size: 12px;
      padding: 5px 8px;
    }}
    .pane-actions button:hover {{ background: #eef2f6; }}
    .message {{ background: #fff; border: 1px solid #d8dee8; border-radius: 6px; padding: 10px; margin: 8px 0; }}
    .message h4 {{ margin: 0 0 6px; text-transform: uppercase; font-size: 11px; letter-spacing: .08em; color: #53616f; }}
    .collapsible-message {{ padding: 0; overflow: hidden; }}
    .collapsible-message summary {{
      align-items: center;
      cursor: pointer;
      display: grid;
      gap: 8px;
      grid-template-columns: auto minmax(0, 1fr) auto;
      list-style: none;
      padding: 9px 10px;
    }}
    .collapsible-message summary::-webkit-details-marker {{ display: none; }}
    .collapsible-message summary:hover {{ background: #f7f9fb; }}
    .message-label {{
      color: #344454;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      white-space: nowrap;
    }}
    .message-preview {{
      color: #53616f;
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .toggle-label {{
      border: 1px solid #c6d0dc;
      border-radius: 6px;
      color: #344454;
      font-size: 11px;
      padding: 3px 7px;
      white-space: nowrap;
    }}
    .toggle-label::before {{ content: "Expand"; }}
    .collapsible-message[open] .toggle-label::before {{ content: "Collapse"; }}
    .message-body {{ border-top: 1px solid #e2e8f0; padding: 10px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 12px; line-height: 1.45; }}
    .tool-call {{ border-top: 1px solid #e2e8f0; margin-top: 8px; padding-top: 8px; }}
    .assistant {{ border-left: 4px solid #4f46e5; }}
    .tool {{ border-left: 4px solid #0f766e; }}
    .user {{ border-left: 4px solid #b45309; }}
    @media (max-width: 900px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>Self-Compaction Rollouts</h1>
  <div class="summary">
    <span>rollouts {html.escape(text_value(summary['rollouts']))}</span>
    <span>solved {html.escape(text_value(summary['solved']))}</span>
    <span>compactions {html.escape(text_value(summary['compactions']))}</span>
  </div>
  {''.join(cards)}
</main>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    output = args.output or args.results_jsonl.with_name(
        f"{args.results_jsonl.stem}_self_compaction.html"
    )
    output.write_text(build_html(build_dashboard_data(args.results_jsonl)), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
