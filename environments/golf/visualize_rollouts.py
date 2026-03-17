from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any


BOARD_MARKER = "Legend: @=player K=key P=goal W=wall .=open"
ROW_PATTERN = re.compile(r"^(\d{2})\s+(.*)$")

CELL_STYLES = {
    ".": {"fill": "#f8f2df", "stroke": "#9b8a68", "label": ""},
    "W": {"fill": "#556270", "stroke": "#36404a", "label": "W"},
    "K": {"fill": "#f6c453", "stroke": "#a87100", "label": "K"},
    "P": {"fill": "#79c27d", "stroke": "#2f6a4f", "label": "P"},
    "@": {"fill": "#ff8f5a", "stroke": "#9f3f1d", "label": "@"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an HTML viewer for Golf eval rollouts from results.jsonl."
    )
    parser.add_argument("results_jsonl", type=Path, help="Path to a results.jsonl file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path. Defaults to <results stem>_viewer.html next to the input file.",
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
                raise ValueError(f"Expected an object on line {line_number} of {path}")
            records.append(payload)
    return records


def parse_json_string(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_tool_call(raw_tool_call: Any) -> dict[str, Any]:
    payload = parse_json_string(raw_tool_call)
    function_payload = payload.get("function")
    if not isinstance(function_payload, dict):
        return {}
    arguments = parse_json_string(function_payload.get("arguments"))
    return {
        "id": payload.get("id"),
        "name": function_payload.get("name"),
        "arguments": arguments,
    }


def split_board_content(content: str) -> tuple[str, str]:
    marker_index = content.find(BOARD_MARKER)
    if marker_index == -1:
        return content, ""
    prefix = content[:marker_index].rstrip()
    board_text = content[marker_index:].strip()
    return prefix, board_text


def parse_board_text(board_text: str) -> dict[str, Any] | None:
    if not board_text.startswith(BOARD_MARKER):
        return None

    lines = board_text.splitlines()
    metadata: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    for line in lines[1:]:
        stripped = line.rstrip()
        if not stripped:
            continue
        if stripped.startswith("Available actions:"):
            metadata["available_actions"] = stripped.removeprefix("Available actions:").strip()
            continue
        match = ROW_PATTERN.match(stripped)
        if match:
            row_number = int(match.group(1))
            row_text = match.group(2).strip()
            tokens = [token.strip() for token in row_text.split(" - ") if token.strip()]
            rows.append({"row_number": row_number, "tokens": tokens})
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            metadata[key.strip()] = value.strip()

    return {"metadata": metadata, "rows": rows, "raw": board_text}


def hex_points(cx: float, cy: float, radius: float) -> str:
    points = []
    for index in range(6):
        angle = math.radians(60 * index - 30)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def render_board_svg(board: dict[str, Any]) -> str:
    rows = board.get("rows", [])
    metadata = board.get("metadata", {})
    if not rows:
        return f'<pre class="raw-board">{html.escape(board.get("raw", ""))}</pre>'

    radius = 24.0
    hex_width = math.sqrt(3) * radius
    hex_height = 2 * radius
    x_step = hex_width
    y_step = radius * 1.5
    margin = radius + 10
    max_cells = max(len(row["tokens"]) for row in rows)
    width = margin * 2 + (max_cells - 1) * x_step + hex_width + radius
    height = margin * 2 + (len(rows) - 1) * y_step + hex_height

    parts = [
        f'<svg class="board-svg" viewBox="0 0 {width:.1f} {height:.1f}" role="img" aria-label="Golf board">'
    ]
    for visual_row_index, row in enumerate(rows):
        row_number = int(row["row_number"])
        row_offset = hex_width / 2 if row_number % 2 else 0.0
        cy = margin + radius + visual_row_index * y_step
        for col_index, token in enumerate(row["tokens"]):
            style = CELL_STYLES.get(token, CELL_STYLES["."])
            cx = margin + radius + row_offset + col_index * x_step
            parts.append(
                (
                    f'<polygon points="{hex_points(cx, cy, radius)}" '
                    f'fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="2.5" />'
                )
            )
            if token == "@":
                parts.append(
                    f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius * 0.34:.1f}" fill="#fff8ef" stroke="{style["stroke"]}" stroke-width="2" />'
                )
            label = style["label"]
            if label:
                parts.append(
                    f'<text x="{cx:.1f}" y="{cy + 6:.1f}" text-anchor="middle" class="cell-label">{html.escape(label)}</text>'
                )
        parts.append(
            f'<text x="{8:.1f}" y="{cy + 4:.1f}" class="row-label">{row_number:02d}</text>'
        )
    parts.append("</svg>")

    badges = []
    for key in ("Turn", "Player", "Key collected", "Key position", "Goal position"):
        value = metadata.get(key)
        if value:
            badges.append(
                f'<span class="board-badge"><strong>{html.escape(key)}:</strong> {html.escape(value)}</span>'
            )
    if metadata.get("available_actions"):
        badges.append(
            f'<span class="board-badge"><strong>Actions:</strong> {html.escape(metadata["available_actions"])}</span>'
        )
    return f'<div class="board-badges">{"".join(badges)}</div>{"".join(parts)}'


def render_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return ""
    items = []
    for tool_call in tool_calls:
        arguments = tool_call.get("arguments")
        pretty_args = json.dumps(arguments, ensure_ascii=True, sort_keys=True)
        items.append(
            f"""
            <div class="tool-call">
              <div><strong>{html.escape(str(tool_call.get("name") or "unknown_tool"))}</strong></div>
              <pre>{html.escape(pretty_args)}</pre>
            </div>
            """
        )
    return f'<div class="tool-calls">{"".join(items)}</div>'


def render_message(message: dict[str, Any], index: int) -> str:
    role = str(message.get("role", "unknown"))
    content = message.get("content")
    safe_content = content if isinstance(content, str) else json.dumps(content, ensure_ascii=True, sort_keys=True)
    prefix, board_text = split_board_content(safe_content)
    board = parse_board_text(board_text) if board_text else None

    extras = ""
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            parsed_tool_calls = [parse_tool_call(raw_tool_call) for raw_tool_call in tool_calls]
            extras = render_tool_calls([call for call in parsed_tool_calls if call])
    elif role == "tool":
        payload = parse_json_string(message.get("content"))
        extras = f'<pre>{html.escape(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))}</pre>'

    content_html = ""
    if prefix:
        content_html += f'<pre>{html.escape(prefix)}</pre>'
    if board:
        content_html += f'<div class="board-panel">{render_board_svg(board)}</div>'
    elif role != "tool":
        content_html += f'<pre>{html.escape(safe_content)}</pre>'

    return f"""
    <article class="message role-{html.escape(role)}">
      <div class="message-head">
        <span class="message-role">{html.escape(role)}</span>
        <span class="message-index">#{index}</span>
      </div>
      {content_html}
      {extras}
    </article>
    """


def extract_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for section_name in ("prompt", "completion"):
        section = record.get(section_name)
        if not isinstance(section, list):
            continue
        for message in section:
            if isinstance(message, dict):
                messages.append(message)
    return messages


def build_rollout_card(record: dict[str, Any]) -> str:
    info = record.get("info", {}) if isinstance(record.get("info"), dict) else {}
    metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
    messages = extract_messages(record)
    summary_bits = [
        f"board={info.get('board_id', 'unknown')}",
        f"difficulty={info.get('difficulty', 'unknown')}",
        f"reward={record.get('reward', 'n/a')}",
        f"solved={metrics.get('_solved_metric', record.get('_solved_metric', 'n/a'))}",
        f"turns={record.get('turn_count', metrics.get('num_turns', 'n/a'))}",
        f"invalid={record.get('invalid_actions', metrics.get('_invalid_actions_metric', 'n/a'))}",
        f"stop={record.get('stop_condition', record.get('termination_reason', 'n/a'))}",
    ]
    messages_html = "\n".join(
        render_message(message, index + 1) for index, message in enumerate(messages)
    )
    optimal_actions = info.get("optimal_actions", [])
    optimal_text = ", ".join(optimal_actions) if isinstance(optimal_actions, list) else "n/a"
    return f"""
    <section class="rollout">
      <details>
        <summary>
          <span class="title">Example {record.get("example_id", "?")}</span>
          <span class="summary">{html.escape(" | ".join(summary_bits))}</span>
        </summary>
        <div class="content">
          <div class="optimal">Optimal actions: {html.escape(optimal_text)}</div>
          <div class="timeline">{messages_html or "<p>No messages found.</p>"}</div>
        </div>
      </details>
    </section>
    """


def build_html(records: list[dict[str, Any]], source_path: Path) -> str:
    solved = sum(
        1
        for record in records
        if bool(record.get("_solved_metric") or record.get("metrics", {}).get("_solved_metric"))
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Golf Rollouts Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe4;
      --panel: rgba(255, 252, 245, 0.92);
      --ink: #203129;
      --muted: #5d6e65;
      --accent: #2f6a4f;
      --border: #d8ccb8;
      --shadow: 0 18px 48px rgba(36, 47, 41, 0.08);
      --system: #eef3f7;
      --user: #fcf2db;
      --assistant: #eef8ef;
      --tool: #f5f0fb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(47, 106, 79, 0.16), transparent 26%),
        radial-gradient(circle at top right, rgba(246, 196, 83, 0.18), transparent 22%),
        linear-gradient(180deg, #fcf8ee 0%, var(--bg) 100%);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 2.25rem;
      letter-spacing: -0.03em;
    }}
    .lead {{
      color: var(--muted);
      margin: 0 0 24px;
      word-break: break-all;
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 22px;
    }}
    .stat {{
      padding: 12px 16px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .rollout {{
      margin-bottom: 16px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
      overflow: hidden;
    }}
    .rollout > details > summary {{
      cursor: pointer;
      list-style: none;
      padding: 18px 20px;
    }}
    .rollout > details > summary::-webkit-details-marker {{
      display: none;
    }}
    .title {{
      display: inline-block;
      margin-right: 12px;
      font-weight: 800;
    }}
    .summary, .optimal {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .content {{
      padding: 0 20px 20px;
    }}
    .timeline {{
      display: grid;
      gap: 14px;
      margin-top: 14px;
    }}
    .message {{
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 16px;
    }}
    .role-system {{ background: var(--system); }}
    .role-user {{ background: var(--user); }}
    .role-assistant {{ background: var(--assistant); }}
    .role-tool {{ background: var(--tool); }}
    .message-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }}
    .message-role {{
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.76rem;
      font-weight: 800;
    }}
    .message-index {{
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .board-panel {{
      margin-top: 10px;
      padding: 14px;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(249,245,236,0.98));
      border: 1px solid rgba(155, 138, 104, 0.26);
    }}
    .board-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .board-badge {{
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255, 252, 245, 0.92);
      border: 1px solid rgba(155, 138, 104, 0.3);
      font-size: 0.88rem;
    }}
    .board-svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .cell-label {{
      fill: #18231c;
      font-weight: 800;
      font-size: 18px;
      dominant-baseline: middle;
    }}
    .row-label {{
      fill: #7f8e86;
      font-weight: 700;
      font-size: 14px;
    }}
    .tool-calls {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }}
    .tool-call {{
      padding: 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.75);
      border: 1px solid rgba(93, 110, 101, 0.18);
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      line-height: 1.45;
    }}
    .raw-board {{
      padding: 12px;
      border-radius: 12px;
      background: #1f241f;
      color: #edf6ee;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Golf Rollouts Viewer</h1>
    <p class="lead">{html.escape(str(source_path))}</p>
    <div class="stats">
      <div class="stat">Rollouts: {len(records)}</div>
      <div class="stat">Solved: {solved}</div>
      <div class="stat">Unsolved: {len(records) - solved}</div>
      <div class="stat">Messages shown: prompt + completion</div>
    </div>
    {''.join(build_rollout_card(record) for record in records)}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    records = load_results(args.results_jsonl)
    output_path = args.output or args.results_jsonl.with_name(f"{args.results_jsonl.stem}_viewer.html")
    output_path.write_text(build_html(records, args.results_jsonl), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
