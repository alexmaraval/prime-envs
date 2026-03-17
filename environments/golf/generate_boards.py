from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from typing import Iterable

from golf_core import DEFAULT_SPECS, GameSpec, shortest_solution, validate_spec


def playable_cells(boundary: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    rows, cols = boundary
    return tuple(
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if row % 2 == col % 2
    )


def _spec_payload(spec: GameSpec) -> dict[str, object]:
    payload = asdict(spec)
    payload["boundary"] = list(spec.boundary)
    payload["init"] = list(spec.init)
    payload["key"] = list(spec.key)
    payload["flag"] = list(spec.flag)
    payload["walls"] = [list(wall) for wall in spec.walls]
    payload["optimal_actions"] = [action.value for action in shortest_solution(spec)]
    payload["optimal_num_actions"] = len(payload["optimal_actions"])
    return payload


def generate_board_from_template(
    template: GameSpec,
    *,
    rng: random.Random,
    board_index: int = 0,
    max_wall_attempts_per_wall: int = 50,
) -> GameSpec:
    cells = list(playable_cells(template.boundary))
    if len(cells) < 3:
        raise ValueError(f"{template.board_id}: need at least three playable cells")

    init, key, flag = rng.sample(cells, 3)
    wall_target = len(template.walls)
    protected = {init, key, flag}
    available_walls = [cell for cell in cells if cell not in protected]
    rng.shuffle(available_walls)

    walls: list[tuple[int, int]] = []
    cursor = 0
    total_attempts = max_wall_attempts_per_wall * max(1, wall_target)
    while len(walls) < wall_target and cursor < len(available_walls) and total_attempts > 0:
        candidate = available_walls[cursor]
        cursor += 1
        total_attempts -= 1
        trial = GameSpec(
            board_id=f"{template.board_id}_generated_{board_index:03d}",
            difficulty=template.difficulty,
            boundary=template.boundary,
            init=init,
            key=key,
            flag=flag,
            walls=tuple(sorted((*walls, candidate))),
        )
        try:
            validate_spec(trial)
            shortest_solution(trial)
        except ValueError:
            continue
        walls.append(candidate)

    spec = GameSpec(
        board_id=f"{template.board_id}_generated_{board_index:03d}",
        difficulty=template.difficulty,
        boundary=template.boundary,
        init=init,
        key=key,
        flag=flag,
        walls=tuple(sorted(walls)),
    )
    validate_spec(spec)
    shortest_solution(spec)
    return spec


def generate_boards(
    *,
    count: int,
    seed: int = 0,
    difficulty: str = "all",
    templates: Iterable[GameSpec] = DEFAULT_SPECS,
) -> list[GameSpec]:
    template_list = [
        template for template in templates if difficulty in ("all", template.difficulty)
    ]
    if not template_list:
        raise ValueError("At least one template board is required")

    rng = random.Random(seed)
    boards: list[GameSpec] = []
    for index in range(count):
        template = rng.choice(template_list)
        boards.append(
            generate_board_from_template(
                template,
                rng=rng,
                board_index=index,
            )
        )
    return boards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate solvable Golf boards from the predefined templates."
    )
    parser.add_argument("--count", type=int, default=8, help="Number of boards to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--difficulty",
        choices=("all", "easy", "medium", "hard"),
        default="all",
        help="Restrict generation to templates with this difficulty.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    templates = [
        spec for spec in DEFAULT_SPECS if args.difficulty in ("all", spec.difficulty)
    ]
    boards = generate_boards(count=args.count, seed=args.seed, templates=templates)
    print(json.dumps([_spec_payload(board) for board in boards], indent=2))


if __name__ == "__main__":
    main()
