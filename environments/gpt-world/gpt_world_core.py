from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import random
from typing import Any


class Action(str, Enum):
    UPRIGHT = "UR"
    RIGHT = "R"
    DOWNRIGHT = "DR"
    DOWNLEFT = "DL"
    LEFT = "L"
    UPLEFT = "UL"
    PICKUP = "Pickup"


ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UPRIGHT: (-1, 1),
    Action.RIGHT: (0, 2),
    Action.DOWNRIGHT: (1, 1),
    Action.DOWNLEFT: (1, -1),
    Action.LEFT: (0, -2),
    Action.UPLEFT: (-1, -1),
    Action.PICKUP: (0, 0),
}

MOVEMENT_ACTIONS: tuple[Action, ...] = (
    Action.UPRIGHT,
    Action.RIGHT,
    Action.DOWNRIGHT,
    Action.DOWNLEFT,
    Action.LEFT,
    Action.UPLEFT,
)

ACTION_ALIASES = {
    "UR": Action.UPRIGHT,
    "UPRIGHT": Action.UPRIGHT,
    "R": Action.RIGHT,
    "RIGHT": Action.RIGHT,
    "DR": Action.DOWNRIGHT,
    "DOWNRIGHT": Action.DOWNRIGHT,
    "DL": Action.DOWNLEFT,
    "DOWNLEFT": Action.DOWNLEFT,
    "L": Action.LEFT,
    "LEFT": Action.LEFT,
    "UL": Action.UPLEFT,
    "UPLEFT": Action.UPLEFT,
    "PICKUP": Action.PICKUP,
    "PICK": Action.PICKUP,
    "PU": Action.PICKUP,
}


@dataclass(frozen=True)
class GameSpec:
    board_id: str
    difficulty: str
    boundary: tuple[int, int]
    init: tuple[int, int]
    key: tuple[int, int]
    flag: tuple[int, int]
    walls: tuple[tuple[int, int], ...]

    def to_info(self) -> dict[str, Any]:
        return {
            "board_id": self.board_id,
            "difficulty": self.difficulty,
            "boundary": list(self.boundary),
            "init": list(self.init),
            "key": list(self.key),
            "flag": list(self.flag),
            "walls": [list(pos) for pos in self.walls],
        }


@dataclass(frozen=True)
class GameState:
    boundary: tuple[int, int]
    player_pos: tuple[int, int]
    key_pos: tuple[int, int] | None
    flag_pos: tuple[int, int]
    walls: frozenset[tuple[int, int]]
    turn_count: int = 0
    picked_key: bool = False


@dataclass(frozen=True)
class Transition:
    state: GameState
    action: Action
    moved: bool
    valid_action: bool
    reward: float
    solved: bool
    reached_goal: bool
    feedback: str
    termination_reason: str | None


DEFAULT_SPECS: tuple[GameSpec, ...] = (
    GameSpec(
        board_id="easy_notebook",
        difficulty="easy",
        boundary=(3, 3),
        init=(0, 0),
        key=(1, 1),
        flag=(2, 2),
        walls=(),
    ),
    GameSpec(
        board_id="easy_wall",
        difficulty="easy",
        boundary=(3, 3),
        init=(0, 0),
        key=(1, 1),
        flag=(2, 2),
        walls=((2, 0),),
    ),
    GameSpec(
        board_id="medium_notebook",
        difficulty="medium",
        boundary=(5, 5),
        init=(0, 0),
        key=(3, 1),
        flag=(4, 4),
        walls=((1, 1),),
    ),
    GameSpec(
        board_id="medium_detour",
        difficulty="medium",
        boundary=(5, 7),
        init=(0, 0),
        key=(2, 2),
        flag=(4, 4),
        walls=((1, 1), (3, 3)),
    ),
    GameSpec(
        board_id="hard_notebook",
        difficulty="hard",
        boundary=(8, 15),
        init=(0, 0),
        key=(3, 1),
        flag=(7, 13),
        walls=((2, 2), (1, 1), (5, 3), (1, 11), (5, 5), (6, 6), (6, 10), (2, 6), (4, 12)),
    ),
    GameSpec(
        board_id="evil_notebook",
        difficulty="hard",
        boundary=(8, 15),
        init=(0, 0),
        key=(5, 1),
        flag=(7, 13),
        walls=((2, 2), (3, 3), (4, 2), (1, 1), (2, 4), (7, 11), (5, 3), (1, 11), (5, 5), (6, 6), (6, 10), (2, 6), (4, 12)),
    ),
    GameSpec(
        board_id="hard_zigzag",
        difficulty="hard",
        boundary=(7, 11),
        init=(0, 0),
        key=(4, 2),
        flag=(6, 10),
        walls=((1, 1), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)),
    ),
    GameSpec(
        board_id="hard_channel",
        difficulty="hard",
        boundary=(7, 13),
        init=(0, 0),
        key=(2, 2),
        flag=(6, 12),
        walls=((1, 1), (2, 4), (3, 3), (3, 5), (4, 6), (5, 5), (5, 7), (6, 8), (4, 10)),
    ),
)


def parse_action(raw_action: str) -> Action:
    normalized = (raw_action or "").strip().upper()
    if normalized not in ACTION_ALIASES:
        raise ValueError(
            "Unknown action. Use one of: UR, R, DR, DL, L, UL, Pickup."
        )
    return ACTION_ALIASES[normalized]


def is_playable_cell(row: int, col: int) -> bool:
    return row % 2 == col % 2


def validate_spec(spec: GameSpec) -> None:
    rows, cols = spec.boundary
    positions = {
        "init": spec.init,
        "key": spec.key,
        "flag": spec.flag,
        **{f"wall_{index}": wall for index, wall in enumerate(spec.walls)},
    }
    for name, pos in positions.items():
        row, col = pos
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"{spec.board_id}: {name} position {pos} is out of bounds")
        if not is_playable_cell(row, col):
            raise ValueError(f"{spec.board_id}: {name} position {pos} is not on a playable hex cell")
    if spec.init in spec.walls or spec.key in spec.walls or spec.flag in spec.walls:
        raise ValueError(f"{spec.board_id}: start, key, and flag must not overlap walls")


def make_initial_state(spec: GameSpec) -> GameState:
    validate_spec(spec)
    return GameState(
        boundary=spec.boundary,
        player_pos=spec.init,
        key_pos=spec.key,
        flag_pos=spec.flag,
        walls=frozenset(spec.walls),
        turn_count=0,
        picked_key=False,
    )


def render_board(state: GameState) -> str:
    rows, cols = state.boundary
    lines = [
        "Legend: @=player K=key P=goal W=wall .=open",
        f"Turn: {state.turn_count}",
        f"Player: {state.player_pos}",
        f"Key collected: {'yes' if state.picked_key else 'no'}",
    ]
    if state.key_pos is None:
        lines.append("Key position: collected")
    else:
        lines.append(f"Key position: {state.key_pos}")
    lines.append(f"Goal position: {state.flag_pos}")
    lines.append("")
    for row in range(rows):
        prefix = "  " if row % 2 else ""
        cells: list[str] = []
        for col in range(cols):
            if not is_playable_cell(row, col):
                continue
            pos = (row, col)
            token = "."
            if pos in state.walls:
                token = "W"
            elif pos == state.flag_pos:
                token = "P"
            elif state.key_pos is not None and pos == state.key_pos:
                token = "K"
            if pos == state.player_pos:
                token = "@"
            cells.append(token)
        lines.append(f"{row:02d} {prefix}{' - '.join(cells)}")
    lines.append("")
    lines.append("Available actions: UR, R, DR, DL, L, UL, Pickup")
    return "\n".join(lines)


def step(state: GameState, action: Action) -> Transition:
    if action is Action.PICKUP:
        if state.key_pos is not None and state.player_pos == state.key_pos:
            next_state = replace(
                state,
                key_pos=None,
                picked_key=True,
                turn_count=state.turn_count + 1,
            )
            solved = next_state.player_pos == next_state.flag_pos
            return Transition(
                state=next_state,
                action=action,
                moved=False,
                valid_action=True,
                reward=1.0 if solved else 0.1,
                solved=solved,
                reached_goal=next_state.player_pos == next_state.flag_pos,
                feedback="Key collected.",
                termination_reason="solved" if solved else None,
            )
        next_state = replace(state, turn_count=state.turn_count + 1)
        return Transition(
            state=next_state,
            action=action,
            moved=False,
            valid_action=False,
            reward=-0.05,
            solved=False,
            reached_goal=state.player_pos == state.flag_pos,
            feedback="Pickup only works when you are standing on the key.",
            termination_reason=None,
        )

    delta_row, delta_col = ACTION_DELTAS[action]
    new_pos = (state.player_pos[0] + delta_row, state.player_pos[1] + delta_col)
    rows, cols = state.boundary
    in_bounds = 0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols
    if (not in_bounds) or (new_pos in state.walls):
        next_state = replace(state, turn_count=state.turn_count + 1)
        return Transition(
            state=next_state,
            action=action,
            moved=False,
            valid_action=False,
            reward=-0.05,
            solved=False,
            reached_goal=state.player_pos == state.flag_pos,
            feedback="That move is blocked. The board is unchanged.",
            termination_reason=None,
        )

    next_state = replace(
        state,
        player_pos=new_pos,
        turn_count=state.turn_count + 1,
    )
    solved = next_state.picked_key and next_state.player_pos == next_state.flag_pos
    reached_goal = next_state.player_pos == next_state.flag_pos
    if reached_goal and not next_state.picked_key:
        feedback = "You reached the goal, but you still need to collect the key first."
    elif solved:
        feedback = "Goal reached with the key. Puzzle solved."
    else:
        feedback = "Move accepted."
    return Transition(
        state=next_state,
        action=action,
        moved=True,
        valid_action=True,
        reward=1.0 if solved else 0.0,
        solved=solved,
        reached_goal=reached_goal,
        feedback=feedback,
        termination_reason="solved" if solved else None,
    )


def shortest_solution(spec: GameSpec) -> tuple[Action, ...]:
    initial_state = make_initial_state(spec)
    start = (initial_state.player_pos, initial_state.key_pos is None)
    queue: deque[tuple[tuple[int, int], bool]] = deque([start])
    parents: dict[tuple[tuple[int, int], bool], tuple[tuple[tuple[int, int], bool], Action] | None] = {
        start: None
    }

    while queue:
        pos, key_collected = queue.popleft()
        if key_collected and pos == spec.flag:
            actions: list[Action] = []
            node: tuple[tuple[int, int], bool] | None = (pos, key_collected)
            while node is not None and parents[node] is not None:
                parent, action = parents[node]
                actions.append(action)
                node = parent
            actions.reverse()
            return tuple(actions)

        if not key_collected and pos == spec.key:
            next_node = (pos, True)
            if next_node not in parents:
                parents[next_node] = ((pos, key_collected), Action.PICKUP)
                queue.append(next_node)

        for action in MOVEMENT_ACTIONS:
            delta_row, delta_col = ACTION_DELTAS[action]
            new_pos = (pos[0] + delta_row, pos[1] + delta_col)
            rows, cols = spec.boundary
            in_bounds = 0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols
            if not in_bounds or new_pos in spec.walls:
                continue
            next_node = (new_pos, key_collected)
            if next_node not in parents:
                parents[next_node] = ((pos, key_collected), action)
                queue.append(next_node)

    raise ValueError(f"{spec.board_id}: no valid solution exists")


def summarize_spec(spec: GameSpec) -> dict[str, Any]:
    solution = shortest_solution(spec)
    return {
        **spec.to_info(),
        "optimal_actions": [action.value for action in solution],
        "optimal_num_actions": len(solution),
    }


def visualize_action_sequence(spec: GameSpec, actions: list[str] | tuple[str, ...]) -> list[str]:
    states = [render_board(make_initial_state(spec))]
    state = make_initial_state(spec)
    for raw_action in actions:
        transition = step(state, parse_action(raw_action))
        state = transition.state
        states.append(render_board(state))
    return states


def make_records(
    specs: tuple[GameSpec, ...] = DEFAULT_SPECS,
    *,
    num_examples: int,
    seed: int,
    split: str,
    difficulty: str = "all",
) -> list[dict[str, Any]]:
    filtered = tuple(spec for spec in specs if difficulty in ("all", spec.difficulty))
    if not filtered:
        raise ValueError(f"No boards found for difficulty={difficulty!r}")
    order = list(filtered)
    shuffle_seed = seed if split == "train" else seed + 10_000
    random.Random(shuffle_seed).shuffle(order)
    total = len(order) if num_examples <= 0 else int(num_examples)
    records: list[dict[str, Any]] = []
    for index in range(total):
        spec = order[index % len(order)]
        info = summarize_spec(spec)
        records.append({"prompt": [], "info": info})
    return records
