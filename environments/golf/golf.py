from __future__ import annotations

import json
from typing import Any, Sequence, cast

from openai.types.chat import ChatCompletionAssistantMessageParam
from datasets import Dataset
import verifiers as vf

from generate_boards import generate_boards
from golf_core import (
    Action,
    GameSpec,
    GameState,
    Transition,
    make_initial_state,
    parse_action,
    render_board,
    shortest_solution,
    step,
)

SYSTEM_PROMPT = """You are playing Golf, a deterministic hex-grid pathfinding game.

Your job is to collect the key and then reach the goal.
On every turn, call the tool `play_move` exactly once with one action.
Valid actions are: UR, R, DR, DL, L, UL, Pickup.

Important rules:
- The board is shown again after every turn.
- If you hit a wall or move out of bounds, the board does not change.
- You must stand on the key and then use `Pickup` to collect it.
- Reaching the goal before collecting the key does not solve the puzzle.

Keep any reasoning minimal and use the board state you are given each turn.
"""

PLAY_MOVE_TOOL = "play_move"
MAX_MODEL_TURNS = 128
DEFAULT_GENERATED_DATASET_SIZE = 4096


def play_move(action: str) -> str:
    """Take exactly one Golf action.

    Args:
        action: One of UR, R, DR, DL, L, UL, Pickup.
    """
    return parse_action(action).value


def _make_dataset(
    *,
    num_examples: int,
    seed: int,
    split: str,
    difficulty: str,
) -> Dataset:
    effective_seed = seed if split == "train" else seed + 10_000
    generated_specs = generate_boards(
        count=num_examples,
        seed=effective_seed,
        difficulty=difficulty,
    )
    records = []
    for spec in generated_specs:
        solution = shortest_solution(spec)
        info = spec.to_info()
        info["optimal_actions"] = [action.value for action in solution]
        info["optimal_num_actions"] = len(solution)
        records.append({"prompt": [], "info": info})
    for record in records:
        record["prompt"] = [{"role": "user", "content": _format_user_prompt(record["info"])}]
    return Dataset.from_list(records)


def _make_dataset_builder(
    *,
    num_examples: int,
    seed: int,
    split: str,
    difficulty: str,
):
    def build() -> Dataset:
        return _make_dataset(
            num_examples=num_examples,
            seed=seed,
            split=split,
            difficulty=difficulty,
        )

    return build


def _last_assistant_message(messages: vf.Messages) -> ChatCompletionAssistantMessageParam:
    if not isinstance(messages, list):
        raise TypeError(f"expected chat messages, got {type(messages).__name__}")
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return cast(ChatCompletionAssistantMessageParam, message)
    raise ValueError("expected an assistant message in the conversation")


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        name = function_payload.get("name")
        if isinstance(name, str):
            return name
    name = tool_call.get("name")
    return str(name) if isinstance(name, str) else ""


def _tool_call_arguments(tool_call: dict[str, Any]) -> Any:
    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict) and "arguments" in function_payload:
        return function_payload.get("arguments")
    return tool_call.get("arguments", "{}")


def _tool_message(tool_call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": json.dumps(payload, ensure_ascii=True, sort_keys=True),
        "tool_call_id": tool_call_id,
    }


def _state_from_info(info: dict[str, Any]) -> GameState:
    spec = GameSpec(
        board_id=str(info["board_id"]),
        difficulty=str(info["difficulty"]),
        boundary=tuple(info["boundary"]),
        init=tuple(info["init"]),
        key=tuple(info["key"]),
        flag=tuple(info["flag"]),
        walls=tuple(tuple(pos) for pos in info["walls"]),
    )
    return make_initial_state(spec)


def _format_user_prompt(info: dict[str, Any]) -> str:
    state = _state_from_info(info)
    optimal_num_actions = int(info["optimal_num_actions"])
    return (
        "Golf board:\n"
        f"Reference shortest solution length: {optimal_num_actions} actions.\n\n"
        f"{render_board(state)}"
    )


def _reward_total(state: vf.State) -> float:
    return float(state.get("total_reward", 0.0))


def _solved_metric(state: vf.State) -> float:
    return 1.0 if state.get("solved") else 0.0


def _picked_key_metric(state: vf.State) -> float:
    return 1.0 if state.get("picked_key") else 0.0


def _invalid_actions_metric(state: vf.State) -> float:
    return float(state.get("invalid_actions", 0.0))


def _turn_count_metric(state: vf.State) -> float:
    return float(state.get("turn_count", 0.0))


def _optimality_gap_metric(state: vf.State) -> float:
    if not state.get("solved"):
        return -1.0
    optimal_num_actions = int(state.get("optimal_num_actions", 0))
    return float(max(0, int(state.get("turn_count", 0)) - optimal_num_actions))


def build_rubric() -> vf.Rubric:
    rubric = vf.Rubric()
    rubric.add_reward_func(_reward_total, weight=1.0)
    rubric.add_metric(_solved_metric)
    rubric.add_metric(_picked_key_metric)
    rubric.add_metric(_invalid_actions_metric)
    rubric.add_metric(_turn_count_metric)
    rubric.add_metric(_optimality_gap_metric)
    return rubric


class GolfEnv(vf.ToolEnv):
    def __init__(self, max_turns: int = MAX_MODEL_TURNS, **kwargs):
        super().__init__(tools=[play_move], max_turns=max_turns, **kwargs)

    async def no_tools_called(self, state: vf.State) -> bool:
        return False

    async def setup_state(self, state: vf.State) -> vf.State:
        board_state = _state_from_info(state["info"])
        state["board_state"] = board_state
        state["turn_count"] = board_state.turn_count
        state["picked_key"] = board_state.picked_key
        state["solved"] = False
        state["invalid_actions"] = 0
        state["total_reward"] = 0.0
        state["last_feedback"] = "Board initialized."
        state["optimal_num_actions"] = int(state["info"]["optimal_num_actions"])
        return await super().setup_state(state)

    def _record_step(self, state: vf.State, transition: Transition, parsed_kind: str) -> None:
        if not state["trajectory"]:
            return
        latest_step = state["trajectory"][-1]
        latest_step["reward"] = transition.reward
        latest_step["extras"] = {
            "action": transition.action.value,
            "parsed_kind": parsed_kind,
            "moved": transition.moved,
            "valid_action": transition.valid_action,
            "feedback": transition.feedback,
            "solved": transition.solved,
            "reached_goal": transition.reached_goal,
            "turn_count": transition.state.turn_count,
        }

    def _invalid_feedback(self, detail: str) -> str:
        return (
            f"{detail} The board is unchanged. "
            f"Retry by calling `{PLAY_MOVE_TOOL}` with one of: UR, R, DR, DL, L, UL, Pickup."
        )

    def _apply_transition(self, state: vf.State, transition: Transition) -> None:
        state["board_state"] = transition.state
        state["turn_count"] = transition.state.turn_count
        state["picked_key"] = transition.state.picked_key
        state["solved"] = transition.solved
        state["last_feedback"] = transition.feedback
        state["total_reward"] = float(state.get("total_reward", 0.0)) + transition.reward
        if not transition.valid_action:
            state["invalid_actions"] = int(state.get("invalid_actions", 0)) + 1
        if transition.termination_reason is not None:
            state["termination_reason"] = transition.termination_reason

    def _reject_tool_calls(
        self,
        state: vf.State,
        *,
        parsed_kind: str,
        feedback: str,
        tool_call_ids: Sequence[str],
    ) -> tuple[list[dict[str, Any]], Transition]:
        board_state = cast(GameState, state["board_state"])
        transition = Transition(
            state=GameState(
                boundary=board_state.boundary,
                player_pos=board_state.player_pos,
                key_pos=board_state.key_pos,
                flag_pos=board_state.flag_pos,
                walls=board_state.walls,
                turn_count=board_state.turn_count + 1,
                picked_key=board_state.picked_key,
            ),
            action=Action.PICKUP,
            moved=False,
            valid_action=False,
            reward=-0.05,
            solved=False,
            reached_goal=board_state.player_pos == board_state.flag_pos,
            feedback=feedback,
            termination_reason=None,
        )
        tool_messages = [
            _tool_message(
                tool_call_id,
                {"status": "error", "reason": parsed_kind, "message": feedback},
            )
            for tool_call_id in tool_call_ids
        ]
        return tool_messages, transition

    def _apply_assistant_action(
        self,
        assistant_message: ChatCompletionAssistantMessageParam,
        state: vf.State,
    ) -> tuple[list[dict[str, Any]], Transition, str]:
        tool_calls = list(assistant_message.get("tool_calls") or [])
        if not tool_calls:
            feedback = self._invalid_feedback(f"You did not call the `{PLAY_MOVE_TOOL}` tool.")
            tool_messages, transition = self._reject_tool_calls(
                state,
                parsed_kind="no_tool_call",
                feedback=feedback,
                tool_call_ids=(),
            )
            return tool_messages, transition, "no_tool_call"

        if len(tool_calls) != 1:
            feedback = self._invalid_feedback(f"Call `{PLAY_MOVE_TOOL}` exactly once per turn.")
            tool_messages, transition = self._reject_tool_calls(
                state,
                parsed_kind="multiple_tool_calls",
                feedback=feedback,
                tool_call_ids=[
                    str(tool_call.get("id", ""))
                    for tool_call in tool_calls
                    if tool_call.get("id")
                ],
            )
            return tool_messages, transition, "multiple_tool_calls"

        tool_call = tool_calls[0]
        tool_call_id = str(tool_call.get("id", ""))
        tool_name = _tool_call_name(tool_call)
        if tool_name != PLAY_MOVE_TOOL:
            feedback = self._invalid_feedback(
                f"Unknown tool `{tool_name or '<empty>'}`. Only `{PLAY_MOVE_TOOL}` is available."
            )
            tool_messages, transition = self._reject_tool_calls(
                state,
                parsed_kind="unknown_tool",
                feedback=feedback,
                tool_call_ids=(tool_call_id,),
            )
            return tool_messages, transition, "unknown_tool"

        raw_arguments = _tool_call_arguments(tool_call)
        if isinstance(raw_arguments, dict):
            parsed_arguments = raw_arguments
        else:
            try:
                parsed_arguments = json.loads(str(raw_arguments))
            except json.JSONDecodeError:
                feedback = self._invalid_feedback(
                    "Tool arguments must be valid JSON like {\"action\": \"DR\"}."
                )
                tool_messages, transition = self._reject_tool_calls(
                    state,
                    parsed_kind="invalid_tool_arguments",
                    feedback=feedback,
                    tool_call_ids=(tool_call_id,),
                )
                return tool_messages, transition, "invalid_tool_arguments"

        if not isinstance(parsed_arguments, dict):
            feedback = self._invalid_feedback(
                "Tool arguments must decode to a JSON object with an `action` field."
            )
            tool_messages, transition = self._reject_tool_calls(
                state,
                parsed_kind="invalid_tool_arguments",
                feedback=feedback,
                tool_call_ids=(tool_call_id,),
            )
            return tool_messages, transition, "invalid_tool_arguments"

        raw_action = parsed_arguments.get("action")
        if not isinstance(raw_action, str):
            feedback = self._invalid_feedback("The `action` argument is required.")
            tool_messages, transition = self._reject_tool_calls(
                state,
                parsed_kind="missing_action_argument",
                feedback=feedback,
                tool_call_ids=(tool_call_id,),
            )
            return tool_messages, transition, "missing_action_argument"

        try:
            action = parse_action(raw_action)
        except ValueError as exc:
            feedback = self._invalid_feedback(str(exc))
            tool_messages, transition = self._reject_tool_calls(
                state,
                parsed_kind="invalid_action_argument",
                feedback=feedback,
                tool_call_ids=(tool_call_id,),
            )
            return tool_messages, transition, "invalid_action_argument"

        transition = step(cast(GameState, state["board_state"]), action)
        tool_messages = [
            _tool_message(
                tool_call_id,
                {
                    "status": "accepted",
                    "action": action.value,
                    "moved": transition.moved,
                    "valid_action": transition.valid_action,
                    "feedback": transition.feedback,
                    "termination_reason": transition.termination_reason,
                },
            )
        ]
        return tool_messages, transition, "valid"

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        assistant_message = _last_assistant_message(messages)
        tool_messages, transition, parsed_kind = self._apply_assistant_action(
            assistant_message, state
        )
        self._apply_transition(state, transition)
        board_text = render_board(cast(GameState, state["board_state"]))
        response_messages: list[dict[str, Any]] = [
            *tool_messages,
            {"role": "user", "content": f"{transition.feedback}\n\n{board_text}"},
        ]
        self._record_step(state, transition, parsed_kind)
        if transition.termination_reason is not None:
            state["final_env_response"] = response_messages
        return response_messages

    @vf.stop(priority=80)
    async def game_ended(self, state: vf.State) -> bool:
        return state.get("termination_reason") is not None


def load_environment(
    difficulty: str = "all",
    seed: int = 0,
    num_examples: int = DEFAULT_GENERATED_DATASET_SIZE,
    eval_examples: int | None = None,
    max_turns: int = MAX_MODEL_TURNS,
) -> vf.Environment:
    if difficulty not in {"all", "easy", "medium", "hard"}:
        raise ValueError("difficulty must be one of: all, easy, medium, hard")
    eval_count = num_examples if eval_examples is None else int(eval_examples)
    return GolfEnv(
        dataset=_make_dataset_builder(
            num_examples=num_examples,
            seed=seed,
            split="train",
            difficulty=difficulty,
        ),
        eval_dataset=_make_dataset_builder(
            num_examples=eval_count,
            seed=seed,
            split="eval",
            difficulty=difficulty,
        ),
        system_prompt=SYSTEM_PROMPT,
        rubric=build_rubric(),
        env_id="golf",
        map_kwargs={"load_from_cache_file": False, "keep_in_memory": True},
        max_turns=max_turns,
    )
