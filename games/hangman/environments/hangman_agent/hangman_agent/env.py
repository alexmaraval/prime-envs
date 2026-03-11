from __future__ import annotations

import json
from typing import Any, Sequence, cast

from openai.types.chat import ChatCompletionAssistantMessageParam
import verifiers as vf

from .game import (
    ParsedGuess,
    apply_guess,
    apply_invalid_action,
    initialize_game_state,
    render_board,
)
from .generator import (
    GenerationConfig,
    filter_lexicon,
    load_lexicon,
    make_dataset_builder,
    resolve_generation_config,
)

SYSTEM_PROMPT = """You are playing Hangman.

Reveal the hidden English word before the hang reaches 100%.
On each turn, call `suggest_letter` exactly once with one new English alphabet character.
Do not answer with XML tags or a plain-text guess.
Think step by step before using a tool call.
"""

SUGGEST_LETTER_TOOL_NAME = "suggest_letter"
MAX_MODEL_TURNS = 64


def suggest_letter(letter: str) -> str:
    """Submit your next Hangman guess.

    Args:
        letter: Exactly one English alphabet character to guess.
    """
    normalized = (letter or "").strip().upper()
    if len(normalized) != 1 or not normalized.isascii() or not normalized.isalpha():
        raise ValueError("`letter` must be exactly one English alphabet character.")
    return normalized


def _last_assistant_message(messages: vf.Messages) -> ChatCompletionAssistantMessageParam:
    if not isinstance(messages, list):
        raise TypeError(f"expected chat messages, got {type(messages).__name__}")
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return cast(ChatCompletionAssistantMessageParam, message)
    raise ValueError("expected an assistant message in the conversation")


def _tool_message(tool_call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": json.dumps(payload, ensure_ascii=True, sort_keys=True),
        "tool_call_id": tool_call_id,
    }


def _turn_reward_sum(state: vf.State) -> float:
    return float(state.get("total_reward", 0.0))


def _solved_metric(state: vf.State) -> float:
    return 1.0 if state.get("solved") else 0.0


def _invalid_outputs_metric(state: vf.State) -> float:
    return float(state.get("num_invalid_outputs", 0.0))


def _repeated_guesses_metric(state: vf.State) -> float:
    return float(state.get("num_repeated_guesses", 0.0))


def _positions_revealed_metric(state: vf.State) -> float:
    return float(state.get("positions_revealed", 0.0))


def build_rubric() -> vf.Rubric:
    rubric = vf.Rubric()
    rubric.add_reward_func(_turn_reward_sum, weight=1.0)
    rubric.add_metric(_solved_metric)
    rubric.add_metric(_invalid_outputs_metric)
    rubric.add_metric(_repeated_guesses_metric)
    rubric.add_metric(_positions_revealed_metric)
    return rubric


class HangmanEnv(vf.ToolEnv):
    def __init__(
        self,
        generation_config: GenerationConfig,
        max_turns: int = MAX_MODEL_TURNS,
        **kwargs,
    ):
        self.generation_config = generation_config
        super().__init__(tools=[suggest_letter], max_turns=max_turns, **kwargs)

    async def no_tools_called(self, state: vf.State) -> bool:
        return False

    async def setup_state(self, state: vf.State) -> vf.State:
        state.update(initialize_game_state(state["info"]))
        return await super().setup_state(state)

    def _record_step(
        self,
        state: vf.State,
        transition: dict[str, Any],
    ) -> None:
        if not state["trajectory"]:
            return

        latest_step = state["trajectory"][-1]
        latest_step["reward"] = transition["step_reward"]
        latest_step["extras"] = {
            "guess": transition["guess"],
            "parsed_kind": transition["parsed_kind"],
            "feedback": transition.get("feedback"),
            "reward_components": transition["reward_components"],
            "termination_reason": transition["termination_reason"],
        }

    def _invalid_feedback(self, detail: str) -> str:
        return (
            f"{detail} The board is unchanged. "
            f"Retry the same board by calling `{SUGGEST_LETTER_TOOL_NAME}` "
            "with exactly one new English alphabet letter in the `letter` argument."
        )

    def _build_valid_guess(self, letter: str) -> ParsedGuess:
        return ParsedGuess(
            kind="valid",
            guess=letter,
            message="ok",
        )

    def _reject_tool_calls(
        self,
        state: vf.State,
        *,
        parsed_kind: str,
        feedback: str,
        tool_call_ids: Sequence[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tool_messages = [
            _tool_message(
                tool_call_id,
                {"status": "error", "reason": parsed_kind, "message": feedback},
            )
            for tool_call_id in tool_call_ids
        ]
        transition = apply_invalid_action(
            state,
            parsed_kind=parsed_kind,
            feedback_message=feedback,
        )
        return tool_messages, transition

    def _apply_assistant_action(
        self,
        assistant_message: ChatCompletionAssistantMessageParam,
        state: vf.State,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tool_calls = list(assistant_message.get("tool_calls") or [])
        if not tool_calls:
            feedback = self._invalid_feedback(
                f"You did not call the `{SUGGEST_LETTER_TOOL_NAME}` tool."
            )
            return self._reject_tool_calls(
                state,
                parsed_kind="no_tool_call",
                feedback=feedback,
                tool_call_ids=(),
            )

        if len(tool_calls) != 1:
            feedback = self._invalid_feedback(
                f"Call `{SUGGEST_LETTER_TOOL_NAME}` exactly once per turn."
            )
            return self._reject_tool_calls(
                state,
                parsed_kind="multiple_tool_calls",
                feedback=feedback,
                tool_call_ids=[
                    str(tool_call.get("id", ""))
                    for tool_call in tool_calls
                    if tool_call.get("id")
                ],
            )

        tool_call = tool_calls[0]
        tool_call_id = str(tool_call.get("id", ""))
        tool_name = str(tool_call.get("function", {}).get("name", ""))
        if tool_name != SUGGEST_LETTER_TOOL_NAME:
            feedback = self._invalid_feedback(
                f"Unknown tool `{tool_name or '<empty>'}`. Only `{SUGGEST_LETTER_TOOL_NAME}` is available."
            )
            return self._reject_tool_calls(
                state,
                parsed_kind="unknown_tool",
                feedback=feedback,
                tool_call_ids=(tool_call_id,),
            )

        raw_arguments = tool_call.get("function", {}).get("arguments", "{}")
        try:
            parsed_arguments = json.loads(str(raw_arguments))
        except json.JSONDecodeError:
            feedback = self._invalid_feedback(
                "Tool arguments must be valid JSON like {\"letter\": \"E\"}."
            )
            return self._reject_tool_calls(
                state,
                parsed_kind="invalid_tool_arguments",
                feedback=feedback,
                tool_call_ids=(tool_call_id,),
            )

        if not isinstance(parsed_arguments, dict):
            feedback = self._invalid_feedback(
                "Tool arguments must decode to a JSON object with a `letter` field."
            )
            return self._reject_tool_calls(
                state,
                parsed_kind="invalid_tool_arguments",
                feedback=feedback,
                tool_call_ids=(tool_call_id,),
            )

        raw_letter = parsed_arguments.get("letter")
        if not isinstance(raw_letter, str):
            feedback = self._invalid_feedback("The `letter` argument is required.")
            return self._reject_tool_calls(
                state,
                parsed_kind="missing_letter_argument",
                feedback=feedback,
                tool_call_ids=(tool_call_id,),
            )

        try:
            letter = suggest_letter(raw_letter)
        except ValueError as exc:
            feedback = self._invalid_feedback(str(exc))
            return self._reject_tool_calls(
                state,
                parsed_kind="invalid_letter_argument",
                feedback=feedback,
                tool_call_ids=(tool_call_id,),
            )

        parsed_guess = self._build_valid_guess(letter)
        transition = apply_guess(state, parsed_guess)
        tool_messages = [
            _tool_message(
                tool_call_id,
                {
                    "status": "accepted",
                    "letter": letter,
                    "outcome": state["last_outcome"],
                    "message": transition.get("feedback"),
                    "termination_reason": transition["termination_reason"],
                },
            )
        ]
        return tool_messages, transition

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        assistant_message = _last_assistant_message(messages)
        tool_messages, transition = self._apply_assistant_action(assistant_message, state)
        board = render_board(
            state,
            reveal_word=state.get("termination_reason") is not None,
        )
        feedback_messages = []
        if not tool_messages and transition.get("feedback"):
            feedback_messages.append({"role": "user", "content": transition["feedback"]})
        response_messages = [
            *tool_messages,
            *feedback_messages,
            {"role": "user", "content": board},
        ]
        self._record_step(state, transition)
        if state.get("termination_reason") is not None:
            state["final_env_response"] = response_messages
        return response_messages

    @vf.stop(priority=80)
    async def game_ended(self, state: vf.State) -> bool:
        return state.get("termination_reason") is not None


def load_environment(
    difficulty: str = "easy",
    seed: int = 0,
    num_examples: int = 128,
    word_length_min: int | None = None,
    word_length_max: int | None = None,
    frequency_tiers: Sequence[str] | str | None = None,
    repeat_density_min: float | None = None,
    repeat_density_max: float | None = None,
    allowed_attempts_min: int | None = None,
    allowed_attempts_max: int | None = None,
    pre_revealed_letters_min: int | None = None,
    pre_revealed_letters_max: int | None = None,
    pre_wrong_letters_min: int | None = None,
    pre_wrong_letters_max: int | None = None,
    ambiguity_min: int | None = None,
    ambiguity_max: int | None = None,
    allow_partial_starts: bool | None = None,
) -> vf.Environment:
    generation_config = resolve_generation_config(
        difficulty=difficulty,
        seed=seed,
        num_examples=num_examples,
        word_length_min=word_length_min,
        word_length_max=word_length_max,
        frequency_tiers=frequency_tiers,
        repeat_density_min=repeat_density_min,
        repeat_density_max=repeat_density_max,
        allowed_attempts_min=allowed_attempts_min,
        allowed_attempts_max=allowed_attempts_max,
        pre_revealed_letters_min=pre_revealed_letters_min,
        pre_revealed_letters_max=pre_revealed_letters_max,
        pre_wrong_letters_min=pre_wrong_letters_min,
        pre_wrong_letters_max=pre_wrong_letters_max,
        ambiguity_min=ambiguity_min,
        ambiguity_max=ambiguity_max,
        allow_partial_starts=allow_partial_starts,
    )
    lexicon = filter_lexicon(load_lexicon(), generation_config)
    map_kwargs = {"load_from_cache_file": False, "keep_in_memory": True}
    return HangmanEnv(
        dataset=make_dataset_builder(generation_config, lexicon, split="train"),
        eval_dataset=make_dataset_builder(generation_config, lexicon, split="eval"),
        generation_config=generation_config,
        system_prompt=SYSTEM_PROMPT,
        rubric=build_rubric(),
        env_id="hangman_agent",
        map_kwargs=map_kwargs,
    )
