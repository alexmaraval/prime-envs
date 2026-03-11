from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

ALPHABET = tuple(string.ascii_uppercase)
ParseKind = Literal["valid", "invalid_format", "invalid_letter"]

OUTCOME_START = "start"
OUTCOME_CORRECT = "correct"
OUTCOME_WRONG = "wrong"
OUTCOME_REPEAT = "repeat"
OUTCOME_INVALID_ACTION = "invalid_action"
OUTCOME_SOLVED = "solved"

TERMINATION_SOLVED = "solved"
TERMINATION_UNWINNABLE_BY_ATTEMPTS = "unwinnable_by_attempts"
TERMINATION_UNWINNABLE_BY_TURNS = "unwinnable_by_turns"
TERMINATION_ATTEMPTS_EXHAUSTED = "attempts_exhausted"
TERMINATION_TURNS_EXHAUSTED = "turns_exhausted"
TERMINATION_TOO_MANY_INVALID_ACTIONS = "too_many_invalid_actions"

REWARD_COMPONENT_KEYS = (
    "valid_guess_bonus",
    "solve_reward",
)


@dataclass(frozen=True, slots=True)
class RewardWeights:
    valid_guess_bonus: float = 0.01
    solve_reward: float = 1.0


DEFAULT_REWARD_WEIGHTS = RewardWeights()


@dataclass(frozen=True, slots=True)
class ParsedGuess:
    kind: ParseKind
    guess: str | None
    message: str


def _already_guessed_letters(state: Mapping[str, Any]) -> list[str]:
    return normalize_letters(
        [*state["correct_guesses"], *state["incorrect_guesses"]]
    )


def _termination_feedback(reason: str | None) -> str | None:
    if reason == TERMINATION_SOLVED:
        return "Game over: the word is solved."
    if reason == TERMINATION_TURNS_EXHAUSTED:
        return "Game over: the hang reached 100%."
    if reason == TERMINATION_TOO_MANY_INVALID_ACTIONS:
        return "Game over: too many invalid actions were made."
    return None


def _merge_feedback(message: str, reason: str | None) -> str:
    termination_message = _termination_feedback(reason)
    if termination_message is None:
        return message
    if not message:
        return termination_message
    return f"{message} {termination_message}"


def _position_word(count: int) -> str:
    return "position" if count == 1 else "positions"


def _hanged_percentage(state: Mapping[str, Any]) -> int:
    initial_turns = max(1, int(state["initial_turns"]))
    turns_remaining = max(0, int(state["turns_remaining"]))
    turns_used = max(0, initial_turns - turns_remaining)
    percentage = round((100 * turns_used) / initial_turns)
    return max(0, min(100, int(percentage)))


def normalize_word(word: str) -> str:
    normalized = (word or "").strip().upper()
    if not normalized or not normalized.isascii() or not normalized.isalpha():
        raise ValueError(f"word must be ASCII alphabetic, got {word!r}")
    return normalized


def normalize_letters(letters: Iterable[str]) -> list[str]:
    normalized = {
        letter.strip().upper()
        for letter in letters
        if isinstance(letter, str) and len(letter.strip()) == 1 and letter.strip().isalpha()
    }
    return sorted(normalized)


def distinct_letters(word: str) -> list[str]:
    return sorted(set(normalize_word(word)))


def compute_repeat_density(word: str) -> float:
    normalized = normalize_word(word)
    if not normalized:
        return 0.0
    return 1.0 - (len(set(normalized)) / len(normalized))


def build_pattern(secret_word: str, correct_guesses: Iterable[str]) -> list[str]:
    guessed = set(normalize_letters(correct_guesses))
    word = normalize_word(secret_word)
    return [letter if letter in guessed else "_" for letter in word]


def count_distinct_unrevealed(secret_word: str, correct_guesses: Iterable[str]) -> int:
    guessed = set(normalize_letters(correct_guesses))
    return len(set(normalize_word(secret_word)) - guessed)


def format_letters(letters: Iterable[str]) -> str:
    normalized = normalize_letters(letters)
    return ", ".join(normalized) if normalized else "-"


def render_board(state: Mapping[str, Any], reveal_word: bool = False) -> str:
    word_tokens = (
        " ".join(list(str(state["secret_word"])))
        if reveal_word
        else " ".join(state["revealed_pattern"])
    )
    lines = [
        f"word: {word_tokens}",
        f"wrong letters: {format_letters(state['incorrect_guesses'])}",
        f"hanged: {_hanged_percentage(state)}%",
        f"turns remaining: {int(state['turns_remaining'])}",
    ]
    if reveal_word and state.get("termination_reason") != TERMINATION_SOLVED:
        lines.append(f"answer: {state['secret_word']}")
    return "\n".join(lines)


def render_initial_prompt(state: Mapping[str, Any]) -> str:
    return render_board(state)


def task_to_info(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "secret_word": normalize_word(str(task["secret_word"])),
        "frequency_tier": str(task["frequency_tier"]),
        "difficulty": str(task["difficulty"]),
        "remaining_attempts": int(task["remaining_attempts"]),
        "turns_remaining": int(task["turns_remaining"]),
        "pre_revealed_letters": normalize_letters(task.get("pre_revealed_letters", [])),
        "pre_wrong_letters": normalize_letters(task.get("pre_wrong_letters", [])),
        "candidate_count": int(task["candidate_count"]),
        "word_length": int(task["word_length"]),
        "distinct_letter_count": int(task["distinct_letter_count"]),
        "repeat_density": float(task["repeat_density"]),
        "seed": int(task["seed"]),
        "config": dict(task.get("config", {})),
    }


def initialize_game_state(task_info: Mapping[str, Any]) -> dict[str, Any]:
    info = task_to_info(task_info)
    secret_word = info["secret_word"]
    correct_guesses = info["pre_revealed_letters"]
    incorrect_guesses = info["pre_wrong_letters"]

    if set(correct_guesses) & set(incorrect_guesses):
        raise ValueError("correct and incorrect guesses must be disjoint")
    if any(letter not in secret_word for letter in correct_guesses):
        raise ValueError("pre-revealed letters must be present in the secret word")
    if any(letter in secret_word for letter in incorrect_guesses):
        raise ValueError("pre-guessed wrong letters must not be present in the secret word")

    revealed_pattern = build_pattern(secret_word, correct_guesses)
    initial_hidden_positions = revealed_pattern.count("_")
    if initial_hidden_positions <= 0:
        raise ValueError("task must not start already solved")

    initial_turns = max(1, int(info["turns_remaining"]))

    return {
        "secret_word": secret_word,
        "word_length": len(secret_word),
        "correct_guesses": correct_guesses,
        "incorrect_guesses": incorrect_guesses,
        "all_guesses_in_order": [],
        "turns_remaining": initial_turns,
        "revealed_pattern": revealed_pattern,
        "initial_hidden_positions": initial_hidden_positions,
        "initial_turns": initial_turns,
        "last_outcome": OUTCOME_START,
        "last_guess": None,
        "last_reward": 0.0,
        "last_feedback": None,
        "reward_history": [],
        "total_reward": 0.0,
        "task_info": info,
        "termination_reason": None,
        "solved": False,
        "num_correct_new_guesses": 0,
        "num_wrong_new_guesses": 0,
        "num_repeated_guesses": 0,
        "num_invalid_outputs": 0,
        "max_invalid_outputs": max(6, initial_turns * 2),
        "positions_revealed": 0,
    }


def blank_reward_components() -> dict[str, float]:
    return {key: 0.0 for key in REWARD_COMPONENT_KEYS}


def termination_reason(state: Mapping[str, Any]) -> str | None:
    distinct_unrevealed = count_distinct_unrevealed(
        str(state["secret_word"]), state["correct_guesses"]
    )
    if distinct_unrevealed == 0:
        return TERMINATION_SOLVED
    if int(state.get("num_invalid_outputs", 0)) >= int(
        state.get("max_invalid_outputs", 0)
    ):
        return TERMINATION_TOO_MANY_INVALID_ACTIONS
    if int(state["turns_remaining"]) <= 0:
        return TERMINATION_TURNS_EXHAUSTED
    return None


def apply_invalid_action(
    state: dict[str, Any],
    *,
    parsed_kind: str,
    feedback_message: str,
    reward_weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> dict[str, Any]:
    reward_components = blank_reward_components()

    state["last_guess"] = None
    state["num_invalid_outputs"] += 1
    state["solved"] = False
    state["last_outcome"] = OUTCOME_INVALID_ACTION
    state["termination_reason"] = termination_reason(state)
    state["last_feedback"] = _merge_feedback(
        feedback_message,
        state["termination_reason"],
    )

    step_reward = round(sum(reward_components.values()), 8)
    state["last_reward"] = step_reward
    state["reward_history"].append(step_reward)
    state["total_reward"] = round(sum(state["reward_history"]), 8)

    return {
        "guess": None,
        "parsed_kind": parsed_kind,
        "step_reward": step_reward,
        "reward_components": reward_components,
        "feedback": state["last_feedback"],
        "termination_reason": state["termination_reason"],
    }


def apply_guess(
    state: dict[str, Any],
    parsed_guess: ParsedGuess,
    reward_weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> dict[str, Any]:
    secret_word = str(state["secret_word"])
    correct_guesses_before = set(state["correct_guesses"])
    incorrect_guesses_before = set(state["incorrect_guesses"])
    guessed_before = correct_guesses_before | incorrect_guesses_before

    if parsed_guess.kind == "invalid_format":
        return apply_invalid_action(
            state,
            parsed_kind=parsed_guess.kind,
            feedback_message=parsed_guess.message,
            reward_weights=reward_weights,
        )
    if parsed_guess.kind == "invalid_letter":
        return apply_invalid_action(
            state,
            parsed_kind=parsed_guess.kind,
            feedback_message=parsed_guess.message,
            reward_weights=reward_weights,
        )

    reward_components = blank_reward_components()
    state["last_guess"] = parsed_guess.guess
    state["last_feedback"] = None

    outcome = OUTCOME_START
    new_positions_revealed = 0
    feedback_message = ""

    if parsed_guess.guess in guessed_before:
        state["turns_remaining"] = max(0, int(state["turns_remaining"]) - 1)
        state["num_repeated_guesses"] += 1
        state["all_guesses_in_order"].append(parsed_guess.guess)
        outcome = OUTCOME_REPEAT
        prior_status = (
            "correct" if parsed_guess.guess in correct_guesses_before else "wrong"
        )
        feedback_message = (
            f"Repeated guess: {parsed_guess.guess} was already tried as a {prior_status} letter. "
            "The word does not change. Choose a new letter."
        )
    elif parsed_guess.guess and parsed_guess.guess in secret_word:
        state["correct_guesses"] = normalize_letters(
            [*state["correct_guesses"], parsed_guess.guess]
        )
        new_positions_revealed = sum(
            1 for letter in secret_word if letter == parsed_guess.guess
        )
        state["positions_revealed"] += new_positions_revealed
        state["num_correct_new_guesses"] += 1
        state["all_guesses_in_order"].append(parsed_guess.guess)
        reward_components["valid_guess_bonus"] = reward_weights.valid_guess_bonus
        outcome = OUTCOME_CORRECT
        feedback_message = (
            f"Accepted: {parsed_guess.guess} reveals "
            f"{new_positions_revealed} {_position_word(new_positions_revealed)}."
        )
    else:
        if parsed_guess.guess:
            state["incorrect_guesses"] = normalize_letters(
                [*state["incorrect_guesses"], parsed_guess.guess]
            )
            state["all_guesses_in_order"].append(parsed_guess.guess)
        state["turns_remaining"] = max(0, int(state["turns_remaining"]) - 1)
        state["num_wrong_new_guesses"] += 1
        reward_components["valid_guess_bonus"] = reward_weights.valid_guess_bonus
        outcome = OUTCOME_WRONG
        feedback_message = f"Accepted: {parsed_guess.guess} is not in the word."

    state["revealed_pattern"] = build_pattern(secret_word, state["correct_guesses"])
    current_termination_reason = termination_reason(state)
    if current_termination_reason == TERMINATION_SOLVED:
        reward_components["valid_guess_bonus"] = 0.0
        reward_components["solve_reward"] = reward_weights.solve_reward
        state["solved"] = True
        state["termination_reason"] = current_termination_reason
        state["last_outcome"] = OUTCOME_SOLVED
        feedback_message = (
            f"Accepted: {parsed_guess.guess} completes the word."
        )
    else:
        state["solved"] = False
        state["termination_reason"] = current_termination_reason
        state["last_outcome"] = outcome

    state["last_feedback"] = _merge_feedback(
        feedback_message,
        state["termination_reason"],
    )
    step_reward = round(sum(reward_components.values()), 8)
    state["last_reward"] = step_reward
    state["reward_history"].append(step_reward)
    state["total_reward"] = round(sum(state["reward_history"]), 8)

    return {
        "guess": parsed_guess.guess,
        "parsed_kind": parsed_guess.kind,
        "step_reward": step_reward,
        "reward_components": reward_components,
        "feedback": state["last_feedback"],
        "termination_reason": state["termination_reason"],
    }
