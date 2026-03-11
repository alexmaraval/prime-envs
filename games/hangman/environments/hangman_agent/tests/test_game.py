from __future__ import annotations

import unittest

from hangman_agent.game import (
    OUTCOME_INVALID_ACTION,
    OUTCOME_REPEAT,
    OUTCOME_SOLVED,
    ParsedGuess,
    TERMINATION_SOLVED,
    TERMINATION_TURNS_EXHAUSTED,
    apply_guess,
    initialize_game_state,
    render_board,
)


def make_task(**overrides):
    base = {
        "secret_word": "APPLE",
        "frequency_tier": "common",
        "difficulty": "medium",
        "remaining_attempts": 5,
        "turns_remaining": 6,
        "pre_revealed_letters": [],
        "pre_wrong_letters": [],
        "candidate_count": 3,
        "word_length": 5,
        "distinct_letter_count": 4,
        "repeat_density": 0.2,
        "seed": 0,
        "config": {"difficulty": "medium"},
    }
    base.update(overrides)
    return base


def make_valid_guess(letter: str) -> ParsedGuess:
    return ParsedGuess(kind="valid", guess=letter.upper(), message="ok")


def make_invalid_guess(kind: str = "invalid_letter") -> ParsedGuess:
    return ParsedGuess(kind=kind, guess=None, message="invalid guess")


class GameStateTests(unittest.TestCase):
    def test_correct_guess_reveals_all_instances_and_gets_small_bonus(self) -> None:
        state = initialize_game_state(make_task())
        transition = apply_guess(state, make_valid_guess("p"))
        self.assertEqual(state["revealed_pattern"], ["_", "P", "P", "_", "_"])
        self.assertAlmostEqual(transition["step_reward"], 0.01, places=6)
        self.assertEqual(state["positions_revealed"], 2)
        self.assertEqual(state["turns_remaining"], 6)

    def test_repeated_guess_consumes_turn_and_gives_no_bonus(self) -> None:
        state = initialize_game_state(
            make_task(
                pre_revealed_letters=["A"],
                pre_wrong_letters=["Q"],
                turns_remaining=5,
            )
        )
        transition = apply_guess(state, make_valid_guess("a"))
        self.assertEqual(state["turns_remaining"], 4)
        self.assertEqual(state["last_outcome"], OUTCOME_REPEAT)
        self.assertEqual(transition["step_reward"], 0.0)
        self.assertIn("already tried as a correct letter", transition["feedback"])

    def test_invalid_guess_keeps_board_resources_unchanged(self) -> None:
        state = initialize_game_state(make_task(turns_remaining=5))
        transition = apply_guess(state, make_invalid_guess())
        self.assertEqual(state["turns_remaining"], 5)
        self.assertEqual(state["last_outcome"], OUTCOME_INVALID_ACTION)
        self.assertEqual(transition["parsed_kind"], "invalid_letter")
        self.assertEqual(transition["step_reward"], 0.0)

    def test_solving_returns_one(self) -> None:
        state = initialize_game_state(
            make_task(
                secret_word="AA",
                turns_remaining=3,
                candidate_count=1,
                word_length=2,
                distinct_letter_count=1,
                repeat_density=0.5,
            )
        )
        transition = apply_guess(state, make_valid_guess("a"))
        self.assertEqual(state["termination_reason"], TERMINATION_SOLVED)
        self.assertEqual(state["last_outcome"], OUTCOME_SOLVED)
        self.assertAlmostEqual(transition["step_reward"], 1.0, places=6)

    def test_wrong_guess_consumes_turn_and_records_wrong_letter(self) -> None:
        state = initialize_game_state(make_task(secret_word="MANGO", turns_remaining=4))
        apply_guess(state, make_valid_guess("z"))
        self.assertEqual(state["turns_remaining"], 3)
        self.assertEqual(state["incorrect_guesses"], ["Z"])

    def test_game_ends_when_hang_reaches_hundred_percent(self) -> None:
        state = initialize_game_state(make_task(secret_word="MANGO", turns_remaining=1))
        transition = apply_guess(state, make_valid_guess("z"))
        self.assertEqual(state["termination_reason"], TERMINATION_TURNS_EXHAUSTED)
        self.assertEqual(transition["step_reward"], 0.01)

    def test_render_board_is_minimal(self) -> None:
        state = initialize_game_state(
            make_task(pre_revealed_letters=["A"], pre_wrong_letters=["Q"])
        )
        board = render_board(state)
        self.assertIn("word: A _ _ _ _", board)
        self.assertIn("wrong letters: Q", board)
        self.assertIn("hanged:", board)
        self.assertNotIn("correct letters:", board)
        self.assertNotIn("last reward:", board)


if __name__ == "__main__":
    unittest.main()
