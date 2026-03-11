from __future__ import annotations

import unittest

from hangman_agent.game import initialize_game_state, render_board
from hangman_agent.generator import (
    build_records,
    filter_lexicon,
    load_lexicon,
    resolve_generation_config,
)


class GeneratorTests(unittest.TestCase):
    def test_presets_resolve_to_expected_defaults(self) -> None:
        config = resolve_generation_config(difficulty="hard", seed=7, num_examples=4)
        self.assertEqual(config.allowed_attempts_max, 5)
        self.assertEqual(config.pre_revealed_letters_max, 0)
        self.assertIn("obscure", config.frequency_tiers)

    def test_easy_is_the_default_generation_preset(self) -> None:
        config = resolve_generation_config(seed=7, num_examples=4)
        self.assertEqual(config.difficulty, "easy")
        self.assertEqual(config.allowed_attempts_min, 8)
        self.assertEqual(config.allowed_attempts_max, 12)

    def test_record_generation_is_deterministic(self) -> None:
        config = resolve_generation_config(difficulty="medium", seed=11, num_examples=5)
        lexicon = filter_lexicon(load_lexicon(), config)
        first = build_records(config, lexicon, split="eval")
        second = build_records(config, lexicon, split="eval")
        self.assertEqual(first, second)

    def test_generated_records_are_solvable(self) -> None:
        config = resolve_generation_config(difficulty="easy", seed=5, num_examples=6)
        lexicon = filter_lexicon(load_lexicon(), config)
        records = build_records(config, lexicon, split="train")
        self.assertEqual(len(records), 6)
        for record in records:
            state = initialize_game_state(record["info"])
            self.assertEqual(record["info"]["pre_revealed_letters"], [])
            self.assertEqual(record["info"]["pre_wrong_letters"], [])
            self.assertGreaterEqual(record["info"]["candidate_count"], 1)
            self.assertEqual(record["prompt"][0]["content"], render_board(state))
            self.assertIn("word: _", record["prompt"][0]["content"])
            self.assertIn("wrong letters: -", record["prompt"][0]["content"])

    def test_easy_games_start_fully_hidden(self) -> None:
        config = resolve_generation_config(difficulty="easy", seed=5, num_examples=1)
        lexicon = filter_lexicon(load_lexicon(), config)
        record = build_records(config, lexicon, split="train")[0]
        self.assertEqual(record["info"]["pre_revealed_letters"], [])
        self.assertEqual(record["info"]["pre_wrong_letters"], [])
        self.assertFalse(config.allow_partial_starts)

    def test_default_easy_generation_has_word_variety(self) -> None:
        config = resolve_generation_config(difficulty="easy", seed=5, num_examples=12)
        lexicon = filter_lexicon(load_lexicon(), config)
        records = build_records(config, lexicon, split="train")
        words = {record["info"]["secret_word"] for record in records}
        self.assertGreater(len(words), 1)

    def test_generation_falls_back_to_duplicates_when_unique_pool_is_too_small(self) -> None:
        config = resolve_generation_config(
            difficulty="easy",
            seed=5,
            num_examples=12,
            ambiguity_min=1,
            ambiguity_max=8,
        )
        lexicon = filter_lexicon(load_lexicon(), config)
        records = build_records(config, lexicon, split="train")
        unique_keys = {
            (
                record["info"]["secret_word"],
                record["info"]["turns_remaining"],
                record["info"]["remaining_attempts"],
            )
            for record in records
        }
        self.assertEqual(len(records), 12)
        self.assertLess(len(unique_keys), len(records))


if __name__ == "__main__":
    unittest.main()
