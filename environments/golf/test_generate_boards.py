from __future__ import annotations

import random
import unittest

from generate_boards import generate_board_from_template, generate_boards
from golf_core import DEFAULT_SPECS, shortest_solution


class GenerateBoardsTests(unittest.TestCase):
    def test_generated_board_is_solvable(self) -> None:
        spec = generate_board_from_template(DEFAULT_SPECS[3], rng=random.Random(7))
        solution = shortest_solution(spec)
        self.assertGreater(len(solution), 0)
        self.assertEqual(len(spec.walls), len(DEFAULT_SPECS[3].walls))

    def test_bulk_generation_respects_count_and_difficulty(self) -> None:
        boards = generate_boards(
            count=5,
            seed=3,
            difficulty="easy",
        )
        self.assertEqual(len(boards), 5)
        self.assertTrue(all(board.difficulty == "easy" for board in boards))
        self.assertTrue(all(board.board_id.endswith(f"{index:03d}") for index, board in enumerate(boards)))

    def test_generation_is_deterministic_for_seed(self) -> None:
        boards_a = generate_boards(count=2, seed=13, difficulty="medium")
        boards_b = generate_boards(count=2, seed=13, difficulty="medium")
        self.assertEqual(boards_a, boards_b)

    def test_generation_produces_new_ids(self) -> None:
        boards = generate_boards(count=3, seed=11, difficulty="easy")
        self.assertTrue(all("_generated_" in board.board_id for board in boards))


if __name__ == "__main__":
    unittest.main()
