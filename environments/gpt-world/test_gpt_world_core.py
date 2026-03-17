from __future__ import annotations

import unittest

from gpt_world_core import (
    Action,
    DEFAULT_SPECS,
    make_initial_state,
    parse_action,
    render_board,
    shortest_solution,
    step,
)


class GPTWorldCoreTests(unittest.TestCase):
    def test_parse_action_supports_aliases(self) -> None:
        self.assertEqual(parse_action("dr"), Action.DOWNRIGHT)
        self.assertEqual(parse_action("Pickup"), Action.PICKUP)
        self.assertEqual(parse_action("pu"), Action.PICKUP)

    def test_invalid_move_keeps_board_in_place(self) -> None:
        spec = DEFAULT_SPECS[0]
        state = make_initial_state(spec)
        transition = step(state, Action.LEFT)
        self.assertFalse(transition.valid_action)
        self.assertEqual(transition.state.player_pos, state.player_pos)

    def test_pickup_requires_being_on_key(self) -> None:
        spec = DEFAULT_SPECS[0]
        state = make_initial_state(spec)
        transition = step(state, Action.PICKUP)
        self.assertFalse(transition.valid_action)
        self.assertFalse(transition.state.picked_key)

    def test_shortest_solution_solves_all_default_specs(self) -> None:
        for spec in DEFAULT_SPECS:
            state = make_initial_state(spec)
            for action in shortest_solution(spec):
                transition = step(state, action)
                state = transition.state
            self.assertTrue(state.picked_key, spec.board_id)
            self.assertEqual(state.player_pos, spec.flag, spec.board_id)

    def test_render_board_mentions_available_actions(self) -> None:
        board = render_board(make_initial_state(DEFAULT_SPECS[0]))
        self.assertIn("Available actions:", board)
        self.assertIn("@", board)
        self.assertIn("K", board)
        self.assertIn("P", board)


if __name__ == "__main__":
    unittest.main()
