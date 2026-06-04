from __future__ import annotations

import unittest

from gpt_world import GPTWorldEnv, load_environment


class GPTWorldEnvTests(unittest.TestCase):
    def test_load_environment_exposes_play_move_tool_schema(self) -> None:
        env = load_environment(seed=3, num_examples=2)
        self.assertIsInstance(env, GPTWorldEnv)
        self.assertEqual(env.oai_tools[0]["function"]["name"], "play_move")

        properties = env.oai_tools[0]["function"]["parameters"]["properties"]
        self.assertEqual(
            properties["action"]["enum"],
            ["UR", "R", "DR", "DL", "L", "UL", "Pickup"],
        )


if __name__ == "__main__":
    unittest.main()
