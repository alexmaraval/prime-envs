from __future__ import annotations

import asyncio
import json
import unittest
from collections import Counter

from hangman_agent.env import HangmanEnv, load_environment


def make_tool_completion(letter: str) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "suggest_letter",
                        "arguments": json.dumps({"letter": letter}),
                    },
                }
            ],
        }
    ]


def make_flat_tool_completion(letter: str) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_flat_1",
                    "name": "suggest_letter",
                    "arguments": {"letter": letter},
                }
            ],
        }
    ]


class EnvironmentTests(unittest.TestCase):
    def test_load_environment_defaults_to_easy_generation(self) -> None:
        env = load_environment(seed=3, num_examples=3)
        self.assertIsInstance(env, HangmanEnv)
        self.assertEqual(env.generation_config.difficulty, "easy")

    def test_load_environment_rejects_removed_generation_knobs(self) -> None:
        with self.assertRaises(TypeError):
            load_environment(**{"ambiguity_min": 1})
        with self.assertRaises(TypeError):
            load_environment(**{"allow_partial_starts": True})

    def test_load_environment_returns_multiturn_env(self) -> None:
        env = load_environment(difficulty="easy", seed=3, num_examples=3)
        self.assertIsInstance(env, HangmanEnv)
        dataset = env.get_eval_dataset(2)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["prompt"][0]["role"], "system")
        self.assertIn("word:", dataset[0]["prompt"][1]["content"])
        self.assertIn("wrong letters:", dataset[0]["prompt"][1]["content"])
        self.assertNotIn("Goal:", dataset[0]["prompt"][1]["content"])
        self.assertEqual(env.oai_tools[0]["function"]["name"], "suggest_letter")

    def test_load_environment_supports_mixed_difficulty_generation(self) -> None:
        env = load_environment(seed=3, num_examples=10, difficulty_mix=[0.3, 0.4, 0.3])
        self.assertIsInstance(env, HangmanEnv)
        self.assertEqual(env.generation_config.difficulty, "mixed")
        self.assertEqual(env.generation_config.difficulty_mix, (0.3, 0.4, 0.3))
        dataset = env.get_eval_dataset(10)
        counts = Counter(example["info"]["difficulty"] for example in dataset)
        self.assertEqual(counts, {"easy": 3, "medium": 4, "hard": 3})

    def test_env_response_records_valid_tool_step(self) -> None:
        env = load_environment(difficulty="easy", seed=9, num_examples=1)
        dataset = env.get_eval_dataset(1)
        prompt = dataset[0]["prompt"]
        completion = make_tool_completion("E")
        state = {
            "info": dataset[0]["info"],
            "prompt": prompt,
            "trajectory": [
                {
                    "prompt": prompt,
                    "completion": completion,
                    "response": None,
                    "tokens": None,
                    "reward": None,
                    "advantage": None,
                    "is_truncated": False,
                    "trajectory_id": "test",
                    "extras": {},
                }
            ],
        }

        asyncio.run(env.setup_state(state))
        response_messages = asyncio.run(
            env.env_response(prompt + state["trajectory"][0]["completion"], state)
        )
        self.assertIsNotNone(state["trajectory"][-1]["reward"])
        self.assertEqual(response_messages[0]["role"], "tool")
        self.assertIn("reward_components", state["trajectory"][-1]["extras"])
        self.assertEqual(state["trajectory"][-1]["extras"]["parsed_kind"], "valid")

    def test_env_response_accepts_flattened_tool_call_shape(self) -> None:
        env = load_environment(difficulty="easy", seed=9, num_examples=1)
        dataset = env.get_eval_dataset(1)
        prompt = dataset[0]["prompt"]
        completion = make_flat_tool_completion("E")
        state = {
            "info": dataset[0]["info"],
            "prompt": prompt,
            "trajectory": [
                {
                    "prompt": prompt,
                    "completion": completion,
                    "response": None,
                    "tokens": None,
                    "reward": None,
                    "advantage": None,
                    "is_truncated": False,
                    "trajectory_id": "test",
                    "extras": {},
                }
            ],
        }

        asyncio.run(env.setup_state(state))
        response_messages = asyncio.run(
            env.env_response(prompt + state["trajectory"][0]["completion"], state)
        )
        tool_payload = json.loads(response_messages[0]["content"])
        self.assertEqual(tool_payload["status"], "accepted")
        self.assertEqual(state["trajectory"][-1]["extras"]["parsed_kind"], "valid")

    def test_missing_tool_call_retries_same_board(self) -> None:
        env = load_environment(difficulty="easy", seed=9, num_examples=1)
        dataset = env.get_eval_dataset(1)
        prompt = dataset[0]["prompt"]
        state = {
            "info": dataset[0]["info"],
            "prompt": prompt,
            "trajectory": [
                {
                    "prompt": prompt,
                    "completion": [{"role": "assistant", "content": "I guess E"}],
                    "response": None,
                    "tokens": None,
                    "reward": None,
                    "advantage": None,
                    "is_truncated": False,
                    "trajectory_id": "test",
                    "extras": {},
                }
            ],
        }

        asyncio.run(env.setup_state(state))
        starting_turns = state["turns_remaining"]
        response_messages = asyncio.run(
            env.env_response(prompt + state["trajectory"][0]["completion"], state)
        )

        self.assertEqual(response_messages[-2]["role"], "user")
        self.assertIn("did not call", response_messages[-2]["content"].lower())
        self.assertEqual(response_messages[-1]["role"], "user")
        self.assertEqual(state["turns_remaining"], starting_turns)
        self.assertEqual(state["last_outcome"], "invalid_action")
        self.assertEqual(state["last_reward"], -0.05)
        self.assertIn("suggest_letter", state["last_feedback"])
        self.assertIn("unchanged", state["last_feedback"].lower())

    def test_repeat_tool_feedback_explains_prior_guess(self) -> None:
        env = load_environment(difficulty="easy", seed=9, num_examples=1)
        state = {
            "info": {
                "secret_word": "APPLE",
                "frequency_tier": "common",
                "difficulty": "easy",
                "turns_remaining": 6,
                "pre_revealed_letters": ["A"],
                "pre_wrong_letters": ["Q"],
                "remaining_attempts": 6,
                "candidate_count": 3,
                "word_length": 5,
                "distinct_letter_count": 4,
                "repeat_density": 0.2,
                "seed": 0,
                "config": {"difficulty": "easy"},
            },
            "prompt": [],
            "trajectory": [
                {
                    "prompt": [],
                    "completion": make_tool_completion("A"),
                    "response": None,
                    "tokens": None,
                    "reward": None,
                    "advantage": None,
                    "is_truncated": False,
                    "trajectory_id": "test",
                    "extras": {},
                }
            ],
        }

        asyncio.run(env.setup_state(state))
        response_messages = asyncio.run(
            env.env_response(state["trajectory"][0]["completion"], state)
        )

        tool_payload = json.loads(response_messages[0]["content"])
        self.assertEqual(tool_payload["outcome"], "repeat")
        self.assertIn("already tried as a correct letter", tool_payload["message"])
        self.assertIn("wrong letters: Q", response_messages[-1]["content"])
        self.assertIn("turns remaining: 5", response_messages[-1]["content"])
        self.assertIn("hanged: 17%", response_messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
