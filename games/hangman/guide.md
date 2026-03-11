# Hangman Environment Guide

This guide explains how the current Hangman environment works, which files matter most, and how execution flows from the entrypoint to saved evaluation results.

## What changed in the current repo

The environment is now simpler than older versions of this workspace:

- every game starts from a fully hidden word
- the visible board is minimal: word pattern, wrong letters, hanged percent, turns remaining
- the live game is driven only by turns, not by a separate attempts budget
- the reward scheme is now a small reward for fresh valid guesses plus terminal uncovered and solved rewards
- repeated and invalid actions no longer apply negative reward
- there is still no `viewer.py`, `parser.py`, or `baselines.py` in the package

If you previously read an older guide or older eval outputs, that is the main drift.

## Directory map

- `environments/hangman_agent/`: installable Prime environment package
- `environments/hangman_agent/hangman_agent/env.py`: entrypoint, tool contract, rollout loop, rubric
- `environments/hangman_agent/hangman_agent/game.py`: board rendering, state transitions, feedback, reward logic
- `environments/hangman_agent/hangman_agent/generator.py`: deterministic task generation and dataset builders
- `environments/hangman_agent/hangman_agent/__init__.py`: exports `HangmanEnv` and `load_environment`
- `environments/hangman_agent/hangman_agent/hangman_agent.py`: thin compatibility re-export
- `hangman/local_eval.py`: helper that launches a local model server and then calls `prime eval run`
- `configs/eval/hangman-vllm.toml`: config-based local vLLM eval
- `configs/endpoints.vllm.py`: endpoint alias builder used by that config
- `environments/hangman_agent/tests/`: environment tests
- `tests/`: workspace tests for the local eval helper and endpoint config

## The main entrypoint

The environment entrypoint is `load_environment(...)` in `environments/hangman_agent/hangman_agent/env.py`.

Prime reaches it through:

1. `prime env install hangman_agent`
2. `environments/hangman_agent/pyproject.toml`
3. `environments/hangman_agent/hangman_agent/__init__.py`
4. `environments/hangman_agent/hangman_agent/env.py:load_environment`

So the runtime starts in `env.py`. The root-level scripts only help with execution.

## End-to-end trace: from `prime eval run` to output files

### 1. `load_environment(...)` assembles the environment

File: `environments/hangman_agent/hangman_agent/env.py`

`load_environment(...)` does the high-level setup:

1. resolve a `GenerationConfig`
2. load the bundled lexicon
3. filter that lexicon to match the config
4. create lazy train and eval dataset builders
5. construct `HangmanEnv(...)` with:
   - the system prompt
   - the `suggest_letter` tool
   - the rubric
   - the dataset builders
   - `env_id="hangman_agent"`

This is the handoff from static config to a runnable `vf.ToolEnv`.

### 2. `generator.py` builds the tasks

File: `environments/hangman_agent/hangman_agent/generator.py`

This module creates the examples before any rollout starts.

Current generation flow:

1. `resolve_generation_config(...)`
   - starts from an `easy`, `medium`, or `hard` preset
   - accepts overrides for word length, frequency tier, repeat density, ambiguity, and turn budget
   - then force-normalizes the current environment back to:
     - no pre-revealed letters
     - no pre-wrong letters
     - no partial starts
     - `turn_slack = 0`
2. `load_lexicon()`
   - reads `hangman_agent/data/lexicon.tsv`
3. `filter_lexicon(...)`
   - keeps only words that match the resolved generation constraints
4. `_attempt_task(...)`
   - samples one candidate word
   - sets `turns_remaining` by sampling within the configured range
   - mirrors that value into `remaining_attempts` in the task metadata, even though the runtime logic is now turn-based
   - computes ambiguity using the fully hidden board
5. `sample_task(...)`
   - tries up to 256 candidates and keeps the one closest to the ambiguity target
6. `build_records(...)`
   - deduplicates tasks
   - renders the initial board from `initialize_game_state(...)`
   - returns records shaped like:
     - `prompt`: the first user-visible board
     - `info`: structured task metadata
7. `build_dataset(...)` / `make_dataset_builder(...)`
   - wrap those records into a Hugging Face `Dataset`

The generator does not run Hangman. It only produces valid starting states.

### 3. `setup_state(...)` converts one task into a live game

Files:

- `environments/hangman_agent/hangman_agent/env.py`
- `environments/hangman_agent/hangman_agent/game.py`

When a rollout starts, `HangmanEnv.setup_state(...)` calls `initialize_game_state(state["info"])`.

`initialize_game_state(...)`:

- normalizes the task metadata
- validates the hidden word and prefilled letters
- computes the initial masked pattern
- sets `initial_turns` from `turns_remaining`
- creates the live state dict used for the rest of the episode

Important runtime fields include:

- `secret_word`
- `correct_guesses`
- `incorrect_guesses`
- `revealed_pattern`
- `turns_remaining`
- `initial_turns`
- `reward_history`
- `total_reward`
- `last_feedback`
- `termination_reason`
- `num_invalid_outputs`
- `num_repeated_guesses`
- `positions_revealed`

One important change from older versions: the live state no longer uses a separate `remaining_attempts` gameplay field. The board and termination logic are driven by turns.

### 4. The model acts through one tool call

File: `environments/hangman_agent/hangman_agent/env.py`

The current environment is tool-driven. Each turn the assistant must call:

`suggest_letter(letter: str)`

with exactly one new English alphabet character.

`HangmanEnv.env_response(...)` is the main rollout loop:

1. read the last assistant message
2. inspect its `tool_calls`
3. reject the turn if:
   - no tool was called
   - multiple tools were called
   - the wrong tool name was used
   - the arguments are not valid JSON
   - the `letter` field is missing or invalid
4. if the call is valid:
   - normalize the letter with `suggest_letter(...)`
   - create `ParsedGuess(kind="valid", ...)`
   - apply the guess through `apply_guess(...)`
5. render the next board
6. return response messages to the model

The returned response shape now depends on the failure mode:

- valid tool call: tool acknowledgement plus the next board
- invalid tool call with a tool id: tool error plus the next board
- missing tool call: a user feedback message plus the next board

### 5. `game.py` owns the Hangman rules

File: `environments/hangman_agent/hangman_agent/game.py`

This module is the state machine behind the environment.

Most important functions:

- `render_board(...)`
- `initialize_game_state(...)`
- `apply_guess(...)`
- `apply_invalid_action(...)`
- `termination_reason(...)`

The board shown to the model is intentionally minimal:

- `word: _ _ _ _ _`
- `wrong letters: ...`
- `hanged: N%`
- `turns remaining: N`

There is no longer a `correct letters:` line, `remaining attempts:` line, or `last reward:` line on the board.

### 6. Current gameplay semantics

The current environment is based on the hang percentage, which is derived from turns used out of `initial_turns`.

Behavior by action type:

- fresh correct guess:
  - reveals matching positions
  - gives `valid_guess_bonus = 0.01`
  - does not reduce `turns_remaining`
- fresh wrong guess:
  - adds the letter to `incorrect_guesses`
  - gives `valid_guess_bonus = 0.01`
  - reduces `turns_remaining` by 1
- any terminal step:
  - also gives `uncovered_percentage_reward = (# unique correct letters guessed) / (# unique letters in the secret word)`
- solved terminal step:
  - also gives `solved_reward = 1.0`
- repeated guess:
  - gives `0.0`
  - reduces `turns_remaining` by 1
  - feedback says whether that repeated letter was already known correct or wrong
- invalid tool action:
  - gives `0.0`
  - does not reduce `turns_remaining`
  - keeps the board unchanged
- solving guess:
  - adds `uncovered_percentage_reward = 1.0`
  - adds `solved_reward = 1.0`
  - terminates the game immediately

That means failed rollouts finish with a terminal reward between `0` and `1`, while solved rollouts finish with a terminal reward of `2.0` before the per-turn valid-guess bonus is added.

### 7. Termination logic

Files:

- `environments/hangman_agent/hangman_agent/game.py`
- `environments/hangman_agent/hangman_agent/env.py`

The episode ends when `termination_reason(state)` returns a reason.

Current terminal states are:

- `solved`
- `turns_exhausted`
- `too_many_invalid_actions`

The older `unwinnable_by_attempts` and `unwinnable_by_turns` branches are no longer part of the live logic.

When the episode ends, `env.py` also stores `state["final_env_response"]`.

### 8. Step metadata and rubric

File: `environments/hangman_agent/hangman_agent/env.py`

After each turn, `_record_step(...)` writes lightweight data into the latest trajectory entry:

- `guess`
- `parsed_kind`
- `feedback`
- `reward_components`
- `termination_reason`

The rubric currently emits:

- total reward
- solved
- invalid outputs
- repeated guesses
- positions revealed

Older references to `oracle_gap` or a dedicated `rollout_trace` state object do not match the current code.

### 9. Where the results come from

The full path to saved results is:

1. `prime eval run hangman_agent ...`
2. `load_environment(...)`
3. dataset generation in `generator.py`
4. state initialization in `game.py`
5. tool validation and response handling in `env.py`
6. reward, feedback, and termination updates in `game.py`
7. metric extraction in `env.py`
8. Prime serialization into the output directory

Run outputs are written under:

`environments/hangman_agent/outputs/evals/...`

The key files are:

- `results.jsonl`
- `metadata.json`
- sometimes `eval.log` for newer runs

## The local helper path

File: `hangman/local_eval.py`

`hangman/local_eval.py` is not the environment itself. It is a wrapper for local inference.

Its job is:

1. parse CLI flags
2. choose a host, port, and log path
3. build either an `mlx_lm.server` or `vllm serve` command
4. wait for `/v1/models`
5. construct a `prime eval run ...` command pointed at that local base URL
6. run the eval
7. stop the server unless `--keep-server` is set

So the separation of responsibility is:

- `env.py` defines Hangman behavior
- `local_eval.py` defines local-server orchestration

## The config-driven local vLLM path

Files:

- `configs/eval/hangman-vllm.toml`
- `configs/endpoints.vllm.py`

This is the alternative to `hangman/local_eval.py` when the local server already exists.

`configs/endpoints.vllm.py`:

- reads `LOCAL_VLLM_MODEL`
- reads `LOCAL_VLLM_BASE_URL`
- normalizes bare hosts to include `/v1`
- exposes the `local-vllm` alias

`configs/eval/hangman-vllm.toml`:

- selects that endpoint alias
- sets default concurrency and save-results behavior
- points the eval at `hangman_agent`

One subtlety remains: the helper/config defaults still request a `rollout_trace` state column, but the current environment no longer builds a standalone `rollout_trace` field.

## Tests that define the current contract

The most useful tests for understanding the current environment are:

- `environments/hangman_agent/tests/test_env.py`
  - verifies the board is minimal
  - verifies tool calls are required
  - verifies repeat-guess feedback explains whether the earlier guess was correct or wrong
- `environments/hangman_agent/tests/test_game.py`
  - verifies the simplified reward scheme
  - verifies correct guesses do not consume turns
  - verifies wrong and repeated guesses do consume turns
  - verifies the game ends when the hang reaches 100%
- `environments/hangman_agent/tests/test_generator.py`
  - verifies generation is deterministic
  - verifies games start fully hidden
  - verifies the rendered initial board matches runtime rendering
- `tests/test_local_eval.py`
  - verifies local-server and `prime eval` command construction
- `tests/test_endpoints_vllm.py`
  - verifies endpoint alias normalization

These tests are the fastest way to confirm which behavior is intentional.

## Quick trace to keep in mind

If you want one mental model for the whole repo, use this path:

1. start at `environments/hangman_agent/hangman_agent/env.py:load_environment`
2. follow into `environments/hangman_agent/hangman_agent/generator.py`
3. jump to `environments/hangman_agent/hangman_agent/game.py:initialize_game_state`
4. return to `environments/hangman_agent/hangman_agent/env.py:env_response`
5. drop into `environments/hangman_agent/hangman_agent/game.py:apply_guess` or `apply_invalid_action`
6. finish in `environments/hangman_agent/outputs/evals/...`

That covers almost everything that determines what the model sees, how the environment reacts, and what ends up in the saved results.
