# Hangman Prime/Verifiers Environment Plan

Last updated: 2026-03-10
Status: Implemented and smoke-evaluated locally on 2026-03-10. Install, unit tests, environment load, viewer smoke checks, and small OpenAI-backed `prime eval run` validations passed.

## Implementation Update

Completed in `environments/hangman_agent/`:

- multi-file `vf.MultiTurnEnv` implementation with deterministic parsing and transition logic
- local curated lexicon, deterministic task generation, difficulty presets, and partial-start support
- dense bounded reward with turn-level component tracking in trajectory extras and state history
- baseline policies (`etaoin` and lexicon-filtering) plus a Rich terminal viewer
- unit test coverage for parser, generator, game logic, baselines, environment loading, and viewer smoke execution

Local validation completed on 2026-03-10:

- `prime env install hangman_agent`
- `uv run python -m unittest discover -s environments/hangman_agent/tests -v`
- `uv run python -c "from hangman_agent import load_environment; ..."`
- `uv run python -m hangman_agent.viewer --difficulty easy --policy lexicon --index 0`
- `set -a; source .env >/dev/null 2>&1; prime eval run hangman_agent -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -n 6 -r 2 -a '{"difficulty":"easy"}' -C 'termination_reason,last_outcome,total_reward' -s --skip-upload -d`
- `set -a; source .env >/dev/null 2>&1; prime eval run hangman_agent -m gpt-5-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -n 6 -r 2 -a '{"difficulty":"hard"}' -C 'termination_reason,last_outcome,total_reward' -s --skip-upload -d`

Remaining validation:

- larger eval sweeps after deciding whether to push the environment to the Hub
- separate follow-up on why `prime eval run ... -e configs/endpoints.toml -m <endpoint_id>` fell back to the default Pinference base URL in this session

## 1. Goal

Build a self-contained Prime/Verifiers environment for a true multi-turn Hangman agent:

- one hidden English word per rollout
- structured assistant action format with optional `<think>...</think>` and exactly one required `<suggest>LETTER</suggest>`
- deterministic parsing, transitions, reward, and termination
- dense turn-level reward with clean debugging signals
- locally stored curated lexicon, deterministic task generation, difficulty presets, and baselines
- local Rich-based terminal viewer for manual inspection

The correct base class is a custom `vf.MultiTurnEnv`. I do not currently see a strong reason to use a tool environment.

## 2. Proposed Environment Identity

- Proposed environment id: `hangman_agent`
- Proposed path after approval: `environments/hangman_agent/`
- Proposed initialization command after approval:

```bash
prime env init hangman_agent --path ./environments --multi-file
```

## 3. Why `MultiTurnEnv`

This task is a native environment-driven dialogue loop rather than tool calling:

- the environment owns hidden state
- each assistant turn causes a state transition
- rewards should be attached to each turn and aggregated deterministically
- episode termination depends on custom game logic, not tool completion
- malformed outputs should be handled inside the game loop without exposing hidden metadata

`vf.MultiTurnEnv` gives the smallest correct abstraction:

- `setup_state(...)` for rollout initialization
- `env_response(...)` for turn parsing and state transitions
- `@vf.stop` methods for solved / unwinnable / exhausted cases
- trajectory-level extras for reward debugging

## 4. Planned Deliverables

After plan approval, implement:

- `environments/hangman_agent/pyproject.toml`
- `environments/hangman_agent/README.md`
- `environments/hangman_agent/hangman_agent/`
- `environments/hangman_agent/hangman_agent/__init__.py`
- `environments/hangman_agent/hangman_agent/env.py`
- `environments/hangman_agent/hangman_agent/game.py`
- `environments/hangman_agent/hangman_agent/parser.py`
- `environments/hangman_agent/hangman_agent/generator.py`
- `environments/hangman_agent/hangman_agent/baselines.py`
- `environments/hangman_agent/hangman_agent/viewer.py`
- `environments/hangman_agent/hangman_agent/data/lexicon.tsv`
- `environments/hangman_agent/tests/...`

I plan to keep the environment multi-file because the generator, parser, baselines, and viewer are substantial enough to justify separation.

## 5. Task Contract

### Assistant output

Each turn the assistant may include zero or more `<think>...</think>` blocks, but must also provide exactly one action:

```text
<suggest>LETTER</suggest>
```

Rules:

- `LETTER` must be exactly one ASCII alphabetic character `[A-Za-z]`
- parsing is case-insensitive
- exactly one `<suggest>` block is allowed
- content outside `<think>` and `<suggest>` is allowed but ignored for reward, except it can still make the output invalid if it creates multiple `<suggest>` blocks or a malformed suggestion payload
- reward never depends on the contents of `<think>`

### Environment response

Each environment turn returns a compact plain-text board. Proposed format:

```game-state
word: _ P P _ E
correct letters: E, P
incorrect letters: B, C, D, I, M
remaining attempts: 2
turns remaining: 5
last outcome: correct
last reward: +0.1800
```

Notes:

- `word` shows one token per character so repeated letters are obvious
- letters are always rendered uppercase and sorted
- `last outcome` is from a fixed enum such as `correct`, `wrong`, `repeat`, `invalid_format`, `invalid_letter`, `solved`, `unwinnable`, `lost`
- the board never reveals the hidden word until the game ends

## 6. Core State Model

Planned per-rollout state fields:

- `secret_word`
- `word_length`
- `correct_guesses`
- `incorrect_guesses`
- `all_guesses_in_order`
- `remaining_attempts`
- `turns_remaining`
- `revealed_pattern`
- `initial_hidden_positions`
- `initial_unrevealed_distinct_letters`
- `initial_attempts`
- `initial_turns`
- `last_outcome`
- `last_guess`
- `last_reward`
- `reward_history`
- `reward_component_history`
- `task_info`
- `oracle_upper_bound`
- `candidate_count`
- `candidate_examples` or small preview only if useful for debugging and excluded from prompt-visible content

Only the board text is shown to the model. Richer fields remain internal for scoring, tests, and debugging.

## 7. Parsing and Anti-Hack Rules

I plan to use a custom parser rather than the generic XML parser because the task needs stricter behavior:

- strip and ignore all `<think>...</think>` spans for action extraction
- extract all `<suggest>...</suggest>` blocks from the raw assistant text
- reject if the count is not exactly one
- reject if the suggestion content is not exactly one English letter after trimming
- normalize accepted letters to uppercase
- repeated guesses are validly parsed but transition to a deterministic `repeat` outcome

Malformed outputs will not leak extra information. The environment will respond with the normal board and generic outcome only.

Planned invalid categories:

- `invalid_format`: zero suggestions, multiple suggestions, broken tag structure, or malformed payload
- `invalid_letter`: suggestion not a single English alphabetic character
- `repeat`: valid letter already guessed before

## 8. Transition Logic

Per turn:

1. Decrement `turns_remaining` by 1 for any assistant response.
2. Parse the assistant action deterministically.
3. If invalid or repeated:
   - decrement `remaining_attempts` by 1
   - do not change revealed pattern
   - assign negative turn reward
4. If a new correct letter:
   - add it to `correct_guesses`
   - reveal all matching positions
   - do not decrement `remaining_attempts`
   - assign positive progress reward
5. If a new wrong letter:
   - add it to `incorrect_guesses`
   - decrement `remaining_attempts` by 1
   - assign negative turn reward
6. Recompute solved / unwinnable conditions.
7. If terminal, set `final_env_response` to the final board plus the hidden word.

This keeps the game deterministic and removes incentives to stall with malformed outputs.

## 9. Precise Termination Definition

The episode ends when any of the following is true:

1. `solved`
   - every position in the word has been revealed
2. `unwinnable_by_attempts`
   - `distinct_unrevealed_letters > remaining_attempts`
3. `unwinnable_by_turns`
   - `distinct_unrevealed_letters > turns_remaining`
4. `attempts_exhausted`
   - `remaining_attempts <= 0` and the word is not solved
5. `turns_exhausted`
   - `turns_remaining <= 0` and the word is not solved

Primary unwinnable definition:

```text
distinct_unrevealed_letters = number of distinct letters in the secret word that are not yet in correct_guesses
```

Using both attempts and turns is stricter than the default suggested in the prompt, but it is the cleanest deterministic definition once turns are an explicit resource.

## 10. Reward Design

I want the total rollout score to stay bounded in `[-1.0, 1.0]` up to minor floating-point error, with most positive mass coming from real progress.

Planned reward components:

- `correct_letter_bonus`
  - total budget across the rollout: `0.4`
  - on a new correct guess: `0.4 / initial_unrevealed_distinct_letters`
- `reveal_progress_bonus`
  - total budget across the rollout: `0.4`
  - on a new correct guess: `0.4 * new_positions_revealed / initial_hidden_positions`
- `wrong_guess_penalty`
  - on a new wrong guess: `-0.6 / initial_attempts`
- `repeat_guess_penalty`
  - on a repeated guess: `-0.8 / initial_attempts`
- `invalid_output_penalty`
  - on invalid format or invalid letter: `-1.0 / initial_attempts`
- `solve_efficiency_bonus`
  - when the board becomes fully revealed: `0.2 * (remaining_attempts / initial_attempts)`

Properties:

- perfect play yields at most `1.0`
- repeated letters are naturally rewarded because they reveal more positions
- wrong, repeated, and invalid actions cannot accidentally create positive reward
- `<think>` adds no reward and no extra state value
- brute-force alphabetical play should underperform informative guessing because progress is normalized while wasted attempts are penalized

Planned debugging visibility:

- store per-turn reward components in trajectory `extras`
- store aggregate metrics such as:
  - `num_correct_new_guesses`
  - `num_wrong_new_guesses`
  - `num_repeated_guesses`
  - `num_invalid_outputs`
  - `positions_revealed`
  - `solved`
  - `termination_reason`
  - `oracle_upper_bound`

## 11. Task Generation

### Lexicon

I plan to vendor a local curated lexicon file inside the environment package, likely TSV/CSV with at least:

- `word`
- `frequency_tier`
- optional precomputed metadata:
  - `word_length`
  - `distinct_letter_count`
  - `repeat_density`
  - `letter_mask`

Constraints:

- lowercase alphabetic words only
- no spaces, hyphens, apostrophes, or accents in the initial version
- no network access required at runtime

### Task shapes

Support both:

- fresh games
- partially played starting states

For partially played states, the generator will:

- choose a target word
- choose a consistent subset of pre-revealed correct letters
- choose a consistent set of pre-guessed wrong letters
- set attempts and turns so the game is still solvable
- reject states that are already solved, already unwinnable, or inconsistent with the lexicon

### Deterministic validation

Each generated task will be checked for:

- board consistency with the secret word
- disjoint correct and incorrect guess sets
- nonnegative attempts and turns
- solvable start state:
  - `distinct_unrevealed_letters <= remaining_attempts`
  - `distinct_unrevealed_letters <= turns_remaining`
- ambiguity metrics computed against the same lexicon used by the environment

### Ambiguity and diversity control

For a visible pattern plus incorrect letters, define:

- `candidate_set`
  - all lexicon words matching length, revealed positions, repeated-letter structure, and excluded wrong letters
- `ambiguity`
  - candidate set size

Generation knobs will let us target candidate-count bands so games are neither trivial nor degenerate.

## 12. Difficulty Controls

Lower-level knobs planned in `load_environment(...)`:

- `seed: int`
- `split: str`
- `num_examples: int`
- `word_length_min: int`
- `word_length_max: int`
- `frequency_tiers: tuple[str, ...]`
- `repeat_density_min: float`
- `repeat_density_max: float`
- `allowed_attempts_min: int`
- `allowed_attempts_max: int`
- `pre_revealed_letters_min: int`
- `pre_revealed_letters_max: int`
- `pre_wrong_letters_min: int`
- `pre_wrong_letters_max: int`
- `ambiguity_min: int`
- `ambiguity_max: int`
- `allow_partial_starts: bool`

Higher-level presets:

- `easy`
  - shorter, common words
  - more attempts
  - 1 to 2 pre-revealed letters
  - low ambiguity target
- `medium`
  - mid-length words
  - moderate attempts
  - optional partial starts
  - moderate ambiguity target
- `hard`
  - longer or more obscure words
  - fewer attempts
  - no or minimal pre-reveals
  - higher ambiguity target
  - more repeated-letter-heavy words allowed

Preset implementation plan:

- presets are thin wrappers over the lower-level knobs
- all sampling remains deterministic under a single seed
- the chosen preset and resolved knob values will be stored in task metadata

## 13. Oracle and Baselines

### Cheap oracle / upper bound

I plan to attach a deterministic optimistic upper bound to every task:

```text
oracle_upper_bound =
  remaining_correct_letter_budget
  + remaining_reveal_budget
  + max_possible_solve_bonus_from_current_state
```

This assumes all future guesses are novel and correct from the current state onward. It is cheap, deterministic, and valid as an upper bound even when not strictly attainable.

I may also include:

- `min_remaining_correct_turns`
- `candidate_count`
- `best_case_remaining_attempts_after_solve`

### Baseline policies

1. Simple baseline
   - fixed letter ordering, for example ETAOIN-style frequency ranking filtered by already guessed letters
   - no lexicon reasoning

2. Stronger baseline
   - maintain the lexicon candidate set from the visible board
   - score each unguessed letter by expected value:
     - probability the letter is present in the candidate set
     - expected number of positions revealed
     - candidate-set reduction / information gain
     - deterministic tie-breaks

The stronger baseline should give a useful difficulty sanity check without pretending to be globally optimal.

## 14. Viewer / Demo Tool

Planned standalone script or module entrypoint using Rich:

- generate a task by seed and difficulty
- print the current board
- step through agent outputs from stdin or a scripted baseline
- show guessed letters, attempts left, turns left, per-turn reward, and reward history
- reveal the hidden word and termination reason at the end

UX target:

- lightweight, text-only, closer in spirit to `prime eval tui` than a game UI

## 15. Testing Plan

Unit tests:

- parser accepts single valid suggestions and ignores `<think>` content
- parser rejects multiple suggestions, malformed tags, empty suggestions, non-letters, and repeats
- transition logic updates board state deterministically
- reward math is correct and bounded
- solved and unwinnable termination conditions trigger exactly when intended
- partial-start generation is consistent and solvable
- dataset generation is deterministic under seed control
- presets resolve to expected low-level parameters
- baseline policies behave deterministically on fixed tasks

Integration tests:

- `load_environment(...)` returns a valid environment
- small local rollout with scripted actions produces expected trajectory extras
- viewer smoke test for one generated task

## 16. Validation Workflow After Approval

Planned implementation loop:

1. Create the environment scaffold with Prime CLI.
2. Implement parser and pure game-state helpers first.
3. Add generator and tests.
4. Wrap the game loop in `vf.MultiTurnEnv`.
5. Add baselines and viewer.
6. Run install and smoke evals repeatedly during development.

Planned commands after approval:

```bash
prime env init hangman_agent --path ./environments --multi-file
prime env install hangman_agent
pytest environments/hangman_agent/tests -q
prime eval run hangman_agent -m qwen3-30b-i -n 6 -r 2 -a difficulty easy
prime eval run hangman_agent -m qwen3-30b-i -n 6 -r 2 -a difficulty medium
prime eval run hangman_agent -m qwen3-30b-t -n 6 -r 2 -a difficulty hard
```

Model choice rationale:

- `qwen3-30b-i` is a reasonable instruct smoke-test target from local endpoints
- `qwen3-30b-t` is a reasonable reasoning-model check if we want to verify `<think>` handling

If those endpoints are unavailable at runtime, I will switch to another alias already present in `configs/endpoints.toml`.

## 17. Main Tradeoffs To Review Before Implementation

1. Invalid and repeated guesses currently consume both a turn and an attempt.
   - I think this is the cleanest anti-stall rule, but it is slightly harsher than classic Hangman.
2. Unwinnable termination uses both attempts remaining and turns remaining.
   - I think this is correct once turns are observable, but it is stricter than the prompt's default attempt-only rule.
3. The initial oracle is an upper bound, not an exact optimal value function.
   - This satisfies the requirement cheaply and deterministically; exact dynamic programming can be added later if useful.
4. The first lexicon version will likely exclude punctuation and multi-word answers.
   - This keeps parsing and difficulty control much cleaner.

## 18. Requested Review

Please review:

- the reward budgets and penalty scaling
- whether invalid / repeated guesses should cost an attempt
- whether the stricter unwinnable rule using both attempts and turns is acceptable
- whether `hangman_agent` is the environment id you want

Once you approve the plan, I’ll scaffold the environment with the Prime CLI and start implementation.
