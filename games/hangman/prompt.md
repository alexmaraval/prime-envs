Make a Prime/Verifiers environment for a Hangman agent.

This should be a true multi-turn environment. Use the smallest correct base class in the framework; I expect a custom `MultiTurnEnv` rather than a tool environment unless you find a strong reason otherwise.

## Core interaction

- Each rollout hides one English word.
- On each turn, the agent may optionally think inside `<think>...</think>` blocks.
- It must then provide exactly one guess in the form `<suggest>LETTER</suggest>`, where `LETTER` is a single English alphabet character.
- Parse guesses case-insensitively.
- Handle invalid format, multiple suggestions, non-letter suggestions, or repeated guesses deterministically and penalize them cleanly.
- No reward should depend on the content of `<think>`.

## Observable state

The observable state should be a simple text-based hangman board. Design it cleanly and keep it compact and unambiguous. A reasonable example is:

```game-state
word: _ P P _ E
correct letters: P, E
incorrect letters: B, C, D, I, M
remaining attempts: 2
turns remaining: 5
```

You may choose a better presentation, but keep it plain text and easy for models to parse.

## Episode termination

- End when the word is fully revealed.
- Also end when the puzzle is no longer winnable from the current state.
- Use a precise deterministic definition and document it clearly. A good default is: terminate when the number of distinct unrevealed letters exceeds the remaining attempts.

## Scoring

- The agent should receive reward at each turn, not only at the end.
- Design a clean dense reward that encourages good-faith play:
- positive reward for new correct guesses and revealing progress,
- negative reward for wrong, repeated, or invalid guesses,
- an efficiency bonus for solving with attempts left,
- no reward for formatting tricks, stalling, or verbose reasoning.
- Keep the final rollout score bounded and deterministic.
- The best strategy should be to infer the hidden word and choose informative letters, not brute-force the alphabet blindly.
- Track turn-level reward components in state or metrics so debugging is easy.

## Task generation

- Programmatically generate tasks from a curated English word list stored locally inside the environment.
- Support both fresh games and partially played starting states if that improves difficulty and diversity.
- Deterministically validate that generated states are consistent and solvable.
- Provide a cheap oracle or upper-bound utility for each task so we can estimate the best attainable score from the hidden word and initial state.
- Prefer tasks where random guessing performs badly.
- Experiment with sampling so tasks are not trivial: avoid states with too many equally good guesses or huge numbers of valid paths unless that is an intentional easy-mode behavior.

## Difficulty control

- Provide fine-grained generation knobs for:
- word length,
- word frequency or obscurity,
- repeated-letter density,
- number of allowed attempts,
- number of pre-revealed letters,
- number of pre-guessed wrong letters,
- ambiguity of the current pattern relative to the lexicon.
- Also provide higher-level presets like `easy`, `medium`, and `hard` that map cleanly onto those lower-level knobs.
- Keep generation deterministic under seed control.

## Robustness and anti-hacks

- Prevent reward hacks and parser tricks.
- Invalid or malformed outputs should not reveal extra information.
- Repeated guesses should never accidentally create positive reward.
- Do not leak metadata that makes the answer trivial.
- The environment should be fully self-contained and require no network calls at runtime.

## Implementation expectations

- Create the environment with the Prime CLI flow (`prime env init`, then implement it cleanly under `environments/<env_name>/`).
- Expose `load_environment(...) -> vf.Environment`.
- Include a strong README with the task contract, generation controls, reward design, and example trajectories.
- Create a detailed `PLAN.md` with the design doc and testing plan before major implementation, then revise `PLAN.md` after major milestones to reflect what was completed and what remains.
- Add unit tests for parsing, transition logic, reward calculation, termination, dataset determinism, and difficulty presets.
- Add at least one simple scripted baseline policy and one stronger lexicon-filtering baseline to sanity-check difficulty.

## Developer tooling

- Include a standalone terminal viewer or playable demo using Rich so a generated game can be inspected in the terminal.
- It should show the current board, guessed letters, attempts left, reward history, and the final hidden word when the game ends.
- Keep the UX similar in spirit to `prime eval tui`, but lightweight.

## Validation

- Run `prime env install <env_name>`.
- Run basic smoke evals throughout development, not only at the end.
- At minimum, run small `prime eval run` checks on a few samples with one instruct model and, if useful, one reasoning model from `configs/endpoints.toml`.
- Make sure the evals exercise several rollouts and multiple difficulty settings.
- You are welcome to use `PRIME_API_KEY` from my environment for inference tests.

Let me know when you are happy with the implementation, and include the exact install and eval commands you used plus any unresolved design tradeoffs.
