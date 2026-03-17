# Golf Environment Plan

## Game Description

This environment is based on the "GPTWorld Golf" notebook from `srush/GPTWorld-Challenge`, adapted into a Prime-compatible task.

The game is a deterministic pathfinding puzzle played on a hex-style board embedded in a 2D array. The board contains:

- `@`: the player start position
- `K`: the key
- `P`: the goal / flag
- `W`: walls
- `.`: playable empty cells
- `" "`: non-playable filler cells used to render the hex layout

The player must:

1. Navigate from the start to the key.
2. Execute the `Pickup` action while standing on the key cell.
3. Navigate to the goal cell.

The puzzle is only solved if the key has been picked up and the player ends on the goal.

## Exact Mechanics From The Notebook

- Board coordinates are `(row, col)`.
- Legal movement actions are:
  - `UR` -> `(-1, +1)`
  - `R` -> `(0, +2)`
  - `DR` -> `(+1, +1)`
  - `DL` -> `(+1, -1)`
  - `L` -> `(0, -2)`
  - `UL` -> `(-1, -1)`
  - `Pickup` -> `(0, 0)`
- Moves that go out of bounds do not change the player position in the notebook implementation.
- Moves into walls also do not change the player position.
- `Pickup` only has an effect when the player is exactly on the key cell; then the key is removed from the board.
- The notebook uses fixed board instances such as:
  - Easy: `boundary=(3, 3)`, `init=(0, 0)`, `key=(1, 1)`, `flag=(2, 2)`, `walls=[]`
  - Medium / Hard / Evil larger boards with hand-authored wall sets.

## Proposed Prime Task Format

I plan to preserve the game rules exactly, but simplify the model interface relative to the notebook.

Instead of asking the model to write Python code that calls `move(...)`, the environment will ask the model to return an action plan directly as a plain-text sequence. For example:

```text
DR
Pickup
DR
```

Why this translation:

- It preserves the actual puzzle mechanics.
- It is much easier to score deterministically.
- It avoids code execution inside the grading loop.
- It is a better fit for Prime eval and RL workflows.

## Planned Environment Shape

I plan to implement this as a single-turn environment:

- Input to model:
  - Natural-language description of the rules
  - ASCII rendering of the current board
  - Explicit action vocabulary
  - Instruction to output only the action sequence
- Model output:
  - One action per line, or a comma/newline separated list
- Scoring:
  - Parse the actions
  - Simulate them with notebook-faithful transition rules
  - Reward success only when the key is picked up and the goal is reached

## Reward Plan

Primary metric:

- `success`: `1.0` if the final simulated state has `key_pos is None` and `player_pos == flag_pos`, else `0.0`

Secondary diagnostics:

- `picked_key`: whether the key was successfully picked up
- `reached_goal`: whether the player ever reached the goal
- `valid_actions_rate`: fraction of parsed actions that are in the allowed action set
- `path_length`: number of parsed actions
- `optimality_gap`: parsed path length minus shortest valid solution length, when solved

Default reward:

- Start with exact success reward only.
- Keep richer diagnostics for analysis.
- Optionally add a small shaped reward later if we want this environment to be more RL-friendly.

## Data Plan

I plan to start with a small curated set of hand-authored boards:

1. Recreate the notebook's named boards: easy, medium, hard, evil.
2. Add several additional boards in the same style.
3. Compute ground-truth shortest solutions with a deterministic solver.

Each example record should contain:

- board size
- start position
- key position
- flag position
- wall positions
- shortest solution
- shortest solution length

## Implementation Plan

1. Recreate the board simulator in `golf.py` using the notebook's exact transition rules.
2. Add a deterministic shortest-path solver over `(player_pos, key_collected)` state.
3. Define a compact dataset of notebook boards plus a few extra boards.
4. Write prompt rendering that explains the hex movement clearly and prints the board in text form.
5. Parse model outputs into action sequences robustly.
6. Score by simulating the parsed actions and comparing against the terminal success condition.
7. Expose the environment through `load_environment(...)`.
8. Update `README.md` and package metadata after the environment is implemented.

## Important Assumptions

- I will treat this as a path-planning environment, not a code-generation environment.
- I will preserve the notebook's move semantics exactly, including "invalid moves leave the player in place".
- I will use deterministic scoring only.
- I will likely start with a small in-file dataset unless a separate JSON dataset becomes cleaner during implementation.

## Open Point To Confirm Before Coding

The main design choice is the interface:

- Planned: model outputs an action sequence directly.
- Alternative: model outputs Python code in the style of the notebook.

My recommendation is the direct action-sequence interface, because it preserves the game while producing a much cleaner and more robust Prime environment.
