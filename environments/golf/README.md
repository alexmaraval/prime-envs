# golf

Multi-turn hex-grid pathfinding environment based on the GPTWorld Golf notebook.

Boards are generated procedurally from the built-in templates each time the dataset is created.

### Overview
- **Environment ID**: `golf`
- **Short description**: Collect the key and then reach the goal on a hex-style board while avoiding walls.
- **Tags**: `games`, `planning`, `tool-use`, `multiturn`, `pathfinding`

### Task
- **Type**: `tool use`
- **Agent interface**: Call the `play_move` tool once per turn with one action: `UR`, `R`, `DR`, `DL`, `L`, `UL`, or `Pickup`.
- **Board feedback**: After each turn, the environment sends an updated ASCII board so the agent can see the current state.
- **Rubric overview**: Deterministic rewards based on valid play, key pickup, and final success.

### Rules
- You start at `@`.
- You must collect `K` before `P` counts as a win.
- `W` are blocked cells.
- Invalid moves leave the board unchanged.
- `Pickup` only works if the player is currently on the key cell.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run golf
```

Pick a subset of boards:

```bash
prime eval run golf -a '{"difficulty":"easy","num_examples":4}'
prime eval run golf -a '{"difficulty":"hard","num_examples":4,"max_turns":96}'
```

### Copy-Paste Eval Command

```bash
set -a; source secrets.env >/dev/null 2>&1

uv run prime eval run golf \
  --model gpt-4o-mini \
  --api-base-url https://api.openai.com/v1 \
  --api-key-var OPENAI_API_KEY \
  --num-examples 25 \
  --rollouts-per-example 4 \
  --max-concurrent 4 \
  --sampling-args '{"max_tokens":16384,"temperature":0.7}' \
  --env-args '{"difficulty":"all","max_turns":30}' \
  --state-columns termination_reason,turn_count,picked_key,invalid_actions,total_reward \
  --save-results \
  --tui
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `difficulty` | `str` | `"all"` | One of `all`, `easy`, `medium`, `hard`. |
| `seed` | `int` | `0` | Seed used to generate board layouts deterministically. |
| `num_examples` | `int` | `4096` | Number of generated examples to build inside the environment when not overridden in `env_args`. This is intentionally large so eval-time `--num-examples` can draw fresh boards instead of being capped by a tiny internal dataset. |
| `eval_examples` | `int \| null` | `null` | Number of eval examples to prebuild for the eval split; defaults to `num_examples`. |
| `max_turns` | `int` | `128` | Maximum number of model turns before truncation. |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Sum of deterministic step rewards |
| `solved` | `1.0` iff the key was collected and the goal was reached |
| `picked_key` | `1.0` iff the key was collected |
| `invalid_actions` | Count of invalid turns |
| `turn_count` | Number of turns taken |
| `optimality_gap` | Extra turns above the shortest valid solution, or `-1` if unsolved |

### Local Helpers

- [`render_board`](/Users/alexandremaraval/Documents/Projects/prime-envs/environments/golf/golf_core.py) renders the board seen by the agent.
- [`shortest_solution`](/Users/alexandremaraval/Documents/Projects/prime-envs/environments/golf/golf_core.py) computes the optimal action sequence for a board.
- [`generate_boards.py`](/Users/alexandremaraval/Documents/Projects/prime-envs/environments/golf/generate_boards.py) procedurally generates solvable boards by re-sampling start, key, goal, and walls from the predefined templates.

### Procedural Board Generation

The environment now uses procedural generation by default. The helper script is still available if you want to preview the kinds of boards that will be produced while keeping the original board sizes and wall counts:

```bash
uv run python generate_boards.py --count 8 --seed 7
uv run python generate_boards.py --count 4 --difficulty hard
```
