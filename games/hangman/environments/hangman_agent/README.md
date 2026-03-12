# hangman_agent

Minimal multi-turn Hangman for Prime/Verifiers. Each rollout starts from a fully hidden English word, shows the model a compact text board, and requires exactly one guess per turn via an OpenAI-style `suggest_letter(letter: str)` tool call.

## Overview

- Environment id: `hangman_agent`
- Package path: `environments/hangman_agent/`
- Base class: custom `vf.ToolEnv`
- Local data: bundled 10000-word TSV lexicon tagged with `easy` / `medium` / `hard` (`hangman_agent/data/lexicon.tsv`), rebuilt with `scripts/build_lexicon.py`
- Default dataset size: 128 train examples and 128 eval examples per resolved config
- Default difficulty: `easy` for development-focused iteration
- Package version: `0.2.9`

Rebuild the lexicon with:

```bash
uv run --with wordfreq python environments/hangman_agent/scripts/build_lexicon.py
```

## Task Contract

The assistant may emit optional reasoning text, but it must call `suggest_letter(letter: str)` exactly once per turn. The system prompt asks for minimal reasoning and the first user message is only the board state.

Rules:

- `letter` must be exactly one ASCII alphabetic character
- missing tool calls, malformed tool arguments, and non-letter payloads receive a small deterministic penalty
- invalid tool usage repeats the same board with explicit feedback and does not change the board
- repeated guesses are accepted but count as wasted turns; the feedback says whether that letter was already known correct or wrong
- reward never depends on assistant free-text content

The model sees a plain-text board like:

```text
word: _ P P _ E
wrong letters: B, C, D, I, M
hanged: 38%
turns remaining: 5
```

The hidden word is only revealed after termination.

## Termination And Reward

The episode ends on the first of these conditions:

- the word is fully revealed
- the hang reaches 100%
- too many invalid tool actions occur in one rollout

Turn reward has four components:

- fresh valid letter guess: `0.0` by default
- progress reward: fraction of initially hidden positions revealed by the current guess
- solved reward: `1.0` if the word is fully uncovered at termination, else `0.0`
- invalid action penalty: `-0.05` for a malformed or missing tool call on a turn

That means solved rollouts can still reach `2.0` from `1.0` progress plus `1.0` solved reward, while repeated guesses give no bonus and invalid tool usage incurs the small penalty above.

Per-turn reward components are attached to trajectory extras for lightweight debugging.

## Generation

Tasks are generated deterministically from a curated local English lexicon with an explicit difficulty tag on each word. Every game starts from a fully hidden word with no pre-filled correct or wrong letters.

Generation controls:

- difficulty tags (`easy`, `medium`, `hard`)
- allowed-attempt range
- optional `difficulty_mix` weights in `easy,medium,hard` order for mixed-difficulty datasets
- deterministic `seed`

For mixed datasets, the generator allocates exact per-difficulty counts from the requested mix, samples directly from the tagged lexicon pools, and shuffles the merged records.

Each generated state is validated to ensure the board is not already solved.

## Quickstart

Install the local environment:

```bash
prime env install hangman_agent
```

Run a local smoke load:

```bash
uv run python -c "from hangman_agent import load_environment; env = load_environment(difficulty='easy', seed=1, num_examples=2); print(type(env).__name__)"
```

## Running Evals

### Hosted eval with `gpt-4.1-mini`

From the workspace root, load your keys and run a small smoke eval against OpenAI:

```bash
set -a; source secrets.env >/dev/null 2>&1

prime eval run hangman_agent \
  -m gpt-4.1-mini \
  -b https://api.openai.com/v1 \
  -k OPENAI_API_KEY \
  -n 6 \
  -r 2 \
  -a '{"difficulty_mix":[0.3,0.4,0.3]}' \
  -C 'termination_reason,last_outcome,total_reward,rollout_trace' \
  -s \
  --skip-upload \
  -d
```

This uses the mixed-difficulty generator added in `0.2.x` and saves full rollout traces locally.

If you prefer a single preset, swap the env args for something like `-a '{"difficulty":"easy"}'`.

### Eval with a locally hosted model

The simplest local path in this workspace is the helper that starts `mlx_lm.server` for you, waits for `/v1/models`, runs `prime eval run`, and then shuts the server down:

```bash
LOCAL_LLM_API_KEY=dummy \
uv run python -m hangman.local_eval \
  --backend mlx-lm \
  --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --difficulty-mix '[0.3, 0.4, 0.3]' \
  --num-examples 6 \
  --rollouts-per-example 2 \
  --max-concurrent 1
```

On the first run, `mlx-lm` may need time to download model weights before the helper sees `http://127.0.0.1:<port>/v1/models`.

If you want to host the server yourself first, use either an OpenAI-compatible local server or the workspace config-driven vLLM path.

Run against a local vLLM server with the dedicated workspace config:

```bash
export LOCAL_VLLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
export LOCAL_VLLM_BASE_URL="http://127.0.0.1:8000"
# Optional if your server enforces auth.
export LOCAL_VLLM_API_KEY="token"

prime eval run configs/eval/hangman-vllm.toml
```

For a one-off local run without the config file, pass the OpenAI-compatible base URL explicitly:

```bash
prime eval run hangman_agent \
  -m "$LOCAL_VLLM_MODEL" \
  -b "http://127.0.0.1:8000/v1" \
  -k LOCAL_VLLM_API_KEY \
  -n 6 \
  -r 2 \
  -a '{"difficulty_mix":[0.3,0.4,0.3]}'
```

`configs/endpoints.vllm.py` normalizes a bare host like `http://127.0.0.1:8000` to `/v1`, so the config-driven path works with the usual vLLM server address.

For a one-off manual local run without the config file, pass the base URL explicitly:

```bash
LOCAL_LLM_API_KEY=dummy prime eval run hangman_agent \
  --env-dir-path environments \
  --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --api-base-url http://127.0.0.1:8080/v1 \
  --api-key-var LOCAL_LLM_API_KEY \
  --num-examples 25 \
  --rollouts-per-example 4 \
  --max-concurrent 4 \
  --env-args '{"difficulty_mix":[0.3,0.4,0.3]}' \
  --state-columns termination_reason,last_outcome,total_reward,rollout_trace \
  --save-results \
  --tui \
  --skip-upload
```

`mlx-lm` is the best-supported backend in this workspace because it is already included in the root project dependencies. `vllm` is also supported by the helper, but only if the `vllm` CLI is installed in the active environment.

### Full eval to Hub
```bash
set -a; source secrets.env >/dev/null 2>&1

uv run prime eval run hangman_agent \
  --model gpt-4.1-mini \
  --api-base-url https://api.openai.com/v1 \
  --api-key-var OPENAI_API_KEY \
  --num-examples 25 \
  --rollouts-per-example 4 \
  --env-args '{"difficulty_mix":[0.3,0.4,0.3]}' \
  --state-columns 'termination_reason,last_outcome,total_reward,rollout_trace' \
  --save-results \
  --tui

RUN_DIR="$(ls -td environments/hangman_agent/outputs/evals/hangman_agent--gpt-4.1-mini/* | head -n1)"
prime eval push "$RUN_DIR"

# push env again to hub
# prime env push --path environments/hangman_agent --visibility PRIVATE
```

## Inspecting Rollouts

Saved eval outputs include the full prompt/completion conversation in `results.jsonl`, including assistant `tool_calls`, tool responses, and the board updates for each turn. The most useful state columns are `termination_reason`, `last_outcome`, `total_reward`, and `rollout_trace`.

After a saved eval, inspect the newest output directory:

```bash
find environments/hangman_agent/outputs/evals \( -name results.jsonl -o -name metadata.json \) | tail
```

Open the rollout file directly:

```bash
less environments/hangman_agent/outputs/evals/.../results.jsonl
```

Or extract a compact summary with `jq`:

```bash
jq '{reward, termination_reason, last_outcome, total_reward}' environments/hangman_agent/outputs/evals/.../results.jsonl
```

If you want the richest traces, make sure your eval command includes:

```bash
-C 'termination_reason,last_outcome,total_reward,rollout_trace' -s
```

That adds rollout-level state fields to the saved records so you can inspect how the board evolved turn by turn.

## Environment Arguments

`load_environment(...)` accepts these knobs:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `difficulty` | `str` | `"easy"` | High-level preset for generation constraints. |
| `difficulty_mix` | `Sequence[float] \| str \| None` | `None` | Optional easy/medium/hard mixture weights. A value like `[0.3, 0.4, 0.3]` yields a dataset with that proportional split, normalized if needed. This cannot be combined with manual generation-range overrides. |
| `seed` | `int` | `0` | Base seed for deterministic train/eval task generation. |
| `num_examples` | `int` | `128` | Number of examples to generate per split. |
| `word_length_min` / `word_length_max` | `int \| None` | preset | Override word-length range. |
| `frequency_tiers` | `Sequence[str] \| str \| None` | preset | Allowed lexicon tiers. |
| `repeat_density_min` / `repeat_density_max` | `float \| None` | preset | Override repeated-letter density range. |
| `allowed_attempts_min` / `allowed_attempts_max` | `int \| None` | preset | Override the wrong-guess budget before the hang reaches 100%. |
| `ambiguity_min` / `ambiguity_max` | `int \| None` | preset | Target candidate-count band for the visible board. |

## Metrics

The rubric emits:

| Metric | Meaning |
| ------ | ------- |
| `reward` | Total rollout reward, computed from the accumulated turn rewards. |
| `solved_metric` | `1.0` when the puzzle is solved, else `0.0`. |
| `invalid_outputs_metric` | Count of invalid or missing tool actions. |
| `repeated_guesses_metric` | Count of repeated guesses. |
| `positions_revealed_metric` | Total number of positions revealed over the rollout. |

## Local Validation

Completed on 2026-03-10:

```bash
prime env install hangman_agent
uv run python -m unittest discover -s environments/hangman_agent/tests -v
set -a; source secrets.env >/dev/null 2>&1; prime eval run hangman_agent -m gpt-4.1-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -n 6 -r 2 -a '{"difficulty":"easy"}' -C 'termination_reason,last_outcome,total_reward' -s --skip-upload -d
set -a; source secrets.env >/dev/null 2>&1; prime eval run hangman_agent -m gpt-5-mini -b https://api.openai.com/v1 -k OPENAI_API_KEY -n 6 -r 2 -a '{"difficulty":"hard"}' -C 'termination_reason,last_outcome,total_reward' -s --skip-upload -d
```

Note: `prime eval run ... -e configs/endpoints.toml -m <endpoint_id>` did not resolve the OpenAI endpoint aliases in this session and instead fell back to the default Pinference base URL. The successful smoke evals therefore passed `-b https://api.openai.com/v1 -k OPENAI_API_KEY` explicitly.

The local vLLM helper added in `configs/endpoints.vllm.py` was verified by loading the endpoint registry directly, but no local vLLM server was available in this session to run a live eval against `http://127.0.0.1` or another self-hosted endpoint.
