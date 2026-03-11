# hangman_agent

Minimal multi-turn Hangman for Prime/Verifiers. Each rollout starts from a fully hidden English word, shows the model a compact text board, and requires exactly one guess per turn via an OpenAI-style `suggest_letter(letter: str)` tool call.

## Overview

- Environment id: `hangman_agent`
- Package path: `environments/hangman_agent/`
- Base class: custom `vf.ToolEnv`
- Local data: bundled TSV lexicon (`hangman_agent/data/lexicon.tsv`)
- Default dataset size: 128 train examples and 128 eval examples per resolved config
- Default difficulty: `easy` for development-focused iteration

## Task Contract

The assistant may emit optional reasoning text, but it must call `suggest_letter(letter: str)` exactly once per turn. The system prompt carries the rules; the first user message is only the board state.

Rules:

- `letter` must be exactly one ASCII alphabetic character
- missing tool calls, malformed tool arguments, and non-letter payloads are penalized deterministically
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

Turn reward is intentionally simple:

- fresh valid letter guess: `0.01`
- solved word: `1.0`
- repeated or invalid action: `0.0`

Per-turn reward components are attached to trajectory extras for lightweight debugging.

## Generation

Tasks are generated deterministically from a curated local English lexicon. Every game starts from a fully hidden word with no pre-filled correct or wrong letters.

Generation controls:

- word length bounds
- frequency tiers (`common`, `standard`, `obscure`)
- repeated-letter density bounds
- allowed-attempt range
- ambiguity band based on lexicon candidate count
- difficulty presets: `easy`, `medium`, `hard`
- deterministic `seed`

Each generated state is validated to ensure:

- the board is not already solved
- candidate count is computed against the filtered lexicon used for generation

## Quickstart

Install the local environment:

```bash
prime env install hangman_agent
```

Run a local smoke load:

```bash
uv run python -c "from hangman_agent import load_environment; env = load_environment(difficulty='easy', seed=1, num_examples=2); print(type(env).__name__)"
```

Run an eval once endpoint credentials are available:

```bash
prime eval run hangman_agent -m qwen3-30b-i -n 6 -r 2 -a '{"difficulty":"easy"}'
prime eval run hangman_agent -m qwen3-30b-t -n 6 -r 2 -a '{"difficulty":"hard"}'
```

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
  -a '{"difficulty":"easy"}'
```

`configs/endpoints.vllm.py` normalizes a bare host like `http://127.0.0.1:8000` to `/v1`, so the config-driven path works with the usual vLLM server address.

For an automatic local-server workflow, use the workspace helper instead of starting the server manually:

```bash
uv run python -m hangman.local_eval \
  --backend mlx-lm \
  --model mlx-community/Qwen3-1.7B-4bit \
  --difficulty easy
```

The helper:

- starts `mlx_lm.server` or `vllm serve` in the background
- waits for the local OpenAI-compatible `/v1/models` endpoint
- runs `prime eval run hangman_agent` against that local base URL
- stops the server when the eval finishes unless `--keep-server` is set

`mlx-lm` is the best-supported backend in this workspace because it is already included in the root project dependencies. `vllm` is also supported by the helper, but only if the `vllm` CLI is installed in the active environment.

## Inspecting Rollouts

Saved eval outputs already include the full prompt/completion conversation in `results.jsonl`, including assistant `tool_calls`, tool responses, and the board updates for each turn. The most useful state columns for quick inspection are `termination_reason`, `last_outcome`, and `total_reward`.

## Environment Arguments

`load_environment(...)` accepts these knobs:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `difficulty` | `str` | `"easy"` | High-level preset for generation constraints. |
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
