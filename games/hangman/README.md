# Hangman Prime Environment

This workspace contains a local Prime/Verifiers environment for multi-turn Hangman at `environments/hangman_agent/`.

## Quickstart

Install the environment:

```bash
prime env install hangman_agent
```

Run a small local smoke eval against a hosted endpoint:

```bash
prime eval run hangman_agent \
  -m gpt-4.1-mini \
  -n 4 \
  -r 1 \
  -a '{"difficulty":"easy"}'
```

## Local MLX Eval

The recommended local-model workflow in this workspace is the auto-start helper:

```bash
LOCAL_LLM_API_KEY=dummy \
uv run python -m hangman.local_eval \
  --backend mlx-lm \
  --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --difficulty-mix '[0.3, 0.4, 0.3]' \
  --num-examples 4 \
  --rollouts-per-example 1 \
  --max-concurrent 1
```

That command:

- starts `mlx_lm.server`
- waits for the OpenAI-compatible `/v1/models` endpoint
- runs `prime eval run hangman_agent` against the local server
- shuts the server down when the eval completes

`vllm` is also supported:

```bash
uv run python -m hangman.local_eval \
  --backend vllm \
  --model Qwen/Qwen3-8B-Instruct
```

This requires a working `vllm` CLI in the active environment.

If you prefer a config-driven eval against a manually started MLX server:

```bash
export LOCAL_MLX_MODEL="mlx-community/Qwen3.5-0.8B-MLX-4bit"
export LOCAL_MLX_BASE_URL="http://127.0.0.1:8080"
export LOCAL_LLM_API_KEY="dummy"

prime eval run configs/eval/hangman-mlx.toml
```

`configs/endpoints.mlx.py` normalizes a bare host like `127.0.0.1:8080` to `/v1`,
so either `127.0.0.1:8080` or `http://127.0.0.1:8080/v1` works.

Note: with Prime CLI `0.5.44`, `prime eval run <config.toml>` may still do its
hosted inference billing preflight before applying local endpoint aliases from
the eval config. If that happens, use the explicit `--api-base-url` command in
the manual flow below, or the `uv run python -m hangman.local_eval ...` helper.

## Manual Local Server Flow

If you want to host the model yourself first:

```bash
./.venv/bin/mlx_lm.server \
  --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --host 127.0.0.1 \
  --port 8080
```

Then in another terminal:

```bash
curl http://127.0.0.1:8080/v1/models

LOCAL_LLM_API_KEY=dummy prime eval run hangman_agent \
  --env-dir-path environments \
  --model mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --api-base-url http://127.0.0.1:8080/v1 \
  --api-key-var LOCAL_LLM_API_KEY \
  --num-examples 4 \
  --rollouts-per-example 1 \
  --max-concurrent 1 \
  --env-args '{"difficulty_mix":[0.3,0.4,0.3]}' \
  --state-columns termination_reason,last_outcome,total_reward,rollout_trace \
  --save-results \
  --skip-upload \
  --debug
```

## Notes

- The infrastructure path for local MLX evaluation is verified in this workspace.
- Small local models may still perform poorly and miss the required `suggest_letter(letter=...)` tool call.
- Full environment details live in `environments/hangman_agent/README.md`.
