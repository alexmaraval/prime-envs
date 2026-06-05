# self-compaction

`self-compaction` is a multi-turn coding-agent RL environment. It adapts the
Prime Sandbox and hidden-test harness from `primeintellect/mini-swe-agent-plus`
and adds a mandatory `compact` tool. When the model calls `compact`, the visible
interaction history after the original task prompt is replaced by the model's
own summary; the full transcript is still retained in state for audit and
visualization.

## Task

The model is given a software issue and a repository sandbox. It can inspect and
edit files, run commands, compact its trajectory, and submit the final patch for
hidden-test scoring.

Visible tools:

- `execute_bash(command: str)`
- `search_files(pattern: str)`
- `edit_via_str_replace(path: str, old_str: str, new_str: str)`
- `compact(summary: str)`
- `submit()`

`submit()` is rejected until `compact()` has succeeded at least once.
Consecutive `compact()` calls are rejected unless the agent does intervening
work, so long rollouts can compact multiple times without collapsing into a
compaction loop.

## Datasets

The default dataset is `R2E-Gym/R2E-Gym-Subset` on the `train` split. The
environment keeps the Mini SWE-compatible harness switch for R2E-Gym and
SWE-bench-style datasets, including:

- `R2E-Gym/R2E-Gym-Subset`
- `R2E-Gym/R2E-Gym-Lite`
- `PrimeIntellect/SWE-Bench-Verified-Quick`
- `SWE-bench/SWE-bench_Verified`

For R2E-Gym tasks, hidden tests are archived and removed from the sandbox during
the rollout, then restored only in `post_rollout` for scoring.

## Quickstart

Install the local environment, then run a one-example smoke eval:

```bash
cd environments/self_compaction
prime env install self-compaction --path ..

prime eval run self-compaction \
  --provider openai \
  --model gpt-4.1-mini \
  --api-key-var OPENAI_API_KEY \
  --num-examples 1 \
  --rollouts-per-example 1 \
  --max-concurrent 1 \
  --max-tokens 4096 \
  --temperature 0.2 \
  --env-args '{"num_examples":1,"max_turns":20,"simple_only":true,"test_timeout":300,"rollout_timeout_seconds":900}' \
  --state-columns submitted,submitted_after_compact,compaction_count,turns_before_first_compact,total_tool_calls,execute_bash_calls,search_files_calls,edit_via_str_replace_calls,compact_calls,submit_calls,command_timeout_count,rollout_duration_seconds,patch_broke_tests,sandbox_oom,sandbox_timeout,sandbox_image_pull_error \
  --save-results \
  --skip-upload \
  --disable-tui
```

Use `--skip-upload` while iterating locally. Drop it when you want the finished
run to be uploaded to the Prime platform.

## Publishing To Env Hub

Run the local gates before pushing:

```bash
cd environments/self_compaction
uv build
uv run pytest tests/test_self_compaction.py tests/test_visualize_rollouts.py -q
prime env install self-compaction --path .. --no-upgrade
```

Then push from the environment directory. Choose `PRIVATE` while iterating and
switch to `PUBLIC` only when you want the environment discoverable:

```bash
prime env push --visibility PRIVATE
# or, when pushing to a specific user/team namespace:
prime env push --owner <owner> --visibility PRIVATE
```

After the push completes, use the returned Hub id everywhere large or hosted
runs refer to the environment:

```bash
prime env status <owner>/self-compaction
prime env info <owner>/self-compaction
```

## Eval Command Modes

### Smoke Mode

Use this for fast environment validation after code or prompt changes. It keeps
the loaded dataset tiny, runs one rollout, and uses short rollout/test timeouts.

```bash
prime eval run self-compaction \
  -p openai \
  -m gpt-4.1-mini \
  -k OPENAI_API_KEY \
  -n 1 \
  -r 1 \
  -c 1 \
  -t 4096 \
  -T 0.2 \
  -a '{"num_examples":1,"max_turns":20,"simple_only":true,"test_timeout":300,"rollout_timeout_seconds":900}' \
  -C submitted,submitted_after_compact,compaction_count,turns_before_first_compact,total_tool_calls,execute_bash_calls,search_files_calls,edit_via_str_replace_calls,compact_calls,submit_calls,command_timeout_count,rollout_duration_seconds,patch_broke_tests \
  -s \
  --skip-upload \
  -d
```

### Compaction Debug Mode

Use this when you need to inspect whether the agent actually compacted useful
state. It saves the full audit transcript, the post-compaction visible transcript,
and every compaction summary. These columns can be large, so keep `-n`, `-r`, and
`-c` small.

```bash
prime eval run self-compaction \
  -p openai \
  -m gpt-4.1-mini \
  -k OPENAI_API_KEY \
  -n 2 \
  -r 1 \
  -c 1 \
  -t 8192 \
  -T 0.2 \
  -a '{"num_examples":2,"max_turns":60,"min_compactions":1,"max_summary_chars":6000}' \
  -C full_rollout_messages,agent_visible_messages,compaction_summaries,last_compaction_summary,submitted,submitted_after_compact,patch_broke_tests,patch_broke_tests_reason \
  -o outputs/evals/debug \
  -s \
  --skip-upload \
  -d
```

### Small Benchmark Mode

Use this after smoke and debug runs pass. It evaluates more examples, uses
multiple rollouts per task, and allows longer coding trajectories. Keep
`--max-concurrent` conservative because each rollout creates a sandbox.

```bash
prime eval run self-compaction \
  -p openai \
  -m gpt-4.1-mini \
  -k OPENAI_API_KEY \
  -n 10 \
  -r 2 \
  -c 2 \
  -t 12000 \
  -T 0.4 \
  -S '{"top_p":0.95}' \
  -a '{"num_examples":10,"max_turns":120,"simple_only":true,"sandbox_command_timeout":120,"test_timeout":900,"rollout_timeout_seconds":3600}' \
  -C submitted,submitted_after_compact,compaction_count,turns_before_first_compact,total_tool_calls,execute_bash_calls,search_files_calls,edit_via_str_replace_calls,compact_calls,submit_calls,command_timeout_count,rollout_duration_seconds,patch_broke_tests,sandbox_oom,sandbox_timeout,sandbox_image_pull_error \
  -s \
  --skip-upload \
  --abbreviated-summary
```

### Alternate Dataset Mode

Pass `dataset_name` and `split` through `--env-args` to switch harnesses. Dataset
names starting with `R2E-Gym/` use the R2E harness; SWE-bench-style datasets use
the SWE-bench harness.

```bash
prime eval run self-compaction \
  -p openai \
  -m gpt-4.1-mini \
  -k OPENAI_API_KEY \
  -n 3 \
  -r 1 \
  -c 1 \
  -t 12000 \
  -T 0.2 \
  -a '{"dataset_name":"PrimeIntellect/SWE-Bench-Verified-Quick","split":"test","simple_only":false,"max_turns":160,"test_timeout":1200,"skip_swebench_install":true}' \
  -C submitted,submitted_after_compact,compaction_count,turns_before_first_compact,total_tool_calls,execute_bash_calls,search_files_calls,edit_via_str_replace_calls,compact_calls,submit_calls,rollout_duration_seconds,patch_broke_tests,patch_broke_tests_reason \
  -s \
  --skip-upload \
  -d
```

### Hosted Mode

For hosted evals, push the environment first and run against the Hub slug. Include
the sandbox/instance access flags because this environment creates Prime
Sandboxes and inspects sandbox instances during rollouts.

```bash
cd environments/self_compaction
prime env push --visibility PRIVATE

prime eval run <owner>/self-compaction \
  --hosted \
  --allow-sandbox-access \
  --allow-instances-access \
  -p openai \
  -m gpt-4.1-mini \
  -k OPENAI_API_KEY \
  -n 20 \
  -r 1 \
  -c 4 \
  -t 12000 \
  -T 0.4 \
  -a '{"max_turns":120,"simple_only":true,"sandbox_command_timeout":120,"test_timeout":900}' \
  -C submitted,submitted_after_compact,compaction_count,turns_before_first_compact,total_tool_calls,execute_bash_calls,search_files_calls,edit_via_str_replace_calls,compact_calls,submit_calls,command_timeout_count,rollout_duration_seconds,patch_broke_tests \
  -s \
  --follow
```

## Training

Use Hosted Training on the Prime Intellect platform for the first RL run. This is
the CPU-launch path and is the right default before setting up self-managed GPU
infrastructure.

1. Push the environment and wait for a healthy Hub action:

```bash
cd environments/self_compaction
prime env push --visibility PRIVATE
prime env status <owner>/self-compaction
```

2. Copy the starter config and replace `your-owner/self-compaction` with the Hub
id returned by the push:

```bash
cp configs/rl/self-compaction-hosted-smoke.toml /tmp/self-compaction-train.toml
$EDITOR /tmp/self-compaction-train.toml
```

3. Launch Hosted Training:

```bash
prime train /tmp/self-compaction-train.toml
```

The starter config keeps rollout pressure low:

- `batch_size = 32`
- `rollouts_per_example = 8`
- `max_inflight_rollouts = 16`
- `max_async_level = 1`
- `max_turns = 120`
- `num_examples = 200`

The reward is binary and sparse, so the config enables conservative online
difficulty filtering. Hosted Training accepts only one of
`max_inflight_rollouts` and `oversampling_factor`; the smoke config keeps
`max_inflight_rollouts = 16` because sandbox fanout is the first thing to
control. If reward diversity looks healthy and you want oversampling later,
remove `max_inflight_rollouts` before adding `oversampling_factor = 2.0`.
Inspect rollouts and reward distributions before scaling:

```bash
prime train list
prime train get <run-id>
prime train progress <run-id>
prime train distributions <run-id>
prime train rollouts <run-id>
prime train checkpoints <run-id>
```

Scale in this order once samples look healthy: increase `num_examples`, then
`max_turns`, then `batch_size`/`max_inflight_rollouts`. Keep `batch_size`
divisible by `rollouts_per_example`, and keep sandbox resource limits
conservative until sandbox OOM/timeout metrics are quiet.

Self-managed `prime-rl` is the power-user path for local GPU infrastructure:

```bash
prime lab setup --prime-rl
```

After setup, generate or adapt a separate `prime-rl` config with the same
`[[env]]` args and conservative rollout settings. Local `prime-rl` runs require
GPU access; Hosted Training is the recommended first launch path for this
sandbox-heavy environment.

## Eval Parameter Reference

Prime eval flags control the evaluation runner:

| Flag | Purpose | Self-compaction notes |
| --- | --- | --- |
| `-m`, `--model` | Model or endpoint id to evaluate. | Use an instruct model for smoke tests; compare reasoning models only after the environment is stable. |
| `-p`, `--provider` | Provider shorthand, such as `openai`, `prime`, `local`, or `vllm`. | Explicit `--api-base-url` and `--api-key-var` override the provider defaults. |
| `-e`, `--endpoints-path` | TOML endpoint registry. | Prefer this for repeated sweeps instead of repeating base URLs and key vars. |
| `-k`, `--api-key-var` | Environment variable containing the API key. | Example: `OPENAI_API_KEY`. |
| `-b`, `--api-base-url` | OpenAI-compatible API base URL. | Useful for local/vLLM or non-default OpenAI-compatible providers. |
| `-n`, `--num-examples` | Number of examples to evaluate. | This is the eval sample count. Pair with `--env-args '{"num_examples":...}'` when you also want to shrink dataset loading. |
| `-r`, `--rollouts-per-example` | Rollouts per dataset example. | Use `1` for smoke/debug, then increase for pass-rate estimates. |
| `-c`, `--max-concurrent` | Concurrent model requests and active rollouts. | Keep low because each rollout creates a sandbox and may run hidden tests. |
| `-w`, `--num-workers` | Env server worker process count. | Leave unset for most local runs; increase only after sandbox startup is stable. |
| `-t`, `--max-tokens` | Maximum generated tokens per assistant response. | Coding tasks usually need more than short QA tasks. |
| `-T`, `--temperature` | Sampling temperature. | Use low values for deterministic smoke checks, higher values for rollout diversity. |
| `-S`, `--sampling-args` | JSON object for provider/model-specific sampling args. | Keys here override `--max-tokens` and `--temperature`. |
| `-a`, `--env-args` | JSON object passed to `load_environment(...)`. | Most self-compaction knobs live here. See the next table. |
| `-C`, `--state-columns` | Comma-separated state keys to save with results. | Use compaction and sandbox diagnostic keys while debugging. |
| `-s`, `--save-results` | Save local result files under `outputs/evals/...`. | Required before using the visualization script. |
| `-o`, `--output-dir` | Custom output directory. | Useful for separating smoke, debug, and benchmark runs. |
| `-R`, `--resume` | Resume a previous incomplete run. | Pass a path or omit the path to auto-detect the latest matching run. |
| `--skip-upload` | Do not upload results to the platform. | Recommended for local iteration. |
| `-d`, `--disable-tui` | Use normal logs instead of the Rich live display. | Better for CI, background runs, and captured logs. |
| `--abbreviated-summary` | Show only settings and stats in the terminal summary. | Useful for larger saved runs. |
| `--hosted` | Run on the platform. | Push the environment first and use the Hub slug. |

## Environment Arguments

Pass these through `-a` / `--env-args` as a JSON object.

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset_name` | `str` | `"R2E-Gym/R2E-Gym-Subset"` | Hugging Face dataset name. |
| `split` | `str \| null` | dataset-dependent | `train` for R2E-Gym subset/lite, `test` for SWE-bench-style datasets. |
| `simple_only` | `bool` | `true` | Keep only small examples when metadata exposes file/line counts. |
| `num_examples` | `int \| null` | `null` | Optional cap applied while loading the environment dataset after shuffle/filtering. |
| `seed` | `int` | `0` | Dataset shuffle seed before optional `num_examples` selection. |
| `filter_repos` | `list[str] \| null` | `null` | Exclude matching `repo` or `repo_name` values. |
| `max_turns` | `int` | `200` | Maximum model turns before truncation. |
| `min_compactions` | `int` | `1` | Required successful `compact()` calls before `submit()` is accepted. |
| `max_summary_chars` | `int` | `6000` | Maximum `compact(summary=...)` length. |
| `sandbox_command_timeout` | `int` | `90` | Per-command timeout in seconds for agent tool calls. |
| `max_command_timeouts` | `int` | `5` | Stop the rollout after this many command timeouts. |
| `rollout_timeout_seconds` | `float` | `5400.0` | Wall-clock cap for one rollout before marking an infra timeout. |
| `test_timeout` | `int` | `900` | Hidden-test timeout in seconds after accepted submit. |
| `total_timeout_minutes` | `int` | `360` | Prime Sandbox lifetime. |
| `cpu_cores` | `int` | `4` | Sandbox CPU cores. |
| `memory_gb` | `int` | `4` | Sandbox memory. |
| `disk_size_gb` | `int` | `2` | Sandbox disk size. |
| `sandbox_client_max_workers` | `int` | `64` | Sandbox client worker pool size used by the environment. |
| `labels` | `list[str] \| null` | `["self-compaction"]` | Labels attached to sandbox requests. |
| `allow_git` | `bool` | `false` | Allow git commands inside `execute_bash`. |
| `skip_swebench_install` | `bool` | `true` | For SWE-bench tasks, skip reinstall steps when the patch does not touch build-sensitive files. |

Common JSON snippets:

```bash
# Tiny R2E smoke dataset.
-a '{"num_examples":1,"max_turns":20,"simple_only":true}'

# Longer R2E run with larger command/test budgets.
-a '{"num_examples":25,"max_turns":160,"sandbox_command_timeout":120,"test_timeout":1200,"rollout_timeout_seconds":5400}'

# Require multiple compactions before submit.
-a '{"min_compactions":2,"max_summary_chars":8000}'

# SWE-bench quick dataset.
-a '{"dataset_name":"PrimeIntellect/SWE-Bench-Verified-Quick","split":"test","simple_only":false}'
```

## Reward And Metrics

Reward is binary:

- `1` when hidden tests pass and `submit()` was accepted after compaction
- `0` otherwise

Metrics include:

- `solved`
- `submitted`
- `submitted_after_compact`
- `compaction_count`
- `turns_before_first_compact`
- `command_timeout_count`
- `rollout_duration_seconds`
- `sandbox_oom`
- `sandbox_timeout`
- `sandbox_image_pull_error`
- `patch_broke_tests`

Useful debug state columns:

- `full_rollout_messages`
- `agent_visible_messages`
- `compaction_summaries`
- `last_compaction_summary`
- `patch_broke_tests_reason`
- `patch_broke_tests_tail`

## Visualizing Rollouts

Generate a side-by-side HTML report from a saved eval `results.jsonl`:

```bash
uv run python scripts/visualize_rollouts.py outputs/evals/<run>/results.jsonl
```

The report shows the full audit transcript next to the transcript reconstructed
from what the model could see after compaction.
