# Self-Compaction RL Plan

## Current State

The hosted signal smoke run `fdatbl6sm0rbpkckffe8nozp` completed cleanly:

- Status: `COMPLETED`
- Runtime: about 1h29 for 10 steps
- Step 0 time: about 603s
- Rollout errors: `0.0`
- Sandbox OOM/timeouts/image-pull errors: `0.0`
- Reward range: both `0` and `1` rewards were present
- Step 0 mean turns: `42.9`
- Step 0 mean decode length: `9045`
- Final eval on 2 examples: `Avg@1 = 0.0`, `Pass@1 = 0.0`

The environment/tool surface is healthy enough to continue:

- No `KeyError("read")`
- No unexpected-keyword tool failures
- `read`, `search_files`, `execute_bash`, `edit_via_str_replace`, `compact`, and `submit` all worked in hosted samples
- The remaining issue is training signal, not infrastructure

The main failure mode is zero-advantage filtering. Several steps generated groups where all rollouts solved or all failed, leaving little policy-gradient signal. This produced very small effective batch sizes even though there were no rollout errors.

## Next Smoke Run

Use the v3 signal smoke config:

```bash
cd /Users/alexandremaraval/Documents/Projects/prime-envs
prime train environments/self_compaction/configs/rl/self-compaction-hosted-signal-smoke-v3.toml
```

The v3 config changes the diagnostic shape:

- `max_steps = 6`
- `batch_size = 32`
- `rollouts_per_example = 8`
- `max_inflight_rollouts = 8`
- `max_turns = 60`
- `sampling.max_tokens = 5000`
- `rollout_timeout_seconds = 1500`
- No hosted eval block
- No expected mid-run checkpoint

The purpose is to test whether larger rollout groups reduce all-pass/all-fail groups while keeping sandbox concurrency capped.

## V3 Go/No-Go Criteria

Proceed if:

- `errored_rollouts/all <= 0.10`
- `dropped_groups/all = 0`
- Reward still has both successes and failures
- `filters/zero_advantage` is meaningfully below v2 levels
- `effective_batch_size/all` improves materially from v2
- Mean turns are at or below `50`
- Mean decode length is at or below `8k`
- Step wall time stays comfortably under 90 minutes

Stop or retune if:

- `filters/zero_advantage >= 0.75` repeats across steps
- `effective_batch_size/all` remains tiny
- Mean turns drift above `50`
- Mean decode length drifts above `8k`
- `max_turns_reached` becomes common
- Any tool-contract or sandbox infrastructure error reappears

## Immediate Diagnostics After V3

Pull run status and progress:

```bash
prime train get <run_id> --output json
prime train progress <run_id>
```

Inspect metrics:

```bash
prime train metrics <run_id> --limit 20
```

Inspect logs and rollout samples:

```bash
prime train logs <run_id> --tail 1000 --raw
prime train rollouts <run_id> --step 0 --num 100
prime train distributions <run_id> --step 0
```

Key fields to compare against v2:

- `filters/all/zero_advantage`
- `effective_batch_size/all`
- `reward/all/mean`
- `reward/all/min`
- `reward/all/max`
- `metrics/alex-maraval/self-compaction/num_turns`
- `decode_len/all/mean`
- `stop_condition/all/max_turns_reached`
- `time/step`
- `time/update_weights`

## Frontier-Lab-Style Next Steps

### 1. Signal Calibration

Build a task difficulty map before scaling. For binary-reward GRPO-style training, the best tasks are not the easiest or hardest tasks; they are tasks where the current policy solves some rollouts but not all.

Target per-task success probability:

- Too easy: near `1.0`, zero advantage
- Too hard: near `0.0`, zero advantage
- Useful: roughly `0.2-0.8`

The goal is to feed training batches with enough within-group contrast.

### 2. Environment Reliability

Keep hard gates before larger runs:

- Deterministic scoring
- No hidden-test leakage
- No reward shortcuts
- No tool contract failures
- No sandbox lifecycle failures
- No repeated command timeout patterns

The current environment mostly clears this bar. Continue monitoring it after every prompt/tool change.

### 3. Trajectory Cost Control

The largest cost drivers are long tool trajectories, large prefill contexts, and long decode lengths.

Continue tightening:

- Earlier compaction
- Better file-reading discipline
- Lower `max_turns`
- Lower `sampling.max_tokens`
- More precise prompt instructions around test commands
- Possibly reward or metric pressure for successful compact-and-submit behavior

Do not treat lower token count as the only objective; keep reward diversity and solution quality intact.

### 4. Curriculum

Train on a curated easy/simple slice first, then widen gradually.

Potential curriculum stages:

1. Single-file simple patches
2. Small multi-file patches
3. Tasks with focused test reproduction
4. Tasks requiring broader repo navigation
5. Full mixed R2E/SWE-style tasks

Each stage should have its own heldout eval split.

### 5. Ablation Grid

Run small, controlled comparisons:

- `rollouts_per_example`: `4`, `8`, `16`
- `max_turns`: `50`, `60`, `70`
- `sampling.max_tokens`: `4000`, `5000`, `6000`
- `temperature`: `0.8`, `1.0`, `1.2`
- difficulty filtering: off, then tuned on
- oversampling: only after stable signal is observed

Compare:

- reward distribution
- effective batch size
- zero-advantage rate
- mean turns
- mean decode length
- max-turn stop rate
- solve rate on heldout eval

### 6. Evaluation Discipline

Separate train reward from heldout evaluation.

The v2 eval was too small and returned `0.0`, so do not overinterpret it. Once training signal improves, create a fixed heldout suite with enough examples to see movement.

Minimum useful eval target:

- 20-50 examples for smoke comparisons
- 100+ examples for stronger claims
- Same seed/task set across runs
- No base-model eval during cheap diagnostics unless explicitly needed

### 7. Scale Only After Signal

Do not scale a low-signal run. Scaling should wait until:

- zero-advantage filtering is controlled
- effective batch size is healthy
- step wall time is predictable
- rollouts are free of infrastructure failures
- heldout eval has a stable baseline

Then consider:

- Increasing `batch_size`
- Increasing `max_steps`
- Re-enabling periodic eval
- Re-enabling checkpoint cadence
- Adding more task shards
- Moving from smoke runs to sustained training runs

## Open Questions

- Should the task set be filtered by repository, patch size, or historical base-model solve rate?
- Should compaction be rewarded explicitly or only enforced structurally?
- Should the rubric expose partial-credit signals, or is binary solved/not-solved enough?
- Should eval use the same task family as training or a harder heldout slice?
- Should future runs use hosted training only, or also self-managed `prime-rl` once the loop is stable?
