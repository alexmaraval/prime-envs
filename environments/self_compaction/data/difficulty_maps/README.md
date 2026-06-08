# Difficulty Maps

Place calibrated self-compaction difficulty JSONL maps here when they should be
packaged with the environment.

Generate a map from saved eval results with:

```bash
uv run python scripts/build_difficulty_map.py outputs/evals/<run>/results.jsonl \
  --dataset-name R2E-Gym/R2E-Gym-Subset \
  --split train \
  --min-rollouts 4 \
  -o data/difficulty_maps/r2e_subset_train.jsonl
```
