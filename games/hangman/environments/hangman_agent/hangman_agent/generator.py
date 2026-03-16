from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from importlib import resources
from math import floor
from typing import Any, Iterable, Sequence

from datasets import Dataset

from .game import (
    MAX_WRONG_GUESSES,
    compute_repeat_density,
    initialize_game_state,
    normalize_word,
    render_board,
)

DEFAULT_DATASET_SIZE = 128
_SPLIT_OFFSETS = {"train": 0, "eval": 100_000}
DIFFICULTY_LEVELS = ("easy", "medium", "hard")


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    word: str
    frequency_tier: str
    word_length: int
    distinct_letter_count: int
    repeat_density: float


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    difficulty: str
    seed: int
    dataset_size: int
    word_length_min: int
    word_length_max: int
    frequency_tiers: tuple[str, ...]
    repeat_density_min: float
    repeat_density_max: float
    allowed_attempts_min: int
    allowed_attempts_max: int
    turn_slack: int
    difficulty_mix: tuple[float, float, float] | None = None


PRESET_CONFIGS: dict[str, GenerationConfig] = {
    "easy": GenerationConfig(
        difficulty="easy",
        seed=0,
        dataset_size=DEFAULT_DATASET_SIZE,
        word_length_min=4,
        word_length_max=10,
        frequency_tiers=("easy",),
        repeat_density_min=0.0,
        repeat_density_max=1.0,
        allowed_attempts_min=MAX_WRONG_GUESSES,
        allowed_attempts_max=MAX_WRONG_GUESSES,
        turn_slack=0,
        difficulty_mix=None,
    ),
    "medium": GenerationConfig(
        difficulty="medium",
        seed=0,
        dataset_size=DEFAULT_DATASET_SIZE,
        word_length_min=4,
        word_length_max=10,
        frequency_tiers=("medium",),
        repeat_density_min=0.0,
        repeat_density_max=1.0,
        allowed_attempts_min=MAX_WRONG_GUESSES,
        allowed_attempts_max=MAX_WRONG_GUESSES,
        turn_slack=0,
        difficulty_mix=None,
    ),
    "hard": GenerationConfig(
        difficulty="hard",
        seed=0,
        dataset_size=DEFAULT_DATASET_SIZE,
        word_length_min=4,
        word_length_max=10,
        frequency_tiers=("hard",),
        repeat_density_min=0.0,
        repeat_density_max=1.0,
        allowed_attempts_min=MAX_WRONG_GUESSES,
        allowed_attempts_max=MAX_WRONG_GUESSES,
        turn_slack=0,
        difficulty_mix=None,
    ),
}


def _coerce_frequency_tiers(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip().lower() for item in value if str(item).strip()]
    return tuple(items)


def _coerce_difficulty_mix(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("difficulty_mix must be non-empty when provided as a string")
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                items = [item.strip() for item in stripped[1:-1].split(",") if item.strip()]
            else:
                if not isinstance(parsed, list):
                    raise ValueError(
                        "difficulty_mix string must decode to a JSON array of three weights"
                    )
                items = [str(item).strip() for item in parsed if str(item).strip()]
        else:
            items = [item.strip() for item in stripped.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]
    if len(items) != len(DIFFICULTY_LEVELS):
        raise ValueError(
            "difficulty_mix must provide exactly three weights in easy,medium,hard order"
        )
    weights = tuple(float(item) for item in items)
    if any(weight < 0.0 for weight in weights):
        raise ValueError("difficulty_mix weights must be non-negative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("difficulty_mix must contain at least one positive weight")
    return tuple(weight / total for weight in weights)


def _allocate_mixture_counts(
    dataset_size: int, difficulty_mix: tuple[float, float, float]
) -> tuple[int, int, int]:
    raw_counts = [dataset_size * weight for weight in difficulty_mix]
    counts = [floor(value) for value in raw_counts]
    remainder = dataset_size - sum(counts)
    ranked_remainders = sorted(
        (
            (raw_counts[index] - counts[index], index)
            for index in range(len(DIFFICULTY_LEVELS))
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for _, index in ranked_remainders[:remainder]:
        counts[index] += 1
    return tuple(counts)


def _build_mixed_generation_config(
    seed: int, dataset_size: int, difficulty_mix: tuple[float, float, float]
) -> GenerationConfig:
    active_presets = [
        PRESET_CONFIGS[difficulty]
        for difficulty, weight in zip(DIFFICULTY_LEVELS, difficulty_mix)
        if weight > 0.0
    ]
    ordered_tiers = tuple(
        dict.fromkeys(
            tier for preset in active_presets for tier in preset.frequency_tiers
        )
    )
    return GenerationConfig(
        difficulty="mixed",
        seed=int(seed),
        dataset_size=dataset_size,
        word_length_min=min(preset.word_length_min for preset in active_presets),
        word_length_max=max(preset.word_length_max for preset in active_presets),
        frequency_tiers=ordered_tiers,
        repeat_density_min=min(preset.repeat_density_min for preset in active_presets),
        repeat_density_max=max(preset.repeat_density_max for preset in active_presets),
        allowed_attempts_min=MAX_WRONG_GUESSES,
        allowed_attempts_max=MAX_WRONG_GUESSES,
        turn_slack=0,
        difficulty_mix=difficulty_mix,
    )


def _mixed_component_configs(config: GenerationConfig) -> tuple[GenerationConfig, ...]:
    if config.difficulty_mix is None:
        return (config,)
    counts = _allocate_mixture_counts(config.dataset_size, config.difficulty_mix)
    components: list[GenerationConfig] = []
    for index, (difficulty, count) in enumerate(zip(DIFFICULTY_LEVELS, counts)):
        if count <= 0:
            continue
        components.append(
            replace(
                PRESET_CONFIGS[difficulty],
                seed=int(config.seed) + ((index + 1) * 10_000),
                dataset_size=count,
                difficulty_mix=None,
            )
        )
    return tuple(components)


def _validate_no_mix_overrides(overrides: dict[str, Any]) -> None:
    conflicting = sorted(key for key, value in overrides.items() if value is not None)
    if conflicting:
        joined = ", ".join(conflicting)
        raise ValueError(
            "difficulty_mix uses the built-in easy/medium/hard presets and cannot be "
            f"combined with manual generation overrides: {joined}"
        )


def resolve_generation_config(
    difficulty: str = "easy",
    seed: int = 0,
    num_examples: int = DEFAULT_DATASET_SIZE,
    difficulty_mix: Sequence[float] | str | None = None,
    word_length_min: int | None = None,
    word_length_max: int | None = None,
    frequency_tiers: Sequence[str] | str | None = None,
    repeat_density_min: float | None = None,
    repeat_density_max: float | None = None,
    allowed_attempts_min: int | None = None,
    allowed_attempts_max: int | None = None,
) -> GenerationConfig:
    dataset_size = DEFAULT_DATASET_SIZE if int(num_examples) <= 0 else int(num_examples)
    overrides: dict[str, Any] = {
        "word_length_min": word_length_min,
        "word_length_max": word_length_max,
        "frequency_tiers": _coerce_frequency_tiers(frequency_tiers),
        "repeat_density_min": repeat_density_min,
        "repeat_density_max": repeat_density_max,
        "allowed_attempts_min": allowed_attempts_min,
        "allowed_attempts_max": allowed_attempts_max,
    }
    normalized_mix = _coerce_difficulty_mix(difficulty_mix)
    if normalized_mix is not None:
        _validate_no_mix_overrides(overrides)
        return _build_mixed_generation_config(
            seed=seed,
            dataset_size=dataset_size,
            difficulty_mix=normalized_mix,
        )

    difficulty_key = (difficulty or "medium").lower()
    if difficulty_key not in PRESET_CONFIGS:
        raise ValueError(
            f"unsupported difficulty {difficulty!r}; expected one of {sorted(PRESET_CONFIGS)}"
        )

    config = replace(PRESET_CONFIGS[difficulty_key], seed=int(seed))
    overrides["dataset_size"] = dataset_size
    normalized = asdict(config)
    for key, value in overrides.items():
        if value is not None:
            normalized[key] = value

    resolved = GenerationConfig(**normalized)
    if resolved.word_length_min > resolved.word_length_max:
        raise ValueError("word_length_min must be <= word_length_max")
    if resolved.allowed_attempts_min > resolved.allowed_attempts_max:
        raise ValueError("allowed_attempts_min must be <= allowed_attempts_max")
    return replace(
        resolved,
        allowed_attempts_min=MAX_WRONG_GUESSES,
        allowed_attempts_max=MAX_WRONG_GUESSES,
        turn_slack=0,
        difficulty_mix=None,
    )


def load_lexicon() -> tuple[LexiconEntry, ...]:
    lexicon_path = resources.files("hangman_agent.data").joinpath("lexicon.tsv")
    entries: list[LexiconEntry] = []
    with lexicon_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            word = normalize_word(row["word"])
            difficulty = (
                row.get("difficulty")
                or row.get("frequency_tier")
                or ""
            ).strip().lower()
            entries.append(
                LexiconEntry(
                    word=word,
                    frequency_tier=difficulty,
                    word_length=len(word),
                    distinct_letter_count=len(set(word)),
                    repeat_density=compute_repeat_density(word),
                )
            )
    return tuple(entries)


def filter_lexicon(
    lexicon: Iterable[LexiconEntry], config: GenerationConfig
) -> tuple[LexiconEntry, ...]:
    tiers = set(config.frequency_tiers)
    filtered = [
        entry
        for entry in lexicon
        if config.word_length_min <= entry.word_length <= config.word_length_max
        and entry.frequency_tier in tiers
        and config.repeat_density_min <= entry.repeat_density <= config.repeat_density_max
    ]
    if not filtered:
        raise ValueError("no lexicon entries match the requested generation config")
    return tuple(filtered)


def _candidate_counts_by_length(
    lexicon: Iterable[LexiconEntry],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for entry in lexicon:
        counts[entry.word_length] = counts.get(entry.word_length, 0) + 1
    return counts


def _split_seed(split: str, seed: int, index: int) -> int:
    return int(seed) + _SPLIT_OFFSETS.get(split, 500_000) + index


def build_records(
    config: GenerationConfig,
    lexicon: Sequence[LexiconEntry],
    split: str = "train",
) -> list[dict[str, Any]]:
    if config.difficulty_mix is not None:
        records: list[dict[str, Any]] = []
        for component_config in _mixed_component_configs(config):
            component_lexicon = filter_lexicon(lexicon, component_config)
            component_records = build_records(
                config=component_config,
                lexicon=component_lexicon,
                split=split,
            )
            for record in component_records:
                info = dict(record["info"])
                info["requested_difficulty_mix"] = {
                    difficulty: weight
                    for difficulty, weight in zip(
                        DIFFICULTY_LEVELS, config.difficulty_mix
                    )
                }
                info["config"] = dict(info["config"])
                info["config"]["difficulty_mix"] = list(config.difficulty_mix)
                records.append({**record, "info": info})
        rng = random.Random(_split_seed(split, config.seed, config.dataset_size))
        rng.shuffle(records)
        return records

    def build_record(task: dict[str, Any]) -> dict[str, Any]:
        initial_state = initialize_game_state(task)
        return {
            "prompt": [{"role": "user", "content": render_board(initial_state)}],
            "info": task,
        }

    def build_task(
        entry: LexiconEntry,
        seed: int,
        candidate_count: int,
    ) -> dict[str, Any]:
        return {
            "secret_word": entry.word,
            "frequency_tier": entry.frequency_tier,
            "difficulty": config.difficulty,
            "max_wrong_guesses": MAX_WRONG_GUESSES,
            "pre_revealed_letters": [],
            "pre_wrong_letters": [],
            "candidate_count": candidate_count,
            "word_length": entry.word_length,
            "distinct_letter_count": entry.distinct_letter_count,
            "repeat_density": entry.repeat_density,
            "seed": seed,
            "config": {
                "difficulty": config.difficulty,
                "word_length_min": config.word_length_min,
                "word_length_max": config.word_length_max,
                "frequency_tiers": list(config.frequency_tiers),
                "repeat_density_min": config.repeat_density_min,
                "repeat_density_max": config.repeat_density_max,
                "allowed_attempts_min": config.allowed_attempts_min,
                "allowed_attempts_max": config.allowed_attempts_max,
                "turn_slack": config.turn_slack,
            },
        }

    candidate_counts = _candidate_counts_by_length(lexicon)
    combinations = list(lexicon)
    if not combinations:
        raise RuntimeError("failed to generate any records for the requested config")
    rng = random.Random(_split_seed(split, config.seed, config.dataset_size))
    rng.shuffle(combinations)

    records: list[dict[str, Any]] = []
    for sample_index in range(config.dataset_size):
        if sample_index < len(combinations):
            entry = combinations[sample_index]
        else:
            entry = combinations[rng.randrange(len(combinations))]
        task = build_task(
            entry=entry,
            seed=_split_seed(split, config.seed, sample_index),
            candidate_count=candidate_counts.get(entry.word_length, 0),
        )
        records.append(build_record(task))
    return records


def build_dataset(
    config: GenerationConfig, lexicon: Sequence[LexiconEntry], split: str = "train"
) -> Dataset:
    return Dataset.from_list(build_records(config=config, lexicon=lexicon, split=split))


def make_dataset_builder(
    config: GenerationConfig, lexicon: Sequence[LexiconEntry], split: str = "train"
):
    def build() -> Dataset:
        return build_dataset(config=config, lexicon=lexicon, split=split)

    return build
