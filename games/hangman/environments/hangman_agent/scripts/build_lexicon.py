from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from wordfreq import zipf_frequency

DEFAULT_SOURCE = Path("/usr/share/dict/words")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "hangman_agent" / "data" / "lexicon.tsv"
)
WORDS_PER_TIER = 2000
MIN_ZIPF = 1.0
COMMON_TARGET_ZIPF = 2.4


@dataclass(frozen=True, slots=True)
class CandidateWord:
    word: str
    zipf: float


def _load_candidates(source: Path) -> list[CandidateWord]:
    candidates: list[CandidateWord] = []
    seen: set[str] = set()
    for raw_line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        word = raw_line.strip()
        if (
            word in seen
            or not word.isascii()
            or not word.isalpha()
            or not word.islower()
            or not 4 <= len(word) <= 10
        ):
            continue
        zipf = zipf_frequency(word, "en")
        if zipf <= MIN_ZIPF:
            continue
        seen.add(word)
        candidates.append(CandidateWord(word=word, zipf=zipf))
    return candidates


def _select_common(candidates: list[CandidateWord]) -> list[CandidateWord]:
    pool = [candidate for candidate in candidates if 4 <= len(candidate.word) <= 6]
    selected = sorted(pool, key=lambda item: (-item.zipf, len(item.word), item.word))
    return selected[:WORDS_PER_TIER]


def _select_obscure(
    candidates: list[CandidateWord], used_words: set[str]
) -> list[CandidateWord]:
    pool = [
        candidate
        for candidate in candidates
        if candidate.word not in used_words and 6 <= len(candidate.word) <= 10
    ]
    selected = sorted(pool, key=lambda item: (item.zipf, -len(item.word), item.word))
    return selected[:WORDS_PER_TIER]


def _select_standard(
    candidates: list[CandidateWord], used_words: set[str]
) -> list[CandidateWord]:
    pool = [
        candidate
        for candidate in candidates
        if candidate.word not in used_words and 5 <= len(candidate.word) <= 8
    ]
    selected = sorted(
        pool,
        key=lambda item: (
            abs(item.zipf - COMMON_TARGET_ZIPF),
            -item.zipf,
            len(item.word),
            item.word,
        ),
    )
    return selected[:WORDS_PER_TIER]


def build_lexicon_rows(source: Path = DEFAULT_SOURCE) -> list[tuple[str, str]]:
    candidates = _load_candidates(source)

    common = _select_common(candidates)
    used_words = {candidate.word for candidate in common}

    obscure = _select_obscure(candidates, used_words)
    used_words.update(candidate.word for candidate in obscure)

    standard = _select_standard(candidates, used_words)

    tiers = {
        "common": sorted(candidate.word for candidate in common),
        "standard": sorted(candidate.word for candidate in standard),
        "obscure": sorted(candidate.word for candidate in obscure),
    }
    return [
        (word, tier)
        for tier in ("common", "standard", "obscure")
        for word in tiers[tier]
    ]


def write_lexicon(rows: list[tuple[str, str]], output: Path = DEFAULT_OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("word", "frequency_tier"))
        writer.writerows(rows)


def main() -> None:
    rows = build_lexicon_rows()
    write_lexicon(rows)
    counts = {
        tier: sum(1 for _, word_tier in rows if word_tier == tier)
        for tier in ("common", "standard", "obscure")
    }
    print(f"Wrote {len(rows)} words to {DEFAULT_OUTPUT}")
    print(counts)


if __name__ == "__main__":
    main()
