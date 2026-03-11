from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from wordfreq import zipf_frequency

DEFAULT_SOURCE = Path("/usr/share/dict/words")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "hangman_agent" / "data" / "lexicon.tsv"
)
WORDS_PER_DIFFICULTY = {"easy": 3334, "medium": 3333, "hard": 3333}
MIN_ZIPF = 1.0
MEDIUM_TARGET_ZIPF = 2.4


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


def _select_easy(candidates: list[CandidateWord]) -> list[CandidateWord]:
    pool = [candidate for candidate in candidates if 4 <= len(candidate.word) <= 6]
    selected = sorted(pool, key=lambda item: (-item.zipf, len(item.word), item.word))
    return selected[:WORDS_PER_DIFFICULTY["easy"]]


def _select_hard(
    candidates: list[CandidateWord], used_words: set[str]
) -> list[CandidateWord]:
    pool = [
        candidate
        for candidate in candidates
        if candidate.word not in used_words and 6 <= len(candidate.word) <= 10
    ]
    selected = sorted(pool, key=lambda item: (item.zipf, -len(item.word), item.word))
    return selected[:WORDS_PER_DIFFICULTY["hard"]]


def _select_medium(
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
            abs(item.zipf - MEDIUM_TARGET_ZIPF),
            -item.zipf,
            len(item.word),
            item.word,
        ),
    )
    return selected[:WORDS_PER_DIFFICULTY["medium"]]


def build_lexicon_rows(source: Path = DEFAULT_SOURCE) -> list[tuple[str, str]]:
    candidates = _load_candidates(source)

    easy = _select_easy(candidates)
    used_words = {candidate.word for candidate in easy}

    hard = _select_hard(candidates, used_words)
    used_words.update(candidate.word for candidate in hard)

    medium = _select_medium(candidates, used_words)

    difficulties = {
        "easy": sorted(candidate.word for candidate in easy),
        "medium": sorted(candidate.word for candidate in medium),
        "hard": sorted(candidate.word for candidate in hard),
    }
    total_words = sum(len(words) for words in difficulties.values())
    if total_words != sum(WORDS_PER_DIFFICULTY.values()):
        raise RuntimeError(f"expected {sum(WORDS_PER_DIFFICULTY.values())} words, got {total_words}")
    return [
        (word, difficulty)
        for difficulty in ("easy", "medium", "hard")
        for word in difficulties[difficulty]
    ]


def write_lexicon(rows: list[tuple[str, str]], output: Path = DEFAULT_OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("word", "difficulty"))
        writer.writerows(rows)


def main() -> None:
    rows = build_lexicon_rows()
    write_lexicon(rows)
    counts = {
        difficulty: sum(1 for _, word_difficulty in rows if word_difficulty == difficulty)
        for difficulty in ("easy", "medium", "hard")
    }
    print(f"Wrote {len(rows)} words to {DEFAULT_OUTPUT}")
    print(counts)


if __name__ == "__main__":
    main()
