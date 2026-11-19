"""Flag filings whose section split looks wrong.

Usage: uv run python -m ingest.check_sections

A 10-K/10-Q is flagged when a key section has <= 2 chunks or 'cover' holds > 25% of chunks.
Flags are suspicions, not confirmed errors. Known gaps are logged in docs/decisions.md.
"""

import json
from collections import Counter
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
KEY_SECTIONS = {
    "10-K": ["item_1a", "item_7", "item_8"],
    "10-Q": ["part_i_item_1", "part_i_item_2"],
}


def check(chunks: list[dict]) -> list[str]:
    """Return the problems found in one filing's chunks (empty list = looks fine)."""
    if not chunks or chunks[0]["form_type"] not in KEY_SECTIONS:
        return []
    counts = Counter(c["section_id"] for c in chunks)
    problems = [
        f"{s} has {counts[s]} chunks"
        for s in KEY_SECTIONS[chunks[0]["form_type"]]
        if counts[s] <= 2
    ]
    if counts["cover"] > 0.25 * len(chunks):
        problems.append(f"cover has {counts['cover']}/{len(chunks)} chunks")
    return problems


def main() -> None:
    checked = flagged = 0
    for path in sorted(PROCESSED_DIR.glob("*/*.jsonl")):
        chunks = [json.loads(line) for line in path.open(encoding="utf-8")]
        if not chunks or chunks[0]["form_type"] not in KEY_SECTIONS:
            continue
        checked += 1
        if problems := check(chunks):
            flagged += 1
            c = chunks[0]
            print(f"{c['ticker']} {c['form_type']} {c['period_of_report']}: {'; '.join(problems)}")
    print(f"\n{flagged} of {checked} 10-K/10-Q filings flagged ({checked - flagged} look clean)")


if __name__ == "__main__":
    main()
