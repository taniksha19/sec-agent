"""Defense 2: rule-based cleaning of retrieved chunks before any prompt sees them.

Neutralizes text that imitates the prompts' own structure (source tags and [n] markers,
Question:/Sources: lines, chat role labels) and strips invisible characters that can hide
instructions. No classifier and no phrase blocklist: filings are full of ordinary sentences
addressed to "you", so those rules would rewrite real text (docs/decisions.md).
"""

import re
import unicodedata

from retrieval.base import SearchResult

# Fullwidth forms (＜ ［ ：) fold to ASCII so the rules below catch them. Not full NFKC: that
# also turns footnote superscripts into digits ("$500¹" -> "$5001"), and 48 chunks have them.
# Line breaks other than \n become \n, so the line rules see every line start.
FOLD = str.maketrans(
    {cp: cp - 0xFEE0 for cp in range(0xFF01, 0xFF5F)}
    | {0xFE64: "<", 0xFE65: ">", "\r": "\n", "\x85": "\n", " ": "\n", " ": "\n"}
)
# < and > become lookalike angle quotes: no tag can be written, whatever is nested inside it,
# and "< 1%" in a table still reads the same.
ANGLES = str.maketrans({"<": "‹", ">": "›"})
# Printable ASCII, tab, newline and common Latin-1/typography: nothing here can be invisible.
INVISIBLE_FREE_RE = re.compile(r"[\t\n\x20-\x7e\xa0-\xac\xae-\xff‐-‧‰-⁞]*")
# [3], the citation and source-block marker. Filings number footnotes as (3), not [3].
MARKER_RE = re.compile(r"\[\s*(\d{1,3})\s*\]")
# A line that starts like a prompt section or a chat turn.
LINE_RE = re.compile(
    r"^(\s*)(question|sources|system|assistant|user|developer)(\s*:)",
    re.IGNORECASE | re.MULTILINE,
)


def visible(ch: str) -> bool:
    """Drops control, format (zero-width, bidi, Unicode tag), private-use and surrogate
    characters; keeps tab and newline."""
    return ch in "\t\n" or unicodedata.category(ch) not in {"Cc", "Cf", "Co", "Cs"}


def clean_text(text: str) -> str:
    text = text.translate(FOLD)
    if not INVISIBLE_FREE_RE.fullmatch(text):  # fast path: most chunks are plain text
        text = "".join(ch for ch in text if visible(ch))
    text = text.translate(ANGLES)
    text = MARKER_RE.sub(r"(\1)", text)
    # "Question:" becomes "Question -": still readable, no longer a prompt section.
    return LINE_RE.sub(r"\1\2 -", text)


def clean_results(results: list[SearchResult]) -> list[SearchResult]:
    out = []
    for r in results:
        c = r.chunk
        update = {"text": clean_text(c.text), "section_title": clean_text(c.section_title)}
        out.append(r.model_copy(update={"chunk": c.model_copy(update=update)}))
    return out


def cleaned(retrieve):
    """Wrap a (query, k, filters) -> results function so every result is cleaned. The agent and
    the MCP filings server both search through this."""

    def wrapper(query, k=None, filters=None):
        return clean_results(retrieve(query, k, filters))

    return wrapper
