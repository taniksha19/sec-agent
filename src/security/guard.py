"""Defense 3: tool arguments must stay within what the user asked about.

lookup_table and compare name companies and years. Those must come from the user's question
or the planner's sub-questions (which only read the question), never from retrieved text
alone. A question with no named company or year leaves that dimension unrestricted.
"""

import re
from dataclasses import dataclass

YEAR_RE = re.compile(r"\b(199\d|20\d\d)\b")


@dataclass(frozen=True)
class Scope:
    tickers: frozenset[str]
    years: frozenset[int]


def scope(question: str, sub_questions: list[tuple[str, str | None]]) -> Scope:
    """sub_questions: (text, ticker) pairs from the planner. Years reach one past each end of
    the years named: growth "in 2025" needs the 2024 figure, and a fiscal year can be labelled
    by the calendar year it starts or ends in."""
    tickers = frozenset(t for _, t in sub_questions if t)
    text = " ".join([question, *(q for q, _ in sub_questions)])
    named = [int(y) for y in YEAR_RE.findall(text)]
    years = frozenset(range(min(named) - 1, max(named) + 2)) if named else frozenset()
    return Scope(tickers, years)


def check(name: str, args: dict, allowed: Scope) -> None:
    """Raise ValueError if a lookup_table or compare call reaches outside the allowed scope."""
    if name == "lookup_table":
        pairs = [(args.get("ticker"), args.get("fiscal_year"))]
    elif name == "compare":
        pairs = [(v.get("ticker"), v.get("fiscal_year")) for v in args.get("values", [])]
    else:
        return
    for ticker, year in pairs:
        if allowed.tickers and str(ticker).upper() not in allowed.tickers:
            raise ValueError(f"ticker {ticker!r} is not part of the question")
        if allowed.years and year not in allowed.years:
            raise ValueError(f"fiscal year {year!r} is not part of the question")
