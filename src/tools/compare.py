"""Cross-company comparison: change from each company's first to last year, ranked."""

from tools.models import CompanyChange, CompareInput, CompareOutput


def compare(inp: CompareInput) -> CompareOutput:
    by_company: dict[str, dict[int, float]] = {}
    for point in inp.values:
        by_company.setdefault(point.ticker, {})[point.fiscal_year] = point.value
    changes = []
    for ticker, by_year in by_company.items():
        first, last = min(by_year), max(by_year)
        a, b = by_year[first], by_year[last]
        pct = round((b - a) / abs(a) * 100, 2) if a else None
        changes.append(
            CompanyChange(
                ticker=ticker,
                first_year=first,
                last_year=last,
                first_value=a,
                last_value=b,
                change_pct=pct,
            )
        )
    # Highest growth first; companies with no computable change go last.
    changes.sort(key=lambda c: (c.change_pct is None, -(c.change_pct or 0)))
    return CompareOutput(metric=inp.metric, changes=changes)
