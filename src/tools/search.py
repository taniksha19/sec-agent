"""Filing search and table lookup, both over the Phase 2 retrieval (hybrid + rerank)."""

from collections.abc import Callable

from ingest.parse import is_body_row
from retrieval.base import Filters, SearchResult
from tools.models import (
    LookupTableInput,
    LookupTableOutput,
    SearchFilingsInput,
    SearchFilingsOutput,
    SourceChunk,
    TableRow,
)

# (query, k, filters) -> ranked results. BaselineRAG.retrieve fits this.
Retrieve = Callable[[str, int, Filters | None], list[SearchResult]]

TABLE_CANDIDATES = 10


def to_source(result: SearchResult) -> SourceChunk:
    c = result.chunk
    return SourceChunk(
        chunk_id=c.chunk_id,
        ticker=c.ticker,
        company=c.company,
        form_type=c.form_type,
        fiscal_year=c.fiscal_year,
        fiscal_period=c.fiscal_period,
        section_title=c.section_title,
        source_url=c.source_url,
        page=c.page,
        text=c.text,
    )


def search_filings(retrieve: Retrieve, inp: SearchFilingsInput) -> SearchFilingsOutput:
    filters: Filters = {}
    if inp.tickers:
        filters["ticker"] = [t.upper() for t in inp.tickers]
    if inp.fiscal_year:
        filters["fiscal_year"] = inp.fiscal_year
    if inp.form_type:
        filters["form_type"] = inp.form_type
    results = retrieve(inp.query, inp.k, filters or None)
    return SearchFilingsOutput(chunks=[to_source(r) for r in results])


def table_rows(text: str, metric: str) -> list[tuple[str, list[str], list[str]]]:
    """(label, header, values) for rows whose label holds every word of the metric.
    The first line of a chunk is its context header; header rows run up to the first body row."""
    words = metric.lower().split()
    header_lines: list[str] = []
    rows = []
    for line in text.splitlines()[1:]:
        if " | " not in line:
            continue
        if not is_body_row(line):
            if not rows:
                header_lines.append(line)
            continue
        label, *values = (cell.strip() for cell in line.split(" | "))
        if all(w in label.lower() for w in words):
            rows.append((label, _header_for(header_lines, len(values)), values))
    return rows


def _header_for(header_lines: list[str], n_values: int) -> list[str]:
    """The last header row whose cells line up with the values (with or without a label cell)."""
    for line in reversed(header_lines):
        cells = [c.strip() for c in line.split(" | ")]
        if len(cells) == n_values + 1:
            cells = cells[1:]
        if len(cells) == n_values:
            return cells
    return []


def lookup_table(retrieve: Retrieve, inp: LookupTableInput) -> LookupTableOutput:
    filters: Filters = {
        "ticker": inp.ticker.upper(),
        "fiscal_year": inp.fiscal_year,
        "is_table": True,
    }
    if inp.fiscal_period:
        filters["fiscal_period"] = inp.fiscal_period
    results = retrieve(inp.metric, TABLE_CANDIDATES, filters)
    rows = []
    for r in results:
        for label, header, values in table_rows(r.chunk.text, inp.metric):
            rows.append(TableRow(label=label, header=header, values=values, source=to_source(r)))
    return LookupTableOutput(rows=rows)
