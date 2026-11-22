import pytest

from ingest.models import Chunk
from retrieval.base import SearchResult
from tools.calculator import calculate
from tools.compare import compare
from tools.models import CalculateInput, CompareInput, LookupTableInput, SearchFilingsInput
from tools.search import lookup_table, search_filings, table_rows

TABLE = """Advanced Micro Devices (AMD) 10-K annual report, fiscal 2025. MD&A. Table: summary.
December 27, 2025 | December 28, 2024
(In millions, except for percentages)
Net revenue | $34,639 | $25,785
Research and development | 8,091 | 6,456
Marketing, general and administrative | 4,000 | 2,783"""


def chunk(text, is_table=True):
    return Chunk(
        chunk_id="acc:item_7:1",
        text=text,
        ticker="AMD",
        company="Advanced Micro Devices",
        form_type="10-K",
        accession_number="acc",
        filing_date="2026-02-01",
        fiscal_year=2025,
        fiscal_period="FY",
        source_url="https://example.com",
        section_id="item_7",
        section_title="MD&A",
        chunk_index=1,
        is_table=is_table,
    )


class FakeRetrieve:
    def __init__(self, results):
        self.results, self.calls = results, []

    def __call__(self, query, k, filters):
        self.calls.append((query, k, filters))
        return self.results


def test_calculate_handles_thousands_separators_and_dollars():
    assert calculate(CalculateInput(expression="($8,091 - 6,456) / 6,456 * 100")).value == (
        pytest.approx(25.325, abs=1e-3)
    )
    assert calculate(CalculateInput(expression="-(2 + 3) * 4")).value == -20


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('x')", "2 ** 100000", "abs(-1)", "[1, 2]", "1 / 0", "x + 1"],
)
def test_calculate_rejects_anything_but_arithmetic(expression):
    with pytest.raises(ValueError):
        calculate(CalculateInput(expression=expression))


def test_compare_ranks_by_growth_from_first_to_last_year():
    out = compare(
        CompareInput(
            metric="R&D",
            values={"AMD": {2023: 5872, 2025: 8091, 2024: 6456}, "X": {2024: 0, 2025: 5}},
        )
    )
    amd, x = out.changes
    assert (amd.ticker, amd.first_year, amd.last_year) == ("AMD", 2023, 2025)
    assert amd.change_pct == pytest.approx(37.79, abs=0.01)
    assert x.change_pct is None


def test_table_rows_match_all_metric_words_and_align_header():
    rows = table_rows(TABLE, "research and development")
    assert rows == [
        ("Research and development", ["December 27, 2025", "December 28, 2024"], ["8,091", "6,456"])
    ]
    assert table_rows(TABLE, "general administrative")[0][0].startswith("Marketing")
    assert table_rows(TABLE, "goodwill") == []


def test_lookup_table_filters_company_year_and_tables():
    retrieve = FakeRetrieve([SearchResult(chunk=chunk(TABLE), score=1.0)])
    out = lookup_table(
        retrieve, LookupTableInput(ticker="amd", fiscal_year=2025, metric="net revenue")
    )
    assert out.rows[0].values == ["$34,639", "$25,785"]
    assert out.rows[0].source.chunk_id == "acc:item_7:1"
    _, _, filters = retrieve.calls[0]
    assert filters == {"ticker": "AMD", "fiscal_year": 2025, "is_table": True}


def test_search_filings_passes_only_given_filters():
    retrieve = FakeRetrieve([SearchResult(chunk=chunk("text", is_table=False), score=1.0)])
    out = search_filings(retrieve, SearchFilingsInput(query="q", tickers=["amd", "nvda"], k=3))
    assert out.chunks[0].ticker == "AMD"
    assert retrieve.calls[0] == ("q", 3, {"ticker": ["AMD", "NVDA"]})
    search_filings(retrieve, SearchFilingsInput(query="q"))
    assert retrieve.calls[1][2] is None
