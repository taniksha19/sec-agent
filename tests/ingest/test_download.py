import pytest

from ingest.download import (
    check_user_agent,
    find_exhibit_99,
    fiscal_periods_by_accession,
    resolve_ciks,
    rows_from_columns,
    select_filings,
)


def row(form, filing_date, report_date="", items="", acc=None):
    return {
        "accessionNumber": acc or f"{form}-{filing_date}",
        "form": form,
        "filingDate": filing_date,
        "reportDate": report_date,
        "items": items,
        "primaryDocument": "doc.htm",
    }


def test_rows_from_columns():
    cols = {"form": ["10-K", "8-K"], "filingDate": ["2025-01-01", "2024-12-01"]}
    assert rows_from_columns(cols) == [
        {"form": "10-K", "filingDate": "2025-01-01"},
        {"form": "8-K", "filingDate": "2024-12-01"},
    ]


def test_select_filings_counts_and_skips_amendments():
    rows = [
        row("8-K", "2025-10-30", items="2.02,9.01"),
        row("10-Q", "2025-10-31", "2025-09-30"),
        row("8-K", "2025-09-01", items="5.02"),  # not an earnings release
        row("10-Q", "2025-07-31", "2025-06-30"),
        row("10-K/A", "2025-04-01", "2024-12-31"),  # amendment, skipped
        row("10-Q", "2025-04-30", "2025-03-31"),
        row("10-K", "2025-02-15", "2024-12-31"),
        row("10-Q", "2024-10-31", "2024-09-30"),
        row("8-K", "2024-10-25", items="2.02,9.01"),
        row("10-Q", "2024-07-31", "2024-06-30"),  # 5th 10-Q, beyond N_10Q
        row("8-K", "2024-07-20", items="2.02"),  # before oldest selected 10-Q period
        row("10-K", "2024-02-15", "2023-12-31"),
        row("10-K", "2023-02-15", "2022-12-31"),
        row("10-K", "2022-02-15", "2021-12-31"),  # 4th 10-K
    ]
    picked = select_filings(rows)
    forms = [r["form"] for r in picked]
    assert forms.count("10-K") == 3
    assert forms.count("10-Q") == 4
    assert "10-K/A" not in forms
    assert [r["filingDate"] for r in picked if r["form"] == "8-K"] == ["2025-10-30", "2024-10-25"]


def test_select_filings_without_10q_returns_only_10k():
    assert [r["form"] for r in select_filings([row("10-K", "2025-02-01", "2024-12-31")])] == [
        "10-K"
    ]


def test_resolve_ciks_pads_and_reports_missing():
    data = {"0": {"cik_str": 320193, "ticker": "AAPL"}, "1": {"cik_str": 789019, "ticker": "MSFT"}}
    assert resolve_ciks(data, ["AAPL"]) == {"AAPL": "0000320193"}
    with pytest.raises(ValueError, match="ZZZZ"):
        resolve_ciks(data, ["AAPL", "ZZZZ"])


def test_fiscal_periods_by_accession_uses_company_tags():
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"accn": "A1", "form": "10-K", "fy": 2024, "fp": "FY"},
                            {"accn": "A2", "form": "10-Q", "fy": 2025, "fp": "Q2"},
                            {"accn": "A3", "form": "8-K", "fy": 2025, "fp": "Q2"},
                        ]
                    }
                }
            }
        }
    }
    assert fiscal_periods_by_accession(facts) == {"A1": (2024, "FY"), "A2": (2025, "Q2")}


INDEX_HTML = """
<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>8-K</td><td><a href="/x/form8-k.htm">form8-k.htm</a></td><td>8-K</td><td>1</td></tr>
<tr><td>2</td><td>SUPPLEMENT</td><td><a href="/x/ex992.htm">ex992.htm</a></td><td>EX-99.2</td><td>1</td></tr>
<tr><td>3</td><td>PRESS RELEASE</td><td><a href="/x/ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>1</td></tr>
</table>
"""


def test_find_exhibit_99_prefers_99_1():
    assert find_exhibit_99(INDEX_HTML) == "ex991.htm"


def test_find_exhibit_99_falls_back_and_handles_none():
    html = INDEX_HTML.replace("EX-99.1", "GRAPHIC")
    assert find_exhibit_99(html) == "ex992.htm"
    assert find_exhibit_99("<table class='tableFile'></table>") is None


@pytest.mark.parametrize("value", [None, "", "sec-agent", "sec-agent you@example.com"])
def test_check_user_agent_rejects_placeholders(value):
    with pytest.raises(SystemExit):
        check_user_agent(value)


def test_check_user_agent_accepts_real_contact():
    assert check_user_agent("sec-agent me@school.edu") == "sec-agent me@school.edu"
