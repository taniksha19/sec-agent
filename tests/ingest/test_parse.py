from bs4 import BeautifulSoup

from ingest.parse import (
    CHUNK_TOKENS,
    Block,
    Unit,
    count_tokens,
    decode,
    extract_blocks,
    pack,
    split_sections,
    table_text,
)


def test_decode_falls_back_to_cp1252():
    assert decode("Company’s".encode()) == "Company’s"
    assert decode("Company’s".encode("cp1252")) == "Company’s"


def test_table_text_merges_dollar_and_paren_cells():
    html = """<table>
      <tr><td></td><td>2025</td><td>2024</td></tr>
      <tr><td>Net income</td><td>$</td><td>1,000</td><td>(50</td><td>)</td></tr>
    </table>"""
    table = BeautifulSoup(html, "lxml").table
    assert table_text(table) == ["2025 | 2024", "Net income | $1,000 | (50)"]


def test_extract_blocks_skips_hidden_xbrl_and_tracks_pages():
    html = """<html><body>
      <div style="display:none">hidden fact</div>
      <p>First page</p>
      <div style="page-break-after: always"></div>
      <p>Second page</p>
    </body></html>"""
    blocks = extract_blocks(html)
    assert [(b.text, b.page) for b in blocks] == [("First page", 1), ("Second page", 2)]


def blocks(*texts):
    return [Block("text", t, 1) for t in texts]


def test_split_sections_skips_table_of_contents():
    bs = blocks(
        "Item 1A. Risk Factors",  # table of contents
        "Item 7. MD&A",  # table of contents
        "Item 1A. Risk Factors",
        "Risks text",
        "Item 7.",
        "Management's Discussion",  # title on its own line
        "MD&A text",
    )
    sections = split_sections(bs, "10-K")
    assert [(s, t) for s, t, _ in sections] == [
        ("cover", "Cover and Table of Contents"),
        ("item_1a", "Risk Factors"),
        ("item_7", "Management's Discussion"),
    ]
    assert [b.text for b in sections[1][2]] == ["Risks text"]


def test_split_sections_10q_keys_include_part():
    bs = blocks("PART I", "Item 2. MD&A", "text", "PART II", "Item 2. Repurchases", "text")
    ids = [s for s, _, _ in split_sections(bs, "10-Q")]
    assert ids == ["cover", "part_i_item_2", "part_ii_item_2"]


def test_split_sections_ignores_long_cross_references():
    long_ref = "Item 7 of this Form 10-K discusses " + "results " * 40
    sections = split_sections(blocks("Item 1. Business", long_ref), "10-K")
    assert [s for s, _, _ in sections] == ["item_1"]


def test_pack_respects_limit_and_overlaps():
    sentence = "Revenue grew because demand for services increased."
    n = count_tokens(sentence)
    units = [Unit(sentence, n, 1, 0) for _ in range(60)]
    chunks = pack(units)
    assert len(chunks) > 1
    assert all(count_tokens(text) <= CHUNK_TOKENS for text, _ in chunks)
    # the last sentence of one chunk starts the next
    assert chunks[1][0].startswith(sentence)


def test_split_sections_10q_item_title_decides_part():
    # BofA-style: "Part II" appears just before MD&A, and Part II has its own Item 2.
    bs = blocks(
        "Part I. Financial Information",
        "Part II. Other Information",
        "Item 2. Management's Discussion and Analysis",
        "MD&A text " * 30,
        "Item 2. Unregistered Sales of Equity Securities",
        "text",
    )
    ids = [s for s, _, _ in split_sections(bs, "10-Q")]
    assert "part_i_item_2" in ids and "part_ii_item_2" in ids


def test_split_sections_replaces_stub_item_with_title():
    # JPMorgan-style: "Item 7." is a one-line pointer; real MD&A comes later under its title.
    bs = blocks(
        "Item 7. Management's Discussion and Analysis",
        "See pages 50-150.",
        "Item 8. Financial Statements",
        "See pages 150-300.",
        "Management's discussion and analysis",
        "Real MD&A " * 40,
    )
    sections = {s: b for s, _, b in split_sections(bs, "10-K")}
    assert sections["item_7"][0].text.startswith("Real MD&A")


def test_is_body_row_skips_header_and_year_rows():
    from ingest.parse import is_body_row

    assert not is_body_row("Year Ended")
    assert not is_body_row("December 27, 2025 | December 28, 2024")
    assert not is_body_row("(In millions) | 2025 | 2024")
    assert is_body_row("Net revenue | $34,639 | $25,785")
    assert is_body_row("Operating loss | ( 2,214 ) | 93")


def test_split_table_repeats_all_header_rows():
    from ingest.parse import table_chunks

    header = ["Year Ended", "December 27, 2025 | December 28, 2024", "(In millions)"]
    body = [f"Line item number {i} with a longer label | {i},000 | {i},500" for i in range(120)]
    block = Block("table", "\n".join(header + body), page=3)
    chunks = table_chunks(block)
    assert len(chunks) > 1
    for text, page in chunks:
        assert text.startswith("\n".join(header))
        assert count_tokens(text) <= CHUNK_TOKENS
        assert page == 3


def test_context_header_names_company_form_period_and_table():
    from datetime import date

    from ingest.models import Company, Filing
    from ingest.parse import context_header

    company = Company(ticker="AMD", name="Advanced Micro Devices", sector="Tech", cik="2488")
    filing = Filing(
        ticker="AMD",
        cik="2488",
        form_type="10-Q",
        accession_number="a",
        filing_date=date(2026, 5, 1),
        period_of_report=date(2026, 3, 28),
        fiscal_year=2026,
        fiscal_period="Q1",
        document_name="d.htm",
        source_url="https://x",
        local_path="p",
    )
    head = context_header(
        filing, company, "Financial Statements", "Condensed Statements of Operations"
    )
    assert head == (
        "Advanced Micro Devices (AMD) 10-Q quarterly report, fiscal 2026 Q1. Financial Statements."
        " Table: Condensed Statements of Operations."
    )
    annual = filing.model_copy(update={"form_type": "10-K", "fiscal_period": "FY"})
    assert "10-K annual report, fiscal 2026. Risk Factors." in context_header(
        annual, company, "Risk Factors", None
    )
