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
