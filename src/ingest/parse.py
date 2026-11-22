"""Parse downloaded filings into section-aware chunks.

Usage: uv run python -m ingest.parse [--tickers AAPL MSFT]

Reads data/raw/<TICKER>/filings.json and writes data/processed/<TICKER>/<accession>.jsonl,
one Chunk per line. Prints the sections found per filing so gaps are visible.
"""

import argparse
import json
import logging
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning
from tokenizers import Tokenizer

from ingest.models import Chunk, Company, Filing

log = logging.getLogger(__name__)
# Inline-XBRL filings are XHTML; the HTML parser handles them fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FASTEMBED_CACHE = REPO_ROOT / "data" / "cache" / "fastembed"

# bge-small reads at most 512 tokens; 400 leaves room for special tokens and a context header.
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50

BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "center"}
ITEM_RE = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[.:\-—–]?\s*(.*)$", re.IGNORECASE)
PART_RE = re.compile(r"^part\s+(iv|i{1,3})\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"“])")
# Headings are short; long blocks starting with "Item 7" are cross-references in prose.
MAX_HEADING_CHARS = 200


@dataclass
class Block:
    kind: str  # "text" or "table"
    text: str
    page: int


@lru_cache(maxsize=1)
def tokenizer() -> Tokenizer:
    """The embedding model's own tokenizer, so chunk sizes match what the model sees."""
    paths = list(FASTEMBED_CACHE.glob("**/*bge-small-en-v1.5*/**/tokenizer.json"))
    if not paths:
        from retrieval.embeddings import Embedder

        _ = Embedder().model  # downloads the model (and tokenizer) into data/cache/fastembed
        paths = list(FASTEMBED_CACHE.glob("**/*bge-small-en-v1.5*/**/tokenizer.json"))
    tok = Tokenizer.from_file(str(paths[0]))
    # The file ships with truncation at 512; we need true lengths to size chunks.
    tok.no_truncation()
    tok.no_padding()
    return tok


def count_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


def decode(raw: bytes) -> str:
    """EDGAR documents are UTF-8 or Windows-1252; lxml sometimes guesses wrong."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def table_text(table: Tag) -> list[str]:
    """One line per row, cells joined by ' | '. Merges EDGAR's split '$', ')' and '%' cells."""
    rows = []
    for tr in table.find_all("tr"):
        cells: list[str] = []
        for td in tr.find_all(["td", "th"]):
            cell = normalize(td.get_text(" "))
            if not cell:
                continue
            if cells and cells[-1] == "$":
                cells[-1] = "$" + cell
            elif cells and cell in (")", "%", ")%"):
                cells[-1] += cell
            else:
                cells.append(cell)
        if cells:
            rows.append(" | ".join(cells))
    return rows


def page_break(tag: Tag) -> tuple[bool, bool]:
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "page-break-before:always" in style, "page-break-after:always" in style


def extract_blocks(html: str) -> list[Block]:
    """Flatten a filing into text and table blocks in reading order, tracking page numbers."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display:\s*none", re.IGNORECASE)):
        tag.decompose()

    blocks: list[Block] = []
    buffer: list[str] = []
    page = 1

    def flush() -> None:
        text = normalize(" ".join(buffer))
        buffer.clear()
        if text:
            blocks.append(Block("text", text, page))

    def walk(el: Tag) -> None:
        nonlocal page
        for child in el.children:
            if isinstance(child, NavigableString):
                buffer.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            before, after = page_break(child)
            if before:
                flush()
                page += 1
            if child.name == "table":
                flush()
                rows = table_text(child)
                # Single-row tables are layout (e.g. "Item 7. | MD&A"), not data.
                if len(rows) == 1:
                    blocks.append(Block("text", rows[0].replace(" | ", " "), page))
                elif rows:
                    blocks.append(Block("table", "\n".join(rows), page))
            elif child.name in BLOCK_TAGS:
                flush()
                walk(child)
                flush()
            else:
                walk(child)
            if after:
                flush()
                page += 1

    walk(soup.body or soup)
    flush()
    return blocks


def heading_key(text: str, part: str | None, form_type: str) -> tuple[str, str] | None:
    """Return (section_id, title) if this block is an Item heading."""
    if len(text) > MAX_HEADING_CHARS:
        return None
    m = ITEM_RE.match(text)
    if not m:
        return None
    item = m.group(1).lower()
    title = m.group(2).strip()
    # Item numbers repeat across Parts in a 10-Q (Part I Item 2 vs Part II Item 2).
    if form_type == "10-Q":
        if part is None:
            return None
        return f"part_{part.lower()}_item_{item}", title
    return f"item_{item}", title


# Some large filers (Intel, BofA, JPMorgan) use plain titles instead of "Item N." headings, or
# keep the Item heading as a one-line stub pointing elsewhere. Used only when the Item heading
# is missing or a stub.
_MDA = r"management'?s discussion and analysis( of financial condition and results of operations)?"
FALLBACK_TITLES: dict[str, dict[str, tuple[str, str]]] = {
    "10-K": {
        "item_1a": ("Risk Factors", r"risk factors"),
        "item_7": ("Management's Discussion and Analysis", _MDA),
        "item_8": (
            "Financial Statements and Supplementary Data",
            r"(consolidated )?financial statements( and supplement(ary|al) (data|details))?"
            r"|consolidated statements? of (income|operations|earnings)"
            r"|reports? of independent registered public accounting firm",
        ),
    },
    "10-Q": {
        "part_i_item_1": (
            "Financial Statements",
            r"(consolidated )?(condensed )?financial statements( \(unaudited\))?"
            r"|consolidated (condensed )?statements? of (income|operations|earnings)"
            r"( \(unaudited\))?",
        ),
        "part_i_item_2": ("Management's Discussion and Analysis", _MDA),
    },
}


PART_I_TITLES = re.compile(
    r"financial statements|discussion and analysis|quantitative and qualitative|controls and proc",
    re.IGNORECASE,
)
PART_II_TITLES = re.compile(
    r"legal proceedings|risk factors|unregistered sales|defaults upon|mine safety"
    r"|other information|exhibits",
    re.IGNORECASE,
)

# An Item section shorter than this is a cross-reference stub ("see pages 50-120").
STUB_CHARS = 1500


def find_title(blocks: list[Block], pattern: str, start: int = 0) -> int | None:
    """Index of the first block that is exactly this title AND is followed by content.

    Whole-line match skips running page headers like 'Risk Factors 37'. Requiring a paragraph
    or table within the next 3 blocks skips table-of-contents lines, which are followed by more
    short lines.
    """
    regex = re.compile(rf"^(?:{pattern})$", re.IGNORECASE)
    for i, block in enumerate(blocks[start:], start):
        text = block.text.replace("’", "'").rstrip(".:").strip()
        if block.kind != "text" or not regex.match(text):
            continue
        following = blocks[i + 1 : i + 4]
        if any(b.kind == "table" or len(b.text) > 200 for b in following):
            return i
    return None


def split_sections(blocks: list[Block], form_type: str) -> list[tuple[str, str, list[Block]]]:
    """Group blocks into (section_id, title, blocks).

    The table of contents repeats every heading, so for each section we keep only its LAST
    heading occurrence; everything before the first kept heading becomes 'cover'.
    """
    if form_type == "8-K":
        return [("exhibit_99_1", "Earnings Release", blocks)]

    candidates: dict[str, tuple[int, str]] = {}
    part = None
    last_item = 0
    for i, block in enumerate(blocks):
        if block.kind != "text":
            continue
        if len(block.text) <= MAX_HEADING_CHARS and (pm := PART_RE.match(block.text)):
            part = pm.group(1)
            last_item = 0
            continue
        if len(block.text) <= MAX_HEADING_CHARS and (m := ITEM_RE.match(block.text)):
            number = int(re.match(r"\d+", m.group(1)).group())
            # Item numbers restarting inside Part I means Part II began without a heading.
            if form_type == "10-Q" and part and part.upper() == "I" and number <= last_item:
                part = "II"
            # A 10-Q item's title fixes its part, even when Part headings are missing or
            # misplaced (BofA lists "Part I" and "Part II" right before its MD&A).
            if form_type == "10-Q":
                if PART_I_TITLES.search(m.group(2)):
                    part = "I"
                elif PART_II_TITLES.search(m.group(2)):
                    part = "II"
            last_item = number
        key = heading_key(block.text, part, form_type)
        if key:
            section_id, title = key
            if not title and i + 1 < len(blocks) and len(blocks[i + 1].text) <= MAX_HEADING_CHARS:
                title = blocks[i + 1].text  # title on its own line after "Item 7."
            candidates[section_id] = (i, title)

    fallbacks = FALLBACK_TITLES.get(form_type, {})
    starts = sorted(i for i, _ in candidates.values())
    for section_id, (title, pattern) in fallbacks.items():
        if section_id not in candidates:
            if (i := find_title(blocks, pattern)) is not None:
                candidates[section_id] = (i, title)
            continue
        i, item_title = candidates[section_id]
        end = next((s for s in starts if s > i), len(blocks))
        if sum(len(b.text) for b in blocks[i + 1 : end]) < STUB_CHARS:
            if (t := find_title(blocks, pattern, start=end)) is not None:
                candidates[section_id] = (t, item_title or title)

    starts = sorted((i, sid, title) for sid, (i, title) in candidates.items())
    sections = []
    first = starts[0][0] if starts else len(blocks)
    if first > 0:
        sections.append(("cover", "Cover and Table of Contents", blocks[:first]))
    for n, (i, sid, title) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(blocks)
        sections.append((sid, title, blocks[i + 1 : end]))
    return sections


@dataclass
class Unit:
    text: str
    tokens: int
    page: int
    paragraph: int


def hard_split(text: str, max_tokens: int) -> list[str]:
    """Split text that has no usable sentence breaks at token boundaries."""
    enc = tokenizer().encode(text, add_special_tokens=False)
    pieces = []
    for start in range(0, len(enc.offsets), max_tokens):
        window = enc.offsets[start : start + max_tokens]
        pieces.append(text[window[0][0] : window[-1][1]])
    return pieces


def text_units(blocks: list[Block]) -> list[Unit]:
    units = []
    for p, block in enumerate(blocks):
        for sentence in SENTENCE_RE.split(block.text):
            n = count_tokens(sentence)
            parts = hard_split(sentence, CHUNK_TOKENS) if n > CHUNK_TOKENS else [sentence]
            for part in parts:
                units.append(Unit(part, count_tokens(part), block.page, p))
    return units


def pack(units: list[Unit]) -> list[tuple[str, int]]:
    """Greedily pack units into chunks of <= CHUNK_TOKENS, carrying ~OVERLAP_TOKENS of trailing
    sentences into the next chunk. Returns (text, page of first unit)."""
    chunks: list[tuple[str, int]] = []
    current: list[Unit] = []

    def render(us: list[Unit]) -> str:
        out = ""
        for prev, u in zip([None, *us], us, strict=False):
            sep = "" if prev is None else (" " if prev.paragraph == u.paragraph else "\n")
            out += sep + u.text
        return out

    for unit in units:
        if current and sum(u.tokens for u in current) + unit.tokens > CHUNK_TOKENS:
            chunks.append((render(current), current[0].page))
            overlap: list[Unit] = []
            for u in reversed(current):
                if sum(o.tokens for o in overlap) + u.tokens > OVERLAP_TOKENS:
                    break
                overlap.insert(0, u)
            # Keep overlap only if the new unit still fits after it.
            if sum(u.tokens for u in overlap) + unit.tokens > CHUNK_TOKENS:
                overlap = []
            current = overlap
        current.append(unit)
    if current:
        chunks.append((render(current), current[0].page))
    return chunks


# A body row: a text label, then a cell that is a figure (not a bare year like 2025).
FIGURE_CELL_RE = re.compile(r"^\(?\$?\s*\d[\d,]*(\.\d+)?\s*\)?%?$")
YEAR_RE = re.compile(r"^(19|20)\d\d$")
MAX_HEADER_ROWS = 4


def is_body_row(row: str) -> bool:
    label, *cells = row.split(" | ")
    return bool(re.search(r"[A-Za-z]", label)) and any(
        FIGURE_CELL_RE.match(c.strip()) and not YEAR_RE.match(c.strip()) for c in cells
    )


def table_chunks(block: Block) -> list[tuple[str, int]]:
    """Tables are their own chunks. Big tables split by rows, repeating the header rows.

    Headers often span several rows ("Year Ended" / the dates / "(In millions)"), so every
    leading row before the first body row repeats; repeating only the first row lost the years.
    """
    rows = block.text.split("\n")
    if count_tokens(block.text) <= CHUNK_TOKENS:
        return [(block.text, block.page)]
    n_header = 0
    while n_header < min(MAX_HEADER_ROWS, len(rows) - 1) and not is_body_row(rows[n_header]):
        n_header += 1
    n_header = max(n_header, 1)
    header, body = "\n".join(rows[:n_header]), rows[n_header:]
    budget = CHUNK_TOKENS - count_tokens(header)
    chunks, current, used = [], [], 0
    for row in body:
        n = count_tokens(row)
        if current and used + n > budget:
            chunks.append(("\n".join([header, *current]), block.page))
            current, used = [], 0
        current.append(row)
        used += n
    if current:
        chunks.append(("\n".join([header, *current]), block.page))
    return chunks


# Short text right above a table is usually its title ("Consolidated Statements of Operations").
CAPTION_CHARS = 150
FORM_KINDS = {"10-K": "annual report", "10-Q": "quarterly report", "8-K": "earnings release"}


def context_header(filing: Filing, company: Company, section: str, caption: str | None) -> str:
    """First line of every chunk: where it came from. A table chunk is often just labels and
    numbers; without this, "AMD R&D in 2025" had nothing to match (the AMD income statement
    ranked below 200 in dense search)."""
    if filing.fiscal_year:
        period = f"fiscal {filing.fiscal_year}"
        if filing.fiscal_period != "FY":
            period += f" {filing.fiscal_period}"
    else:
        period = f"period ended {filing.period_of_report}"
    kind = FORM_KINDS[filing.form_type]
    header = f"{company.name} ({filing.ticker}) {filing.form_type} {kind}, {period}. {section}."
    return f"{header} Table: {caption}." if caption else header


def chunk_filing(filing: Filing, company: Company) -> tuple[list[Chunk], list[str]]:
    html = decode((REPO_ROOT / filing.local_path).read_bytes())
    sections = split_sections(extract_blocks(html), filing.form_type)

    chunks: list[Chunk] = []
    for section_id, title, blocks in sections:
        pieces: list[tuple[str, int, bool, str | None]] = []
        run: list[Block] = []  # consecutive text blocks, packed together
        for block in [*blocks, None]:
            if block is not None and block.kind == "text":
                run.append(block)
                continue
            caption = run[-1].text if run and len(run[-1].text) <= CAPTION_CHARS else None
            pieces += [(t, pg, False, None) for t, pg in pack(text_units(run))]
            run = []
            if block is not None:
                pieces += [(t, pg, True, caption) for t, pg in table_chunks(block)]

        for index, (text, page, is_table, caption) in enumerate(pieces):
            header = context_header(filing, company, title, caption)
            chunks.append(
                Chunk(
                    chunk_id=Chunk.make_id(filing.accession_number, section_id, index),
                    text=f"{header}\n{text}",
                    ticker=filing.ticker,
                    company=company.name,
                    form_type=filing.form_type,
                    accession_number=filing.accession_number,
                    filing_date=filing.filing_date,
                    period_of_report=filing.period_of_report,
                    fiscal_year=filing.fiscal_year,
                    fiscal_period=filing.fiscal_period,
                    source_url=filing.source_url,
                    section_id=section_id,
                    section_title=title,
                    chunk_index=index,
                    page=page,
                    is_table=is_table,
                )
            )
    return chunks, [sid for sid, _, _ in sections]


# Sections every filing of this form should have; missing ones are reported.
REQUIRED = {"10-K": {"item_1a", "item_7", "item_8"}, "10-Q": {"part_i_item_1", "part_i_item_2"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers", nargs="*", help="Subset of tickers (default: all)")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR, help="Output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    dirs = sorted(d for d in RAW_DIR.iterdir() if (d / "filings.json").exists())
    if args.tickers:
        dirs = [d for d in dirs if d.name in {t.upper() for t in args.tickers}]

    total = 0
    for d in dirs:
        company = Company.model_validate_json((d / "company.json").read_text(encoding="utf-8"))
        out_dir = args.out / d.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for raw in json.loads((d / "filings.json").read_text(encoding="utf-8")):
            filing = Filing.model_validate(raw)
            chunks, section_ids = chunk_filing(filing, company)
            out = out_dir / f"{filing.accession_number}.jsonl"
            out.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
            missing = REQUIRED.get(filing.form_type, set()) - set(section_ids)
            level = logging.WARNING if missing else logging.INFO
            log.log(
                level,
                "%s %s %s: %d chunks, %d sections%s",
                filing.ticker,
                filing.form_type,
                filing.period_of_report,
                len(chunks),
                len(section_ids),
                f", MISSING {sorted(missing)}" if missing else "",
            )
            total += len(chunks)
    log.info("Total chunks: %d", total)


if __name__ == "__main__":
    main()
