from datetime import date

import pytest

from ingest.models import Chunk


def make_chunk(ticker: str, form_type: str, fiscal_year: int, section_id: str, text: str) -> Chunk:
    accession = f"0000000000-{fiscal_year % 100:02d}-{ticker}"
    return Chunk(
        chunk_id=Chunk.make_id(accession, section_id, 0),
        text=text,
        ticker=ticker,
        company=f"{ticker} Inc.",
        form_type=form_type,
        accession_number=accession,
        filing_date=date(fiscal_year, 11, 1),
        fiscal_year=fiscal_year,
        fiscal_period="FY",
        source_url=f"https://www.sec.gov/Archives/{accession}.htm",
        section_id=section_id,
        section_title="Risk Factors" if section_id == "item_1a" else "MD&A",
        chunk_index=0,
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk("AAPL", "10-K", 2024, "item_1a", "Apple supply chain risk."),
        make_chunk("AAPL", "10-K", 2023, "item_7", "Apple revenue discussion."),
        make_chunk("MSFT", "10-K", 2024, "item_1a", "Microsoft cloud competition risk."),
        make_chunk("NVDA", "10-Q", 2024, "item_7", "Nvidia data center growth."),
    ]
