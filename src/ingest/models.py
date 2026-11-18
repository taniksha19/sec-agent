"""Shared data models for filings and chunks.

These are the contract between ingestion and retrieval. Change them only on main.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

FormType = Literal["10-K", "10-Q", "8-K"]
FiscalPeriod = Literal["FY", "Q1", "Q2", "Q3", "Q4"]


class Company(BaseModel):
    ticker: str
    name: str
    sector: str
    cik: str = Field(description="10-digit zero-padded CIK, from EDGAR")
    fiscal_year_end: str | None = Field(
        default=None, description="MMDD as reported by EDGAR, e.g. '0928' for Apple"
    )


class Filing(BaseModel):
    """One downloaded document. For 8-Ks this is the Exhibit 99.1 earnings release."""

    ticker: str
    cik: str
    form_type: FormType
    accession_number: str = Field(description="e.g. '0000320193-24-000123'")
    filing_date: date
    period_of_report: date | None = None
    fiscal_year: int | None = Field(default=None, description="From EDGAR XBRL metadata")
    fiscal_period: FiscalPeriod | None = None
    document_name: str = Field(description="File name of the downloaded document")
    source_url: str
    local_path: str = Field(description="Path relative to the repo root")


class Chunk(BaseModel):
    """A retrievable unit of text. Metadata is denormalized so search filters need no joins."""

    chunk_id: str = Field(description="Deterministic: '{accession_number}:{section_id}:{index}'")
    text: str

    # Filing metadata
    ticker: str
    company: str
    form_type: FormType
    accession_number: str
    filing_date: date
    period_of_report: date | None = None
    fiscal_year: int | None = None
    fiscal_period: FiscalPeriod | None = None
    source_url: str

    # Position within the filing
    section_id: str = Field(description="Normalized, e.g. 'item_1a', 'item_7', 'exhibit_99_1'")
    section_title: str = Field(description="As written in the filing, e.g. 'Risk Factors'")
    chunk_index: int = Field(description="Order of this chunk within its section, from 0")
    page: int | None = Field(default=None, description="Only when the filing marks page breaks")
    is_table: bool = False

    @staticmethod
    def make_id(accession_number: str, section_id: str, index: int) -> str:
        return f"{accession_number}:{section_id}:{index}"
