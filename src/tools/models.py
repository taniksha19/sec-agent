"""Tool inputs and outputs: the contract between the agent, the MCP servers and the tools.

Change them only on main. Every tool is a plain function from one input model to one output
model, so the LangGraph agent and the MCP servers wrap the same code.
"""

from pydantic import BaseModel, Field

from ingest.models import FiscalPeriod, FormType


class SourceChunk(BaseModel):
    """A retrieved chunk with what a citation needs."""

    chunk_id: str
    ticker: str
    company: str
    form_type: FormType
    fiscal_year: int | None
    fiscal_period: FiscalPeriod | None
    section_title: str
    source_url: str
    page: int | None = None
    text: str


class SearchFilingsInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    tickers: list[str] | None = Field(default=None, description="Only these companies")
    fiscal_year: int | None = None
    form_type: FormType | None = None
    k: int = Field(default=5, ge=1, le=10)


class SearchFilingsOutput(BaseModel):
    chunks: list[SourceChunk]


class LookupTableInput(BaseModel):
    ticker: str
    fiscal_year: int
    fiscal_period: FiscalPeriod | None = Field(default=None, description="None means any")
    metric: str = Field(
        min_length=1, max_length=200, description="Row label, e.g. 'research and development'"
    )


class TableRow(BaseModel):
    label: str
    header: list[str] = Field(description="Column headers, e.g. ['2026', '2025', '2024']")
    values: list[str] = Field(description="Cells as written in the filing")
    source: SourceChunk


class LookupTableOutput(BaseModel):
    rows: list[TableRow]


class CalculateInput(BaseModel):
    expression: str = Field(
        min_length=1, max_length=200, description="Arithmetic only: numbers, + - * / ( )"
    )


class CalculateOutput(BaseModel):
    value: float


class CompareInput(BaseModel):
    metric: str
    values: dict[str, dict[int, float]] = Field(
        description="ticker -> fiscal year -> value, e.g. {'NVDA': {2024: 8675, 2026: 18497}}"
    )


class CompanyChange(BaseModel):
    ticker: str
    first_year: int
    last_year: int
    first_value: float
    last_value: float
    change_pct: float | None = Field(description="None when the first value is 0")


class CompareOutput(BaseModel):
    metric: str
    changes: list[CompanyChange] = Field(description="Sorted by change_pct, highest first")
