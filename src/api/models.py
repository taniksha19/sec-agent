"""FastAPI request and response models: the contract with the gateway and any client.

Change them only on main.
"""

from typing import Any

from pydantic import BaseModel, Field

from ingest.models import FiscalPeriod, FormType


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Citation(BaseModel):
    n: int = Field(description="The [n] used in the answer text")
    chunk_id: str
    company: str
    ticker: str
    form_type: FormType
    fiscal_year: int | None
    fiscal_period: FiscalPeriod | None
    section_title: str
    source_url: str
    page: int | None = None
    quote: str = Field(description="Start of the cited chunk")


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]
    ok: bool


class Usage(BaseModel):
    llm_calls: int
    input_tokens: int
    output_tokens: int


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    sub_questions: list[str] = Field(description="How the planner split the question")
    rounds: int = Field(description="Most retrieval rounds any sub-question needed (max 3)")
    tool_calls: list[ToolCall]
    usage: Usage
    latency_ms: int
