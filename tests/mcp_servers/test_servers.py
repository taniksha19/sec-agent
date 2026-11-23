"""The MCP servers expose the same tools as the agent, over the MCP protocol (in memory)."""

import pytest
from mcp import Client

from mcp_servers import filings_server, math_server
from retrieval.base import SearchResult
from tests.tools.test_tools import TABLE, FakeRetrieve, chunk

COMPANIES = {"AMD": "Advanced Micro Devices", "NVDA": "NVIDIA"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def filings(results=None):
    fake = FakeRetrieve(results or [SearchResult(chunk=chunk(TABLE), score=1.0)])
    return filings_server.create_server(retrieve=fake, companies=COMPANIES), fake


@pytest.mark.anyio
async def test_tools_are_listed_read_only():
    server, _ = filings()
    for srv, names in [
        (server, {"search_filings", "lookup_table"}),
        (math_server.create_server(), {"calculate", "compare"}),
    ]:
        async with Client(srv) as client:
            tools = (await client.list_tools()).tools
            assert {t.name for t in tools} == names
            assert all(t.annotations.read_only_hint for t in tools)
    async with Client(server) as client:
        for tool in (await client.list_tools()).tools:
            assert tool.annotations.open_world_hint  # third-party text
            assert "untrusted" in tool.description
            assert tool.input_schema["properties"]["fiscal_year"]  # limits are published


@pytest.mark.anyio
async def test_calculate_and_compare():
    async with Client(math_server.create_server()) as client:
        res = await client.call_tool("calculate", {"expression": "(12,914 - 7,339) / 7,339 * 100"})
        assert not res.is_error
        assert round(res.structured_content["value"], 1) == 76.0

        res = await client.call_tool(
            "compare",
            {
                "metric": "R&D",
                "values": [
                    {"ticker": "AMD", "fiscal_year": 2023, "value": 5872},
                    {"ticker": "AMD", "fiscal_year": 2025, "value": 8091},
                    {"ticker": "NVDA", "fiscal_year": 2023, "value": 7339},
                    {"ticker": "NVDA", "fiscal_year": 2025, "value": 12914},
                ],
            },
        )
        assert [c["ticker"] for c in res.structured_content["changes"]] == ["NVDA", "AMD"]


@pytest.mark.anyio
async def test_bad_input_is_a_tool_error_not_a_crash():
    async with Client(math_server.create_server()) as client:
        res = await client.call_tool("calculate", {"expression": "__import__('os')"})
        assert res.is_error
        # The server keeps working after an error.
        assert not (await client.call_tool("calculate", {"expression": "1+1"})).is_error


@pytest.mark.anyio
async def test_lookup_table_returns_rows_with_source():
    server, fake = filings()
    async with Client(server) as client:
        res = await client.call_tool(
            "lookup_table",
            {"ticker": "amd", "fiscal_year": 2025, "metric": "research and development"},
        )
    assert not res.is_error
    row = res.structured_content["rows"][0]
    assert row["values"] == ["8,091", "6,456"]
    assert row["source"]["source_url"] == "https://example.com"
    assert fake.calls[0][2] == {"ticker": "AMD", "fiscal_year": 2025, "is_table": True}


@pytest.mark.anyio
async def test_search_filters_and_unknown_tickers():
    server, fake = filings()
    async with Client(server) as client:
        res = await client.call_tool(
            "search_filings", {"query": "R&D", "tickers": ["nvda"], "form_type": "10-K", "k": 3}
        )
        assert not res.is_error
        assert fake.calls[-1] == ("R&D", 3, {"ticker": ["NVDA"], "form_type": "10-K"})

        res = await client.call_tool("search_filings", {"query": "R&D", "tickers": ["XXXX"]})
        assert res.is_error
        res = await client.call_tool("search_filings", {"query": "R&D", "k": 50})
        assert res.is_error
    assert len(fake.calls) == 1  # rejected calls never reach retrieval


@pytest.mark.anyio
async def test_instructions_mark_filing_text_untrusted():
    server, _ = filings()
    assert "untrusted" in server.instructions


@pytest.mark.anyio
async def test_errors_are_readable_and_skip_retrieval():
    server, fake = filings()
    async with Client(server) as client:
        res = await client.call_tool(
            "lookup_table", {"ticker": "XXXX", "fiscal_year": 2025, "metric": "r"}
        )
        assert res.is_error and "available: AMD, NVDA" in res.content[0].text
        res = await client.call_tool("search_filings", {"query": "q", "k": 50})
        assert res.is_error and "less than or equal to 10" in res.content[0].text
    assert fake.calls == []


@pytest.mark.anyio
async def test_paid_calls_are_rate_limited(monkeypatch):
    monkeypatch.setattr(filings_server, "MAX_CALLS_PER_MINUTE", 2)
    server, fake = filings()
    async with Client(server) as client:
        args = {"query": "R&D"}
        assert not (await client.call_tool("search_filings", args)).is_error
        assert not (await client.call_tool("search_filings", args)).is_error
        res = await client.call_tool("search_filings", args)
        assert res.is_error and "rate limit" in res.content[0].text
    assert len(fake.calls) == 2


@pytest.mark.anyio
async def test_out_of_range_math_is_an_error():
    async with Client(math_server.create_server()) as client:
        for expr in ["1e308*10", "1e308*10-1e308*10"]:
            res = await client.call_tool("calculate", {"expression": expr})
            assert res.is_error
