"""MCP server `sec-math`: calculate and compare. Pure functions: no keys, network or data.

Run (stdio, for MCP clients such as Claude Desktop):
    uv run python -m mcp_servers.math_server
"""

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from tools.calculator import calculate as run_calculate
from tools.compare import compare as run_compare
from tools.models import CalculateInput, CalculateOutput, CompareInput, CompareOutput, DataPoint

PURE = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


def create_server() -> MCPServer:
    server = MCPServer("sec-math", instructions="Exact arithmetic for financial figures.")

    @server.tool(annotations=PURE)
    def calculate(expression: str) -> CalculateOutput:
        """Evaluate arithmetic: numbers, + - * / and parentheses. Commas and $ are ignored,
        e.g. "(12,914 - 7,339) / 7,339 * 100"."""
        return run_calculate(CalculateInput(expression=expression))

    @server.tool(annotations=PURE)
    def compare(metric: str, values: list[DataPoint]) -> CompareOutput:
        """Rank companies by percent change in a metric from their first to last year.
        values: one item per company and year, e.g. {"ticker": "NVDA", "fiscal_year": 2024,
        "value": 8675}."""
        return run_compare(CompareInput(metric=metric, values=values))

    return server


if __name__ == "__main__":
    create_server().run()
