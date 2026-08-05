"""Smoke test: verify the AviationStack MCP server starts and exposes its tools."""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Resolve paths relative to this file so the test runs from any directory.
HERE = Path(__file__).parent.parent  # Multi_agent_system_MCP/
load_dotenv(HERE / ".env", override=True)

from langchain_mcp_adapters.client import MultiServerMCPClient

AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY") or os.getenv("AVIATIONSTACK_API_KEY")
)

# Dedicated venv created by `uv sync` inside aviationstack-mcp/.
# Can be overridden via AVIATION_MCP_PYTHON env var (e.g. on a VPS).
if os.name == "nt":
    _default = HERE / "aviationstack-mcp" / ".venv" / "Scripts" / "python.exe"
else:
    _default = HERE / "aviationstack-mcp" / ".venv" / "bin" / "python"

AVIATION_PYTHON = os.getenv("AVIATION_MCP_PYTHON", str(_default))

client = MultiServerMCPClient({
    "aviationstack": {
        "transport": "stdio",
        "command": AVIATION_PYTHON,
        "args": ["-m", "aviationstack_mcp", "mcp", "run"],
        "env": {"AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY},
    }
})


async def main():
    print("Connecting to AviationStack MCP server...")
    tools = await client.get_tools()

    print(f"\nAvailable tools ({len(tools)}):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    print("\nCalling list_airports for Tokyo...")
    airports_tool = next(t for t in tools if t.name == "list_airports")
    result = await airports_tool.ainvoke({"search": "Tokyo", "limit": 5, "offset": 0})
    print("Result:", result)


if __name__ == "__main__":
    asyncio.run(main())
