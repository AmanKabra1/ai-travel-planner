"""Smoke test: verify the weather MCP server starts and exposes its tools."""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Resolve paths relative to this file so the test runs from any directory.
HERE = Path(__file__).parent.parent  # Multi_agent_system_MCP/
load_dotenv(HERE / ".env", override=True)

from langchain_mcp_adapters.client import MultiServerMCPClient

WEATHER_SERVER = str(HERE / "weather_mcp_server.py")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

client = MultiServerMCPClient({
    "weather": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [WEATHER_SERVER],
        "env": {"OPENWEATHER_API_KEY": OPENWEATHER_API_KEY},
    }
})


async def main():
    print("Connecting to weather MCP server...")
    tools = await client.get_tools()

    print(f"\nAvailable tools ({len(tools)}):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    print("\nCalling get_current_weather for Tokyo...")
    weather_tool = next(t for t in tools if t.name == "get_current_weather")
    result = await weather_tool.ainvoke({"city": "Tokyo"})
    print("Result:", result)


if __name__ == "__main__":
    asyncio.run(main())
