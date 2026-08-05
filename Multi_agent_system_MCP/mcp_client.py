import os
import sys
import traceback

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import (
    AVIATION_STACK_API_KEY,
    OPENWEATHER_API_KEY,
    TAVILY_API_KEY,
)

# ---------------------------------------------------------------------------
# Paths are resolved relative to this file so the project runs on any machine
# (Windows or Linux/VPS) instead of relying on hardcoded absolute paths.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

# The weather server only needs mcp + requests, which live in the same
# interpreter that runs this app, so reuse the current Python executable.
APP_PYTHON = sys.executable

# Dedicated venv that has the aviationstack_mcp package installed.
# Can be overridden via the AVIATION_MCP_PYTHON env var (useful on a VPS
# where the aviationstack-mcp repo may live outside the project folder).
if os.name == "nt":
    _default_aviation_python = os.path.join(HERE, "aviationstack-mcp", ".venv", "Scripts", "python.exe")
else:
    _default_aviation_python = os.path.join(HERE, "aviationstack-mcp", ".venv", "bin", "python")

AVIATION_PYTHON = os.getenv("AVIATION_MCP_PYTHON", _default_aviation_python)

WEATHER_SERVER = os.path.join(HERE, "weather_mcp_server.py")

# Create MCP Client
client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        },

        "aviationstack": {
            "transport": "stdio",
            "command": AVIATION_PYTHON,
            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run",
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY,
            },
        },

        "weather": {
            "transport": "stdio",
            "command": APP_PYTHON,
            "args": [
                WEATHER_SERVER,
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY,
            },
        },
    }
)

# Cache tools so we don't load them repeatedly
_tools_cache = None


async def get_tools():
    global _tools_cache

    if _tools_cache is None:
        try:
            _tools_cache = await client.get_tools()

        except Exception as e:
            print("\n========== FULL TRACEBACK ==========")
            traceback.print_exc()

            if hasattr(e, "exceptions"):
                print("\n========== SUB EXCEPTIONS ==========")

                for i, sub in enumerate(e.exceptions):
                    print(f"\n--- Exception {i + 1} ---")
                    traceback.print_exception(
                        type(sub),
                        sub,
                        sub.__traceback__,
                    )

            raise

    return _tools_cache


async def call_tool(tool_name: str, args: dict = None):
    tools = await get_tools()

    tool = next(
        (tool for tool in tools if tool.name == tool_name),
        None,
    )

    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found")

    return await tool.ainvoke(args or {})


# ------------------------
# Tavily MCP Tools
# ------------------------

async def tavily_search(query: str):
    return await call_tool(
        "tavily_search",
        {"query": query},
    )


# ------------------------
# AviationStack MCP Tools
# ------------------------

async def list_airports(search: str = "", limit: int = 10):
    return await call_tool(
        "list_airports",
        {"search": search, "limit": limit, "offset": 0},
    )


async def list_airlines(search: str = "", limit: int = 10):
    return await call_tool(
        "list_airlines",
        {"search": search, "limit": limit, "offset": 0},
    )


# ------------------------
# Weather MCP Tools
# ------------------------

async def current_weather(city: str):
    return await call_tool(
        "get_current_weather",
        {"city": city},
    )


async def forecast(city: str):
    return await call_tool(
        "get_forecast",
        {"city": city},
    )
