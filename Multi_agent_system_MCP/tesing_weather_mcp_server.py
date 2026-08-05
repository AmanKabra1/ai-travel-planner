import os
import sys
from dotenv import load_dotenv
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
load_dotenv(override=True)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

HERE = os.path.dirname(os.path.abspath(__file__))
WEATHER_SERVER_SCRIPT = os.path.join(HERE, "custom_weather_mcp_server.py")

client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                WEATHER_SERVER_SCRIPT
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY
            }
        }
    }
)

async def main():

    print("Loading tools...")

    tools = await client.get_tools()

    print("Tools loaded!")

    for tool in tools:
        print(tool.name)

if __name__ == "__main__":
    asyncio.run(main())