# Weather MCP server — uses wttr.in (free, no API key required).

import sys

from mcp.server.fastmcp import FastMCP
import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

mcp = FastMCP("Weather Server")

_HEADERS = {"User-Agent": "ai-travel-planner/1.0"}
_TIMEOUT = 10


@mcp.tool()
def get_current_weather(city: str) -> dict:
    """Return current weather conditions for a city (no API key needed)."""
    try:
        resp = requests.get(
            f"https://wttr.in/{city}?format=j1",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data    = resp.json()
        current = data["current_condition"][0]
        area    = data.get("nearest_area", [{}])[0]
        name    = area.get("areaName", [{}])[0].get("value", city)

        return {
            "city":          name,
            "temperature_c": int(current["temp_C"]),
            "feels_like_c":  int(current["FeelsLikeC"]),
            "humidity_%":    int(current["humidity"]),
            "condition":     current["weatherDesc"][0]["value"],
            "wind_speed_kmh":int(current["windspeedKmph"]),
            "visibility_km": int(current["visibility"]),
        }
    except Exception as exc:
        return {"error": str(exc), "city": city}


@mcp.tool()
def get_forecast(city: str) -> dict:
    """Return a 3-day weather forecast for a city (no API key needed)."""
    try:
        resp = requests.get(
            f"https://wttr.in/{city}?format=j1",
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        forecast = []
        for day in data.get("weather", []):
            hourly = day.get("hourly", [])
            # Pick noon slot (index 4 out of 8 three-hourly slots)
            noon = hourly[4] if len(hourly) > 4 else (hourly[0] if hourly else {})
            forecast.append({
                "date":        day["date"],
                "max_temp_c":  int(day["maxtempC"]),
                "min_temp_c":  int(day["mintempC"]),
                "condition":   day["hourly"][0]["weatherDesc"][0]["value"] if day.get("hourly") else "",
                "rain_mm":     float(day.get("hourly", [{}])[0].get("precipMM", 0)),
                "sunrise":     day.get("astronomy", [{}])[0].get("sunrise", ""),
                "sunset":      day.get("astronomy", [{}])[0].get("sunset", ""),
            })

        return {"city": city, "forecast": forecast}
    except Exception as exc:
        return {"error": str(exc), "city": city}


if __name__ == "__main__":
    mcp.run()
