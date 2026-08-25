import asyncio
import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import GROQ_FALLBACKS, get_llm, get_retry_models
from mcp_client import current_weather, forecast, list_airlines, list_airports, tavily_search
from nearby_api import (
    bucket_pois,
    format_pois_text,
    geocode_city,
    overpass_nearby,
    wikivoyage_local_tips,
)
from state import TravelState

logger = logging.getLogger(__name__)

llm = get_llm()


# ── Utilities ─────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove <think>…</think> chain-of-thought blocks from reasoning models."""
    return _THINK_RE.sub("", text).strip()


def _llm_text(system: str, prompt: str) -> str:
    """Call the LLM, auto-switching to the next model on any API error.

    Uses the live Groq model list (fetched at startup) so we only try real,
    current model IDs — not stale ones from the static GROQ_FALLBACKS list.
    Strips <think> blocks produced by reasoning models (e.g. DeepSeek R1).
    """
    global llm
    msgs = [SystemMessage(content=system), HumanMessage(content=prompt)]
    retry_list = get_retry_models()   # live IDs from Groq /models API
    tried: set[str] = set()
    for _ in range(len(retry_list) + 1):
        current = getattr(llm, "model_name", "") or getattr(llm, "model", "")
        try:
            return _strip_think(llm.invoke(msgs).content)
        except Exception as exc:
            tried.add(current)
            next_models = [m for m in retry_list if m not in tried]
            if not next_models:
                raise RuntimeError(f"All Groq models failed. Last: {current} — {exc}") from exc
            nxt = next_models[0]
            logger.warning("Groq model %s failed (%s); switching to %s",
                           current, str(exc)[:80], nxt)
            llm = get_llm(nxt)
    raise RuntimeError("All Groq models exhausted")


def _extract_text(mcp_result) -> str:
    """Normalise MCP tool output to plain text.

    MCP tools may return a bare string or a list of content blocks
    like [{"type": "text", "text": "..."}].  Both shapes end up as a
    single string so every agent can treat the result uniformly.
    """
    if isinstance(mcp_result, str):
        return mcp_result
    if isinstance(mcp_result, list):
        parts = []
        for block in mcp_result:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(mcp_result)


def _parse_json(text: str) -> dict:
    """Extract and parse the first JSON object found in *text*."""
    start = text.index("{")
    end   = text.rindex("}") + 1
    return json.loads(text[start:end])


def _extract_tavily_with_urls(raw) -> str:
    """Parse Tavily MCP output and return markdown-formatted results with links."""
    text = _extract_text(raw)
    try:
        data = json.loads(text)
        results = data.get("results", [])
        if results:
            parts = []
            for r in results[:6]:
                title   = r.get("title", "Result")
                url     = r.get("url", "")
                content = r.get("content", "")[:450]
                if url:
                    parts.append(f"**[{title}]({url})**\n{content}")
                else:
                    parts.append(f"**{title}**\n{content}")
            return "\n\n".join(parts)
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass
    return text


async def _parallel_tavily(*queries: str) -> list[str]:
    """Run multiple Tavily searches concurrently and return formatted strings."""
    raw_list = await asyncio.gather(*[tavily_search(q) for q in queries], return_exceptions=True)
    out = []
    for raw in raw_list:
        if isinstance(raw, Exception):
            logger.warning("Parallel Tavily search failed: %s", raw)
            out.append("")
        else:
            out.append(_extract_tavily_with_urls(raw)[:1500])
    return out


# ── Supervisor ────────────────────────────────────────────────────────────────

def supervisor_agent(state: TravelState):
    query = state["user_query"]
    logger.info("Supervisor received query: %s", query[:80])

    # ── Input guardrail ──────────────────────────────────────────────────────
    guardrail_raw = _llm_text(
        "You are an input validation guardrail. Return strict JSON only.",
        f"""Decide whether the request is a valid travel-planning request.
Return only JSON:
{{
    "allowed": true,
    "reason": ""
}}

User request:
{query}""",
    )

    try:
        guardrail = _parse_json(guardrail_raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Guardrail JSON parse failed: %s", exc)
        guardrail = {"allowed": True, "reason": ""}

    logger.debug("Guardrail result: %s", guardrail)

    if not guardrail.get("allowed", True):
        reason = guardrail.get("reason", "Request rejected by input guardrail.")
        logger.info("Guardrail blocked request: %s", reason)
        return {
            "selected_agents":      [],
            "trip_constraints":     {},
            "supervisor_reasoning": reason,
            "final_response":       reason,
            "messages":             [AIMessage(content=f"Blocked: {reason}")],
            "llm_calls":            state.get("llm_calls", 0) + 1,
        }

    # ── Routing decision ─────────────────────────────────────────────────────
    routing_raw = _llm_text(
        "You route work to specialist agents. Return strict JSON only.",
        f"""You are the supervisor of a multi-agent travel planning system.

Decide which specialist agents are needed for the user request.

Available agents:
- flight_agent    : flights, airports, airlines, routes, airfare
- transport_agent : trains, buses, ground transport, overland routes between cities
- hotel_agent     : hotels, accommodation, neighbourhood guides
- weather_agent   : weather, climate, seasonal advice, packing
- nearby_agent    : nearby attractions, temples, parks, local food, points of interest
- budget_agent    : cost breakdown, affordability, money-saving tips
- itinerary_agent : always include — produces the actual travel plan

IMPORTANT: If the request mentions both an origin city AND a destination city, ALWAYS include transport_agent to find trains and buses.

Return ONLY JSON:
{{
  "selected_agents": ["flight_agent", "transport_agent", "hotel_agent", "weather_agent", "nearby_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}""",
    )

    try:
        parsed = _parse_json(routing_raw)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Routing JSON parse failed: %s. Defaulting all agents.", exc)
        parsed = {
            "selected_agents":  ["flight_agent", "transport_agent", "hotel_agent",
                                  "weather_agent", "nearby_agent", "budget_agent",
                                  "itinerary_agent"],
            "trip_constraints": {},
            "reasoning":        "Default routing (parse error).",
        }

    logger.info("Selected agents: %s", parsed.get("selected_agents"))

    # Merge LLM-extracted constraints on top of the frontend-provided ones so that
    # start_date, end_date, members, transport_mode, interests are never lost.
    existing_constraints = state.get("trip_constraints", {}) or {}
    llm_constraints      = parsed.get("trip_constraints", {}) or {}
    merged_constraints   = dict(existing_constraints)
    for k, v in llm_constraints.items():
        if v not in (None, "", [], "Not specified"):
            merged_constraints[k] = v

    # Force transport_agent when both origin and destination are given
    selected = parsed.get("selected_agents", [])
    if merged_constraints.get("origin") and merged_constraints.get("destination"):
        if "transport_agent" not in selected:
            # Insert after flight_agent (or at position 1)
            idx = selected.index("flight_agent") + 1 if "flight_agent" in selected else 1
            selected.insert(idx, "transport_agent")

    return {
        "selected_agents":      selected,
        "trip_constraints":     merged_constraints,
        "supervisor_reasoning": parsed.get("reasoning", ""),
        "messages":             [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls":            state.get("llm_calls", 0) + 1,
    }


# ── Flight agent ──────────────────────────────────────────────────────────────

def flight_agent(state: TravelState):
    query       = state["user_query"]
    constraints = state.get("trip_constraints", {})
    destination = constraints.get("destination", "")
    logger.info("Flight agent — destination: %s", destination)

    try:
        airports = asyncio.run(list_airports(destination, limit=10))
        airlines = asyncio.run(list_airlines("", limit=10))
        airport_text = _extract_text(airports)[:3000]
        airline_text = _extract_text(airlines)[:3000]
    except Exception as exc:
        logger.warning("Flight MCP call failed: %s", exc)
        airport_text = airline_text = "Live data unavailable."

    result = _llm_text(
        "You are a flight planning specialist.",
        f"""Create flight guidance for this trip.

User request:
{query}

Trip constraints:
{constraints}

Airport data:
{airport_text}

Airline data:
{airline_text}

Cover: likely airports, relevant airlines, estimated duration,
fare range, peak-season warnings, and booking tips.
""",
    )

    return {
        "flight_results": result,
        "messages":       [AIMessage(content="Flight agent completed.")],
        "llm_calls":      state.get("llm_calls", 0) + 1,
    }


# ── Hotel agent ───────────────────────────────────────────────────────────────

def hotel_agent(state: TravelState):
    destination  = (state.get("trip_constraints") or {}).get("destination", "")
    search_query = f"Best hotels to book in {destination} for: {state['user_query']}"
    logger.info("Hotel agent — query: %s", search_query[:80])

    try:
        raw          = asyncio.run(tavily_search(search_query))
        search_text  = _extract_tavily_with_urls(raw)
    except Exception as exc:
        logger.warning("Hotel Tavily search failed: %s", exc)
        search_text = "Live search unavailable."

    result = _llm_text(
        "You are a hotel recommendation specialist.",
        f"""Based on these web search results, recommend the best hotels and areas to stay.

User request:
{state['user_query']}

Search results (with booking links):
{search_text[:4500]}

Return a clean, readable markdown summary:
- Group by area/neighbourhood
- List 2–4 specific hotels per area with name, price range, vibe
- For each hotel, include the booking/source link as a Markdown hyperlink like [Hotel Name](URL)
- Include one direct booking site link per hotel where available (Booking.com, Hotels.com, Agoda, etc.)
Do not output raw JSON.
""",
    )

    return {
        "hotel_results": result,
        "messages":      [AIMessage(content="Hotel agent completed.")],
        "llm_calls":     state.get("llm_calls", 0) + 1,
    }


# ── Transport agent ──────────────────────────────────────────────────────────

def transport_agent(state: TravelState):
    constraints = state.get("trip_constraints", {})
    origin      = constraints.get("origin", "")
    destination = constraints.get("destination", "")
    logger.info("Transport agent — %s → %s", origin, destination)

    route_str = f"{origin} to {destination}" if origin else destination

    # ── All 4 transport searches run in parallel ──────────────────────────────
    train_direct, train_via, bus_search, general = asyncio.run(_parallel_tavily(
        f"train {route_str} 2026 direct schedule timing fare IRCTC booking",
        f"how to reach {destination} from {origin} by train 2026 via connecting junction route change",
        f"bus {route_str} 2026 Redbus Abhibus schedule timing booking fare",
        f"{route_str} travel options 2026 how to reach fastest cheapest transport",
    ))

    result = _llm_text(
        "You are a ground transport specialist. You always show real schedules, fares, and booking links.",
        f"""Provide a comprehensive transport guide for this route.

User request: {state['user_query']}
Route: {route_str}

Direct train search results:
{train_direct}

Connecting / via-route train results:
{train_via}

Bus search results:
{bus_search}

General transport search:
{general}

Return a clean markdown guide with these sections:

## ✈️ Flights
Mention if any direct/connecting flight exists for this route.

## 🚂 Trains
### Direct Trains
List every direct train found: name/number, departure time, arrival time, duration, fare (sleeper/3AC/2AC), [Book on IRCTC](https://www.irctc.co.in)
### Connecting Trains (if no direct)
If there is no direct train, show the best via-route:
- Leg 1: [Train Name] — Origin → Junction City — depart HH:MM, arrive HH:MM, X hrs, Rs. Y (class)
- Leg 2: [Train Name] — Junction City → Destination — depart HH:MM, arrive HH:MM, X hrs, Rs. Y (class)
- Total journey time including connection wait
- [Book Leg 1 on IRCTC](https://www.irctc.co.in) | [Book Leg 2 on IRCTC](https://www.irctc.co.in)

## 🚌 Buses
List direct buses AND multi-leg bus options:
- Operator with [Book Now](URL), departure time, duration, fare
- If no direct bus: show via-route (e.g. City A → City B → City C)

## 🚗 Self-Drive / Cab
- Approximate road distance (km) and drive time
- Highway route (e.g. NH-X via City Y)
- Estimated fuel cost + toll cost
- Ola/Uber inter-city option if available

## 💡 Best Option Summary
One-line recommendation for each budget tier (budget / mid / premium).

Use real train names and numbers from the search results. For Indian routes, always mention IRCTC for booking.
Do not output raw JSON.
""",
    )

    return {
        "transport_results": result,
        "messages":          [AIMessage(content="Transport agent completed.")],
        "llm_calls":         state.get("llm_calls", 0) + 1,
    }


# ── Weather agent ─────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract the first JSON object from *text* (handles prefix/suffix prose)."""
    import json as _j
    # Try direct parse first
    try:
        return _j.loads(text)
    except Exception:
        pass
    # Regex: find the outermost { … }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return _j.loads(m.group())
    raise ValueError("No JSON object found in text")


def _fmt_weather(now_raw: str, fore_raw: str) -> str:
    """Format raw MCP weather text (JSON or prose+JSON) into readable markdown."""
    lines: list[str] = []

    # ── Current conditions ──────────────────────────────────────────────────
    try:
        now        = _extract_json(now_raw)
        city       = now.get("city", "")
        cond       = now.get("condition", "-")
        temp       = now.get("temperature_c", "-")
        feels      = now.get("feels_like_c", "-")
        humidity   = now.get("humidity_%", "-")
        wind       = now.get("wind_speed_kmh", "-")
        visibility = now.get("visibility_km", "-")
        lines.append(f"## Weather in {city}\n")
        lines.append("### Current Conditions")
        lines.append("| | |")
        lines.append("|---|---|")
        lines.append(f"| Condition     | {cond} |")
        lines.append(f"| Temperature   | {temp} C (feels like {feels} C) |")
        lines.append(f"| Humidity      | {humidity}% |")
        lines.append(f"| Wind speed    | {wind} km/h |")
        lines.append(f"| Visibility    | {visibility} km |")
    except Exception:
        lines.append("### Current Conditions")
        lines.append(now_raw)

    lines.append("")

    # ── Forecast ────────────────────────────────────────────────────────────
    try:
        fore = _extract_json(fore_raw)
        days = fore.get("forecast", [])
        if days:
            lines.append("### Forecast (next 3 days)")
            lines.append("| Date | Max | Min | Condition | Rain |")
            lines.append("|------|-----|-----|-----------|------|")
            for d in days:
                date = d.get("date", "")
                hi   = d.get("max_temp_c", "-")
                lo   = d.get("min_temp_c", "-")
                cond = d.get("condition", "-")
                rain = d.get("rain_mm", 0)
                lines.append(f"| {date} | {hi} C | {lo} C | {cond} | {rain} mm |")
            lines.append("")
            sr = days[0].get("sunrise", "")
            ss = days[0].get("sunset", "")
            if sr or ss:
                lines.append(f"Sunrise: {sr}  |  Sunset: {ss}")
    except Exception:
        lines.append("### Forecast")
        lines.append(fore_raw)

    return "\n".join(lines)


def weather_agent(state: TravelState):
    city = (state.get("trip_constraints") or {}).get("destination", "")
    logger.info("Weather agent — city: %s", city)

    try:
        weather_now  = _extract_text(asyncio.run(current_weather(city)))
        weather_fore = _extract_text(asyncio.run(forecast(city)))
        result = _fmt_weather(weather_now, weather_fore)
    except Exception as exc:
        logger.warning("Weather MCP call failed: %s", exc)
        result = "Live weather data unavailable."

    return {
        "weather_results": result,
        "messages":        [AIMessage(content="Weather agent completed.")],
    }


# ── Nearby attractions agent ──────────────────────────────────────────────────

def nearby_agent(state: TravelState):
    destination = (state.get("trip_constraints") or {}).get("destination", "")
    logger.info("Nearby agent — destination: %s", destination)

    if not destination:
        return {
            "nearby_results": "No destination specified — nearby attractions skipped.",
            "messages":       [AIMessage(content="Nearby agent skipped (no destination).")],
        }

    # ── Geocode ──────────────────────────────────────────────────────────────
    geo = geocode_city(destination)
    if not geo:
        return {
            "nearby_results": f"Could not locate {destination} on the map — nearby search skipped.",
            "messages":       [AIMessage(content="Nearby agent: geocoding failed.")],
        }
    lat, lon, geo_meta = geo
    region  = geo_meta.get("region", "")
    country = geo_meta.get("country", "")
    place_type = geo_meta.get("place_type", "")
    logger.info("Nearby agent — '%s' (%s) coords: %.4f, %.4f", destination, place_type, lat, lon)

    dest_enc = destination.replace(" ", "+")
    gmaps    = f"https://www.google.com/maps/search/{dest_enc}"

    # ── Overpass POIs — 6 per bucket max ─────────────────────────────────────
    pois    = overpass_nearby(lat, lon, radius_m=100_000)
    buckets = bucket_pois(pois)
    logger.info("Nearby agent — %d POIs found", len(pois))

    # ── 3 Tavily searches run IN PARALLEL ─────────────────────────────────────
    t_attr, t_food, t_review = asyncio.run(_parallel_tavily(
        f"best places to visit {destination} {region} {country} 2026 tourist attractions complete guide",
        f"famous local food street food restaurants {destination} 2026 must eat where to find price",
        f"tripadvisor {destination} hidden gems travel review 2026 offbeat things to do",
    ))

    # ── Compact POI summary (6 per bucket) ───────────────────────────────────
    bucket_txt = "\n".join([
        f"≤10km: {format_pois_text(buckets['within_10km'], 6)}",
        f"10-30km: {format_pois_text(buckets['10_to_30km'], 6)}",
        f"30-50km: {format_pois_text(buckets['30_to_50km'], 6)}",
        f"50-100km: {format_pois_text(buckets['50_to_100km'], 6)}",
    ])

    web_data = "\n---\n".join(t for t in [t_attr, t_food, t_review] if t)

    result = _llm_text(
        "You are a local travel expert with 2026 knowledge. Be specific, use real names, include Google Maps links.",
        f"""Local guide for {destination}, {region}, {country}.

OSM MAP DATA:
{bucket_txt}

WEB SEARCH DATA (2026):
{web_data[:4000]}

Write Markdown with these sections (be concise but specific):

## 📍 Nearby Attractions
### ≤ 10 km
- **Name** (type) — why visit · entry fee · [Map]({gmaps}+name)
### 10–30 km — Day Trips
### 30–100 km — Excursions
(4–6 places per section; use web data where OSM is sparse)

## 🍜 Local Food & Restaurants
- Dish name — where to find — Rs. price · [Find]({gmaps}+food)
(List 6–8 items)

## 🛍️ Markets & Shopping
- Market name — what's sold — best time

## 🎭 Culture & Festivals 2026
- Key temples / events / customs

## 🚕 Getting Around
- Auto / bus / cab rates and tips

## 💎 Hidden Gems
- 3–4 offbeat spots from reviews
""",
    )

    return {
        "nearby_results": result,
        "messages":       [AIMessage(content="Nearby attractions agent completed.")],
        "llm_calls":      state.get("llm_calls", 0) + 1,
    }


# ── Budget agent ──────────────────────────────────────────────────────────────

def budget_agent(state: TravelState):
    logger.info("Budget agent running")
    constraints = state.get("trip_constraints", {}) or {}
    destination = constraints.get("destination", "")
    members     = constraints.get("members", 2)

    # Live 2026 price search for the destination
    price_data = ""
    if destination:
        try:
            results = asyncio.run(_parallel_tavily(
                f"{destination} travel cost 2026 hotel price food budget per day per person",
                f"{destination} entry fees tourist places 2026 ticket prices activities cost",
            ))
            price_data = "\n---\n".join(r for r in results if r)[:2500]
        except Exception as exc:
            logger.warning("Budget price search failed: %s", exc)

    result = _llm_text(
        "You are a precise travel budget analyst. Always show per-person AND total-party costs.",
        f"""Build a detailed 2026 budget breakdown for this trip.

User request: {state['user_query']}

Trip details:
- Destination: {destination}
- Travellers: {members}
- Dates: {constraints.get('start_date','')} to {constraints.get('end_date','')}
- Duration: {constraints.get('duration', 'see dates')} days
- Budget preference: {constraints.get('budget', 'not specified')}

Live 2026 price data for {destination}:
{price_data}

Flight info (use fares from here):
{state.get('flight_results', 'N/A')[:800]}

Transport info (use fares from here):
{state.get('transport_results', 'N/A')[:800]}

Hotel info (use prices from here):
{state.get('hotel_results', 'N/A')[:800]}

Nearby attractions (use entry fees from here):
{state.get('nearby_results', 'N/A')[:600]}

Produce a budget breakdown in this exact Markdown format:

## 💰 Budget Breakdown — {destination}

### Per Person Costs
| Category | Budget Option | Mid-Range | Premium |
|----------|--------------|-----------|---------|
| ✈️ Transport (to/from) | Rs. X | Rs. X | Rs. X |
| 🏨 Hotels (per night × nights) | Rs. X | Rs. X | Rs. X |
| 🍜 Food (per day × days) | Rs. X | Rs. X | Rs. X |
| 🎫 Entry fees & activities | Rs. X | Rs. X | Rs. X |
| 🚕 Local transport (per day) | Rs. X | Rs. X | Rs. X |
| 🛡️ Miscellaneous & buffer | Rs. X | Rs. X | Rs. X |
| **TOTAL per person** | **Rs. X** | **Rs. X** | **Rs. X** |

### For {members} Traveller(s)
| | Budget | Mid-Range | Premium |
|--|--------|-----------|---------|
| **Grand Total** | **Rs. X** | **Rs. X** | **Rs. X** |

### 💡 Money-Saving Tips
1. [specific tip for this destination]
2. ...

### ⚠️ Budget Risks
- [what might cost more than expected]

### ✅ Feasibility
One paragraph on whether the user's stated budget is realistic.

Use real fares from the search data where possible. Label estimates clearly.
""",
    )

    return {
        "budget_results": result,
        "messages":       [AIMessage(content="Budget agent completed.")],
        "llm_calls":      state.get("llm_calls", 0) + 1,
    }


# ── Master itinerary system prompt ────────────────────────────────────────────
_ITINERARY_SYSTEM = """
You are an AI Travel Itinerary Planner. Given a user's trip details and
pre-fetched research data, generate a complete, editable, budget-aware plan.

═══════════════════════════════
DATA YOU WILL RECEIVE
═══════════════════════════════
The user message contains pre-fetched specialist data:
- starting_location, destination_location, start_date, end_date
- number_of_members, budget, preferred_transport_mode, interests
- Flights data (live search results)
- Trains & buses data (live search results with booking links)
- Hotels data (live search results with booking links)
- Weather data (live forecasts)
- Budget analysis

Do NOT invent hotel names, prices, or transport fares that were not in the
provided data. Clearly mark any gap as "approx. — verify before booking."

═══════════════════════════════
WHAT TO PRODUCE
═══════════════════════════════
A) HOTELS — use data from the pre-fetched hotel results. List:
   budget pick, best-value pick, premium pick with price/night & booking link.

B) LOCAL FOOD — top 5–8 must-try local dishes/street food with where to find them.

C) LOCAL MARKETS — 2–3 popular markets/bazaars, known for, best time to visit.

D) NEARBY ATTRACTIONS (within ~100 km of destination) — temples, waterfalls,
   historic sites, parks, viewpoints. For each: name, distance, entry fee,
   ideal visit duration, best time.

E) TRANSPORT — use data from pre-fetched flight/train/bus results.

═══════════════════════════════
ITINERARY RULES
═══════════════════════════════
- Build a DAY-WISE plan: arrival → hotel check-in → each day → departure.
- Group nearby attractions to minimise travel. Balance must-see + offbeat.
- Show time slots, estimated travel time between stops, cost per activity.
- Breakfast / lunch / dinner suggestions every day.

═══════════════════════════════
BUDGET RULES
═══════════════════════════════
- Multiply per-person costs × number_of_members.
- Add 10 % contingency buffer.
- If no fixed budget given, show Budget / Standard / Premium tiers.
- Show total trip cost AND cost per person.

═══════════════════════════════
OUTPUT FORMAT
═══════════════════════════════
Return TWO blocks separated by ---PROSE---:

Block 1 — valid JSON (fenced ```json ... ```) following this exact schema:
{
  "trip_summary": {"from":"","to":"","start_date":"","end_date":"","members":0,"transport_mode":"","total_budget_estimate":""},
  "hotels": [{"name":"","type":"budget|best_value|premium","price_per_night":"","rating":"","booking_link":"","notes":""}],
  "days": [{"day":1,"date":"","activities":[{"time":"","place":"","category":"","duration_hrs":"","entry_fee":"","notes":""}],"meals":{"breakfast":"","lunch":"","dinner":""},"estimated_day_cost":""}],
  "local_food": [{"dish":"","where":"","price_range":""}],
  "local_markets": [{"name":"","known_for":"","best_time":""}],
  "nearby_attractions": [{"name":"","category":"","distance_km":"","entry_fee":"","duration_hrs":"","best_time":""}],
  "transport": {"to_destination":[{"mode":"","cost_per_person":"","duration":"","booking_link":""}],"local":[{"mode":"","cost_per_day":""}]},
  "budget_breakdown": {"transport_total":"","hotel_total":"","food_total":"","activities_total":"","local_transport_total":"","buffer_10pct":"","grand_total":"","cost_per_person":""}
}

Block 2 (after ---PROSE---) — a clean, human-readable markdown itinerary
summarising the same plan. This is shown to the traveller for review.

TONE: specific real names, clearly label estimates, no invented data.
""".strip()


# ── Itinerary agent ───────────────────────────────────────────────────────────

def itinerary_agent(state: TravelState):
    logger.info("Itinerary agent running")
    constraints   = state.get("trip_constraints", {})
    user_choices  = state.get("user_choices") or {}

    uc_transport = user_choices.get("transport", constraints.get("transport_mode", "Mixed"))
    uc_hotel     = user_choices.get("hotel",     "Mid-range")
    uc_food      = user_choices.get("food",      "No preference")
    uc_style     = user_choices.get("style",     "Explorer")
    uc_special   = user_choices.get("special_requests", "")

    result = _llm_text(
        _ITINERARY_SYSTEM,
        f"""Generate a complete travel itinerary using the data below.

Trip details:
- From: {constraints.get('origin', 'Not specified')}
- To: {constraints.get('destination', 'Not specified')}
- Start date: {constraints.get('start_date', 'Not specified')}
- End date: {constraints.get('end_date', 'Not specified')}
- Travellers: {constraints.get('members', 2)}
- Budget: {constraints.get('budget', 'Not specified')}
- Transport preference (USER CHOSE): {uc_transport}
- Hotel tier (USER CHOSE): {uc_hotel}
- Food preference (USER CHOSE): {uc_food}
- Travel style (USER CHOSE): {uc_style}
- Special requests: {uc_special if uc_special else 'None'}
- Interests: {constraints.get('interests', [])}

User's full request:
{state['user_query']}

═══ PRE-FETCHED FLIGHTS DATA ═══
{state.get('flight_results', 'Not fetched.')}

Flight info:
{state.get('flight_results', '')}

═══ PRE-FETCHED HOTELS DATA (with booking links) ═══
{state.get('hotel_results', 'Not fetched.')}

═══ WEATHER DATA ═══
{state.get('weather_results', 'Not fetched.')}

═══ NEARBY ATTRACTIONS DATA ═══
{state.get('nearby_results', 'Not fetched.')[:2000]}

═══ BUDGET ANALYSIS ═══
{state.get('budget_results', 'Not fetched.')}

Remember: output JSON block first, then ---PROSE--- separator, then markdown summary.
""",
    )

    # Extract the JSON block
    itinerary_json = ""
    try:
        m = re.search(r"```json\s*([\s\S]*?)\s*```", result, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
        else:
            start = result.index("{")
            end   = result.rindex("}") + 1
            candidate = result[start:end]
        json.loads(candidate)          # validate — raises if invalid
        itinerary_json = candidate
    except (ValueError, json.JSONDecodeError):
        logger.warning("Itinerary JSON extraction failed; prose-only mode")

    # Extract the prose section (after ---PROSE--- if present, else use full text)
    if "---PROSE---" in result:
        prose = result.split("---PROSE---", 1)[1].strip()
    else:
        prose = result

    approval_request = (
        f"Please review this draft travel plan.\n\n{prose}\n\nApprove or request changes."
    )

    return {
        "itinerary":        prose,
        "itinerary_json":   itinerary_json,
        "approval_request": approval_request,
        "messages":         [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls":        state.get("llm_calls", 0) + 1,
    }


# ── Human approval node ───────────────────────────────────────────────────────

def human_approval_agent(state: TravelState):
    """Pause after all research agents so the user can review findings and
    choose their transport, hotel tier, food and style preferences before
    the itinerary is generated."""
    logger.info("Human approval — waiting for user choices")

    choices = interrupt({
        "type":              "user_choices",
        "flight_results":    state.get("flight_results", ""),
        "transport_results": state.get("transport_results", ""),
        "hotel_results":     state.get("hotel_results", ""),
        "budget_results":    state.get("budget_results", ""),
    })

    logger.info("User choices received: %s", choices)
    return {
        "user_choices":   choices,
        "human_feedback": choices.get("special_requests", ""),
        "approved":       True,
        "messages":       [AIMessage(content="User choices collected.")],
    }


# ── Final response agent ──────────────────────────────────────────────────────

def final_response_agent(state: TravelState):
    approved = state.get("approved", True)
    logger.info("Final agent — approved=%s", approved)

    if approved:
        prompt = f"""The user approved the draft itinerary.

Produce the final, polished travel plan. Make it beautiful and practical.
Include hotel booking links and train/bus booking links where available.

Draft itinerary:
{state.get('itinerary', '')}

Train & bus options (with links):
{state.get('transport_results', '')}

Hotel recommendations (with links):
{state.get('hotel_results', '')}

Budget notes:
{state.get('budget_results', '')}
"""
    else:
        prompt = f"""The user requested revisions.

Original request:
{state.get('user_query', '')}

Draft itinerary:
{state.get('itinerary', '')}

User feedback:
{state.get('human_feedback', '')}

Train & bus options (with links):
{state.get('transport_results', '')}

Hotel recommendations (with links):
{state.get('hotel_results', '')}

Budget notes:
{state.get('budget_results', '')}

Revise the itinerary to address the user's feedback. Make it final and polished.
Include clickable links for hotels and transport bookings.
"""

    result = _llm_text("You produce final, user-ready travel plans.", prompt)

    return {
        "final_response":  result,
        "itinerary_json":  state.get("itinerary_json", ""),   # carry structured data forward
        "messages":        [AIMessage(content=result)],
        "llm_calls":       state.get("llm_calls", 0) + 1,
    }
