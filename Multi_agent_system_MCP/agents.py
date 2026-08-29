import asyncio
import concurrent.futures
import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import GROQ_FALLBACKS, get_gemini_llm, get_llm, get_retry_models
from mcp_client import current_weather, forecast, tavily_search
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

_THINK_RE  = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*",         re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks; also removes unclosed <think> to end-of-string."""
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN.sub("", text)
    return text.strip()


def _extract_content(result) -> str:
    """Normalize any LLM response to a plain markdown string.

    Handles both response shapes:
    - Groq / most LLMs: result.content is a plain str
    - Gemini 3.6-flash: result.content is a list of typed content blocks
      (may include "thinking" blocks that should be discarded)
    """
    raw = result.content if hasattr(result, "content") else str(result)
    if isinstance(raw, list):
        parts = [
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in raw
            if not (isinstance(b, dict) and b.get("type") == "thinking")
        ]
        return "\n\n".join(p for p in parts if p).strip() or str(raw)
    return str(raw).strip()


def _llm_text(system: str, prompt: str, max_tokens: int = 8192) -> str:
    """Call the best available LLM: Gemini primary, Groq retry-loop fallback.

    Both providers pass through _extract_content() so markdown structure
    (headers, tables, bullets) is preserved regardless of which model responds.
    If a reasoning model returns only a <think> block (empty after stripping),
    the result is discarded and the next model is tried automatically.
    """
    global llm
    msgs = [SystemMessage(content=system), HumanMessage(content=prompt)]

    # ── Primary: Google Gemini Flash (no TPM limit, 1M context) ──────────────
    gemini = get_gemini_llm(max_output_tokens=max_tokens)
    if gemini is not None:
        try:
            text = _extract_content(gemini.invoke(msgs))
            cleaned = _strip_think(text)
            if cleaned:
                logger.info("Gemini responded OK (%d chars)", len(cleaned))
                return cleaned
            logger.warning("Gemini returned only a think block — falling back to Groq")
        except Exception as exc:
            logger.warning("Gemini failed, falling back to Groq: %s", str(exc)[:300])

    # ── Fallback: Groq retry loop (tries best models first by quality score) ──
    retry_list = get_retry_models()
    tried: set[str] = set()
    for _ in range(len(retry_list) + 1):
        current = getattr(llm, "model_name", "") or getattr(llm, "model", "")
        try:
            text = _extract_content(llm.invoke(msgs))
            cleaned = _strip_think(text)
            if not cleaned:
                # Reasoning model returned only <think> — try next model
                logger.warning("Groq %s returned only a think block; switching model", current)
                tried.add(current)
                next_models = [m for m in retry_list if m not in tried]
                if next_models:
                    llm = get_llm(next_models[0])
                    continue
                # All models only think — return raw content with tags removed as last resort
                return text.replace("<think>", "").replace("</think>", "").strip()
            logger.info("Groq %s responded OK (%d chars)", current, len(cleaned))
            return cleaned
        except Exception as exc:
            tried.add(current)
            exc_str = str(exc)
            logger.error("Groq model %s error: %s", current, exc_str[:300])
            if "429" in exc_str or "rate_limit" in exc_str.lower():
                import time as _time; _time.sleep(2)
            next_models = [m for m in retry_list if m not in tried]
            if not next_models:
                raise RuntimeError(f"All models failed. Last Groq: {current} — {exc_str[:200]}") from exc
            nxt = next_models[0]
            logger.warning("Groq model %s failed; switching to %s", current, nxt)
            llm = get_llm(nxt)
    raise RuntimeError("All LLM providers exhausted")


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


def _tavily_one(query: str, cap: int = 1500) -> str:
    """Run a single Tavily search synchronously (safe to call from any thread)."""
    try:
        raw = asyncio.run(tavily_search(query))
        return _extract_tavily_with_urls(raw)[:cap]
    except Exception as exc:
        logger.warning("Tavily search failed (%s): %s", query[:60], exc)
        return ""


def _parallel_tavily(*queries: str, cap: int = 1500) -> list[str]:
    """Run multiple Tavily searches in parallel threads — each thread owns its
    own event loop so there are no asyncio nesting / loop-reuse conflicts."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = [pool.submit(_tavily_one, q, cap) for q in queries]
        return [f.result() for f in futures]


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
- transport_agent : flights, trains, buses, airlines, airports, airfare, ground transport, overland routes between cities
- hotel_agent     : hotels, accommodation, neighbourhood guides
- weather_agent   : weather, climate, seasonal advice, packing
- nearby_agent    : nearby attractions, temples, parks, local food, points of interest
- budget_agent    : cost breakdown, affordability, money-saving tips
- itinerary_agent : always include — produces the actual travel plan

IMPORTANT: If the request mentions both an origin city AND a destination city, ALWAYS include transport_agent to find trains and buses.

Return ONLY JSON:
{{
  "selected_agents": ["transport_agent", "hotel_agent", "weather_agent", "nearby_agent", "budget_agent", "itinerary_agent"],
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
            "selected_agents":  ["transport_agent", "hotel_agent",
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
    # Remove flight_agent if LLM returned it (it's now part of transport_agent)
    selected = [a for a in selected if a != "flight_agent"]
    if merged_constraints.get("origin") and merged_constraints.get("destination"):
        if "transport_agent" not in selected:
            selected.insert(0, "transport_agent")

    return {
        "selected_agents":      selected,
        "trip_constraints":     merged_constraints,
        "supervisor_reasoning": parsed.get("reasoning", ""),
        "messages":             [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls":            state.get("llm_calls", 0) + 1,
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
        "You are a hotel specialist. Use REAL names and prices from the search data. Write each area ONCE.",
        f"""Hotel recommendations for: {destination}
User request: {state['user_query']}

Search results (with booking links):
{search_text[:4000]}

Group hotels by area/neighbourhood. For each area:

### [Area / Neighbourhood Name]
1–2 sentence description of the area and why it suits travellers.

| Hotel | Price/Night (₹) | Type | Rating | Highlights | Book |
|-------|----------------|------|--------|------------|------|
(budget / standard / premium rows; hyperlink hotel name to booking URL if available)

Cover at least 3 areas. Include a mix of budget, mid-range, and premium options.

After all area tables add:

## 💡 Booking Tips
2–3 practical tips (best booking platform, when to book, what to watch out for).

## ⚠️ Watch Out
One short paragraph on common issues (hidden charges, location traps, etc.).
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

    # ── 5 parallel searches: flights + trains (direct + via) + buses + general ──
    flights_search, train_direct, train_via, bus_search, general = _parallel_tavily(
        f"flight {route_str} 2026 airline airfare cheap air tickets booking IndiGo SpiceJet AirAsia",
        f"train {route_str} 2026 direct schedule timing fare IRCTC booking",
        f"how to reach {destination} from {origin} by train 2026 via connecting junction route change",
        f"bus {route_str} 2026 Redbus Abhibus schedule timing booking fare",
        f"{route_str} travel options 2026 how to reach fastest cheapest transport",
    )

    result = _llm_text(
        "You are a transport specialist. Use REAL train/flight/bus names from the data. Write each section ONCE.",
        f"""Transport guide for: {route_str}
User request: {state['user_query']}

FLIGHT DATA: {flights_search[:1200]}
TRAIN (DIRECT): {train_direct[:1200]}
TRAIN (VIA): {train_via[:800]}
BUS DATA: {bus_search[:800]}
GENERAL: {general[:600]}

## ✈️ Flights
Brief note on flight availability (1 sentence).
| Airline | Route | Fare (₹) | Duration | Book |
|---------|-------|----------|----------|------|
(if no direct flight: "Via [Hub] – [Airline]"; if no data: "No flights — nearest airport: [X] ([Y] km)")

## 🚂 Trains
Best train recommendation in 1 sentence, then the table:
| Train Name | No. | Departs | Arrives | Duration | Fare Sl/3A/2A | Book |
|-----------|-----|---------|---------|----------|---------------|------|
(connecting trains: label rows "Leg 1" / "Leg 2" with junction city)
[Book on IRCTC](https://www.irctc.co.in)

## 🚌 Buses
| Operator | Departs | Duration | Fare (₹) | Type | Book |
|---------|---------|----------|----------|------|------|

## 🚗 Self-Drive / Cab
| Route (Highway) | Distance | Drive Time | Est. Fuel+Toll |
|----------------|---------|-----------|----------------|

## 💡 Recommendation
2–3 sentences: which option is best for budget / comfort / speed travellers on this specific route, and why.
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

    # ── Geocode (non-fatal — Tavily runs regardless) ─────────────────────────
    geo = geocode_city(destination)
    lat = lon = None
    region = country = place_type = ""
    if geo:
        lat, lon, geo_meta = geo
        region     = geo_meta.get("region", "")
        country    = geo_meta.get("country", "")
        place_type = geo_meta.get("place_type", "")
        logger.info("Nearby agent — '%s' (%s) coords: %.4f, %.4f", destination, place_type, lat, lon)
    else:
        logger.warning("Nearby agent — geocoding failed for '%s'; continuing with web-only data", destination)

    # ── Overpass POIs (only when coordinates are available) ──────────────────
    pois    = []
    buckets = {"within_10km": [], "10_to_30km": [], "30_to_50km": [], "50_to_100km": []}
    if lat is not None:
        try:
            pois    = overpass_nearby(lat, lon, radius_m=50_000)
            buckets = bucket_pois(pois)
            logger.info("Nearby agent — %d POIs found", len(pois))
        except Exception as exc:
            logger.warning("Overpass failed, continuing without map data: %s", exc)

    # ── 6 parallel Tavily searches: attractions + food + markets + experiences + gems + transport ──
    try:
        t_attr, t_food, t_markets, t_exp, t_gems, t_tips = _parallel_tavily(
            f"best tourist places to visit {destination} {region} {country} 2026 sightseeing complete guide top attractions",
            f"famous local food street food best restaurants {destination} 2026 must-eat dishes where to find price specialty",
            f"famous local markets bazaars shopping {destination} {region} 2026 what to buy handicrafts souvenirs",
            f"local experiences activities things to do {destination} 2026 festivals events culture spiritual",
            f"hidden gems offbeat places {destination} {region} tripadvisor 2026 unique unusual underrated",
            f"{destination} local transport auto rickshaw taxi prices tips 2026 getting around practical guide",
        )
    except Exception as exc:
        logger.warning("Nearby parallel Tavily failed: %s", exc)
        t_attr = t_food = t_markets = t_exp = t_gems = t_tips = ""

    # ── Compact POI summary (6 per bucket) ───────────────────────────────────
    bucket_txt = "\n".join([
        f"<=10km: {format_pois_text(buckets['within_10km'], 6)}",
        f"10-30km: {format_pois_text(buckets['10_to_30km'], 6)}",
        f"30-50km: {format_pois_text(buckets['30_to_50km'], 6)}",
        f"50-100km: {format_pois_text(buckets['50_to_100km'], 6)}",
    ])

    web_data = "\n---\n".join(t for t in [t_attr, t_food, t_markets, t_exp, t_gems, t_tips] if t)
    if not web_data and not any(buckets.values()):
        return {
            "nearby_results": f"Web search data unavailable for {destination} — please check your internet connection.",
            "messages":       [AIMessage(content="Nearby agent returned no data.")],
        }

    result = _llm_text(
        "You are a local expert. Use REAL names from the data. Mix tables with brief descriptions where helpful.",
        f"""Local guide for {destination}, {region}, {country}.

MAP DATA (POIs): {bucket_txt[:600]}
ATTRACTIONS: {t_attr[:800]}
LOCAL FOOD: {t_food[:900]}
MARKETS: {t_markets[:700]}
EXPERIENCES: {t_exp[:600]}
HIDDEN GEMS: {t_gems[:500]}
LOCAL TRANSPORT: {t_tips[:500]}

## 📍 Nearby Attractions
1–2 sentences about what makes {destination} special for visitors.

### Within 10 km
| Place | Type | Distance | Entry Fee | Best Time |
|-------|------|---------|-----------|-----------|

### 10–30 km — Day Trips
| Place | Type | Distance | Entry Fee | Best Time |
|-------|------|---------|-----------|-----------|

### 30–100 km — Excursions
| Place | Type | Distance | Entry Fee | Best Time |
|-------|------|---------|-----------|-----------|

## 🍜 Local Food Guide
Brief intro (1 sentence) on the food culture of {destination}.

### Must-Eat Dishes
| Dish | Description | Where to Find | Price (₹) |
|------|------------|--------------|-----------|

### Best Restaurants & Dhabas
| Name | Specialty | Price/Person (₹) | Area |
|------|-----------|-----------------|------|

## 🛍️ Famous Markets & Bazaars
| Market | Sells | Best Time | Bargaining Tip |
|--------|-------|-----------|---------------|

## 🎭 Local Experiences
1 sentence on the must-do experience in {destination}, then the table:
| Activity | Where | Timing | Cost (₹) |
|---------|-------|--------|---------|

## 🔍 Hidden Gems
| Place | Why Special | Distance | Tip |
|-------|------------|---------|-----|

## 🚌 Getting Around Locally
| Mode | Typical Fare | Notes |
|------|-------------|-------|

## 💡 Insider Tips
2–3 practical tips a first-time visitor to {destination} must know.

Use REAL place names from MAP DATA and WEB RESEARCH.
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

    # Live 2026 price search for the destination (parallel threads)
    price_data = ""
    if destination:
        try:
            results = _parallel_tavily(
                f"{destination} travel cost 2026 hotel price food budget per day per person",
                f"{destination} entry fees tourist places 2026 ticket prices activities cost",
                f"{destination} local transport auto rickshaw taxi bus fare 2026",
                f"{destination} street food restaurant cost cheap eats budget 2026",
            )
            price_data = "\n---\n".join(r for r in results if r)[:4000]
        except Exception as exc:
            logger.warning("Budget price search failed: %s", exc)

    result = _llm_text(
        "You are a precise travel budget analyst. Write each section ONCE only. Never repeat category names. End after the Feasibility paragraph. No extra text after that.",
        f"""Build a detailed 2026 budget breakdown for this trip.

User request: {state['user_query']}

Trip details:
- Destination: {destination}
- Travellers: {members}
- Dates: {constraints.get('start_date','')} to {constraints.get('end_date','')}
- Duration: {constraints.get('duration', 'see dates')} days
- Budget preference: {constraints.get('budget', 'not specified')}

Live 2026 price data for {destination} (hotels, food, transport, entry fees):
{price_data}

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

### 🍽️ Food Cost Guide
| Meal Type | Estimated Cost per Person |
|-----------|--------------------------|
| Street food / snacks | Rs. X |
| Local restaurant (thali / meal) | Rs. X |
| Mid-range restaurant | Rs. X |
| Branded café / fast food | Rs. X |

### 🚌 Local Transport Guide
| Mode | Typical Fare |
|------|-------------|
| Auto rickshaw (3 km) | Rs. X |
| Local bus | Rs. X |
| Taxi / cab app | Rs. X/km |
| Cycle rickshaw | Rs. X |

### 💡 Money-Saving Tips
1. [specific tip for this destination]
2. [another specific tip]
3. [third specific tip]

### ⚠️ Budget Risks
- [what might cost more than expected]
- [seasonal price surge risk]

### ✅ Feasibility
One paragraph on whether the user's stated budget is realistic, with a specific recommendation.

Use real fares from the search data where possible. Label estimates clearly.
""",
    )

    return {
        "budget_results": result,
        "messages":       [AIMessage(content="Budget agent completed.")],
        "llm_calls":      state.get("llm_calls", 0) + 1,
    }


# ── Parallel research runner ──────────────────────────────────────────────────

def research_all_agent(state: TravelState):
    """Run all 5 research agents simultaneously in parallel threads.

    This replaces the sequential supervisor-routed chain and reduces total
    research time from ~5× slowest-agent to ~1× slowest-agent.
    """
    selected = state.get("selected_agents") or []

    _agent_map = {
        "transport_agent": transport_agent,
        "hotel_agent":     hotel_agent,
        "weather_agent":   weather_agent,
        "nearby_agent":    nearby_agent,
        "budget_agent":    budget_agent,
    }
    order = ["transport_agent", "hotel_agent", "weather_agent", "nearby_agent", "budget_agent"]
    to_run = [_agent_map[n] for n in order if n in selected]
    if not to_run:
        return {"messages": [AIMessage(content="No research agents selected.")]}

    merged: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(to_run)) as ex:
        futures = {ex.submit(fn, state): fn.__name__ for fn in to_run}
        for fut in concurrent.futures.as_completed(futures):
            name = futures[fut]
            try:
                res = fut.result()
                for k, v in res.items():
                    if k != "messages" and v:
                        merged[k] = v
                logger.info("✅ %s done", name)
            except Exception as exc:
                logger.error("❌ %s failed: %s", name, exc)

    merged["messages"] = [AIMessage(content="All research agents completed in parallel.")]
    merged["llm_calls"] = state.get("llm_calls", 0) + len(to_run)
    return merged


# ── Master itinerary system prompt (kept short to fit Groq free-tier 8k TPM) ──
_ITINERARY_SYSTEM = """
You are an AI Travel Itinerary Planner. Build a complete, budget-aware plan from the pre-fetched data.

RULES:
- Use ACTUAL names from research data. If a hotel/transport name is NOT confirmed in the data, write  _______________  (blank) so the planner can fill it in manually.
- VEGETARIAN mode: mark hotels 🌿, recommend only veg dishes every meal.
- Day 1 MUST START FROM THE ORIGIN CITY. First activity = departure from origin (train/flight/bus boarding time, station/airport name, platform). Next activities = journey travel time, arrival at destination, check-in at hotel.
- Day 1 MUST use the exact transport mode chosen by the user (Train / Flight / Bus / Self-drive).
- Every meal: specific dish + specific restaurant name from nearby data, or _______________ if unknown.
- Include at least 1 local market with bargaining tips.
- Budget: per-person × members + 10% buffer, show grand total and per-person cost.
- Add a "## 💬 What to Say & Do" section: Hindi/English phrases for hotel check-in, train platform, auto fare bargaining, market bargaining + emergency numbers (Police 100, Ambulance 108).

OUTPUT — exactly TWO blocks separated by ---PROSE---:

Block 1: valid JSON fenced as ```json — keep values SHORT (under 60 chars each)
{"trip_summary":{"from":"","to":"","start_date":"","end_date":"","members":0,"transport_mode":"","total_budget_estimate":""},"days":[{"day":1,"date":"","activities":[{"time":"","place":"","duration_hrs":"","entry_fee":"","notes":""}],"meals":{"breakfast":"","lunch":"","dinner":""},"hotel":"","estimated_day_cost":""}],"budget_breakdown":{"transport_total":"","hotel_total":"","food_total":"","activities_total":"","buffer_10pct":"","grand_total":"","cost_per_person":""}}
CRITICAL: Include ALL days (every single day from start to end date). Do NOT stop at day 2 if the trip is 3 days.

Block 2 (after ---PROSE---): Markdown tables ONLY — no prose paragraphs, no bullet lists.
Generate ALL days without stopping early. For EVERY day:

## Day N — [Theme] | [Date]
| Time | Activity | Duration | Cost | Notes |
|------|----------|----------|------|-------|
| 06:00 AM | [ORIGIN] Depart [Station] | — | Rs.0 | Platform info |
| [arrival] | Arrive [DESTINATION] Station | — | — | — |
| [check-in] | Hotel check-in | 1h | Rs.XX | hotel name |
| [time] | [Attraction] | Xh | Rs.XX | tip |
Meals: BF:[dish@place] Lunch:[dish@place] Dinner:[dish@place]  Budget: Rs.XXXX/person
---
(Day 2+: skip departure, start directly at destination)

After all days, one summary table:
| Item | Per Person | Total |
|------|-----------|-------|
| Transport | Rs.XX | Rs.XX |
| Hotels | Rs.XX | Rs.XX |
| Food | Rs.XX | Rs.XX |
| Activities | Rs.XX | Rs.XX |
| **GRAND TOTAL** | **Rs.XX** | **Rs.XX** |
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
- Interests: {constraints.get('interests', [])}

User's full request:
{state['user_query']}

USER CHOICES (apply strictly):
- Transport chosen: {uc_transport}
- Hotel tier chosen: {uc_hotel}
- Food preference: {uc_food} {"→ VEGETARIAN MODE: recommend ONLY veg-friendly hotels and pure-veg dishes" if "Vegetarian" in uc_food else ""}
- Travel style: {uc_style}
- Special requests: {uc_special if uc_special else "None"}

═══ PRE-FETCHED TRANSPORT DATA (flights, trains, buses) ═══
{(state.get('transport_results') or 'Not fetched.')[:2000]}

═══ PRE-FETCHED HOTELS DATA (with booking links) ═══
{(state.get('hotel_results') or 'Not fetched.')[:1500]}

═══ WEATHER DATA ═══
{(state.get('weather_results') or 'Not fetched.')[:800]}

═══ NEARBY ATTRACTIONS, LOCAL FOOD & MARKETS DATA (USE for day-by-day planning) ═══
{(state.get('nearby_results') or 'Not fetched.')[:2500]}

═══ BUDGET ANALYSIS ═══
{(state.get('budget_results') or 'Not fetched.')[:1200]}

Remember: output JSON block first, then ---PROSE--- separator, then markdown summary.
""",
        max_tokens=8192,
    )

    # Extract the JSON block
    itinerary_json = ""
    _candidate = ""
    try:
        m = re.search(r"```json\s*([\s\S]*?)\s*```", result, re.IGNORECASE)
        if m:
            _candidate = m.group(1).strip()
        else:
            start = result.index("{")
            end   = result.rindex("}") + 1
            _candidate = result[start:end]
        # Fix trailing commas — common LLM error: "key": , or [,,,{...}]
        _fixed = re.sub(r',(\s*[}\]])', r'\1', _candidate)
        json.loads(_fixed)          # validate — raises if still invalid
        itinerary_json = _fixed
    except (ValueError, json.JSONDecodeError):
        # Store raw candidate so frontend can strip it verbatim / do best-effort parse
        itinerary_json = _candidate
        if _candidate:
            logger.warning("Itinerary JSON invalid; stored raw for display")
        else:
            logger.warning("Itinerary JSON extraction failed; prose-only mode")

    # Extract the prose section (after ---PROSE--- if present, else strip JSON from full text)
    if "---PROSE---" in result:
        prose = result.split("---PROSE---", 1)[1].strip()
        # Only strip ```json blocks embedded in prose (not ```markdown or plain ```)
        prose = re.sub(r"`{3,}json[ \t]*\n[\s\S]*?`{3,}", "", prose, flags=re.IGNORECASE)
        prose = re.sub(r"`{3,}json[ \t]*\n[\s\S]*",        "", prose, flags=re.IGNORECASE)
    else:
        # LLM skipped the separator — strip ALL fenced blocks (closed + unclosed)
        prose = re.sub(r"`{3,}[^\n]*\n[\s\S]*?`{3,}", "", result, flags=re.IGNORECASE)
        prose = re.sub(r"`{3,}[^\n]*\n[\s\S]*",        "", prose,  flags=re.IGNORECASE)
        prose = prose.strip()
        # Strip bare { … } and [ … ] JSON blocks via bracket-counting
        # If block is unclosed (truncated JSON), drop everything from that point
        def _bracket_strip(s, open_c, close_c):
            s = s.lstrip()
            if not s.startswith(open_c):
                return s
            _bd, _ei, _found = 0, 0, False
            for _ci, _ch in enumerate(s):
                if _ch == open_c:  _bd += 1
                elif _ch == close_c:
                    _bd -= 1
                    if _bd == 0:
                        _ei = _ci + 1; _found = True; break
            if _found:
                return s[_ei:].strip()
            # Unclosed — check if it looks like JSON; if so, discard everything
            if '": ' in s[:300] or '":"' in s[:300]:
                return ""
            return s
        prose = _bracket_strip(prose, "{", "}")
        prose = _bracket_strip(prose, "[", "]")

    # Strip any leaked instruction text before the first real content line
    # (LLM sometimes echoes back instruction phrases before the actual ## Day 1 heading)
    prose_lines = prose.split("\n")
    first_real = 0
    for idx, ln in enumerate(prose_lines):
        s = ln.strip()
        if s.startswith("##") or s.startswith("|") or s.startswith("**"):
            first_real = idx
            break
    if first_real > 0:
        prose = "\n".join(prose_lines[first_real:]).strip()

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


# ── Final response agent — pass-through, no extra LLM call ───────────────────
# The itinerary_agent already produces the complete polished plan. Calling the
# LLM again here was adding 15-30s latency and risking Groq 8k-token overflows.

def final_response_agent(state: TravelState):
    logger.info("Final agent — pass-through (itinerary is complete)")
    itinerary = state.get("itinerary", "")
    return {
        "final_response": itinerary,
        "itinerary_json": state.get("itinerary_json", ""),
        "messages":       [AIMessage(content="Travel plan ready.")],
        "llm_calls":      state.get("llm_calls", 0),
    }
