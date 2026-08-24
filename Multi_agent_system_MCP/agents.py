import asyncio
import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import get_llm
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

def _llm_text(system: str, prompt: str) -> str:
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ])
    return response.content


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
- transport_agent : trains, buses, ferries, local ground transport between cities
- hotel_agent     : hotels, accommodation, neighbourhood guides, booking links
- weather_agent   : weather, climate, seasonal advice, packing
- nearby_agent    : discovers nearby attractions within 10/30/50/100 km — temples, waterfalls, parks, viewpoints, local food & culture from open data
- budget_agent    : cost breakdown, affordability, money-saving tips
- itinerary_agent : always include — produces the actual travel plan

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
                                  "weather_agent", "nearby_agent", "budget_agent", "itinerary_agent"],
            "trip_constraints": {},
            "reasoning":        "Default routing (parse error).",
        }

    logger.info("Selected agents: %s", parsed.get("selected_agents"))

    return {
        "selected_agents":      parsed["selected_agents"],
        "trip_constraints":     parsed.get("trip_constraints", {}),
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

    try:
        train_raw = asyncio.run(tavily_search(
            f"train {route_str} booking schedule tickets site:booking.com OR site:irctc.co.in OR site:raileurope.com OR site:trainline.com"
        ))
        train_text = _extract_tavily_with_urls(train_raw)
    except Exception as exc:
        logger.warning("Transport train search failed: %s", exc)
        train_text = "Train search unavailable."

    try:
        bus_raw = asyncio.run(tavily_search(
            f"bus {route_str} booking tickets site:redbus.in OR site:busbud.com OR site:flixbus.com OR site:megabus.com"
        ))
        bus_text = _extract_tavily_with_urls(bus_raw)
    except Exception as exc:
        logger.warning("Transport bus search failed: %s", exc)
        bus_text = "Bus search unavailable."

    result = _llm_text(
        "You are a ground transport specialist for travel planning.",
        f"""Summarise train and bus options for this traveller.

User request:
{state['user_query']}

Route: {route_str}

Train search results (with links):
{train_text[:2500]}

Bus search results (with links):
{bus_text[:2500]}

Return a clean markdown summary with two sections: **Trains** and **Buses**.
For each option include:
- Operator / service name with a booking link as [Operator Name](URL)
- Approximate journey time and frequency
- Indicative fare range
- Any important tips (advance booking, passes, etc.)
If trains or buses are not relevant for this route (e.g. international long haul), say so briefly.
Do not output raw JSON.
""",
    )

    return {
        "transport_results": result,
        "messages":          [AIMessage(content="Transport agent completed.")],
        "llm_calls":         state.get("llm_calls", 0) + 1,
    }


# ── Weather agent ─────────────────────────────────────────────────────────────

def weather_agent(state: TravelState):
    city = (state.get("trip_constraints") or {}).get("destination", "")
    logger.info("Weather agent — city: %s", city)

    try:
        weather_now  = _extract_text(asyncio.run(current_weather(city)))
        weather_fore = _extract_text(asyncio.run(forecast(city)))
    except Exception as exc:
        logger.warning("Weather MCP call failed: %s", exc)
        weather_now = weather_fore = "Live weather data unavailable."

    result = f"**Current conditions:**\n{weather_now}\n\n**Forecast:**\n{weather_fore}"

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
    coords = geocode_city(destination)
    if not coords:
        return {
            "nearby_results": f"Could not locate {destination} on the map — nearby search skipped.",
            "messages":       [AIMessage(content="Nearby agent: geocoding failed.")],
        }
    lat, lon = coords
    logger.info("Nearby agent — coords: %.4f, %.4f", lat, lon)

    # ── Overpass POIs (100 km radius) ────────────────────────────────────────
    pois    = overpass_nearby(lat, lon, radius_m=100_000)
    buckets = bucket_pois(pois)
    logger.info("Nearby agent — %d POIs found", len(pois))

    # ── Wikivoyage local culture ──────────────────────────────────────────────
    wiki_tips = wikivoyage_local_tips(destination)

    # ── Tavily for hidden gems ────────────────────────────────────────────────
    try:
        hidden_raw  = asyncio.run(tavily_search(
            f"hidden gem local attractions near {destination} worth visiting"
        ))
        hidden_text = _extract_tavily_with_urls(hidden_raw)[:2000]
    except Exception as exc:
        logger.warning("Nearby Tavily search failed: %s", exc)
        hidden_text = ""

    # ── LLM format ────────────────────────────────────────────────────────────
    bucket_summary = "\n\n".join([
        f"**Within 10 km:**\n{format_pois_text(buckets['within_10km'], 10)}",
        f"**10 – 30 km away:**\n{format_pois_text(buckets['10_to_30km'], 10)}",
        f"**30 – 50 km away:**\n{format_pois_text(buckets['30_to_50km'], 10)}",
        f"**50 – 100 km away:**\n{format_pois_text(buckets['50_to_100km'], 10)}",
    ])

    wiki_summary = "\n\n".join(
        f"**{k.title()}:**\n{v[:600]}" for k, v in wiki_tips.items()
    ) or "No Wikivoyage data available."

    result = _llm_text(
        "You are a local travel expert who knows the area intimately.",
        f"""Produce a rich, structured guide to places and experiences near {destination}.

Raw OSM data grouped by distance:
{bucket_summary}

Wikivoyage local tips:
{wiki_summary}

Hidden gems from web search:
{hidden_text}

Format as clean Markdown with these sections:
## 📍 Nearby Attractions by Distance
For each bucket (≤10 km / 10–30 km / 30–50 km / 50–100 km) list:
- Best 4–6 places with name, category, distance, why it's worth visiting
- Add a Google Maps search link like: [View on Map](https://www.google.com/maps/search/{destination.replace(' ', '+')}+{{}})

## 🍜 Local Food & Specialties
Top dishes, street food, and where to eat — use the Wikivoyage eat section.

## 🛍️ Shopping & Markets
Local markets and what they're famous for — use the Wikivoyage buy section.

## 🎭 Culture & Traditions
What makes this place unique — festivals, customs, local people's way of life.

## 💎 Hidden Gems
Underrated spots or experiences that most tourists miss.

Be specific and vivid. Use real names from the data provided.
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

    result = _llm_text(
        "You are a practical travel budget analyst.",
        f"""Analyse whether this trip plan is realistic for the user's budget.

User request:
{state['user_query']}

Constraints:
{state.get('trip_constraints', {})}

Flight info:
{state.get('flight_results', 'N/A')}

Train & bus info:
{state.get('transport_results', 'N/A')}

Hotel info:
{state.get('hotel_results', 'N/A')}

Weather info:
{state.get('weather_results', 'N/A')}

Nearby attractions:
{state.get('nearby_results', 'N/A')[:800]}

Provide:
1. Estimated cost by category (flights, ground transport, accommodation, food, activities)
2. Risk areas where costs might blow out
3. Money-saving tips
4. Overall feasibility verdict
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
    constraints = state.get("trip_constraints", {})

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
- Transport preference: {constraints.get('transport_mode', 'Mixed')}
- Interests: {constraints.get('interests', [])}

User's full request:
{state['user_query']}

═══ PRE-FETCHED FLIGHTS DATA ═══
{state.get('flight_results', 'Not fetched.')}

═══ PRE-FETCHED TRAINS & BUSES DATA (with booking links) ═══
{state.get('transport_results', 'Not fetched.')}

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
    logger.info("Human approval — waiting for user input via interrupt()")

    feedback = interrupt({
        "question":           "Do you approve this itinerary?",
        "draft_itinerary":    state.get("itinerary", ""),
        "approval_request":   state.get("approval_request", ""),
        "expected_response":  {"approved": True, "feedback": "Optional revision notes"},
    })

    approved       = feedback.get("approved", True)
    human_feedback = feedback.get("feedback", "")
    logger.info("Human decision — approved=%s", approved)

    return {
        "approved":      approved,
        "human_feedback": human_feedback,
        "messages":      [AIMessage(content="Human approval step completed.")],
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
