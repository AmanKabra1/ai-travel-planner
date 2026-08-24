import asyncio
import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import GROQ_FALLBACKS, get_llm
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
    """Call the LLM, auto-switching to the next model on any API error."""
    global llm
    msgs = [SystemMessage(content=system), HumanMessage(content=prompt)]
    tried: set[str] = set()
    for _ in range(len(GROQ_FALLBACKS) + 1):
        current = getattr(llm, "model_name", "") or getattr(llm, "model", "")
        try:
            return llm.invoke(msgs).content
        except Exception as exc:
            tried.add(current)
            next_models = [m for m in GROQ_FALLBACKS if m not in tried]
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
- flight_agent  : flights, airports, airlines, routes, airfare
- hotel_agent   : hotels, accommodation, neighbourhood guides
- weather_agent : weather, climate, seasonal advice, packing
- budget_agent  : cost breakdown, affordability, money-saving tips
- itinerary_agent : always include — produces the actual travel plan

Return ONLY JSON:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
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
            "selected_agents":  ["flight_agent", "hotel_agent", "weather_agent",
                                  "budget_agent", "itinerary_agent"],
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

    # ── Overpass POIs (auto-expands for sparse/small places) ─────────────────
    pois    = overpass_nearby(lat, lon, radius_m=100_000)
    buckets = bucket_pois(pois)
    logger.info("Nearby agent — %d POIs found", len(pois))

    # ── Wikivoyage local culture (falls back to region → country) ────────────
    wiki_tips = wikivoyage_local_tips(destination, region=region, country=country)

    # ── Tavily searches — broader for small/unknown places ───────────────────
    tavily_parts: list[str] = []

    def _tsearch(q: str) -> str:
        try:
            raw = asyncio.run(tavily_search(q))
            return _extract_tavily_with_urls(raw)[:1800]
        except Exception as exc:
            logger.warning("Nearby Tavily '%s' failed: %s", q[:50], exc)
            return ""

    if len(pois) < 10:
        # Small / obscure place — run multiple targeted searches
        tavily_parts.append(_tsearch(f"top tourist places to visit near {destination} {region} {country}"))
        tavily_parts.append(_tsearch(f"famous things to do in {destination} travel guide local attractions"))
        tavily_parts.append(_tsearch(f"local food specialties street food {destination} {country}"))
    else:
        tavily_parts.append(_tsearch(f"hidden gem local attractions near {destination} worth visiting"))
        tavily_parts.append(_tsearch(f"local food specialties {destination}"))

    hidden_text = "\n\n".join(t for t in tavily_parts if t)

    # ── LLM format ────────────────────────────────────────────────────────────
    bucket_summary = "\n\n".join([
        f"**Within 10 km:**\n{format_pois_text(buckets['within_10km'], 12)}",
        f"**10 – 30 km away:**\n{format_pois_text(buckets['10_to_30km'], 12)}",
        f"**30 – 50 km away:**\n{format_pois_text(buckets['30_to_50km'], 12)}",
        f"**50 – 100 km away:**\n{format_pois_text(buckets['50_to_100km'], 12)}",
    ])

    wiki_summary = "\n\n".join(
        f"**{k.title()}:**\n{v[:600]}" for k, v in wiki_tips.items()
    ) or f"No Wikivoyage page found for {destination} — using web search data below."

    context_note = (
        f"NOTE: This is a small/lesser-known location in {region}, {country}."
        if len(pois) < 10 else
        f"Location: {destination}, {region}, {country}."
    )

    dest_enc = destination.replace(" ", "+")
    result = _llm_text(
        "You are a knowledgeable local travel expert. Even for small, obscure, or lesser-known places, "
        "you produce rich, detailed, accurate guides using whatever data is available.",
        f"""Produce a detailed, vivid guide to {destination} and its surroundings.

{context_note}

OSM map data grouped by distance from {destination}:
{bucket_summary}

Wikivoyage / regional tips:
{wiki_summary}

Web search results (may include specific local info for small places):
{hidden_text[:4000]}

Format as clean Markdown with these exact sections:

## 📍 Nearby Attractions by Distance

### ≤ 10 km — Right Here
List every named place found within 10 km. For each:
- **Name** (category) — X km · Why it's worth visiting / what it's known for
- [View on Google Maps](https://www.google.com/maps/search/{dest_enc}+attraction)

### 10 – 30 km
### 30 – 50 km
### 50 – 100 km

(If a distance bucket has no OSM data, mention 2–3 places from the web search.)

## 🍜 Local Food & Specialties
- Top 5–8 dishes or street foods this region is famous for
- Where / how to find them (market, roadside stall, specific restaurant if known)
- Price range

## 🛍️ Shopping & Markets
- Local markets, bazaars, or craft centres
- What they're known for (textiles, spices, handicrafts…)
- Best time to visit

## 🎭 Culture, Festivals & Traditions
- What makes this place and its people unique
- Any festivals, religious events, or seasonal highlights
- Local customs travellers should know

## 💎 Hidden Gems & Local Secrets
- Spots most tourists never find
- Offbeat experiences, viewpoints, or local favourites
- Any insider tips from the web search

Be specific and use real place names from the data. If information is limited for this place,
draw on what's available for the broader region and clearly say so.
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
