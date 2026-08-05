import asyncio
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import get_llm
from mcp_client import current_weather, forecast, list_airlines, list_airports, tavily_search
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
    search_query = f"Best hotels and areas to stay for: {state['user_query']}"
    logger.info("Hotel agent — query: %s", search_query[:80])

    try:
        raw         = asyncio.run(tavily_search(search_query))
        search_text = _extract_text(raw)
    except Exception as exc:
        logger.warning("Hotel Tavily search failed: %s", exc)
        search_text = "Live search unavailable."

    result = _llm_text(
        "You are a hotel recommendation specialist.",
        f"""Based on these web search results, recommend the best hotels and areas to stay.

User request:
{state['user_query']}

Search results:
{search_text[:4000]}

Return a clean, readable summary: list recommended areas and specific hotels
with a short note on vibe and budget level for each. Do not output raw JSON.
""",
    )

    return {
        "hotel_results": result,
        "messages":      [AIMessage(content="Hotel agent completed.")],
        "llm_calls":     state.get("llm_calls", 0) + 1,
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

Hotel info:
{state.get('hotel_results', 'N/A')}

Weather info:
{state.get('weather_results', 'N/A')}

Provide:
1. Estimated cost by category (flights, accommodation, food, transport, activities)
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


# ── Itinerary agent ───────────────────────────────────────────────────────────

def itinerary_agent(state: TravelState):
    logger.info("Itinerary agent running")

    result = _llm_text(
        "You are an expert travel itinerary planner.",
        f"""Create a detailed, day-by-day travel itinerary.

User request:
{state['user_query']}

Trip constraints:
{state.get('trip_constraints', {})}

Flight info:
{state.get('flight_results', '')}

Hotel info:
{state.get('hotel_results', '')}

Weather info:
{state.get('weather_results', '')}

Budget analysis:
{state.get('budget_results', '')}

Format the output clearly with Day-by-Day sections, practical tips,
and a summary. Make it ready for human review and approval.
""",
    )

    approval_request = (
        f"Please review this draft travel plan.\n\n{result}\n\nApprove or provide feedback."
    )

    return {
        "itinerary":        result,
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

Draft itinerary:
{state.get('itinerary', '')}

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

Budget notes:
{state.get('budget_results', '')}

Revise the itinerary to address the user's feedback. Make it final and polished.
"""

    result = _llm_text("You produce final, user-ready travel plans.", prompt)

    return {
        "final_response": result,
        "messages":       [AIMessage(content=result)],
        "llm_calls":      state.get("llm_calls", 0) + 1,
    }
