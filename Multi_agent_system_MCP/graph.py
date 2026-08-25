from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from agents import (
    budget_agent,
    final_response_agent,
    hotel_agent,
    human_approval_agent,
    itinerary_agent,
    nearby_agent,
    supervisor_agent,
    transport_agent,
    weather_agent,
)
from config import DATABASE_URL
from state import TravelState

AGENT_ORDER = [
    "transport_agent",
    "hotel_agent",
    "weather_agent",
    "nearby_agent",
    "budget_agent",
    "itinerary_agent",
]

ROUTE_MAP = {
    "transport_agent": "transport_agent",
    "hotel_agent":     "hotel_agent",
    "weather_agent":   "weather_agent",
    "nearby_agent":    "nearby_agent",
    "budget_agent":    "budget_agent",
    "itinerary_agent": "itinerary_agent",
    "human_approval":  "human_approval",   # must be here or routing silently fails
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents") or []
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)
        for next_agent in AGENT_ORDER[current_index + 1:]:
            # Skip itinerary_agent — it runs AFTER human_approval
            if next_agent in selected and next_agent != "itinerary_agent":
                return next_agent
        # All research agents done → pause for user choices
        return "human_approval"
    return route


def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("supervisor",      supervisor_agent)
    graph.add_node("transport_agent", transport_agent)
    graph.add_node("hotel_agent",     hotel_agent)
    graph.add_node("weather_agent",   weather_agent)
    graph.add_node("nearby_agent",    nearby_agent)
    graph.add_node("budget_agent",    budget_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("human_approval",  human_approval_agent)
    graph.add_node("final_response",  final_response_agent)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor",       route_from_supervisor,              ROUTE_MAP)
    graph.add_conditional_edges("transport_agent",  route_after_agent("transport_agent"), ROUTE_MAP)
    graph.add_conditional_edges("hotel_agent",      route_after_agent("hotel_agent"),     ROUTE_MAP)
    graph.add_conditional_edges("weather_agent",    route_after_agent("weather_agent"),   ROUTE_MAP)
    graph.add_conditional_edges("nearby_agent",     route_after_agent("nearby_agent"),    ROUTE_MAP)
    graph.add_conditional_edges("budget_agent",     route_after_agent("budget_agent"),    ROUTE_MAP)
    # human_approval fires BEFORE itinerary so user choices shape the plan
    graph.add_edge("human_approval",  "itinerary_agent")
    graph.add_edge("itinerary_agent", "final_response")
    graph.add_edge("final_response",  END)

    if DATABASE_URL:
        # ConnectionPool auto-reconnects when Neon closes idle connections.
        # autocommit=True + prepare_threshold=0 are required for Neon's
        # PgBouncer pooler (DDL outside transactions, no prepared statements).
        pool = ConnectionPool(
            conninfo=DATABASE_URL,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            min_size=1,
            max_size=5,
            open=True,
        )
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        return graph.compile(checkpointer=checkpointer)

    return graph.compile()


app = build_graph()
