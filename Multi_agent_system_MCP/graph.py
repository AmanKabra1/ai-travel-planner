from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from agents import (
    final_response_agent,
    human_approval_agent,
    itinerary_agent,
    research_all_agent,
    supervisor_agent,
)
from config import DATABASE_URL
from state import TravelState


def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("supervisor",     supervisor_agent)
    graph.add_node("research_all",   research_all_agent)   # runs all 5 agents in parallel
    graph.add_node("human_approval", human_approval_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("final_response", final_response_agent)

    graph.add_edge(START,            "supervisor")
    graph.add_edge("supervisor",     "research_all")
    graph.add_edge("research_all",   "human_approval")
    graph.add_edge("human_approval", "itinerary_agent")
    graph.add_edge("itinerary_agent","final_response")
    graph.add_edge("final_response", END)

    if DATABASE_URL:
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
