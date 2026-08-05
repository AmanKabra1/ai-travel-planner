# CLI runner for the Multi-Agent Travel Planner (modular architecture).
# For the web UI use:  streamlit run frontend.py

import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import app


def run():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_id": "cli_user",
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    # The graph pauses at the human_approval node via interrupt().
    if "__interrupt__" in result:
        draft = result["__interrupt__"][0].value.get("draft_itinerary", "")

        print("\n================ DRAFT ITINERARY ================\n")
        print(draft)
        print("\n=================================================\n")

        answer = input("Approve this itinerary? (y/n): ").strip().lower()
        approved = answer in ("y", "yes")

        feedback = ""
        if not approved:
            feedback = input("What would you like changed? ")

        result = app.invoke(
            Command(resume={"approved": approved, "feedback": feedback}),
            config=config,
        )

    print("\n================ FINAL TRAVEL PLAN ================\n")
    print(result.get("final_response", "No final response generated."))


if __name__ == "__main__":
    run()
