"""Helpers to list, load, and delete conversation threads stored by the
LangGraph Postgres checkpointer."""

import psycopg

from config import DATABASE_URL
from graph import app


def list_threads(user_id: str):
    """Return this user's threads (newest first) as
    [{"thread_id", "ts", "query"}]. A thread belongs to the user when its id
    equals the user_id or starts with '<user_id>_'."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT thread_id, MAX(checkpoint->>'ts') AS ts
                FROM checkpoints
                GROUP BY thread_id
                """
            )
            rows = cur.fetchall()

    prefix = f"{user_id}_"
    rows = [
        (tid, ts) for tid, ts in rows
        if tid == user_id or tid.startswith(prefix)
    ]
    rows.sort(key=lambda r: r[1] or "", reverse=True)

    threads = []
    for thread_id, ts in rows:
        query = ""
        try:
            state = app.get_state({"configurable": {"thread_id": thread_id}})
            if state and state.values:
                query = state.values.get("user_query", "") or ""
        except Exception:
            pass
        threads.append({"thread_id": thread_id, "ts": ts or "", "query": query})

    return threads


def load_thread(thread_id: str):
    """Return (saved_state_values, waiting_for_approval) for a thread."""
    state = app.get_state({"configurable": {"thread_id": thread_id}})
    values = dict(state.values) if state and state.values else {}
    waiting = bool(state.next) and ("human_approval" in state.next)
    return values, waiting


def delete_thread(thread_id: str):
    """Remove all checkpoints/writes for a thread from Postgres."""
    app.checkpointer.delete_thread(thread_id)
