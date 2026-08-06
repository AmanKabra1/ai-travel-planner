"""Helpers to list, load, and delete conversation threads stored by the
LangGraph Postgres checkpointer."""

import logging

import psycopg

from config import DATABASE_URL
from graph import app

logger = logging.getLogger(__name__)


def list_threads(user_id: str):
    """Return this user's threads (newest first) as
    [{"thread_id", "ts", "query"}]. A thread belongs to the user when its id
    equals the user_id (case-insensitive) or starts with '<user_id>_'."""
    user_id = user_id.strip().lower()

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Fetch all threads with their latest timestamp
                cur.execute(
                    """
                    SELECT thread_id, MAX(checkpoint->>'ts') AS ts
                    FROM checkpoints
                    GROUP BY thread_id
                    ORDER BY MAX(checkpoint->>'ts') DESC
                    """
                )
                rows = cur.fetchall()
                logger.debug("Found %d total threads in DB", len(rows))
    except Exception as exc:
        logger.warning("list_threads DB query failed: %s", exc)
        return []

    # Filter to this user's threads: exact match or starts with "user_id_"
    prefix = f"{user_id}_"
    user_threads = [
        (tid, ts) for tid, ts in rows
        if (tid == user_id or tid.startswith(prefix))
    ]
    logger.debug("Filtered to %d threads for user '%s'", len(user_threads), user_id)

    # Sort by timestamp, newest first
    user_threads.sort(key=lambda r: r[1] or "", reverse=True)

    threads = []
    for thread_id, ts in user_threads:
        query = ""
        try:
            state = app.get_state({"configurable": {"thread_id": thread_id}})
            if state and state.values:
                query = state.values.get("user_query", "") or ""
        except Exception as exc:
            logger.debug("Could not load query for thread %s: %s", thread_id, exc)

        threads.append({"thread_id": thread_id, "ts": ts or "", "query": query})

    logger.info("list_threads returning %d threads for user '%s'", len(threads), user_id)
    return threads


def load_thread(thread_id: str):
    """Return (saved_state_values, waiting_for_approval) for a thread."""
    try:
        state = app.get_state({"configurable": {"thread_id": thread_id}})
        values = dict(state.values) if state and state.values else {}
        waiting = bool(state.next) and ("human_approval" in state.next)
        logger.info("Loaded thread %s: waiting=%s", thread_id, waiting)
        return values, waiting
    except Exception as exc:
        logger.warning("load_thread failed for %s: %s", thread_id, exc)
        return {}, False


def delete_thread(thread_id: str):
    """Remove all checkpoints/writes for a thread from Postgres."""
    try:
        app.checkpointer.delete_thread(thread_id)
        logger.info("Deleted thread %s", thread_id)
    except Exception as exc:
        logger.warning("delete_thread failed for %s: %s", thread_id, exc)
