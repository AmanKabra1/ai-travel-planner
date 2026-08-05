"""Dummy auth module — stores hashed passwords in the same PostgreSQL DB.
This is intentionally simple (SHA-256, no salt) and is meant for portfolio
demonstration only, not production security.
"""

import hashlib
import logging

import psycopg

from config import DATABASE_URL

logger = logging.getLogger(__name__)


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def setup() -> None:
    """Create the users table if it does not already exist."""
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username      TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at    TIMESTAMPTZ DEFAULT NOW()
            )
        """)


def create_user(username: str, password: str) -> tuple[bool, str]:
    """Register a new user.  Returns (success, error_message)."""
    u = username.strip().lower()
    if len(u) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (u, _hash(password)),
            )
        return True, ""
    except Exception as exc:
        msg = str(exc).lower()
        if "unique" in msg or "duplicate" in msg:
            return False, "Username already taken — try another."
        logger.warning("create_user error: %s", exc)
        return False, "Could not create account. Please try again."


def verify_user(username: str, password: str) -> bool:
    """Return True when credentials match a stored user."""
    u = username.strip().lower()
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = %s", (u,)
            ).fetchone()
    except Exception as exc:
        logger.warning("verify_user error: %s", exc)
        return False
    return bool(row) and row[0] == _hash(password)
