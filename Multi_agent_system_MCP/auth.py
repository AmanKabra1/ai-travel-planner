"""Auth module — hashed passwords + persistent session tokens + shareable plans in PostgreSQL.
Sessions survive page refreshes; logout explicitly invalidates the token.
"""

import hashlib
import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import psycopg

from config import DATABASE_URL

logger = logging.getLogger(__name__)

SESSION_DAYS = 30


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def setup() -> None:
    """Create all tables if they do not already exist."""
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username      TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at    TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                username   TEXT NOT NULL,
                expires_at TIMESTAMPTZ NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_plans (
                share_id   TEXT PRIMARY KEY,
                username   TEXT NOT NULL,
                plan_json  TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)


def create_user(username: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, error_message)."""
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


# ── Session tokens ────────────────────────────────────────────────────────────

def create_session(username: str) -> str:
    """Generate a random token, persist it, and return it."""
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute(
                "INSERT INTO sessions (token, username, expires_at) VALUES (%s, %s, %s)",
                (token, username.strip().lower(), expires),
            )
    except Exception as exc:
        logger.warning("create_session error: %s", exc)
    return token


def validate_session(token: str) -> str | None:
    """Return the username if the token is valid and not expired, else None."""
    if not token:
        return None
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            row = conn.execute(
                "SELECT username FROM sessions WHERE token = %s AND expires_at > NOW()",
                (token,),
            ).fetchone()
        return row[0] if row else None
    except Exception as exc:
        logger.warning("validate_session error: %s", exc)
        return None


def delete_session(token: str) -> None:
    """Invalidate a session token (called on logout)."""
    if not token:
        return
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute("DELETE FROM sessions WHERE token = %s", (token,))
    except Exception as exc:
        logger.warning("delete_session error: %s", exc)


# ── Shareable plans ───────────────────────────────────────────────────────────

def create_share(username: str, plan_state: dict) -> str:
    """Persist a plan state dict and return a share UUID."""
    share_id = str(uuid.uuid4())
    try:
        plan_json = json.dumps(plan_state, default=str)
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute(
                "INSERT INTO shared_plans (share_id, username, plan_json) VALUES (%s, %s, %s)",
                (share_id, username.strip().lower(), plan_json),
            )
    except Exception as exc:
        logger.warning("create_share error: %s", exc)
    return share_id


def load_share(share_id: str) -> dict | None:
    """Return the plan state dict for a share_id, or None if not found."""
    if not share_id:
        return None
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            row = conn.execute(
                "SELECT plan_json FROM shared_plans WHERE share_id = %s",
                (share_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None
    except Exception as exc:
        logger.warning("load_share error: %s", exc)
        return None
