import logging
import os
import sys

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ── Encoding fix ──────────────────────────────────────────────────────────────
# Windows consoles default to cp1252, which cannot encode characters such as
# the rupee sign (₹) or degree symbol (°C) that appear in LLM output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ── Environment ───────────────────────────────────────────────────────────────
# Always load the .env sitting next to this file, regardless of CWD.
HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"), override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# Accept both spellings of the AviationStack key.
AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATABASE_URL        = os.getenv("DATABASE_URL")


GROQ_FALLBACKS = [
    "llama3-70b-8192",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-8b-8192",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

# Prefer 70b / large models; avoid embed/whisper/guard models
_PREFER_KEYWORDS  = ["70b", "versatile", "405b", "large"]
_EXCLUDE_KEYWORDS = ["embed", "whisper", "guard", "tts", "vision"]

_log = logging.getLogger(__name__)
_verified_model: str | None = None   # cached after first successful test-call


def _find_working_groq_model() -> str:
    """
    Query Groq /models, then test-call each candidate until one actually works.
    Returns the first model that responds successfully.
    Cached for the lifetime of the process.
    """
    import json as _json
    import urllib.request
    from langchain_core.messages import HumanMessage as _HMsg

    api_key = os.getenv("GROQ_API_KEY", "")

    # 1. Get the live model list from Groq
    live_ids: list[str] = []
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = _json.loads(resp.read())
        live_ids = [
            m["id"] for m in data.get("data", [])
            if not any(x in m["id"].lower() for x in _EXCLUDE_KEYWORDS)
        ]
        _log.info("Groq reported %d chat models", len(live_ids))
    except Exception as exc:
        _log.warning("Could not fetch Groq model list: %s", exc)

    # 2. Build candidate list: our preferred order first, then live list
    candidates: list[str] = []
    for m in GROQ_FALLBACKS:
        if m not in candidates:
            candidates.append(m)
    for m in live_ids:
        if m not in candidates:
            if any(k in m.lower() for k in _PREFER_KEYWORDS):
                candidates.insert(0, m)
            else:
                candidates.append(m)

    # 3. Test-call each candidate — first one that responds is the winner
    for candidate in candidates:
        try:
            ChatGroq(model=candidate).invoke([_HMsg(content="hi")])
            _log.info("Verified working Groq model: %s", candidate)
            return candidate
        except Exception as exc:
            _log.warning("Model %s failed (%s) — trying next", candidate, str(exc)[:80])

    _log.error("No working Groq model found; defaulting to %s", GROQ_FALLBACKS[0])
    return GROQ_FALLBACKS[0]


def get_llm(model: str | None = None) -> ChatGroq:
    """Return a ChatGroq instance.

    Priority:
      1. explicit model arg  — used by the retry loop in agents.py
      2. auto-detect + verify — tests each model until one works; cached
         (GROQ_MODEL secret is ignored so a stale value can't break the app)
    """
    global _verified_model
    if model:
        return ChatGroq(model=model)
    if _verified_model is None:
        _verified_model = _find_working_groq_model()
    return ChatGroq(model=_verified_model)
