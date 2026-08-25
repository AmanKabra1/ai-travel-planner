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


# ── Known model names to try (newest / most likely first) ─────────────────────
# Includes 2025-era models and speculative 2026 successors.
# The app also queries the live /models API so real names always win.
GROQ_FALLBACKS = [
    # Llama 4 family (Meta, released ~Apr 2025)
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-4-maverick-17b-128e-instruct",
    "llama-4-scout-17b-16e-instruct",
    # Llama 3.3 / 3.1
    "llama-3.3-70b-specdec",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    # Qwen / DeepSeek (added to Groq ~early 2025)
    "qwen-qwq-32b",
    "deepseek-r1-distill-llama-70b",
    # Compound
    "compound-beta",
    "compound-beta-mini",
    # Llama 3 / Gemma / Mixtral (legacy, may be decommissioned)
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

_PREFER_KEYWORDS  = ["70b", "maverick", "scout", "versatile", "qwq", "deepseek", "compound"]
_EXCLUDE_KEYWORDS = ["embed", "whisper", "guard", "tts", "vision"]
_SMALL_KEYWORDS   = ["8b", "7b", "9b", "gemma2", "3b", "1b"]  # avoid for complex tasks


def _model_score(model_id: str) -> int:
    m = model_id.lower()
    if any(x in m for x in _EXCLUDE_KEYWORDS):
        return -1
    if any(x in m for x in ["maverick", "scout"]):
        return 10
    if any(x in m for x in ["70b", "qwq", "deepseek", "compound"]):
        return 8
    if any(x in m for x in _SMALL_KEYWORDS):
        return 1   # too small — usable as last resort only
    return 5

_log = logging.getLogger(__name__)
_verified_model: str | None = None
_live_model_ids: list[str] = []   # cached after first successful API call


def _fetch_live_model_ids(api_key: str) -> list[str]:
    """Return live Groq model IDs via the groq Python client (fallback: raw HTTP).
    Also populates _live_model_ids for use by the retry loop."""
    global _live_model_ids

    # Method 1 — groq client (most reliable, handles auth correctly)
    try:
        from groq import Groq as _Groq
        client = _Groq(api_key=api_key)
        resp = client.models.list()
        ids = [m.id for m in resp.data
               if not any(x in m.id.lower() for x in _EXCLUDE_KEYWORDS)]
        _log.info("Groq client listed %d models: %s", len(ids), ids)
        _live_model_ids = ids
        return ids
    except Exception as exc:
        _log.warning("groq client model list failed: %s", exc)

    # Method 2 — raw HTTP
    import json as _j, urllib.request as _ur
    try:
        req = _ur.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
        )
        with _ur.urlopen(req, timeout=8) as resp:
            data = _j.loads(resp.read())
        ids = [m["id"] for m in data.get("data", [])
               if not any(x in m["id"].lower() for x in _EXCLUDE_KEYWORDS)]
        _log.info("HTTP model list: %d models: %s", len(ids), ids)
        _live_model_ids = ids
        return ids
    except Exception as exc:
        _log.warning("HTTP model list failed: %s", exc)

    return []


def get_retry_models() -> list[str]:
    """Return the list of real Groq model IDs to try when the primary fails.
    Uses the live API-sourced list (populated at startup); falls back to the
    static GROQ_FALLBACKS if the API call hasn't run yet."""
    return list(_live_model_ids) if _live_model_ids else list(GROQ_FALLBACKS)


def _find_working_groq_model() -> str:
    """
    1. Fetch live model list from Groq API.
    2. Build candidate list (live list first, then our fallback list).
    3. Test-call each candidate with a tiny message until one works.
    4. Return and cache the first working model.
    """
    from langchain_core.messages import HumanMessage as _HMsg
    api_key = os.getenv("GROQ_API_KEY", "")

    live_ids = _fetch_live_model_ids(api_key)

    # Build candidate list: live models first, then static fallback list.
    # Sort by quality score so large capable models are tried first.
    seen: set[str] = set()
    candidates: list[str] = []
    for m in live_ids:
        if m not in seen and _model_score(m) >= 0:
            seen.add(m)
            candidates.append(m)
    for m in GROQ_FALLBACKS:
        if m not in seen and _model_score(m) >= 0:
            seen.add(m)
            candidates.append(m)

    candidates.sort(key=_model_score, reverse=True)
    _log.info("Candidates (sorted by quality): %s", candidates)

    for candidate in candidates:
        try:
            ChatGroq(model=candidate, max_tokens=64).invoke([_HMsg(content="hi")])
            _log.info("✅ Verified working Groq model: %s", candidate)
            return candidate
        except Exception as exc:
            _log.warning("Model %s failed (%s) — trying next", candidate, str(exc)[:100])

    _log.error("No working Groq model found! Check your GROQ_API_KEY.")
    return candidates[0] if candidates else GROQ_FALLBACKS[0]


def get_llm(model: str | None = None, max_tokens: int = 4096) -> ChatGroq:
    """Return a verified, working ChatGroq instance.

    Priority:
      1. explicit model arg  — used by the per-call retry loop in agents.py
      2. auto-detect + verify — queries live API + test-calls until one works;
         result is cached for the whole process lifetime.
         (GROQ_MODEL secret is intentionally ignored so stale values can't
          break the app — delete it from Streamlit secrets if set.)
    """
    global _verified_model
    if model:
        return ChatGroq(model=model, max_tokens=max_tokens, temperature=0.3)
    if _verified_model is None:
        _verified_model = _find_working_groq_model()
    return ChatGroq(model=_verified_model, max_tokens=max_tokens, temperature=0.3)
