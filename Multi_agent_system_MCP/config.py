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
_auto_model: str | None = None   # cached after first successful detection


def _detect_best_groq_model() -> str:
    """Query Groq /models, pick the best chat model available."""
    import urllib.request, json as _json
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return GROQ_FALLBACKS[0]
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = _json.loads(resp.read())
        ids = [m["id"] for m in data.get("data", [])
               if not any(x in m["id"].lower() for x in _EXCLUDE_KEYWORDS)]

        # Prefer our known-good list first (order matters)
        for candidate in GROQ_FALLBACKS:
            if candidate in ids:
                _log.info("Auto-selected Groq model: %s", candidate)
                return candidate

        # Fallback: pick any 70b or large model from the live list
        for mid in ids:
            if any(k in mid.lower() for k in _PREFER_KEYWORDS):
                _log.info("Auto-selected Groq model (live list): %s", mid)
                return mid

        # Last resort: first available model
        if ids:
            _log.info("Auto-selected Groq model (first available): %s", ids[0])
            return ids[0]
    except Exception as exc:
        _log.warning("Could not fetch Groq model list (%s); using fallback", exc)
    return GROQ_FALLBACKS[0]


def get_llm(model: str | None = None) -> ChatGroq:
    """Return a ChatGroq instance.

    Priority:
      1. explicit model arg (used by the retry loop in agents.py)
      2. GROQ_MODEL env var  (set in Streamlit secrets — optional)
      3. auto-detect from Groq /models API
      4. hardcoded fallback list
    """
    global _auto_model
    if model:
        return ChatGroq(model=model)
    env_model = os.getenv("GROQ_MODEL", "").strip()
    if env_model:
        return ChatGroq(model=env_model)
    # Auto-detect once per process and cache the result
    if _auto_model is None:
        _auto_model = _detect_best_groq_model()
    return ChatGroq(model=_auto_model)
