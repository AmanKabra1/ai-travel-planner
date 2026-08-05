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


def get_llm():
    """Return a configured ChatGroq instance using the env-specified model."""
    return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
