import json
import sys
import uuid
from datetime import date, datetime, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

import auth
from graph import app
from thread_history import list_threads, load_thread, delete_thread

auth.setup()

# ── Auto-login: restore session from URL query param on every page load ────────
if not st.session_state.get("authenticated"):
    _tok = st.query_params.get("s", "")
    if _tok:
        _saved = auth.validate_session(_tok)
        if _saved:
            st.session_state["authenticated"]  = True
            st.session_state["username"]        = _saved
            st.session_state["session_token"]   = _tok

st.set_page_config(
    page_title="Wandr — AI Travel Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"]   { background: #060d1f; }
[data-testid="stSidebar"]            { background: #0a1628; border-right: 1px solid #1a3a6b; }
[data-testid="stHeader"]             { background: #060d1f !important; border-bottom: 1px solid #0f2341 !important; }
[data-testid="stMainBlockContainer"] { padding-top: 3.5rem !important; }
[data-testid="stBottom"]             { background: #060d1f; }
[data-testid="stDecoration"]         { display: none; }
.stApp { background: #060d1f; }

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #060d1f 0%, #0a1f3d 50%, #062040 100%);
    border: 1px solid #1a3a6b;
    border-radius: 20px;
    padding: 2.8rem 3rem 2rem;
    margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(14,165,233,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-wrap::after {
    content: '';
    position: absolute; bottom: -40px; left: 20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #0ea5e9; margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #bfdbfe 50%, #67e8f9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2; padding: 0.1em 0; margin-bottom: 0.6rem;
    display: inline-block;
}
.hero-sub {
    color: #64748b; font-size: 0.95rem; line-height: 1.6;
    margin-bottom: 0;
}
.hero-pills {
    display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1.2rem;
}
.hero-pill {
    background: rgba(14,165,233,0.1); border: 1px solid rgba(14,165,233,0.25);
    border-radius: 20px; padding: 4px 14px;
    color: #7dd3fc; font-size: 0.78rem; font-weight: 600;
}

/* ── Destination chips ── */
.chips-label {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: #475569; margin-bottom: 0.5rem;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1.2rem; }
.chip {
    background: #0c1b35; border: 1px solid #1a3a6b;
    border-radius: 20px; padding: 5px 14px;
    color: #94a3b8; font-size: 0.8rem; cursor: pointer;
    transition: all 0.2s;
}
.chip:hover { background: #0f2341; border-color: #0ea5e9; color: #7dd3fc; }

/* ── Travel style tags ── */
.style-grid {
    display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 1rem;
}
.style-tag {
    background: #0c1b35; border: 1px solid #1a3a6b;
    border-radius: 8px; padding: 6px 14px;
    color: #64748b; font-size: 0.82rem; font-weight: 500;
    cursor: default;
}

/* ── Auth card ── */
.auth-card {
    background: #0a1628; border: 1px solid #1a3a6b;
    border-radius: 20px; padding: 2.8rem 3rem;
    box-shadow: 0 30px 80px rgba(0,0,0,.7);
}
.auth-icon  { text-align: center; font-size: 3.5rem; margin-bottom: 0.6rem; }
.auth-title {
    text-align: center; font-size: 1.9rem; font-weight: 800;
    background: linear-gradient(135deg, #0ea5e9, #06b6d4, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; display: block; margin-bottom: 0.2rem;
    line-height: 1.3; padding: 0.05em 0;
}
.auth-sub {
    text-align: center; color: #475569; font-size: 0.88rem;
    margin-bottom: 1.8rem; line-height: 1.5;
}
.auth-features {
    display: flex; justify-content: center; gap: 20px;
    flex-wrap: wrap; margin-bottom: 1.8rem;
}
.auth-feature {
    color: #334155; font-size: 0.8rem;
    display: flex; align-items: center; gap: 5px;
}
.auth-feature-dot { color: #0ea5e9; }

/* ── Typography ── */
h1 { color: #e2e8f0 !important; }
h2, h3 { color: #bfdbfe !important; }
h4 { color: #94a3b8 !important; }
p, li { color: #cbd5e1; }
.stMarkdown p { color: #cbd5e1; }

/* ── Inputs ── */
textarea, input[type="text"], input[type="password"] {
    background: #0c1b35 !important; color: #e2e8f0 !important;
    border: 1px solid #1a3a6b !important; border-radius: 10px !important;
    font-size: 0.95rem !important;
}
textarea:focus, input[type="text"]:focus, input[type="password"]:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,.15) !important;
}
/* ── Kill ALL red/orange borders — Streamlit uses data-baseweb="base-input" ── */
[data-baseweb="base-input"],
[data-baseweb="input"],
[data-baseweb="base-input"] > div,
[data-baseweb="input"] > div { border-color: #1a3a6b !important; box-shadow: none !important; }

[data-baseweb="base-input"]:focus-within,
[data-baseweb="input"]:focus-within {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,.15) !important;
}
/* Catch inline-style red borders Streamlit injects on validation */
input:focus { outline: none !important; }
input:invalid, input[aria-invalid="true"],
textarea:invalid { border-color: #1a3a6b !important; box-shadow: none !important; }
/* Nuke any remaining red with an attribute-value wildcard */
[style*="rgb(255, 75, 75)"] { color: #94a3b8 !important; }
[style*="border-color: rgb(255"] { border-color: #1a3a6b !important; }
label { color: #94a3b8 !important; }

/* ── Tab active indicator — override Streamlit's default red/pink ── */
[data-baseweb="tab-highlight"] {
    background-color: #0ea5e9 !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #0ea5e9 !important;
}
[data-baseweb="tab-border"] { background-color: #1a3a6b !important; }

/* ── Kill browser spellcheck red underlines on all inputs ── */
input, textarea {
    -webkit-text-decoration: none !important;
    text-decoration: none !important;
}
.stTextArea label { display: none; }

/* ── No text-wrap on all buttons ── */
button p { white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important; }

/* ── Route row (From → To) ── */
.route-row {
    display: flex; align-items: center; gap: 0; margin-bottom: 1rem;
}
.route-arrow {
    color: #0ea5e9; font-size: 1.6rem; font-weight: 300;
    padding: 0 0.6rem; padding-top: 1.6rem; flex-shrink: 0;
    user-select: none;
}
.route-label {
    font-size: 0.73rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #334155; margin-bottom: 3px;
}
/* Highlight the from/to inputs differently from the query textarea */
div[data-testid="stTextInput"]:has(input[aria-label="From"]) input,
div[data-testid="stTextInput"]:has(input[aria-label="To"]) input {
    border-color: #1a3a6b !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
div[data-testid="stTextInput"]:has(input[aria-label="To"]) input {
    border-color: #0369a1 !important;
    background: #0c1f38 !important;
}

/* ── Primary button ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    border: none !important; color: #fff !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 0.95rem !important; transition: all 0.25s !important;
    letter-spacing: 0.01em !important;
}
button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 30px rgba(14,165,233,.4) !important;
    background: linear-gradient(135deg, #0284c7, #38bdf8) !important;
}

/* ── Secondary button ── */
button[kind="secondary"] {
    background: #0c1b35 !important; color: #64748b !important;
    border: 1px solid #1a3a6b !important; border-radius: 8px !important;
    font-size: 0.8rem !important; transition: all 0.2s !important;
}
button[kind="secondary"]:hover {
    border-color: #0ea5e9 !important; color: #7dd3fc !important;
    background: #0f2341 !important;
}

/* ── Pipeline strip ── */
.pipeline {
    display: flex; align-items: center; flex-wrap: wrap; gap: 3px;
    background: #0a1628; border: 1px solid #1a3a6b; border-radius: 12px;
    padding: 0.8rem 1.2rem; margin: 1rem 0;
}
.step {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 13px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; white-space: nowrap;
}
.step-done { background: #0c2e1a; color: #6ee7b7; border: 1px solid #065f46; }
.step-run  {
    background: #0c1f38; color: #7dd3fc; border: 1px solid #0369a1;
    animation: pulse 1.4s ease-in-out infinite;
}
.step-wait { background: #0a1628; color: #334155; border: 1px solid #1a3a6b; }
.sep       { color: #1a3a6b; font-size: 0.7rem; user-select: none; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.55} }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0a1628; border: 1px solid #1a3a6b;
    border-radius: 12px; padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700 !important; font-size: 1.5rem !important; }
[data-testid="stMetricLabel"] { color: #475569  !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid #1a3a6b !important;
    background: transparent !important;
}
[data-baseweb="tab"] {
    color: #475569 !important; font-weight: 600;
    border-radius: 0 !important; background: transparent !important;
    padding: 0.6rem 1.2rem !important;
    font-size: 0.88rem !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #0ea5e9 !important;
}
[data-baseweb="tab-panel"] {
    background: transparent !important; padding-top: 1.2rem;
}

/* ── Result content cards ── */
.result-card {
    background: #0a1628; border: 1px solid #1a3a6b;
    border-radius: 12px; padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
}
.result-card h3, .result-card h4 { color: #7dd3fc !important; }

/* ── Approval box ── */
.approval-box {
    background: linear-gradient(135deg, #0a1628, #0c1f38);
    border: 2px solid #0369a1; border-radius: 16px;
    padding: 1.8rem 2rem; margin: 1.2rem 0;
    position: relative; overflow: hidden;
}
.approval-box::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #0ea5e9, #06b6d4, #67e8f9);
}
.approval-title {
    color: #7dd3fc; font-size: 1.2rem; font-weight: 700;
    margin-bottom: 0.4rem; display: flex; align-items: center; gap: 8px;
}
.approval-sub { color: #475569; font-size: 0.88rem; margin-bottom: 0; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #0c1b35 !important; border: 1px solid #1a3a6b !important;
    color: #64748b !important; border-radius: 10px !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #0ea5e9 !important; color: #7dd3fc !important;
    background: #0f2341 !important;
}

/* ── Divider ── */
hr { border-color: #1a3a6b !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] .stTextInput input { background: #060d1f !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #7dd3fc !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stCaption { color: #475569 !important; }

.thread-item {
    padding: 0.5rem 0; border-bottom: 1px solid #0c1b35;
}
.user-badge {
    background: linear-gradient(135deg, #062040, #0c2340);
    border: 1px solid #1a3a6b;
    border-radius: 10px; padding: 0.6rem 0.9rem;
    color: #7dd3fc; font-size: 0.87rem; font-weight: 600;
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.5rem;
}
.sidebar-brand {
    font-size: 1.15rem; font-weight: 800; letter-spacing: -0.01em;
    background: linear-gradient(135deg, #0ea5e9, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; display: inline-block; line-height: 1.3; padding: 0.05em 0;
}
.sidebar-tagline {
    color: #334155; font-size: 0.75rem; margin-bottom: 0;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0a1628 !important; border: 1px solid #1a3a6b !important;
    border-radius: 10px !important;
}
summary { color: #64748b !important; }
summary:hover { color: #94a3b8 !important; }

/* ── Status widget ── */
[data-testid="stStatusWidget"] {
    background: #0a1628 !important; border: 1px solid #1a3a6b !important;
    border-radius: 10px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #060d1f; }
::-webkit-scrollbar-thumb { background: #1a3a6b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #0369a1; }

/* ── Tip card ── */
.tip-card {
    background: #0c1b35; border: 1px solid #1a3a6b;
    border-radius: 10px; padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.tip-card-icon { font-size: 1rem; margin-bottom: 2px; }
.tip-card-text { color: #64748b; font-size: 0.78rem; line-height: 1.5; }

/* ── Final plan section ── */
.final-header {
    background: linear-gradient(135deg, #062040, #0c2340);
    border: 1px solid #0369a1; border-radius: 14px;
    padding: 1.4rem 1.8rem; margin-bottom: 1rem;
    border-left: 4px solid #0ea5e9;
}
.final-title {
    font-size: 1.3rem; font-weight: 700; color: #7dd3fc; margin-bottom: 0.2rem;
}
.final-sub { color: #475569; font-size: 0.85rem; }

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 1rem; padding-bottom: 0.7rem;
    border-bottom: 1px solid #1a3a6b;
}
.section-icon { font-size: 1.2rem; }
.section-title { font-size: 1.1rem; font-weight: 700; color: #bfdbfe !important; }

/* ── Preferences panel ── */
.prefs-panel {
    background: linear-gradient(135deg, #0a1628 0%, #0c1f3a 100%);
    border: 1px solid #1a3a6b; border-radius: 16px;
    padding: 2rem 2.2rem; margin-top: 1.5rem;
}
.prefs-panel-title { font-size: 1.35rem; font-weight: 700; color: #7dd3fc; margin-bottom: 0.2rem; }
.prefs-panel-sub { color: #475569; font-size: 0.84rem; margin-bottom: 1.5rem; }
.prefs-choices { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.5rem; }
.pref-pill { background: #0f2341; border: 1px solid #1a3a6b; border-radius: 20px;
    padding: 0.28rem 0.75rem; font-size: 0.78rem; color: #7dd3fc; }
.confirm-box {
    background: linear-gradient(135deg, #0c2340, #062040);
    border: 1px solid #0ea5e9; border-left: 4px solid #0ea5e9;
    border-radius: 12px; padding: 1.2rem 1.5rem; margin: 1.5rem 0 0.5rem;
}
.confirm-box-title { font-size: 1.05rem; font-weight: 700; color: #7dd3fc; margin-bottom: 0.15rem; }
.confirm-box-sub { color: #64748b; font-size: 0.83rem; }
</style>
<script>
(function disableSpellcheck() {
    function patch(root) {
        root.querySelectorAll('input, textarea').forEach(function(el) {
            el.setAttribute('spellcheck', 'false');
            el.setAttribute('autocomplete', 'off');
            el.setAttribute('autocorrect', 'off');
            el.setAttribute('autocapitalize', 'off');
        });
    }
    patch(document);
    new MutationObserver(function(mutations) {
        mutations.forEach(function(m) {
            m.addedNodes.forEach(function(n) {
                if (n.nodeType === 1) patch(n);
            });
        });
    }).observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
_PIPELINE = [
    ("🗺️", "Planning",   "supervisor_reasoning"),
    ("🚀", "Transport",   "transport_results"),
    ("🏨", "Hotels",     "hotel_results"),
    ("🌤️", "Weather",   "weather_results"),
    ("📍", "Nearby",     "nearby_results"),
    ("💰", "Budget",     "budget_results"),
    ("📋", "Itinerary",  "itinerary"),
    ("✍️", "Your Review","approved"),
    ("🎉", "Ready",      "final_response"),
]

_NODE_LABELS = {
    "supervisor":       "🗺️  Planning your trip…",
    "transport_agent":  "🚀  Searching flights, trains & buses…",
    "hotel_agent":      "🏨  Finding the best accommodation…",
    "weather_agent":    "🌤️  Checking destination weather…",
    "nearby_agent":     "📍  Discovering nearby attractions…",
    "budget_agent":     "💰  Calculating costs & budget…",
    "itinerary_agent":  "📋  Crafting your day-by-day itinerary…",
    "human_approval":   "✍️  Ready for your review…",
    "final_response":   "🎉  Putting the finishing touches…",
}

_QUICK_DESTINATIONS = [
    ("🇯🇵", "Japan"),
    ("🇫🇷", "Paris"),
    ("🇮🇩", "Bali"),
    ("🇦🇪", "Dubai"),
    ("🇬🇧", "London"),
    ("🇹🇭", "Bangkok"),
    ("🇮🇹", "Rome"),
    ("🇸🇬", "Singapore"),
    ("🇳🇵", "Nepal"),
    ("🇲🇻", "Maldives"),
]

_TRAVEL_STYLES = [
    ("🎒", "Budget"),
    ("🌟", "Luxury"),
    ("👨‍👩‍👧", "Family"),
    ("🏔️", "Adventure"),
    ("🏖️", "Beach"),
    ("🍜", "Food & Culture"),
    ("🧘", "Wellness"),
    ("📸", "Photography"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _render_pipeline(state: dict, active: str | None = None) -> None:
    html = '<div class="pipeline">'
    for i, (icon, label, key) in enumerate(_PIPELINE):
        val  = state.get(key)
        done = bool(val) if not isinstance(val, bool) else (val is not None)
        is_active = active and label.lower() in active.lower()

        if done:
            cls, txt = "step step-done", f"✓ {label}"
        elif is_active:
            cls, txt = "step step-run", f"⟳ {label}"
        else:
            cls, txt = "step step-wait", f"{icon} {label}"

        html += f'<span class="{cls}">{txt}</span>'
        if i < len(_PIPELINE) - 1:
            html += '<span class="sep"> › </span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _build_markdown_export(state: dict) -> str:
    dest  = (state.get("trip_constraints") or {}).get("destination", "your destination")
    query = state.get("user_query", "")
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# ✈️ Travel Plan — {dest}",
        f"*Generated by Wandr AI Travel Planner · {now}*\n",
        f"## Your Request\n{query}\n",
    ]
    for section, key in [
        ("Transport (Flights, Trains & Buses)", "transport_results"),
        ("Accommodation",        "hotel_results"),
        ("Weather & Climate",    "weather_results"),
        ("Nearby Attractions",   "nearby_results"),
        ("Budget Breakdown",     "budget_results"),
        ("Day-by-Day Itinerary", "itinerary"),
        ("Your Complete Travel Plan", "final_response"),
    ]:
        content = state.get(key, "")
        if content:
            lines.append(f"## {section}\n{content}\n")
    lines.append("---\n*Wandr — Your AI Travel Expert*")
    return "\n".join(lines)


def _render_structured_itinerary(itin_json: str) -> None:
    """Render structured itinerary JSON as rich Streamlit UI."""
    try:
        data = json.loads(itin_json)
    except (json.JSONDecodeError, TypeError):
        return

    # ── Trip summary card ──
    ts = data.get("trip_summary", {})
    if ts:
        cols = st.columns(4)
        cols[0].metric("From → To",  f"{ts.get('from','?')} → {ts.get('to','?')}")
        cols[1].metric("Dates",      f"{ts.get('start_date','?')} – {ts.get('end_date','?')}")
        cols[2].metric("Travellers", ts.get("members", "?"))
        cols[3].metric("Est. Total", ts.get("total_budget_estimate", "?"))

    # ── Day-by-day ──
    days = data.get("days", [])
    if days:
        st.markdown("### 🗓 Day-by-Day Plan")
        for day in days:
            label = f"Day {day.get('day','')}  {day.get('date','')}  —  ~{day.get('estimated_day_cost','?')}"
            with st.expander(label, expanded=(day.get("day") == 1)):
                meals = day.get("meals", {})
                if any(meals.values()):
                    st.markdown(
                        f"🍳 **Breakfast:** {meals.get('breakfast','—')} &nbsp;&nbsp;"
                        f"🍜 **Lunch:** {meals.get('lunch','—')} &nbsp;&nbsp;"
                        f"🍽️ **Dinner:** {meals.get('dinner','—')}"
                    )
                acts = day.get("activities", [])
                if acts:
                    rows = ["| Time | Place | Duration | Entry Fee | Notes |",
                            "|------|-------|----------|-----------|-------|"]
                    for a in acts:
                        rows.append(
                            f"| {a.get('time','—')} | **{a.get('place','—')}** "
                            f"| {a.get('duration_hrs','—')} hr | {a.get('entry_fee','—')} "
                            f"| {a.get('notes','—')} |"
                        )
                    st.markdown("\n".join(rows))

    # ── Hotels ──
    hotels = data.get("hotels", [])
    if hotels:
        st.markdown("### 🏨 Recommended Hotels")
        for h in hotels:
            tag = {"budget": "💚 Budget", "best_value": "⭐ Best Value", "premium": "💎 Premium"}.get(h.get("type",""), "🏨")
            link = f" — [Book]({h['booking_link']})" if h.get("booking_link") else ""
            st.markdown(
                f"**{tag} — {h.get('name','?')}** {link}  \n"
                f"₹/$ {h.get('price_per_night','?')}/night · ⭐ {h.get('rating','?')} · {h.get('notes','')}"
            )

    # ── Transport ──
    transport = data.get("transport", {})
    to_dest = transport.get("to_destination", [])
    if to_dest:
        st.markdown("### 🚀 Getting There")
        for t in to_dest:
            link = f" — [Book]({t['booking_link']})" if t.get("booking_link") else ""
            st.markdown(f"**{t.get('mode','?')}** {link} · {t.get('duration','?')} · ~{t.get('cost_per_person','?')} per person")

    # ── Budget breakdown ──
    bb = data.get("budget_breakdown", {})
    if bb:
        st.markdown("### 💰 Budget Breakdown")
        items = [
            ("✈️ Transport (to/from)", bb.get("transport_total","")),
            ("🏨 Accommodation",       bb.get("hotel_total","")),
            ("🍜 Food & Dining",       bb.get("food_total","")),
            ("🎫 Activities & Entry",  bb.get("activities_total","")),
            ("🚌 Local Transport",     bb.get("local_transport_total","")),
            ("🛡️ Buffer (10%)",       bb.get("buffer_10pct","")),
        ]
        for label, val in items:
            if val:
                col_l, col_r = st.columns([4, 2])
                col_l.markdown(label)
                col_r.markdown(f"**{val}**")
        st.divider()
        total_col, pp_col = st.columns(2)
        total_col.metric("Grand Total",    bb.get("grand_total","—"))
        pp_col.metric("Per Person",         bb.get("cost_per_person","—"))

    # ── Food & Markets ──
    food = data.get("local_food", [])
    if food:
        st.markdown("### 🍜 Local Food Highlights")
        for f in food:
            st.markdown(f"**{f.get('dish','?')}** — {f.get('where','?')} · {f.get('price_range','')}")

    markets = data.get("local_markets", [])
    if markets:
        st.markdown("### 🛍️ Local Markets")
        for m in markets:
            st.markdown(f"**{m.get('name','?')}** — {m.get('known_for','?')} · Best time: {m.get('best_time','?')}")

    # ── Nearby attractions ──
    attractions = data.get("nearby_attractions", [])
    if attractions:
        st.markdown("### 📍 Nearby Attractions")
        for a in attractions:
            st.markdown(
                f"**{a.get('name','?')}** ({a.get('category','')}) — "
                f"{a.get('distance_km','?')} km away · {a.get('duration_hrs','?')} hr · Entry: {a.get('entry_fee','?')}"
            )


def _build_print_html(state: dict) -> str:
    """Generate a print-ready HTML document for PDF export."""
    dest        = (state.get("trip_constraints") or {}).get("destination", "Trip")
    s_date      = (state.get("trip_constraints") or {}).get("start_date", "")
    e_date      = (state.get("trip_constraints") or {}).get("end_date", "")
    members     = (state.get("trip_constraints") or {}).get("members", "")
    query       = state.get("user_query", "")
    final       = state.get("final_response", "") or state.get("itinerary", "")
    transport   = state.get("transport_results", "")
    hotels      = state.get("hotel_results", "")
    weather     = state.get("weather_results", "")
    nearby      = state.get("nearby_results", "")
    budget      = state.get("budget_results", "")
    now         = datetime.now().strftime("%d %b %Y")

    def md_section(title, content):
        if not content:
            return ""
        paras = "".join(f"<p>{line}</p>" for line in content.split("\n") if line.strip())
        return f"<div class='section'><h2>{title}</h2>{paras}</div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Travel Plan — {dest}</title>
<style>
  body {{ font-family: 'Georgia', serif; max-width: 860px; margin: 40px auto; color: #1a202c; line-height: 1.7; }}
  h1 {{ color: #0369a1; border-bottom: 3px solid #0ea5e9; padding-bottom: 8px; }}
  h2 {{ color: #0369a1; margin-top: 28px; border-left: 4px solid #0ea5e9; padding-left: 12px; }}
  .meta {{ background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 16px 20px; margin: 20px 0; }}
  .meta span {{ margin-right: 24px; font-size: 0.95rem; color: #0369a1; }}
  .section {{ margin-bottom: 28px; }}
  .section p {{ margin: 4px 0; }}
  .footer {{ margin-top: 40px; color: #94a3b8; font-size: 0.82rem; border-top: 1px solid #e2e8f0; padding-top: 12px; }}
  @media print {{ body {{ margin: 20px; }} }}
</style>
</head>
<body>
<h1>✈️ Travel Plan — {dest}</h1>
<div class="meta">
  <span>📅 {s_date} – {e_date}</span>
  <span>👥 {members} traveller(s)</span>
  <span>🗓 Generated: {now}</span>
</div>

{md_section("Your Request", query)}
{md_section("Complete Itinerary", final)}
{md_section("Transport (Flights, Trains &amp; Buses)", transport)}
{md_section("Hotels", hotels)}
{md_section("Weather", weather)}
{md_section("Nearby Attractions", nearby)}
{md_section("Budget Breakdown", budget)}

<div class="footer">Wandr — AI Travel Planner &nbsp;|&nbsp; Open in browser and press Ctrl+P / Cmd+P to save as PDF</div>
</body>
</html>"""


def _build_pdf_bytes(state: dict) -> bytes:
    """Generate a professional travel-itinerary PDF using fpdf2."""
    import re
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    tc        = state.get("trip_constraints") or {}
    dest      = tc.get("destination", "Trip")
    s_date    = tc.get("start_date", "")
    e_date    = tc.get("end_date", "")
    members   = tc.get("members", "")
    budget    = tc.get("budget", "")
    query     = state.get("user_query", "")
    now       = datetime.now().strftime("%d %b %Y")

    uch = state.get("user_choices") or {}

    # ── Character translation table (Unicode → Latin-1 equivalents) ──────────
    _LATIN1 = str.maketrans({
        "—": "-",   "–": "-",    # em/en dash
        "‘": "’",   "’": "’",    # curly apostrophes
        0x201c: 0x22,  0x201d: 0x22,   # curly double-quotes -> straight "
        "₹": "Rs.", "°": "deg",  # rupee, degree
        "•": "-",   "…": "...",  # bullet, ellipsis
        "é": "e",   "è": "e",   "ê": "e",
        "à": "a",   "â": "a",   "ä": "a",
        "ü": "u",   "û": "u",
        "ö": "o",   "ô": "o",
        "î": "i",   "ï": "i",
        "ç": "c",   "ñ": "n",
    })

    def _clean(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
        text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)
        text = re.sub(r"#{1,6}\s*", "", text)
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[-|: ]+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.translate(_LATIN1)
        # Strip ALL remaining non-Latin-1 characters (emojis, Devanagari, etc.)
        text = re.sub(r"[^\x00-\xFF]", "", text)
        return text.encode("latin-1", errors="replace").decode("latin-1").strip()

    # ── Palette ───────────────────────────────────────────────────────────────
    # Navy: (15, 23, 42)   Sky: (14, 165, 233)   Gold: (202, 138, 4)
    # Slate: (71, 85, 105) Light: (241, 245, 249) Ink: (30, 41, 59)

    class WandrPDF(FPDF):
        def header(self):
            # Thin top stripe on every content page
            if self.page_no() > 1:
                self.set_fill_color(15, 23, 42)
                self.rect(0, 0, 210, 6, "F")
                self.set_y(9)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(148, 163, 184)
            self.cell(0, 4, f"Wandr AI Travel Planner  |  {_clean(dest)}  |  Generated {now}  |  Page {self.page_no()}", align="C")

    pdf = WandrPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(left=16, top=8, right=16)
    eff_w = 210 - 32   # 178 mm

    dest_clean = _clean(dest)

    # ════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ════════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # Full navy background
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 297, "F")

    # Gold accent bar at top
    pdf.set_fill_color(202, 138, 4)
    pdf.rect(0, 0, 210, 5, "F")

    # Centred logo text
    pdf.set_y(38)
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, "WANDR", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 7, "AI Travel Planner", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Gold divider line
    pdf.ln(10)
    pdf.set_draw_color(202, 138, 4)
    pdf.set_line_width(0.8)
    pdf.line(40, pdf.get_y(), 170, pdf.get_y())
    pdf.ln(14)

    # Destination
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(0, 12, f"Trip to {dest_clean}", align="C")
    pdf.ln(4)

    # Date & members row
    info_parts = []
    if s_date and e_date:
        info_parts.append(f"{s_date} - {e_date}")
    elif s_date:
        info_parts.append(s_date)
    if members:
        info_parts.append(f"{members} traveller(s)")
    if info_parts:
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(202, 138, 4)
        pdf.cell(0, 7, "  |  ".join(info_parts), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(20)

    # User preferences card
    pref_items = []
    for lbl, key in [("Transport", "transport"), ("Hotel tier", "hotel"), ("Food", "food"), ("Style", "style")]:
        val = uch.get(key, "")
        if val:
            pref_items.append((lbl, _clean(val)))
    sp = _clean(uch.get("special_requests") or uch.get("special", ""))
    if sp:
        pref_items.append(("Special", sp[:80]))

    if pref_items:
        # Card background
        card_y = pdf.get_y()
        pdf.set_fill_color(30, 41, 59)
        pdf.rect(28, card_y, 154, 10 + len(pref_items) * 8 + 6, "F")
        # Gold left edge
        pdf.set_fill_color(202, 138, 4)
        pdf.rect(28, card_y, 2, 10 + len(pref_items) * 8 + 6, "F")

        pdf.set_xy(34, card_y + 5)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(202, 138, 4)
        pdf.cell(0, 5, "YOUR PREFERENCES", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(34)
        for lbl, val in pref_items:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(148, 163, 184)
            pdf.cell(28, 7, f"{lbl}:", new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(220, 230, 240)
            pdf.cell(0, 7, val[:70], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_x(34)
        pdf.ln(6)

    # Trip request quote
    if query:
        q_y = pdf.get_y() + 6
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.set_xy(28, q_y)
        pdf.multi_cell(154, 6, f"\"{_clean(query)[:200]}\"")

    # Bottom strip
    pdf.set_fill_color(202, 138, 4)
    pdf.rect(0, 292, 210, 5, "F")

    # ════════════════════════════════════════════════════════════════════════
    # CONTENT SECTIONS
    # ════════════════════════════════════════════════════════════════════════
    SECTION_COLORS = {
        "Complete Itinerary":   (14, 165, 233),   # sky blue
        "Transport":            (16, 185, 129),   # emerald
        "Hotels":               (245, 158, 11),   # amber
        "Weather":              (99,  102, 241),  # indigo
        "Nearby Attractions":   (236,  72,  153), # pink
        "Budget Breakdown":     (234,  88,   12), # orange
    }

    sections = [
        ("Complete Itinerary",   state.get("final_response") or state.get("itinerary", "")),
        ("Transport",            state.get("transport_results", "")),
        ("Hotels",               state.get("hotel_results", "")),
        ("Weather",              state.get("weather_results", "")),
        ("Nearby Attractions",   state.get("nearby_results", "")),
        ("Budget Breakdown",     state.get("budget_results", "")),
    ]

    def _draw_section_header(pdf, title, color):
        """Draw a full-width colored section header banner."""
        y = pdf.get_y()
        r, g, b = color
        pdf.set_fill_color(r, g, b)
        pdf.rect(0, y, 210, 11, "F")
        # White text
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(16, y + 1.5)
        pdf.cell(178, 8, title.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(pdf.l_margin)
        pdf.set_text_color(30, 41, 59)
        pdf.ln(3)

    def _is_day_header(line: str) -> bool:
        return bool(re.match(r"^\s*(Day\s*\d+|DAY\s*\d+)", line.strip()))

    def _draw_day_header(pdf, line: str, color):
        """Draw a tinted day header pill."""
        r, g, b = color
        # Light tint (mix toward white)
        pdf.set_fill_color(min(255, r+180), min(255, g+180), min(255, b+180))
        y = pdf.get_y()
        pdf.rect(16, y, 178, 8, "F")
        # Colored left mark
        pdf.set_fill_color(r, g, b)
        pdf.rect(16, y, 3, 8, "F")
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(r, g, b)
        pdf.set_xy(22, y + 1)
        txt = re.sub(r"\s+", " ", line.strip())
        pdf.cell(172, 6, _clean(txt), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(pdf.l_margin)
        pdf.set_text_color(30, 41, 59)
        pdf.ln(1)

    for title, content in sections:
        if not content or not content.strip():
            continue

        color = SECTION_COLORS.get(title, (14, 165, 233))
        cleaned = _clean(content)
        if not cleaned.strip():
            continue

        pdf.add_page()
        _draw_section_header(pdf, title, color)

        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(30, 41, 59)

        paragraphs = cleaned.split("\n")
        for para in paragraphs:
            raw = para.rstrip()
            stripped = raw.strip()

            if not stripped:
                pdf.ln(2.5)
                continue

            # Day headers (Day 1 / Day 2 etc.)
            if _is_day_header(stripped):
                if pdf.get_y() > 255:
                    pdf.add_page()
                    _draw_section_header(pdf, title, color)
                else:
                    pdf.ln(2)
                _draw_day_header(pdf, stripped, color)
                pdf.set_font("Helvetica", "", 9.5)
                pdf.set_text_color(30, 41, 59)
                continue

            # Bullet points — indent
            if stripped.startswith(("-", "*", "+")):
                bullet_text = stripped.lstrip("-*+ ").strip()
                try:
                    pdf.set_x(22)
                    pdf.set_font("Helvetica", "", 9.5)
                    pdf.cell(4, 5, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
                    pdf.set_x(26)
                    pdf.multi_cell(eff_w - 12, 5, bullet_text)
                except Exception:
                    pdf.ln(3)
                continue

            # Sub-heading (ends with colon or all-caps short line)
            if (stripped.endswith(":") and len(stripped) < 60) or (stripped.isupper() and len(stripped) < 50):
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 10)
                r, g, b = color
                pdf.set_text_color(r, g, b)
                try:
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(eff_w, 5.5, stripped)
                except Exception:
                    pdf.ln(3)
                pdf.set_font("Helvetica", "", 9.5)
                pdf.set_text_color(30, 41, 59)
                continue

            # Normal paragraph
            try:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(eff_w, 5, stripped)
            except Exception:
                pdf.ln(3)

    return bytes(pdf.output())


def _merge(base: dict, update: dict) -> dict:
    for k, v in update.items():
        if k == "messages" and isinstance(v, list):
            base.setdefault("messages", [])
            base["messages"] = base["messages"] + v
        elif v is not None:
            base[k] = v
    return base


# ── Shared plan public view — ?share=<uuid> works for everyone ───────────────
_share_param = st.query_params.get("share", "")
if _share_param:
    _shared = auth.load_share(_share_param)
    if _shared:
        st.markdown(
            "<style>[data-testid='stSidebar']{display:none!important}</style>",
            unsafe_allow_html=True,
        )
        _sdest = (_shared.get("trip_constraints") or {}).get("destination", "Travel Plan")
        st.markdown(
            f'<div class="final-header">'
            f'<div class="final-title">🌍 Shared Travel Plan — {_sdest}</div>'
            f'<div class="final-sub">Created with Wandr AI Travel Planner &amp; shared with you</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if _shared.get("final_response"):
            st.markdown(_shared["final_response"])
        elif _shared.get("itinerary"):
            st.markdown(_shared["itinerary"])
        if _shared.get("nearby_results"):
            st.divider()
            st.markdown(_shared["nearby_results"])
        st.divider()
        _back_col, _brand_col = st.columns([1, 3])
        with _back_col:
            if st.button("← Back to my trips", use_container_width=True):
                st.query_params.clear()
                st.rerun()
        with _brand_col:
            st.markdown(
                "<div style='padding-top:0.5rem;color:#334155;font-size:0.82rem'>"
                "Plan your own trip at <b style='color:#0ea5e9'>Wandr</b> — AI Travel Planner</div>",
                unsafe_allow_html=True,
            )
        st.stop()
    else:
        st.warning("This share link is invalid or has expired.")
        st.query_params.clear()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH GATE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.get("authenticated"):
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none!important}</style>",
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown('<div class="auth-icon">🌍</div>', unsafe_allow_html=True)
        st.markdown('<span class="auth-title">Wandr</span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-sub">Your personal AI travel expert.<br>'
            'Flights · Hotels · Weather · Budget · Custom Itineraries</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="auth-features">'
            '<span class="auth-feature"><span class="auth-feature-dot">✈️</span> Real-time flights</span>'
            '<span class="auth-feature"><span class="auth-feature-dot">🏨</span> Hotel search</span>'
            '<span class="auth-feature"><span class="auth-feature-dot">🌤️</span> Live weather</span>'
            '<span class="auth-feature"><span class="auth-feature-dot">💰</span> Budget planning</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        tab_in, tab_up = st.tabs(["Sign In", "Create Account"])

        with tab_in:
            li_user = st.text_input("Username", key="li_user", placeholder="your username")
            li_pass = st.text_input("Password", key="li_pass", type="password", placeholder="••••••••")

            if st.button("Sign In", type="primary", use_container_width=True, key="li_btn"):
                if not li_user.strip() or not li_pass:
                    st.error("Please enter both username and password.")
                elif auth.verify_user(li_user, li_pass):
                    _tok = auth.create_session(li_user)
                    st.session_state["authenticated"]  = True
                    st.session_state["username"]        = li_user.strip().lower()
                    st.session_state["session_token"]   = _tok
                    st.query_params["s"] = _tok
                    st.rerun()
                else:
                    st.error("Incorrect username or password. Please try again.")

            st.markdown(
                "<div style='text-align:center;color:#334155;font-size:0.78rem;margin-top:1rem'>"
                "New here? Switch to <b style='color:#0ea5e9'>Create Account</b> tab.</div>",
                unsafe_allow_html=True,
            )

        with tab_up:
            su_user    = st.text_input("Username",         key="su_user",    placeholder="choose a username (min 3 chars)")
            su_pass    = st.text_input("Password",         key="su_pass",    type="password", placeholder="at least 4 characters")
            su_confirm = st.text_input("Confirm Password", key="su_confirm", type="password", placeholder="repeat your password")

            if st.button("Create Account", type="primary", use_container_width=True, key="su_btn"):
                if not su_user.strip() or not su_pass:
                    st.error("Username and password are required.")
                elif su_pass != su_confirm:
                    st.error("Passwords don't match — please try again.")
                else:
                    ok, err = auth.create_user(su_user, su_pass)
                    if ok:
                        _tok = auth.create_session(su_user)
                        st.session_state["authenticated"]  = True
                        st.session_state["username"]        = su_user.strip().lower()
                        st.session_state["session_token"]   = _tok
                        st.query_params["s"] = _tok
                        st.success("Welcome aboard! Your account is ready.")
                        st.rerun()
                    else:
                        st.error(err)

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
username = st.session_state["username"].strip().lower()

if "thread_id" not in st.session_state or not st.session_state["thread_id"].startswith(f"{username}_"):
    st.session_state["thread_id"] = f"{username}_{uuid.uuid4().hex[:8]}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:0.3rem 0 0.6rem">'
        '<span class="sidebar-brand">🌍 Wandr</span><br>'
        '<span class="sidebar-tagline">AI Travel Planner</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        f'<div class="user-badge">👤 &nbsp;{username}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Sign Out", use_container_width=True):
        auth.delete_session(st.session_state.get("session_token", ""))
        st.query_params.clear()
        for key in list(st.session_state.keys()):
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()

    if st.button("✚  New Trip Plan", use_container_width=True, type="primary"):
        st.session_state["thread_id"] = f"{username}_{uuid.uuid4().hex[:8]}"
        for k in ("latest_result", "waiting_for_approval", "current_thread_query"):
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()

    st.markdown("**Trip History**")
    threads = list_threads(username)

    current_tid   = st.session_state.get("thread_id", "")
    current_query = st.session_state.get("current_thread_query", "")
    if current_query and current_tid:
        thread_ids_in_list = [t["thread_id"] for t in threads]
        if current_tid not in thread_ids_in_list:
            threads.insert(0, {
                "thread_id": current_tid,
                "ts": datetime.now().isoformat(),
                "query": current_query,
            })

    if not threads:
        st.markdown(
            '<div class="tip-card">'
            '<div class="tip-card-icon">🗺️</div>'
            '<div class="tip-card-text">Your saved trips will appear here once you create a plan.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for t in threads:
            tid    = t["thread_id"]
            active = tid == st.session_state["thread_id"]
            title  = (t["query"] or tid)
            title  = title[:32] + "…" if len(title) > 32 else title
            ts     = t["ts"][:16].replace("T", " ") if t["ts"] else "—"
            dot    = "🟢 " if active else ""

            st.markdown(
                f'<div class="thread-item">'
                f'<span style="color:#bfdbfe;font-size:0.82rem">{dot}<b>{title}</b></span><br>'
                f'<span style="color:#334155;font-size:0.72rem">{ts}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Open", key=f"load_{tid}", use_container_width=True):
                    vals, waiting = load_thread(tid)
                    st.session_state["thread_id"]            = tid
                    st.session_state["latest_result"]        = vals
                    st.session_state["waiting_for_approval"] = waiting
                    st.rerun()
            with c2:
                if st.button("Delete", key=f"del_{tid}", use_container_width=True):
                    delete_thread(tid)
                    if active:
                        for k in ("latest_result", "waiting_for_approval"):
                            st.session_state.pop(k, None)
                        st.session_state["thread_id"] = f"{username}_{uuid.uuid4().hex[:8]}"
                    st.rerun()

    st.divider()

    st.markdown(
        '<div class="tip-card">'
        '<div class="tip-card-icon">💡</div>'
        '<div class="tip-card-text"><b style="color:#475569">Pro tip:</b> Be specific — mention budget, travel style, hotel preference, and any must-do activities for the best itinerary.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tip-card">'
        '<div class="tip-card-icon">✅</div>'
        '<div class="tip-card-text">Every plan includes a <b style="color:#475569">review step</b> before finalising — you can request changes at any time.</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-eyebrow">✦ AI Travel Planner</div>'
    '<div class="hero-title">Plan Your Dream Trip</div>'
    '<div class="hero-sub">'
    'Tell us where you want to go, your budget, and travel style — '
    'we\'ll handle flights, hotels, weather, and build a day-by-day itinerary just for you.'
    '</div>'
    '<div class="hero-pills">'
    '<span class="hero-pill">✈️ Real-time Flights</span>'
    '<span class="hero-pill">🏨 Hotel Search</span>'
    '<span class="hero-pill">🌤️ Live Weather</span>'
    '<span class="hero-pill">💰 Budget Planner</span>'
    '<span class="hero-pill">📋 Day-by-Day Itinerary</span>'
    '<span class="hero-pill">✍️ Human Review Step</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

config = {"configurable": {"thread_id": st.session_state["thread_id"]}}


# ── Detect running state & consume pending run ────────────────────────────────
_pending_run = st.session_state.pop("pending_run", None)
_is_running  = st.session_state.get("is_running", False)

if _is_running:
    st.markdown("""
<style>
@keyframes bar-slide {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
<div style="
    background: linear-gradient(135deg, #0c1f38 0%, #062040 100%);
    border: 1px solid #0369a1; border-radius: 16px;
    padding: 1.6rem 2rem 1.4rem; margin-bottom: 1.2rem;
    text-align: center; position: relative; overflow: hidden;
">
  <div style="
      position: absolute; top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(90deg, #0369a1, #0ea5e9, #06b6d4, #67e8f9, #0369a1);
      background-size: 200% 100%;
      animation: bar-slide 1.6s linear infinite;
  "></div>

  <div style="
      width: 40px; height: 40px; border-radius: 50%;
      border: 3px solid #1a3a6b; border-top-color: #0ea5e9;
      animation: spin 0.9s linear infinite;
      margin: 0 auto 0.9rem;
  "></div>

  <div style="color:#7dd3fc; font-size:1.1rem; font-weight:700; margin-bottom:0.35rem;">
      ✈️ &nbsp;Building Your Travel Plan…
  </div>
  <div style="color:#475569; font-size:0.83rem; line-height:1.6;">
      Flights &amp; Trains &amp; Buses &nbsp;·&nbsp; Hotels &nbsp;·&nbsp; Weather &nbsp;·&nbsp;
      Nearby attractions &nbsp;·&nbsp; Budget &nbsp;·&nbsp; Itinerary<br>
      <span style="color:#334155; font-size:0.78rem;">
          All inputs are locked. This usually takes 30–60 seconds.
      </span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Quick destinations ────────────────────────────────────────────────────────
st.markdown('<div class="chips-label">Popular Destinations — click to fill To field</div>', unsafe_allow_html=True)
clicked_dest = None
for row in [_QUICK_DESTINATIONS[:5], _QUICK_DESTINATIONS[5:]]:
    chip_cols = st.columns(5)
    for i, (flag, dest) in enumerate(row):
        with chip_cols[i]:
            if st.button(f"{flag} {dest}", key=f"chip_{dest}",
                         use_container_width=True, disabled=_is_running):
                clicked_dest = dest

if clicked_dest:
    st.session_state["to_input"] = clicked_dest


# ── From / To route row ───────────────────────────────────────────────────────
col_from, col_arrow, col_to = st.columns([10, 1, 10])
with col_from:
    st.markdown('<div class="route-label">🛫 From — Departure city</div>', unsafe_allow_html=True)
    origin = st.text_input(
        "From", key="from_input",
        placeholder="e.g. Mumbai, London, New York",
        label_visibility="collapsed",
        disabled=_is_running,
    )
with col_arrow:
    st.markdown(
        "<div style='text-align:center;padding-top:1.55rem;color:#0ea5e9;"
        "font-size:1.5rem;user-select:none'>→</div>",
        unsafe_allow_html=True,
    )
with col_to:
    st.markdown('<div class="route-label">🛬 To — Destination</div>', unsafe_allow_html=True)
    destination = st.text_input(
        "To", key="to_input",
        placeholder="e.g. Tokyo, Paris, Bali",
        label_visibility="collapsed",
        disabled=_is_running,
    )


# ── Dates · Members · Transport ───────────────────────────────────────────────
col_s, col_e, col_m, col_tp = st.columns([3, 3, 2, 3])
with col_s:
    st.markdown('<div class="route-label">📅 Departure Date</div>', unsafe_allow_html=True)
    start_date = st.date_input(
        "start_date", key="start_date",
        value=date.today(),
        label_visibility="collapsed", disabled=_is_running,
    )
with col_e:
    st.markdown('<div class="route-label">📅 Return Date</div>', unsafe_allow_html=True)
    end_date = st.date_input(
        "end_date", key="end_date",
        value=date.today() + timedelta(days=5),
        min_value=start_date,
        label_visibility="collapsed", disabled=_is_running,
    )
with col_m:
    st.markdown('<div class="route-label">👥 Travellers</div>', unsafe_allow_html=True)
    members = st.number_input(
        "members", key="n_members",
        min_value=1, max_value=20, value=2,
        label_visibility="collapsed", disabled=_is_running,
    )
with col_tp:
    st.markdown('<div class="route-label">🚀 Main Transport</div>', unsafe_allow_html=True)
    transport_pref = st.selectbox(
        "transport_pref", key="transport_pref",
        options=["Mixed (Auto)", "Flight", "Train", "Bus", "Car / Self-drive"],
        label_visibility="collapsed", disabled=_is_running,
    )

# Interests
st.markdown('<div class="route-label" style="margin-top:0.5rem">🎯 Interests (optional)</div>', unsafe_allow_html=True)
interests = st.multiselect(
    "interests", key="interests",
    options=["Temples & Heritage", "Nature & Outdoors", "Food & Street Food",
             "Shopping & Markets", "Adventure Sports", "Art & Museums",
             "Beach & Water", "Wellness & Spa", "Nightlife & Entertainment", "Photography"],
    placeholder="Select your travel interests…",
    label_visibility="collapsed",
    disabled=_is_running,
)


# ── Travel details textarea ───────────────────────────────────────────────────
query = st.text_area(
    "Travel details",
    placeholder=(
        "Budget, hotel preference, must-do activities, dietary needs…\n"
        "Example: Budget ₹1.5 lakh per person, love street food, prefer mid-range hotels, no overnight trains."
    ),
    height=90,
    label_visibility="collapsed",
    disabled=_is_running,
)


# ── Travel style quick-select ─────────────────────────────────────────────────
st.markdown(
    '<div class="chips-label" style="margin-top:0.6rem">Travel Style</div>',
    unsafe_allow_html=True,
)
for row in [_TRAVEL_STYLES[:4], _TRAVEL_STYLES[4:]]:
    style_cols = st.columns(4)
    for i, (icon, style) in enumerate(row):
        with style_cols[i]:
            if st.button(f"{icon} {style}", key=f"style_{style}",
                         use_container_width=True, disabled=_is_running):
                st.session_state["_append_style"] = style

if st.session_state.get("_append_style"):
    st.session_state.pop("_append_style")


# ── Run button ────────────────────────────────────────────────────────────────
col_btn, col_hint = st.columns([2, 5])
with col_btn:
    run = st.button(
        "✈️  Create My Travel Plan" if not _is_running else "⏳  Working…",
        type="primary", use_container_width=True, disabled=_is_running,
    )
with col_hint:
    st.markdown(
        "<div style='padding-top:0.65rem;color:#334155;font-size:0.83rem'>"
        "We research flights, hotels &amp; weather first — then you choose before we build your itinerary."
        "</div>",
        unsafe_allow_html=True,
    )


# ── Stage 1: Button clicked → save inputs, set running, rerun ─────────────────
if run and not _is_running:
    _origin    = st.session_state.get("from_input",    "").strip()
    _dest      = st.session_state.get("to_input",      "").strip()
    _s_date    = st.session_state.get("start_date",    date.today())
    _e_date    = st.session_state.get("end_date",      date.today() + timedelta(days=14))
    _members   = int(st.session_state.get("n_members", 2))
    _tpref     = st.session_state.get("transport_pref","Mixed (Auto)")
    _interests = st.session_state.get("interests",     [])
    _duration  = (_e_date - _s_date).days

    if not _dest and not query.strip():
        st.warning("Please enter a destination (To field) or describe your trip.")
    else:
        parts = []
        if _origin and _dest:
            parts.append(f"Travelling from {_origin} to {_dest}")
        elif _dest:
            parts.append(f"Destination: {_dest}")
        elif _origin:
            parts.append(f"Departing from {_origin}")
        parts.append(f"from {_s_date} to {_e_date} ({_duration} days)")
        parts.append(f"for {_members} traveller{'s' if _members > 1 else ''}")
        if _tpref != "Mixed (Auto)":
            parts.append(f"preferring {_tpref} for main travel")
        if _interests:
            parts.append(f"interests: {', '.join(_interests)}")

        enriched_query = "; ".join(parts) + ". " + query.strip()

        st.session_state["pending_run"] = {
            "enriched_query": enriched_query,
            "origin":         _origin,
            "dest":           _dest,
            "start_date":     str(_s_date),
            "end_date":       str(_e_date),
            "members":        _members,
            "transport_mode": _tpref.split()[0].lower(),
            "interests":      _interests,
            "budget":         query.strip(),
        }
        st.session_state["is_running"] = True
        st.rerun()


# ── Stage 2: Pending run exists → execute agents ──────────────────────────────
if _pending_run:
    enriched_query = _pending_run["enriched_query"]
    _origin        = _pending_run.get("origin", "")
    _dest          = _pending_run.get("dest", "")

    _trip_constraints = {
        "origin":         _origin,
        "destination":    _dest,
        "start_date":     _pending_run.get("start_date", ""),
        "end_date":       _pending_run.get("end_date", ""),
        "members":        _pending_run.get("members", 2),
        "transport_mode": _pending_run.get("transport_mode", "mixed"),
        "interests":      _pending_run.get("interests", []),
        "budget":         _pending_run.get("budget", ""),
    }

    state: dict = {
        "supervisor_reasoning": "", "selected_agents": [],
        "trip_constraints":     _trip_constraints,
        "flight_results": "",   "transport_results": "",
        "hotel_results":  "",   "weather_results":   "",
        "nearby_results": "",   "budget_results":    "",
        "itinerary":      "",   "itinerary_json":    "",
        "final_response": "",   "approved": None,
        "llm_calls":      0,
    }
    interrupted = False

    input_data = {
        "messages":          [HumanMessage(content=enriched_query)],
        "user_id":           username,
        "user_query":        enriched_query,
        "trip_constraints":  _trip_constraints,
        "flight_results":    "", "transport_results": "",
        "hotel_results":     "", "weather_results":   "",
        "nearby_results":    "", "budget_results":    "",
        "itinerary":         "", "itinerary_json":    "",
        "final_response":    "", "llm_calls":         0,
    }

    with st.status("✈️  Researching your perfect trip…", expanded=True) as status_box:
        try:
            for chunk in app.stream(input_data, config=config, stream_mode="updates"):
                for node_name, node_update in chunk.items():
                    if node_name == "__interrupt__":
                        interrupted = True
                    else:
                        status_box.write(_NODE_LABELS.get(node_name, f"⚙️  Working…"))
                        state = _merge(state, node_update)

            saved = app.get_state(config)
            if saved and saved.values:
                state = _merge(state, dict(saved.values))
                if not interrupted and saved.next:
                    interrupted = "human_approval" in saved.next

            if interrupted:
                status_box.update(label="✍️  Your itinerary draft is ready — please review below", state="running")
            else:
                status_box.update(label="🎉  Your travel plan is ready!", state="complete")

        except Exception as exc:
            status_box.update(label="Something went wrong — please try again", state="error")
            st.error(f"We hit an error while planning your trip: {exc}")
            st.exception(exc)

    st.session_state["latest_result"]        = state
    st.session_state["waiting_for_approval"] = interrupted
    st.session_state["current_thread_query"] = enriched_query
    st.session_state["is_running"]           = False
    st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.get("latest_result")

# When a saved trip is opened but had no content (e.g. a failed previous run),
# show a friendly message instead of a blank page.
if result is not None and not any(
    result.get(k) for k in (
        "supervisor_reasoning", "hotel_results",
        "transport_results", "itinerary", "final_response",
    )
):
    st.info(
        "⚠️ This trip didn't complete — it may have failed during planning. "
        "Fill in the details below and click **Create My Travel Plan** to try again.",
        icon=None,
    )

if result and any(result.get(k) for k in ("supervisor_reasoning", "hotel_results", "transport_results", "itinerary", "final_response")):
    st.divider()

    _render_pipeline(result)

    searches_done = sum(
        1 for k in ("transport_results", "hotel_results", "weather_results", "nearby_results", "budget_results")
        if result.get(k)
    )
    pending       = st.session_state.get("waiting_for_approval")
    finished      = bool(result.get("final_response"))
    status_label  = "⏸ Awaiting Review" if pending else ("✅ Complete" if finished else "🔄 In Progress")
    dest          = (result.get("trip_constraints") or {}).get("destination", "—")
    duration      = (result.get("trip_constraints") or {}).get("duration", "—")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Searches Done", searches_done)
    m2.metric("Destination",   dest)
    m3.metric("Duration",      duration)
    m4.metric("Status",        status_label)

    # ── Approval panel ABOVE tabs (visible from all tabs while waiting) ─────────
    if st.session_state.get("waiting_for_approval"):
        _apr_res = st.session_state.get("latest_result", {}) or {}
        st.markdown(
            '<div class="approval-box">'
            '<div class="approval-title">Research complete — choose your preferences</div>'
            '<div class="approval-sub">'
            'We found transport, hotels, weather &amp; nearby attractions. '
            'Browse the tabs below to review results, make your selections here, then generate.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        _tr = _apr_res.get("transport_results", "") or ""
        _ht = _apr_res.get("hotel_results", "") or ""

        # Always show all transport modes — agent covers flights, trains, buses & drive
        _transport_options = ["Flight", "Train", "Bus", "Self-drive / Car rental", "No preference"]
        _ex1, _ex2 = st.columns(2)
        with _ex1:
            if _tr:
                with st.expander("🚀 Transport options found — review", expanded=False):
                    st.markdown(_tr[:3000] + ("…" if len(_tr) > 3000 else ""))
        with _ex2:
            if _ht:
                with st.expander("🏨 Hotel options found — review", expanded=False):
                    st.markdown(_ht[:2500] + ("…" if len(_ht) > 2500 else ""))

        _c1, _c2 = st.columns(2)
        with _c1:
            ch_transport = st.radio(
                "Which transport mode do you prefer?",
                _transport_options, horizontal=False,
            )
            ch_food = st.radio(
                "Dietary preference?",
                ["Vegetarian", "Non-vegetarian", "No preference"],
                horizontal=False,
                help="Vegetarian filters hotels & meals to pure-veg only.",
            )
        with _c2:
            ch_hotel = st.radio(
                "Hotel budget tier?",
                ["Budget  (< Rs.2,000/night)", "Mid-range  (Rs.2,000–5,000)", "Premium  (Rs.5,000+)"],
                horizontal=False,
            )
            ch_style = st.radio(
                "Travel style?",
                ["Explorer (lots of sightseeing)", "Relaxed (slow pace)", "Foodie (local cuisine focus)", "Cultural / Spiritual"],
                horizontal=False,
            )
        ch_special = st.text_input(
            "Any special requests? (optional)",
            placeholder="Wheelchair access, halal food, honeymoon couple…",
        )
        if st.button("Generate My Personalised Itinerary", type="primary", use_container_width=True):
            st.session_state["user_choices_display"] = {
                "transport": ch_transport, "hotel": ch_hotel,
                "food": ch_food, "style": ch_style, "special": ch_special,
            }
            with st.spinner("Crafting your personalised day-by-day plan…"):
                _final_r = app.invoke(
                    Command(resume={
                        "transport": ch_transport, "hotel": ch_hotel,
                        "food": ch_food, "style": ch_style, "special_requests": ch_special,
                    }),
                    config=config,
                )
            st.session_state["latest_result"]        = _final_r
            st.session_state["waiting_for_approval"] = False
            st.rerun()

    # ── Tabs (research results) ───────────────────────────────────────────────
    tab_tr, tab_ht, tab_wx, tab_nb, tab_bud, tab_itin = st.tabs(
        ["🚀  Transport", "🏨  Hotels", "🌤️  Weather", "📍  Nearby", "💰  Budget", "📋  Itinerary"]
    )
    with tab_tr:
        _c = result.get("transport_results")
        if _c:
            st.markdown(_c)
        else:
            st.markdown('<div class="tip-card"><div class="tip-card-text">Transport options — flights, trains, buses, and self-drive — appear here. Enter a departure city in the "From" field.</div></div>', unsafe_allow_html=True)
    with tab_ht:
        _c = result.get("hotel_results")
        if _c:
            st.markdown(_c)
        else:
            st.markdown('<div class="tip-card"><div class="tip-card-text">Hotel recommendations will appear here.</div></div>', unsafe_allow_html=True)
    with tab_wx:
        _c = result.get("weather_results", "")
        if _c:
            if '"temperature_c"' in _c or _c.strip().lstrip("* \n").startswith("{"):
                import re as _re, json as _j
                _nm = _re.search(r'\{[^{}]*"temperature_c"[^{}]*\}', _c, _re.DOTALL)
                _fm = _re.search(r'\{[^{}]*"forecast".*?\}', _c, _re.DOTALL)
                if _nm or _fm:
                    from agents import _fmt_weather
                    _c = _fmt_weather(_nm.group() if _nm else "{}", _fm.group() if _fm else "{}")
            st.markdown(_c)
        else:
            st.markdown('<div class="tip-card"><div class="tip-card-text">Weather data will appear here once the destination is confirmed.</div></div>', unsafe_allow_html=True)
    with tab_nb:
        import re as _re2
        _c = _re2.sub(r"<think>.*?</think>", "", result.get("nearby_results", ""), flags=_re2.DOTALL | _re2.IGNORECASE).strip()
        if _c:
            st.markdown(_c)
        else:
            st.markdown('<div class="tip-card"><div class="tip-card-text">Nearby attractions, local food & markets will appear here.</div></div>', unsafe_allow_html=True)
    with tab_bud:
        _c = result.get("budget_results")
        if _c:
            st.markdown(_c)
        else:
            st.markdown('<div class="tip-card"><div class="tip-card-text">Budget breakdown will appear here. Mention your budget in the request for a cost estimate.</div></div>', unsafe_allow_html=True)

    # ── Itinerary tab — shows final plan after generation ──────────────────────
    with tab_itin:
        _final = st.session_state.get("latest_result") or {}
        if _final.get("final_response"):
            _dest_lbl = (_final.get("trip_constraints") or {}).get("destination", "your destination")

            # Non-editable choices summary
            _uch = _final.get("user_choices") or st.session_state.get("user_choices_display") or {}
            if _uch:
                _pils = "".join(
                    f'<span class="pref-pill"><b>{lbl}:</b> {val}</span>'
                    for lbl, val in [
                        ("Transport", _uch.get("transport","")),
                        ("Hotel",     _uch.get("hotel","")),
                        ("Food",      _uch.get("food","")),
                        ("Style",     _uch.get("style","")),
                    ] if val
                )
                _sp = _uch.get("special_requests") or _uch.get("special","")
                if _sp:
                    _pils += f'<span class="pref-pill"><b>Special:</b> {_sp}</span>'
                st.markdown(
                    f'<div class="prefs-panel" style="margin-top:0;margin-bottom:1rem;padding:1rem 1.4rem">'
                    f'<div class="prefs-panel-title" style="font-size:0.85rem;margin-bottom:0.5rem">Your confirmed preferences</div>'
                    f'<div class="prefs-choices">{_pils}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="final-header">'
                f'<div class="final-title">Your Complete Travel Plan — {_dest_lbl}</div>'
                f'<div class="final-sub">Personalised itinerary · transport · hotels · weather · budget · what to say</div>'
                f'</div>', unsafe_allow_html=True,
            )
            st.markdown(_final["final_response"])
            st.divider()

            _col_pdf, _col_share = st.columns([3, 2])
            with _col_pdf:
                st.download_button(
                    label="Download PDF",
                    data=_build_pdf_bytes(_final),
                    file_name=f"wandr_{_dest_lbl.replace(' ','_').lower()}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with _col_share:
                if st.button("Get Share Link", use_container_width=True):
                    _sh_state = {k: _final.get(k, "") for k in (
                        "user_query", "trip_constraints", "transport_results",
                        "hotel_results", "weather_results", "nearby_results",
                        "budget_results", "itinerary", "itinerary_json", "final_response",
                    )}
                    _sid = auth.create_share(username, _sh_state)
                    st.session_state["share_url"] = f"https://wandr-ai.streamlit.app/?share={_sid}"
                    st.session_state["share_id"]  = _sid
            if st.session_state.get("share_url"):
                st.success("Share link ready — copy below and send to anyone:")
                st.code(st.session_state["share_url"], language=None)
                st.caption("Anyone with this link can view your plan — no login needed.")

        elif _final.get("itinerary"):
            st.markdown(_final["itinerary"])
        elif st.session_state.get("waiting_for_approval"):
            st.info("Use the panel above the tabs to set your preferences, then click Generate.")
