import sys
import uuid
from datetime import datetime

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
label { color: #94a3b8 !important; }
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
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
_PIPELINE = [
    ("🗺️", "Planning",   "supervisor_reasoning"),
    ("✈️", "Flights",    "flight_results"),
    ("🚂", "Transport",  "transport_results"),
    ("🏨", "Hotels",     "hotel_results"),
    ("🌤️", "Weather",   "weather_results"),
    ("💰", "Budget",     "budget_results"),
    ("📋", "Itinerary",  "itinerary"),
    ("✍️", "Your Review","approved"),
    ("🎉", "Ready",      "final_response"),
]

_NODE_LABELS = {
    "supervisor":       "🗺️  Planning your trip…",
    "flight_agent":     "✈️  Searching flights & routes…",
    "transport_agent":  "🚂  Finding trains & buses…",
    "hotel_agent":      "🏨  Finding the best accommodation…",
    "weather_agent":    "🌤️  Checking destination weather…",
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
        ("Flights",              "flight_results"),
        ("Trains & Buses",       "transport_results"),
        ("Accommodation",        "hotel_results"),
        ("Weather & Climate",    "weather_results"),
        ("Budget Breakdown",     "budget_results"),
        ("Day-by-Day Itinerary", "itinerary"),
        ("Your Complete Travel Plan", "final_response"),
    ]:
        content = state.get(key, "")
        if content:
            lines.append(f"## {section}\n{content}\n")
    lines.append("---\n*Wandr — Your AI Travel Expert*")
    return "\n".join(lines)


def _merge(base: dict, update: dict) -> dict:
    for k, v in update.items():
        if k == "messages" and isinstance(v, list):
            base.setdefault("messages", [])
            base["messages"] = base["messages"] + v
        elif v is not None:
            base[k] = v
    return base


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
                    st.session_state["authenticated"] = True
                    st.session_state["username"]      = li_user.strip().lower()
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
                        st.session_state["authenticated"] = True
                        st.session_state["username"]      = su_user.strip().lower()
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
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
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


# ── Quick destinations ────────────────────────────────────────────────────────
st.markdown('<div class="chips-label">Popular Destinations — click to fill To field</div>', unsafe_allow_html=True)
clicked_dest = None
for row in [_QUICK_DESTINATIONS[:5], _QUICK_DESTINATIONS[5:]]:
    chip_cols = st.columns(5)
    for i, (flag, dest) in enumerate(row):
        with chip_cols[i]:
            if st.button(f"{flag} {dest}", key=f"chip_{dest}", use_container_width=True):
                clicked_dest = dest

# Fill To field when chip clicked
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
    )


# ── Travel details textarea ───────────────────────────────────────────────────
query = st.text_area(
    "Travel details",
    placeholder=(
        "Duration, budget, number of travellers, hotel preference, must-do activities…\n"
        "Example: 7 days for 2 people, budget ₹1.5 lakh per person, love street food and temples, prefer mid-range hotels."
    ),
    height=100,
    label_visibility="collapsed",
)


# ── Travel style quick-select ─────────────────────────────────────────────────
st.markdown(
    '<div class="chips-label" style="margin-top:0.6rem">Travel Style — add to your request</div>',
    unsafe_allow_html=True,
)
for row in [_TRAVEL_STYLES[:4], _TRAVEL_STYLES[4:]]:
    style_cols = st.columns(4)
    for i, (icon, style) in enumerate(row):
        with style_cols[i]:
            if st.button(f"{icon} {style}", key=f"style_{style}", use_container_width=True):
                st.session_state["_append_style"] = style

if st.session_state.get("_append_style"):
    st.session_state.pop("_append_style")


# ── Run button ────────────────────────────────────────────────────────────────
col_btn, col_hint = st.columns([2, 5])
with col_btn:
    run = st.button("✈️  Create My Travel Plan", type="primary", use_container_width=True)
with col_hint:
    st.markdown(
        "<div style='padding-top:0.65rem;color:#334155;font-size:0.83rem'>"
        "Flights · Hotels · Weather · Budget · Itinerary · Review &amp; Finalise"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Run agents ────────────────────────────────────────────────────────────────
if run:
    _origin = st.session_state.get("from_input", "").strip()
    _dest   = st.session_state.get("to_input",   "").strip()

    if not _dest and not query.strip():
        st.warning("Please enter a destination or describe your trip above.")
    else:
        # Build a complete query that makes origin/destination explicit
        route_prefix = ""
        if _origin and _dest:
            route_prefix = f"Travelling from {_origin} to {_dest}. "
        elif _dest:
            route_prefix = f"Destination: {_dest}. "
        elif _origin:
            route_prefix = f"Departing from {_origin}. "

        enriched_query = (route_prefix + query).strip()

        state: dict = {
            "supervisor_reasoning": "", "selected_agents": [],
            "trip_constraints":     {"origin": _origin, "destination": _dest} if (_origin or _dest) else {},
            "flight_results": "",   "transport_results": "",
            "hotel_results":  "",   "weather_results":   "",
            "budget_results": "",   "itinerary":         "",
            "final_response": "",   "approved": None,
            "llm_calls":      0,
        }
        interrupted = False

        input_data = {
            "messages":          [HumanMessage(content=enriched_query)],
            "user_id":           username,
            "user_query":        enriched_query,
            "flight_results":    "", "transport_results": "",
            "hotel_results":     "", "weather_results":   "",
            "budget_results":    "", "itinerary":         "",
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
        st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.get("latest_result")

if result and any(result.get(k) for k in ("supervisor_reasoning", "flight_results", "hotel_results", "transport_results")):
    st.divider()

    _render_pipeline(result)

    searches_done = sum(
        1 for k in ("flight_results", "transport_results", "hotel_results", "weather_results", "budget_results")
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

    tab_fl, tab_tr, tab_ht, tab_wx, tab_bud, tab_itin = st.tabs(
        ["✈️  Flights", "🚂  Trains & Buses", "🏨  Hotels", "🌤️  Weather", "💰  Budget", "📋  Itinerary"]
    )
    with tab_fl:
        content = result.get("flight_results")
        if content:
            st.markdown(content)
        else:
            st.markdown(
                '<div class="tip-card"><div class="tip-card-text">Flight information was not requested for this plan. Try adding specific travel dates and a departure city.</div></div>',
                unsafe_allow_html=True,
            )
    with tab_tr:
        content = result.get("transport_results")
        if content:
            st.markdown(content)
        else:
            st.markdown(
                '<div class="tip-card"><div class="tip-card-text">Train and bus options were not included in this plan.</div></div>',
                unsafe_allow_html=True,
            )
    with tab_ht:
        content = result.get("hotel_results")
        if content:
            st.markdown(content)
        else:
            st.markdown(
                '<div class="tip-card"><div class="tip-card-text">Hotel recommendations were not included in this plan.</div></div>',
                unsafe_allow_html=True,
            )
    with tab_wx:
        content = result.get("weather_results")
        if content:
            st.markdown(content)
        else:
            st.markdown(
                '<div class="tip-card"><div class="tip-card-text">Weather data will appear here once the destination is confirmed.</div></div>',
                unsafe_allow_html=True,
            )
    with tab_bud:
        content = result.get("budget_results")
        if content:
            st.markdown(content)
        else:
            st.markdown(
                '<div class="tip-card"><div class="tip-card-text">Budget breakdown was not included. Mention your budget in the request for a cost estimate.</div></div>',
                unsafe_allow_html=True,
            )
    with tab_itin:
        content = result.get("itinerary")
        if content:
            st.markdown(content)
        else:
            st.markdown(
                '<div class="tip-card"><div class="tip-card-text">The day-by-day itinerary will appear here.</div></div>',
                unsafe_allow_html=True,
            )


# ── Human Approval panel ──────────────────────────────────────────────────────
if st.session_state.get("waiting_for_approval"):
    st.divider()
    st.markdown(
        '<div class="approval-box">'
        '<div class="approval-title">✍️  Review Your Itinerary Draft</div>'
        '<div class="approval-sub">Look through the itinerary above. Approve it to get your finalised travel plan, or request changes below.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    decision = st.radio(
        "What would you like to do?",
        ["✅  Looks great — finalise my plan!", "✏️  I'd like some changes"],
        horizontal=True,
    )

    feedback = ""
    if "changes" in decision.lower():
        feedback = st.text_area(
            "What changes would you like?",
            placeholder="E.g. Add more budget accommodation options, replace Day 3 with a beach day, include vegetarian restaurant recommendations…",
            height=100,
        )

    if st.button("Submit", type="primary"):
        with st.spinner("✨  Applying your feedback and finalising the plan…"):
            final = app.invoke(
                Command(resume={"approved": "Approve" in decision or "great" in decision, "feedback": feedback}),
                config=config,
            )
        st.session_state["latest_result"]        = final
        st.session_state["waiting_for_approval"] = False
        st.rerun()


# ── Final plan ────────────────────────────────────────────────────────────────
final = st.session_state.get("latest_result")
if final and final.get("final_response"):
    st.divider()

    dest_label = (final.get("trip_constraints") or {}).get("destination", "your destination")
    st.markdown(
        f'<div class="final-header">'
        f'<div class="final-title">🎉 Your Complete Travel Plan — {dest_label}</div>'
        f'<div class="final-sub">Personalised itinerary with flights, hotels, weather & budget breakdown</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(final["final_response"])

    st.download_button(
        label="📥  Download as Markdown",
        data=_build_markdown_export(final).encode("utf-8"),
        file_name=f"travel_plan_{dest_label.replace(' ', '_').lower()}_{st.session_state['thread_id'][-6:]}.md",
        mime="text/markdown",
    )
