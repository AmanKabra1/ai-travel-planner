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

# ── One-time DB setup ─────────────────────────────────────────────────────────
auth.setup()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Travel Planner · Multi-Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"]   { background: #0f172a; }
[data-testid="stSidebar"]            { background: #1e293b; border-right: 1px solid #334155; }
[data-testid="stHeader"]             { background: #0f172a !important; }
[data-testid="stMainBlockContainer"] { padding-top: 3.5rem !important; }
[data-testid="stBottom"]             { background: #0f172a; }

/* ── Typography ── */
.hero-title {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.4; padding: 0.1em 0; margin-bottom: 0.1rem;
    display: inline-block;
}
.hero-sub { color: #64748b; font-size: 0.95rem; margin-bottom: 1.8rem; }
h2, h3 { color: #e2e8f0 !important; }
p, li   { color: #cbd5e1; }

/* ── Auth card ── */
.auth-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 16px; padding: 2.5rem 2.8rem;
    box-shadow: 0 25px 60px rgba(0,0,0,.6);
}
.auth-icon  { text-align: center; font-size: 3.2rem; margin-bottom: 0.4rem; }
.auth-title {
    text-align: center; font-size: 1.7rem; font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; display: block; margin-bottom: 0.2rem;
}
.auth-sub   { text-align: center; color: #64748b; font-size: 0.85rem; margin-bottom: 1.5rem; }

/* ── Inputs ── */
textarea, input[type="text"], input[type="password"] {
    background: #0f172a !important; color: #e2e8f0 !important;
    border: 1px solid #334155 !important; border-radius: 8px !important;
}
textarea:focus, input[type="text"]:focus, input[type="password"]:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,.2) !important;
}
label { color: #94a3b8 !important; }

/* ── Primary button ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important; color: #fff !important;
    border-radius: 8px !important; font-weight: 700 !important;
    font-size: 0.95rem !important; transition: all 0.25s !important;
}
button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 25px rgba(99,102,241,.4) !important;
}

/* ── Secondary button ── */
button[kind="secondary"] {
    background: #1e293b !important; color: #94a3b8 !important;
    border: 1px solid #334155 !important; border-radius: 6px !important;
    font-size: 0.8rem !important;
}
button[kind="secondary"]:hover {
    border-color: #6366f1 !important; color: #a78bfa !important;
}

/* ── Pipeline strip ── */
.pipeline {
    display: flex; align-items: center; flex-wrap: wrap; gap: 3px;
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 0.7rem 1rem; margin: 1rem 0;
}
.step {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 11px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; white-space: nowrap;
}
.step-done { background: #14532d; color: #86efac; border: 1px solid #166534; }
.step-run  { background: #312e81; color: #a5b4fc; border: 1px solid #4338ca;
             animation: blink 1.4s ease-in-out infinite; }
.step-wait { background: #1e293b; color: #475569; border: 1px solid #2d3748; }
.sep       { color: #334155; font-size: 0.7rem; user-select: none; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 0.6rem 1rem;
}
[data-testid="stMetricValue"] { color: #a78bfa !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #64748b  !important; font-size: 0.78rem !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"]                  { gap: 0; border-bottom: 1px solid #334155 !important; background: transparent !important; }
[data-baseweb="tab"]                       { color: #64748b !important; font-weight: 600; border-radius: 0 !important; background: transparent !important; }
[data-baseweb="tab"][aria-selected="true"] { color: #a78bfa !important; border-bottom: 2px solid #6366f1 !important; }
[data-baseweb="tab-panel"]                 { background: transparent !important; padding-top: 1rem; }

/* ── Approval box ── */
.approval-box {
    background: #160f2a; border: 2px solid #7c3aed; border-radius: 12px;
    padding: 1.5rem 1.8rem; margin: 1rem 0;
}
.approval-title { color: #c4b5fd; font-size: 1.15rem; font-weight: 700; margin-bottom: 0.4rem; }
.approval-sub   { color: #7c3aed; font-size: 0.87rem; margin-bottom: 1rem; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #1e293b !important; border: 1px solid #334155 !important;
    color: #94a3b8 !important; border-radius: 8px !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #6366f1 !important; color: #a78bfa !important;
}

/* ── Divider ── */
hr { border-color: #334155 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] .stTextInput input { background: #0f172a !important; }
.thread-item { padding: 0.45rem 0; border-bottom: 1px solid #1e293b; }
.user-badge {
    background: #312e81; border: 1px solid #4338ca;
    border-radius: 8px; padding: 0.5rem 0.8rem;
    color: #a5b4fc; font-size: 0.85rem; font-weight: 600;
    display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;
}

/* ── Expander ── */
[data-testid="stExpander"] { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 10px !important; }
summary { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
_PIPELINE = [
    ("🧠", "Supervisor",  "supervisor_reasoning"),
    ("✈️", "Flights",     "flight_results"),
    ("🏨", "Hotels",      "hotel_results"),
    ("🌤️", "Weather",    "weather_results"),
    ("💰", "Budget",      "budget_results"),
    ("📋", "Itinerary",   "itinerary"),
    ("👤", "Approval",    "approved"),
    ("✅", "Final",       "final_response"),
]

_NODE_LABELS = {
    "supervisor":      "🧠 Supervisor — routing to agents…",
    "flight_agent":    "✈️  Flight Agent — researching routes & fares…",
    "hotel_agent":     "🏨 Hotel Agent — finding accommodation…",
    "weather_agent":   "🌤️  Weather Agent — checking forecasts…",
    "budget_agent":    "💰 Budget Agent — analysing costs…",
    "itinerary_agent": "📋 Itinerary Agent — drafting your plan…",
    "human_approval":  "👤 Requesting human approval…",
    "final_response":  "✅ Finalising your travel plan…",
}


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
        f"*Generated: {now}*\n",
        f"## Request\n{query}\n",
    ]
    for section, key in [
        ("Flight Information",    "flight_results"),
        ("Hotel Recommendations", "hotel_results"),
        ("Weather & Climate",     "weather_results"),
        ("Budget Analysis",       "budget_results"),
        ("Itinerary Draft",       "itinerary"),
        ("Final Travel Plan",     "final_response"),
    ]:
        content = state.get(key, "")
        if content:
            lines.append(f"## {section}\n{content}\n")
    lines.append("---\n*AI Travel Planner · Multi-Agent System · LangGraph + Groq LLaMA 3.3*")
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
# AUTH GATE — show login/signup page if not authenticated
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.get("authenticated"):

    # Hide sidebar on auth page
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none!important}</style>",
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown('<div class="auth-icon">✈️</div>', unsafe_allow_html=True)
        st.markdown('<span class="auth-title">AI Travel Planner</span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="auth-sub">Multi-Agent AI · LangGraph · Groq LLaMA 3.3 70B</div>',
            unsafe_allow_html=True,
        )

        tab_in, tab_up = st.tabs(["🔑  Sign In", "🆕  Create Account"])

        # ── Sign In ──────────────────────────────────────────────────────────
        with tab_in:
            li_user = st.text_input("Username", key="li_user", placeholder="your username")
            li_pass = st.text_input("Password", key="li_pass", type="password", placeholder="••••••••")

            if st.button("Sign In", type="primary", use_container_width=True, key="li_btn"):
                if not li_user.strip() or not li_pass:
                    st.error("Please enter both username and password.")
                elif auth.verify_user(li_user, li_pass):
                    st.session_state["authenticated"] = True
                    st.session_state["username"]      = li_user.strip().lower()  # Store lowercase
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

            st.markdown(
                "<div style='text-align:center;color:#475569;font-size:0.78rem;margin-top:1rem'>"
                "No account yet? Switch to <b>Create Account</b> tab above.</div>",
                unsafe_allow_html=True,
            )

        # ── Create Account ───────────────────────────────────────────────────
        with tab_up:
            su_user    = st.text_input("Username",         key="su_user",    placeholder="choose a username")
            su_pass    = st.text_input("Password",         key="su_pass",    type="password", placeholder="••••••••")
            su_confirm = st.text_input("Confirm Password", key="su_confirm", type="password", placeholder="••••••••")

            if st.button("Create Account", type="primary", use_container_width=True, key="su_btn"):
                if not su_user.strip() or not su_pass:
                    st.error("Username and password are required.")
                elif su_pass != su_confirm:
                    st.error("❌ Passwords do not match.")
                else:
                    ok, err = auth.create_user(su_user, su_pass)
                    if ok:
                        st.session_state["authenticated"] = True
                        st.session_state["username"]      = su_user.strip().lower()  # Store lowercase
                        st.success("Account created! Welcome 🎉")
                        st.rerun()
                    else:
                        st.error(f"❌ {err}")

    st.stop()   # ← nothing below renders until the user is logged in


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP  (only reached when authenticated)
# ══════════════════════════════════════════════════════════════════════════════
username = st.session_state["username"].strip().lower()

# Ensure each user always has an active thread_id with username in the prefix
if "thread_id" not in st.session_state or not st.session_state["thread_id"].startswith(f"{username}_"):
    st.session_state["thread_id"] = f"{username}_{uuid.uuid4().hex[:8]}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ✈️ Travel Planner")
    st.caption("Multi-Agent AI System")
    st.divider()

    # ── Logged-in user + logout
    st.markdown(
        f'<div class="user-badge">👤 &nbsp;{username}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Logout", use_container_width=True):
        # Clear all session state on logout
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()

    # ── New thread
    st.markdown("**Session**")
    st.caption(f"Thread: `…{st.session_state['thread_id'][-8:]}`")

    if st.button("➕  New Thread", use_container_width=True):
        st.session_state["thread_id"] = f"{username}_{uuid.uuid4().hex[:8]}"  # username already lowercased
        for k in ("latest_result", "waiting_for_approval"):
            st.session_state.pop(k, None)
        st.rerun()

    st.divider()

    # ── Thread history — auto-refreshes every rerun
    st.markdown("**🗂  Thread History**")
    threads = list_threads(username)

    if not threads:
        st.caption("No saved threads yet. Run a plan to create one.")
    else:
        for t in threads:
            tid    = t["thread_id"]
            active = tid == st.session_state["thread_id"]
            title  = (t["query"] or tid)
            title  = title[:34] + "…" if len(title) > 34 else title
            ts     = t["ts"][:16].replace("T", " ") if t["ts"] else "—"
            dot    = "🟢 " if active else ""

            st.markdown(
                f'<div class="thread-item">'
                f'<span style="color:#e2e8f0;font-size:0.82rem">{dot}<b>{title}</b></span><br>'
                f'<span style="color:#475569;font-size:0.72rem">{ts}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("📂 Open", key=f"load_{tid}", use_container_width=True):
                    vals, waiting = load_thread(tid)
                    st.session_state["thread_id"]            = tid
                    st.session_state["latest_result"]        = vals
                    st.session_state["waiting_for_approval"] = waiting
                    st.rerun()
            with c2:
                if st.button("🗑 Del", key=f"del_{tid}", use_container_width=True):
                    delete_thread(tid)
                    if active:
                        for k in ("latest_result", "waiting_for_approval"):
                            st.session_state.pop(k, None)
                        # Create new thread with lowercased username
                        st.session_state["thread_id"] = f"{username}_{uuid.uuid4().hex[:8]}"
                    st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem;color:#475569;line-height:1.7'>"
        "<b style='color:#6366f1'>Agent Pipeline</b><br>"
        "Supervisor → Flight → Hotel<br>"
        "→ Weather → Budget → Itinerary<br>"
        "→ Human Approval → Final Plan"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">✈️ AI Travel Planner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">'
    'Multi-agent AI · LangGraph &nbsp;|&nbsp; Groq LLaMA 3.3 70B &nbsp;|&nbsp; '
    'Real-time MCP data (flights, weather, hotels)'
    '</div>',
    unsafe_allow_html=True,
)

config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

# ── Query input ───────────────────────────────────────────────────────────────
query = st.text_area(
    "Travel request",
    placeholder=(
        "Plan a 7-day Japan trip under ₹2 lakh. "
        "I prefer budget hotels, no overnight flights, and love street food."
    ),
    height=105,
    label_visibility="collapsed",
)

col_btn, col_hint = st.columns([2, 5])
with col_btn:
    run = st.button("🚀  Create Travel Plan", type="primary", use_container_width=True)
with col_hint:
    st.markdown(
        "<div style='padding-top:0.65rem;color:#475569;font-size:0.83rem'>"
        "7 AI agents · live flights &amp; weather · human approval · persistent memory"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Run agents ────────────────────────────────────────────────────────────────
if run:
    if not query.strip():
        st.warning("Please enter a travel request first.")
    else:
        state: dict = {
            "supervisor_reasoning": "", "selected_agents": [],
            "trip_constraints":     {},
            "flight_results": "",  "hotel_results":  "",
            "weather_results": "", "budget_results": "",
            "itinerary": "",       "final_response": "",
            "approved": None,      "llm_calls":      0,
        }
        interrupted = False

        input_data = {
            "messages":       [HumanMessage(content=query)],
            "user_id":        username,
            "user_query":     query,
            "flight_results": "", "hotel_results":  "",
            "weather_results":"", "budget_results": "",
            "itinerary":      "", "final_response": "",
            "llm_calls":      0,
        }

        with st.status("🤖 Agents are planning your trip…", expanded=True) as status_box:
            try:
                for chunk in app.stream(input_data, config=config, stream_mode="updates"):
                    for node_name, node_update in chunk.items():
                        if node_name == "__interrupt__":
                            interrupted = True
                        else:
                            status_box.write(_NODE_LABELS.get(node_name, f"⚙️  {node_name}…"))
                            state = _merge(state, node_update)

                # Re-read authoritative state from checkpointer
                saved = app.get_state(config)
                if saved and saved.values:
                    state = _merge(state, dict(saved.values))
                    if not interrupted and saved.next:
                        interrupted = "human_approval" in saved.next

                if interrupted:
                    status_box.update(label="⏸️  Awaiting your approval", state="running")
                else:
                    status_box.update(label="✅  Plan ready!", state="complete")

            except Exception as exc:
                status_box.update(label="❌  Error occurred", state="error")
                st.error(f"**Error:** {exc}")
                st.exception(exc)

        st.session_state["latest_result"]        = state
        st.session_state["waiting_for_approval"] = interrupted
        # Rerun so the sidebar thread list picks up the newly saved thread
        st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.get("latest_result")

if result and any(result.get(k) for k in ("supervisor_reasoning", "flight_results", "hotel_results")):
    st.divider()

    _render_pipeline(result)

    agents_ran   = sum(1 for k in ("flight_results","hotel_results","weather_results","budget_results") if result.get(k))
    pending      = st.session_state.get("waiting_for_approval")
    finished     = bool(result.get("final_response"))
    status_label = "⏸ Awaiting Approval" if pending else ("✅ Complete" if finished else "🔄 In Progress")
    dest         = (result.get("trip_constraints") or {}).get("destination", "—")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Agents Run",  agents_ran)
    m2.metric("LLM Calls",   result.get("llm_calls", "—"))
    m3.metric("Status",      status_label)
    m4.metric("Destination", dest)

    if result.get("supervisor_reasoning"):
        with st.expander("🧠  Supervisor — routing decision", expanded=False):
            st.markdown(f"**Agents selected:** `{'` · `'.join(result.get('selected_agents', []))}`")
            st.markdown(result["supervisor_reasoning"])

    tab_fl, tab_ht, tab_wx, tab_bud, tab_itin = st.tabs(
        ["✈️  Flights", "🏨  Hotels", "🌤️  Weather", "💰  Budget", "📋  Itinerary"]
    )
    with tab_fl:
        st.markdown(result.get("flight_results") or "_Not requested for this plan._")
    with tab_ht:
        st.markdown(result.get("hotel_results") or "_Not requested for this plan._")
    with tab_wx:
        st.markdown(result.get("weather_results") or "_Not requested for this plan._")
    with tab_bud:
        st.markdown(result.get("budget_results") or "_Not requested for this plan._")
    with tab_itin:
        st.markdown(result.get("itinerary") or "_Not yet generated._")


# ── Human Approval panel ──────────────────────────────────────────────────────
if st.session_state.get("waiting_for_approval"):
    st.divider()
    st.markdown(
        '<div class="approval-box">'
        '<div class="approval-title">👤 Human-in-the-Loop Approval</div>'
        '<div class="approval-sub">Review the draft itinerary above, then approve or request revisions.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    decision = st.radio(
        "Decision",
        ["✅  Approve — looks great!", "✏️  Request revisions"],
        horizontal=True,
        label_visibility="collapsed",
    )

    feedback = ""
    if "revisions" in decision.lower():
        feedback = st.text_area(
            "Revision notes",
            placeholder="E.g. Replace Day 3 with a cooking class and add a per-day budget breakdown.",
        )

    if st.button("Submit Decision", type="primary"):
        with st.spinner("✨  Finalising your travel plan…"):
            final = app.invoke(
                Command(resume={"approved": "Approve" in decision, "feedback": feedback}),
                config=config,
            )
        st.session_state["latest_result"]        = final
        st.session_state["waiting_for_approval"] = False
        st.rerun()


# ── Final plan ────────────────────────────────────────────────────────────────
final = st.session_state.get("latest_result")
if final and final.get("final_response"):
    st.divider()
    st.markdown("## ✅  Your Personalised Travel Plan")
    st.markdown(final["final_response"])

    st.download_button(
        label="📥  Download plan as Markdown",
        data=_build_markdown_export(final).encode("utf-8"),
        file_name=f"travel_plan_{st.session_state['thread_id'][-8:]}.md",
        mime="text/markdown",
    )
