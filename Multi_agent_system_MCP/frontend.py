import json
import re
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

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

# ── User timezone (set by JS on first load via ?tz= query param) ──────────────
_raw_tz = st.query_params.get("tz", "Asia/Kolkata")   # default IST for Indian users
try:
    _USER_TZ = ZoneInfo(_raw_tz)
except Exception:
    _USER_TZ = ZoneInfo("Asia/Kolkata")


def _fmt_ts(ts_iso: str) -> str:
    """Convert a UTC ISO timestamp to IST (or user's local time)."""
    if not ts_iso:
        return "—"
    try:
        # LangGraph stores timestamps like "2026-08-25T12:09:00.123456+00:00"
        # Python <3.11 fromisoformat can't parse "+00:00" on some builds; replace it
        ts_clean = ts_iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(_USER_TZ)
        return local.strftime("%d %b %Y, %H:%M IST")
    except Exception:
        return ts_iso[:16].replace("T", " ")

st.set_page_config(
    page_title="Wandr — AI Travel Planner",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Timezone auto-detect: inject JS once to set ?tz= query param ──────────────
if not st.query_params.get("tz"):
    import streamlit.components.v1 as _stc_tz
    _stc_tz.html("""
<script>
(function(){
    var tz = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
    var u  = new URL(window.parent.location.href);
    if (!u.searchParams.has('tz')) {
        u.searchParams.set('tz', tz);
        window.parent.location.replace(u.toString());
    }
})();
</script>""", height=0)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"]   { background: #060d1f; }
[data-testid="stSidebar"]            { background: #0a1628; border-right: 1px solid #1a3a6b; }
[data-testid="stHeader"]             { background: #060d1f !important; border-bottom: none !important; height: 0 !important; min-height: 0 !important; padding: 0 !important; }
[data-testid="stMainBlockContainer"] { padding-top: 1rem !important; }
[data-testid="stBottom"]             { background: #060d1f; }
.stApp { background: #060d1f; }

/* ── Hide Streamlit Cloud "Manage app" button (bottom-right) ── */
a[href*="manage=true"],
[class*="viewerBadge"],
[data-testid="manage-app-button"] { display: none !important; }

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
/* Hide "No results" in multiselect when all options are already chosen */
[data-baseweb="menu"] li:only-child { display: none !important; }
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
/* Compact Select All / Clear buttons in interests row */
div[data-testid="stButton"]:has(button[kind="secondary"]) button[data-testid="baseButton-secondary"] {
    padding: 2px 8px !important;
    font-size: 0.7rem !important;
    height: 26px !important;
    min-height: 26px !important;
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

/* ── Hide Streamlit chrome: running dot, three-dot menu, deploy/fork buttons, footer ── */
[data-testid="stStatusWidget"]      { display: none !important; }
[data-testid="stToolbar"]           { display: none !important; }
[data-testid="stDecoration"]        { display: none !important; }
[data-testid="stToolbarActionButton"] { display: none !important; }
#MainMenu                           { display: none !important; }
footer                              { display: none !important; }
footer:after                        { display: none !important; }
.stDeployButton                     { display: none !important; }
[data-testid="stAppDeployButton"]   { display: none !important; }
button[kind="header"]               { display: none !important; }
/* Hide any remaining top-right icon buttons inside the header */
[data-testid="stHeader"] button,
[data-testid="stHeader"] a,
[data-testid="stHeader"] svg        { display: none !important; }

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

(function injectTimezone() {
    try {
        var params = new URLSearchParams(window.location.search);
        if (!params.has('tz')) {
            var tz = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
            params.set('tz', tz);
            window.location.search = params.toString();
        }
    } catch(e) {}
})();

</script>
""", unsafe_allow_html=True)


# ── Tab content cleaner ───────────────────────────────────────────────────────
def _clean_tab(text: str, max_chars: int = 10000) -> str:
    """Strip JSON, think tags, repetition; cap length for tab display."""
    if not text:
        return text
    import re as _re_ct
    # ── Strip all JSON before showing to the user ─────────────────────────────
    # Fenced blocks: closed first, then unclosed/truncated
    text = _re_ct.sub(r'```[a-z]*[ \t]*\n[\s\S]*?```', '', text, flags=_re_ct.IGNORECASE)
    text = _re_ct.sub(r'```[a-z]*[ \t]*\n[\s\S]*',     '', text, flags=_re_ct.IGNORECASE)
    # Bracket-counting: strip bare {…} and […] blocks that look like JSON
    _out, _ci, _cn = [], 0, len(text)
    while _ci < _cn:
        _c0 = text[_ci]
        if _c0 in ('{', '['):
            _oc = _c0; _cc = '}' if _c0 == '{' else ']'
            _d, _j, _ins, _cl = 0, _ci, False, False
            while _j < _cn:
                _ch = text[_j]
                if _ch == '"' and (_j == 0 or text[_j - 1] != '\\'):
                    _ins = not _ins
                if not _ins:
                    if _ch == _oc:  _d += 1
                    elif _ch == _cc:
                        _d -= 1
                        if _d == 0:
                            _j += 1; _cl = True; break
                _j += 1
            _pk = text[_ci:min(_ci + 300, _cn)]
            if not _cl and ('": ' in _pk or '":"' in _pk):
                _ci = _cn; continue          # truncated JSON → drop to end
            if _cl and ('": ' in text[_ci:_j] or '":"' in text[_ci:_j]):
                _ci = _j; continue           # closed JSON block → skip
        _out.append(text[_ci]); _ci += 1
    text = ''.join(_out)
    # Line-level filter: drop any remaining pure JSON lines
    _jln = _re_ct.compile(
        r'^\s*(?:`+[a-z]*`*|json|"[^"]*"\s*:.*'
        r'|"[^"]*"\s*,?\s*$|[{\}\[\]]\s*,?\s*$'
        r'|null\s*,?\s*$|true\s*,?\s*$|false\s*,?\s*$)\s*$',
        _re_ct.IGNORECASE
    )
    text = '\n'.join(ln for ln in text.split('\n') if not _jln.match(ln))
    text = _re_ct.sub(r'^\s*[,\[\]\{\}]\s*$', '', text, flags=_re_ct.MULTILINE)
    text = _re_ct.sub(r'\n{3,}', '\n\n', text).strip()
    # ── Strip <think> tags ────────────────────────────────────────────────────
    # Strip complete <think>...</think> pairs
    text = _re_ct.sub(r'<think>.*?</think>', '', text, flags=_re_ct.DOTALL | _re_ct.IGNORECASE)
    # Strip any unclosed <think> tag (LLM truncated mid-think — cut from tag to end)
    text = _re_ct.sub(r'<think>.*', '', text, flags=_re_ct.DOTALL | _re_ct.IGNORECASE)
    text = text.strip()
    # Detect repetition: same 120-char window appearing 3+ times → cut before 3rd
    window = 120
    t = text
    for i in range(0, min(len(t) - window * 3, 2000), 60):
        chunk = t[i : i + window]
        p1 = t.find(chunk, i + window)
        if p1 != -1:
            p2 = t.find(chunk, p1 + window)
            if p2 != -1:
                t = t[: max(i + window, p1 - 5)].rstrip()
                t += "\n\n> *Repetitive content removed. Full data is used for the itinerary.*"
                break
    # Length cap
    if len(t) > max_chars:
        cut = t.rfind("\n", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        t = t[:cut].rstrip() + "\n\n> *Display trimmed. Full data is used for the itinerary.*"
    return t


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
    "research_all":     "🔍  Researching flights · hotels · weather · nearby · budget (all at once)…",
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
    import re, json as _json
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    try:
        import markdown as _md_lib
        _has_md = True
    except ImportError:
        _has_md = False

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
        text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)       # bold/italic
        text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)
        text = re.sub(r"#{1,6}\s*", "", text)                      # headings
        text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)      # blockquotes
        text = re.sub(r"^\\\s*", "", text, flags=re.MULTILINE)     # leading backslash
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\(https?://[^\)]+\)", r"\1", text)  # markdown links
        text = re.sub(r"<https?://[^>]+>", "", text)               # raw <URL> links
        text = re.sub(r"^\|.*$", "", text, flags=re.MULTILINE)     # ALL table rows
        text = re.sub(r"^[-|:= ]{3,}$", "", text, flags=re.MULTILINE)  # table borders / hr
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.translate(_LATIN1)
        text = re.sub(r"[^\x00-\xFF]", "", text)                   # strip emojis etc.
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
        """Draw a compact full-width colored section header banner."""
        y = pdf.get_y()
        r, g, b = color
        pdf.set_fill_color(r, g, b)
        pdf.rect(0, y, 210, 8, "F")
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(16, y + 1)
        pdf.cell(178, 6, title.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(pdf.l_margin)
        pdf.set_text_color(30, 41, 59)
        pdf.ln(2)

    def _split_blocks(text: str) -> list:
        """Split markdown into alternating ('text', ...) and ('table', ...) blocks."""
        blocks, current, in_table = [], [], False
        for line in text.split("\n"):
            is_tbl = bool(re.match(r"^\s*\|", line.rstrip()))
            if is_tbl != in_table:
                if current:
                    blocks.append(("table" if in_table else "text", "\n".join(current)))
                current, in_table = [], is_tbl
            current.append(line)
        if current:
            blocks.append(("table" if in_table else "text", "\n".join(current)))
        return blocks

    def _parse_md_table(text: str):
        """Return (headers, rows) from a markdown table string."""
        headers, rows = [], []
        for line in text.strip().split("\n"):
            s = line.strip()
            if not s.startswith("|"):
                continue
            if re.match(r"^\|[-:\s|]+\|$", s):
                continue   # separator row
            cells = [c.strip() for c in s.split("|")[1:-1]]
            if not headers:
                headers = cells
            else:
                rows.append(cells)
        return headers, rows

    def _render_table(pdf, headers: list, rows: list):
        """Render a markdown table via fpdf2 table() with proportional column widths."""
        from fpdf import FontFace
        try:
            from fpdf.enums import TableBordersLayout, TableCellFillMode
        except ImportError:
            return
        if not headers:
            return
        n = len(headers)
        # Proportional widths from content length
        char_w = []
        for i in range(n):
            mx = len(_clean(headers[i]))
            for row in rows:
                if i < len(row):
                    mx = max(mx, len(_clean(row[i])))
            char_w.append(max(mx, 4))
        total = sum(char_w)
        col_mm = [max(14, int(eff_w * cw / total)) for cw in char_w]
        diff = eff_w - sum(col_mm)
        col_mm[-1] = max(10, col_mm[-1] + diff)
        if pdf.get_y() + 18 > 272:
            pdf.add_page()
        hstyle = FontFace(emphasis="BOLD", color=(255, 255, 255),
                          fill_color=(15, 23, 42), size_pt=8)
        try:
            pdf.set_font("Helvetica", "", 8)
            with pdf.table(
                col_widths=tuple(col_mm),
                first_row_as_headings=True,
                headings_style=hstyle,
                line_height=5,
                borders_layout=TableBordersLayout.ALL,
                cell_fill_mode=TableCellFillMode.ROWS,
                cell_fill_color=(241, 245, 249),
            ) as tbl:
                hr = tbl.row()
                for h in headers:
                    hr.cell(_clean(h)[:100])
                for row_data in rows:
                    dr = tbl.row()
                    for i in range(n):
                        dr.cell(_clean(row_data[i]) if i < len(row_data) else "")
            pdf.ln(2)
        except Exception:
            # Cell-based fallback
            pdf.set_fill_color(15, 23, 42)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(255, 255, 255)
            pdf.set_x(pdf.l_margin)
            for i, h in enumerate(headers):
                mx_c = max(1, int(col_mm[i] / 2.1))
                pdf.cell(col_mm[i], 6, _clean(h)[:mx_c], border=1, fill=True,
                         new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln(6)
            for ridx, row_data in enumerate(rows):
                if ridx % 2 == 0:
                    pdf.set_fill_color(241, 245, 249)
                else:
                    pdf.set_fill_color(255, 255, 255)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(30, 41, 59)
                pdf.set_x(pdf.l_margin)
                for i in range(n):
                    txt = _clean(row_data[i]) if i < len(row_data) else ""
                    mx_c = max(1, int(col_mm[i] / 2.1))
                    pdf.cell(col_mm[i], 5.5, txt[:mx_c], border=1, fill=True,
                             new_x=XPos.RIGHT, new_y=YPos.TOP)
                pdf.ln(5.5)
            pdf.ln(2)

    def _render_section(pdf, content: str):
        """Render section: tables AND prose (headings, bullets, descriptions)."""
        _MAX_TABLES = 6
        _MAX_ROWS   = 20
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<think>.*",          "", text,    flags=re.DOTALL | re.IGNORECASE)
        # Strip fenced blocks (closed then unclosed) and bare JSON lines
        text = re.sub(r'```[a-z]*[ \t]*\n[\s\S]*?```', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```[a-z]*[ \t]*\n[\s\S]*',     '', text, flags=re.IGNORECASE)
        _jln = re.compile(
            r'^\s*(?:`+[a-z]*`*|json|"[^"]*"\s*:.*'
            r'|"[^"]*"\s*,?\s*$|[{\}\[\]]\s*,?\s*$'
            r'|null\s*,?\s*$|true\s*,?\s*$|false\s*,?\s*$)\s*$',
            re.IGNORECASE
        )
        text = '\n'.join(ln for ln in text.split('\n') if not _jln.match(ln))
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\(https?://[^\)]+\)", r"\1", text)
        text = re.sub(r"<https?://[^>]+>", "", text)
        text = text.strip()
        if not text:
            return
        table_count = 0
        for block_type, block_text in _split_blocks(text):
            if not block_text.strip():
                continue
            if block_type == "table":
                if table_count >= _MAX_TABLES:
                    continue
                headers, rows = _parse_md_table(block_text)
                if headers:
                    _render_table(pdf, headers, rows[:_MAX_ROWS])
                    table_count += 1
            else:
                # Render prose: headings bold, bullets as lines
                for ln in block_text.split("\n"):
                    s = ln.strip()
                    if not s:
                        pdf.ln(1)
                        continue
                    if s.startswith("#"):
                        s = re.sub(r"^#{1,6}\s*", "", s)
                        s = _clean(s)
                        if not s:
                            continue
                        pdf.set_font("Helvetica", "B", 9)
                        pdf.set_text_color(15, 23, 42)
                        pdf.set_x(pdf.l_margin)
                        pdf.multi_cell(eff_w, 5.5, s)
                        pdf.ln(0.5)
                    elif s.startswith(("- ", "* ", "• ")):
                        s = _clean(s.lstrip("-*• ").strip())
                        if not s:
                            continue
                        pdf.set_font("Helvetica", "", 8.5)
                        pdf.set_text_color(30, 41, 59)
                        pdf.set_x(pdf.l_margin + 3)
                        pdf.multi_cell(eff_w - 3, 5, f"- {s}")
                    else:
                        s = _clean(s)
                        if not s:
                            continue
                        pdf.set_font("Helvetica", "", 8.5)
                        pdf.set_text_color(71, 85, 105)
                        pdf.set_x(pdf.l_margin)
                        pdf.multi_cell(eff_w, 5, s)
                pdf.ln(1)

    # ── Day-table renderer from itinerary_json ────────────────────────────────
    def _draw_day_table(pdf, day_data: dict, color):
        r, g, b = color
        day_num  = day_data.get("day", "?")
        day_date = _clean(str(day_data.get("date", "")))
        acts     = day_data.get("activities") or []
        meals    = day_data.get("meals") or {}
        cost     = _clean(str(day_data.get("estimated_day_cost", "")))

        # Day header row
        pdf.ln(3)
        y = pdf.get_y()
        pdf.set_fill_color(r, g, b)
        pdf.rect(16, y, 178, 9, "F")
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(19, y + 1.5)
        label = f"Day {day_num}"
        if day_date:
            label += f"  |  {day_date}"
        pdf.cell(172, 6, label, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_x(pdf.l_margin)
        pdf.ln(1)

        if acts:
            # Table header
            COL = [22, 82, 18, 26, 30]  # Time | Activity | Duration | Cost | Notes
            HDR = ["Time", "Activity", "Duration", "Cost", "Notes"]
            pdf.set_fill_color(min(255, r+180), min(255, g+180), min(255, b+180))
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(r, g, b)
            pdf.set_x(16)
            for i, h in enumerate(HDR):
                pdf.cell(COL[i], 6, h, border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln(6)

            # Table rows
            for idx, act in enumerate(acts):
                t    = _clean(str(act.get("time", "")))
                plc  = _clean(str(act.get("place", "")))
                dur  = _clean(str(act.get("duration_hrs", "")))
                fee  = _clean(str(act.get("entry_fee", "")))
                nt   = _clean(str(act.get("notes", "")))

                fill_bg = (245, 250, 255) if idx % 2 == 0 else (255, 255, 255)
                pdf.set_fill_color(*fill_bg)
                pdf.set_text_color(30, 41, 59)
                pdf.set_font("Helvetica", "B" if not t else "", 8)
                pdf.set_x(16)

                row_h = 5.5
                # Time cell with arrow prefix
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(r, g, b)
                pdf.cell(COL[0], row_h, (f"-> {t}" if t else "->"), border=1, fill=True,
                         new_x=XPos.RIGHT, new_y=YPos.TOP)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(30, 41, 59)
                # Activity
                plc_txt = plc[:38] if plc else "_______________"
                pdf.cell(COL[1], row_h, plc_txt, border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
                # Duration
                dur_txt = (dur + "h") if dur and dur != "—" and "h" not in dur else (dur or "—")
                pdf.cell(COL[2], row_h, dur_txt[:10], border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
                # Cost
                fee_txt = fee if fee else "—"
                pdf.cell(COL[3], row_h, fee_txt[:14], border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
                # Notes
                pdf.cell(COL[4], row_h, nt[:28] if nt else "—", border=1, fill=True,
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)

        # Meals row
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(71, 85, 105)
        pdf.set_x(16)
        meal_parts = []
        for slot, key in [("BF", "breakfast"), ("Lunch", "lunch"), ("Dinner", "dinner")]:
            v = _clean(str(meals.get(key, "")))
            if v:
                meal_parts.append(f"{slot}: {v[:35]}")
        if meal_parts:
            pdf.multi_cell(178, 5, "  |  ".join(meal_parts))
        # Day cost
        if cost:
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.set_text_color(r, g, b)
            pdf.set_x(16)
            pdf.cell(178, 5, f"Day Budget: {cost} per person", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    # Parse itinerary_json for structured day tables
    _itin_json_str = state.get("itinerary_json", "") or ""
    _itin_data = {}
    if _itin_json_str.strip():
        try:
            _itin_data = _json.loads(_itin_json_str)
        except Exception:
            try:
                _itin_data = _json.loads(re.sub(r',(\s*[}\]])', r'\1', _itin_json_str))
            except Exception:
                _itin_data = {}

    first_section = True
    for title, content in sections:
        if not content or not content.strip():
            continue

        color = SECTION_COLORS.get(title, (14, 165, 233))

        # First section always starts on a new page (after cover);
        # subsequent sections only break when < 55mm remains.
        if first_section or pdf.get_y() > 242:
            pdf.add_page()
        else:
            pdf.ln(3)
        first_section = False
        _draw_section_header(pdf, title, color)

        # For the itinerary section: always use structured day tables from JSON
        if title == "Complete Itinerary":
            # Filter null/empty day entries (LLM sometimes emits [null, null, {…}])
            if _itin_data.get("days"):
                _itin_data["days"] = [d for d in _itin_data["days"] if d and isinstance(d, dict)]
            if _itin_data.get("days"):
                ts = _itin_data.get("trip_summary", {})
                badge = "AI-Generated — verify bookings before travel"
                if ts.get("total_budget_estimate"):
                    badge += f"   |   Total: {_clean(str(ts['total_budget_estimate']))}"
                pdf.set_font("Helvetica", "I", 7.5)
                pdf.set_text_color(202, 138, 4)
                pdf.set_x(pdf.l_margin)
                pdf.cell(0, 5, badge, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(1)
                for day in _itin_data["days"]:
                    if pdf.get_y() > 248:
                        pdf.add_page()
                    _draw_day_table(pdf, day, color)
            else:
                # JSON unavailable — strip JSON from prose before rendering
                # Step 1: fenced blocks (closed then unclosed/truncated)
                _clean_content = re.sub(r'```[a-z]*[ \t]*\n[\s\S]*?```', '',
                                        content, flags=re.IGNORECASE)
                _clean_content = re.sub(r'```[a-z]*[ \t]*\n[\s\S]*', '',
                                        _clean_content, flags=re.IGNORECASE)
                # Step 2: bare {…} and […] blocks via bracket-counting
                # Unclosed blocks → skip to end of string (truncated JSON)
                _out, _ci, _cn = [], 0, len(_clean_content)
                while _ci < _cn:
                    _c0 = _clean_content[_ci]
                    if _c0 in ('{', '['):
                        _oc, _cc2 = _c0, ('}' if _c0 == '{' else ']')
                        _d, _j, _ins, _cl = 0, _ci, False, False
                        while _j < _cn:
                            _ch = _clean_content[_j]
                            if _ch == '"' and (_j == 0 or _clean_content[_j-1] != '\\'):
                                _ins = not _ins
                            if not _ins:
                                if _ch == _oc: _d += 1
                                elif _ch == _cc2:
                                    _d -= 1
                                    if _d == 0:
                                        _j += 1; _cl = True; break
                            _j += 1
                        _peek = _clean_content[_ci:min(_ci + 300, _cn)]
                        if not _cl and ('": ' in _peek or '":"' in _peek):
                            _ci = _cn; continue  # truncated JSON → skip to end
                        if _cl and ('": ' in _clean_content[_ci:_j] or
                                    '":"' in _clean_content[_ci:_j]):
                            _ci = _j; continue
                    _out.append(_clean_content[_ci])
                    _ci += 1
                _clean_content = ''.join(_out)
                # Step 3: line-by-line pass — drop any remaining JSON lines
                _json_ln = re.compile(
                    r'^\s*(?:`+[a-z]*`*|json|"[^"]*"\s*:.*'
                    r'|"[^"]*"\s*,?\s*$|[{\}\[\]]\s*,?\s*$'
                    r'|null\s*,?\s*$|true\s*,?\s*$|false\s*,?\s*$)\s*$',
                    re.IGNORECASE
                )
                _clean_content = '\n'.join(
                    ln for ln in _clean_content.split('\n')
                    if not _json_ln.match(ln)
                )
                _clean_content = re.sub(r'^\s*[,\[\]\{\}]\s*$', '',
                                        _clean_content, flags=re.MULTILINE)
                _clean_content = re.sub(r'\n{3,}', '\n\n', _clean_content).strip()
                if _clean_content:
                    _render_section(pdf, _clean_content)
                else:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(148, 163, 184)
                    pdf.multi_cell(eff_w, 6, "Itinerary data not available in this export.")
            continue

        # ── Render section: tables and prose (Nearby, Budget, etc.) ──────────
        _render_section(pdf, content)

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
            st.markdown(_clean_tab(_shared["final_response"]))
        elif _shared.get("itinerary"):
            st.markdown(_clean_tab(_shared["itinerary"]))
        if _shared.get("nearby_results"):
            st.divider()
            st.markdown(_clean_tab(_shared["nearby_results"]))
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
            ts_iso = t["ts"] if t["ts"] else ""
            dot    = "🟢 " if active else ""

            ts_label = _fmt_ts(ts_iso) if ts_iso else "—"
            ts_html  = f'<span style="color:#334155;font-size:0.72rem">{ts_label}</span>'

            st.markdown(
                f'<div class="thread-item">'
                f'<span style="color:#bfdbfe;font-size:0.82rem">{dot}<b>{title}</b></span><br>'
                f'{ts_html}'
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
_pending_run    = st.session_state.pop("pending_run", None)
_is_running     = st.session_state.get("is_running", False)
_is_busy        = _is_running or bool(st.session_state.get("generating_final", False))

# Block ALL interactions while any generation is in progress
if _is_busy:
    st.markdown("""
<style>
/* Freeze every interactive element while the plan is building */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] a,
[data-testid="stSidebarContent"] * { pointer-events: none !important; }

[data-testid="stMain"] button,
[data-testid="stMain"] input,
[data-testid="stMain"] textarea,
[data-testid="stMain"] select,
[data-testid="stMain"] [role="tab"],
[data-testid="stMain"] [role="radio"],
[data-testid="stMain"] [role="checkbox"],
[data-testid="stMain"] [data-baseweb="select"],
[data-testid="stMain"] [data-baseweb="input"],
[data-testid="stMain"] [data-baseweb="tab"] {
    pointer-events: none !important;
    opacity: 0.45 !important;
    cursor: not-allowed !important;
}
</style>
""", unsafe_allow_html=True)

# (Building banner moved into the Itinerary tab — see tab_itin below)


# Defaults (overridden inside the form when it is shown)
run   = False
query = ""

# Determine whether to show the input form
_saved_result       = st.session_state.get("latest_result")
_result_has_content = bool(_saved_result and any(
    _saved_result.get(k) for k in (
        "supervisor_reasoning", "hotel_results",
        "transport_results", "itinerary", "final_response",
    )
))
_show_form = not _is_running and not _result_has_content

if _show_form:
    # ── Quick destinations ────────────────────────────────────────────────────
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

    # ── From / To route row ───────────────────────────────────────────────────
    col_from, col_arrow, col_to = st.columns([10, 1, 10])
    with col_from:
        st.markdown('<div class="route-label">🛫 From — Departure city</div>', unsafe_allow_html=True)
        st.text_input(
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
        st.text_input(
            "To", key="to_input",
            placeholder="e.g. Tokyo, Paris, Bali",
            label_visibility="collapsed",
        )

    # ── Dates · Members · Transport ───────────────────────────────────────────
    col_s, col_e, col_m, col_tp = st.columns([3, 3, 2, 3])
    with col_s:
        st.markdown('<div class="route-label">📅 Departure Date</div>', unsafe_allow_html=True)
        start_date = st.date_input(
            "start_date", key="start_date",
            value=date.today(),
            label_visibility="collapsed",
        )
    with col_e:
        st.markdown('<div class="route-label">📅 Return Date</div>', unsafe_allow_html=True)
        end_date = st.date_input(
            "end_date", key="end_date",
            value=date.today() + timedelta(days=5),
            min_value=start_date,
            label_visibility="collapsed",
        )
    with col_m:
        st.markdown('<div class="route-label">👥 Travellers</div>', unsafe_allow_html=True)
        st.number_input(
            "members", key="n_members",
            min_value=1, max_value=20, value=2,
            label_visibility="collapsed",
        )
    with col_tp:
        st.markdown('<div class="route-label">🚀 Main Transport</div>', unsafe_allow_html=True)
        st.selectbox(
            "transport_pref", key="transport_pref",
            options=["Mixed (Auto)", "Flight", "Train", "Bus", "Car / Self-drive"],
            label_visibility="collapsed",
        )

    # Interests — label row with Select All / Clear buttons
    _ALL_INTERESTS = [
        "Temples & Heritage", "Nature & Outdoors", "Food & Street Food",
        "Shopping & Markets", "Adventure Sports", "Art & Museums",
        "Beach & Water", "Wellness & Spa", "Nightlife & Entertainment", "Photography",
    ]
    _int_lbl, _int_all, _int_clr = st.columns([5, 1, 1])
    with _int_lbl:
        st.markdown('<div class="route-label" style="margin-top:0.5rem">🎯 Interests (optional)</div>', unsafe_allow_html=True)
    with _int_all:
        if st.button("Select All", key="sel_all_interests", use_container_width=True):
            st.session_state["interests"] = list(_ALL_INTERESTS)
            st.rerun()
    with _int_clr:
        if st.button("Clear", key="clr_interests", use_container_width=True):
            st.session_state["interests"] = []
            st.rerun()

    st.multiselect(
        "interests", key="interests",
        options=_ALL_INTERESTS,
        placeholder="Select your travel interests…",
        label_visibility="collapsed",
    )

    # ── Travel details textarea ───────────────────────────────────────────────
    query = st.text_area(
        "Travel details",
        placeholder=(
            "Budget, hotel preference, must-do activities, dietary needs…\n"
            "Example: Budget ₹1.5 lakh per person, love street food, prefer mid-range hotels, no overnight trains."
        ),
        height=90,
        label_visibility="collapsed",
    )

    # ── Travel style quick-select ─────────────────────────────────────────────
    st.markdown(
        '<div class="chips-label" style="margin-top:0.6rem">Travel Style</div>',
        unsafe_allow_html=True,
    )
    for row in [_TRAVEL_STYLES[:4], _TRAVEL_STYLES[4:]]:
        style_cols = st.columns(4)
        for i, (icon, style) in enumerate(row):
            with style_cols[i]:
                if st.button(f"{icon} {style}", key=f"style_{style}",
                             use_container_width=True):
                    st.session_state["_append_style"] = style

    if st.session_state.get("_append_style"):
        st.session_state.pop("_append_style")

    # ── Run button ────────────────────────────────────────────────────────────
    col_btn, col_hint = st.columns([2, 5])
    with col_btn:
        run = st.button(
            "✈️  Create My Travel Plan",
            type="primary", use_container_width=True,
        )
    with col_hint:
        st.markdown(
            "<div style='padding-top:0.65rem;color:#334155;font-size:0.83rem'>"
            "We research flights, hotels &amp; weather first — then you choose before we build your itinerary."
            "</div>",
            unsafe_allow_html=True,
        )


# ── Building banner — shown while research agents are running ──────────────────
if _is_running:
    st.markdown("""
<style>
@keyframes _wdr_pulse{0%,100%{opacity:1}50%{opacity:0.55}}
@keyframes _wdr_spin{to{transform:rotate(360deg)}}
._wdr_spinner{
    display:inline-block;width:28px;height:28px;
    border:3px solid rgba(14,165,233,0.25);
    border-top-color:#0ea5e9;border-radius:50%;
    animation:_wdr_spin 0.9s linear infinite;
    vertical-align:middle;margin-right:10px;
}
._wdr_dots span{animation:_wdr_pulse 1.4s ease-in-out infinite}
._wdr_dots span:nth-child(2){animation-delay:.2s}
._wdr_dots span:nth-child(3){animation-delay:.4s}
</style>
<div style="
    background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
    border-radius:16px;padding:2rem 2.5rem;text-align:center;
    margin:1.5rem 0;border:1px solid rgba(14,165,233,0.35);
">
  <div style="font-size:2.4rem;margin-bottom:0.6rem;">🗺️</div>
  <div style="color:#fff;font-size:1.35rem;font-weight:700;margin-bottom:0.5rem;">
    <span class="_wdr_spinner"></span>Building Your Travel Plan
    <span class="_wdr_dots"><span>.</span><span>.</span><span>.</span></span>
  </div>
  <div style="color:#94a3b8;font-size:0.92rem;margin-bottom:0.9rem;">
      AI agents are researching flights, hotels, weather &amp; attractions for you
  </div>
  <div style="color:#0ea5e9;font-size:0.82rem;">
      ✈️ &nbsp;Research takes 30–60 seconds — progress updates appear below
  </div>
</div>""", unsafe_allow_html=True)


# ── Stage 1: Button clicked → save inputs, set running, rerun ─────────────────
if run and not _is_running:
    # Always start a fresh thread so we don't collide with old checkpoints
    # (opening a failed/interrupted trip from history sets the old thread_id)
    st.session_state["thread_id"] = f"{username}_{uuid.uuid4().hex[:8]}"
    st.session_state.pop("latest_result",        None)
    st.session_state.pop("waiting_for_approval", None)
    st.session_state.pop("generating_final",     None)
    st.session_state.pop("pending_choices",      None)
    st.session_state.pop("cached_pdf_bytes",     None)

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

    # ── Structured itinerary HTML renderer (from itinerary_json) ────────────────
    def _render_itin_html(itin_data: dict) -> str:
        _COLORS = ["#0ea5e9","#10b981","#f59e0b","#8b5cf6",
                   "#ef4444","#06b6d4","#ec4899","#84cc16"]
        parts = []
        ts = itin_data.get("trip_summary") or {}
        if ts:
            _m = []
            if ts.get("total_days"):
                _m.append(f"<b>{ts['total_days']} Days</b>")
            if ts.get("total_budget_estimate"):
                _m.append(f"Estimated: <b>{ts['total_budget_estimate']}</b>")
            if ts.get("travel_style"):
                _m.append(str(ts["travel_style"]))
            parts.append(f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);border-radius:12px;
padding:1.1rem 1.5rem;margin-bottom:1.4rem;border-left:4px solid #f59e0b;">
  <div style="color:#f59e0b;font-size:0.7rem;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;margin-bottom:.25rem;">Trip Summary</div>
  <div style="color:#fff;font-size:1rem;font-weight:700;margin-bottom:.3rem;">{ts.get('destination','')}</div>
  <div style="color:#94a3b8;font-size:.83rem;">{" &nbsp;·&nbsp; ".join(_m)}</div>
</div>""")

        for idx, day in enumerate(itin_data.get("days") or []):
            col = _COLORS[idx % len(_COLORS)]
            dn = day.get("day", idx + 1)
            hdr = f"Day {dn}"
            if day.get("theme"):
                hdr += f" — {day['theme']}"
            if day.get("date"):
                hdr += f" &nbsp;|&nbsp; {day['date']}"
            dc = day.get("estimated_day_cost", "")

            rows_html = ""
            for ri, act in enumerate(day.get("activities") or []):
                bg  = "#ffffff" if ri % 2 == 0 else "#f8fafc"
                t   = act.get("time", "") or "—"
                pl  = act.get("place", "") or "—"
                dur = str(act.get("duration_hrs", "") or "—")
                if dur not in ("—","") and "h" not in dur and "min" not in dur:
                    dur += "h"
                fee  = act.get("entry_fee", "") or "—"
                note = act.get("notes", "") or ""
                rows_html += f"""
<tr style="background:{bg};">
  <td style="padding:.4rem .7rem;font-weight:600;color:{col};font-size:.82rem;
  white-space:nowrap;border-bottom:1px solid #f1f5f9;">{t}</td>
  <td style="padding:.4rem .7rem;color:#1e293b;font-size:.85rem;
  border-bottom:1px solid #f1f5f9;">{pl}</td>
  <td style="padding:.4rem .7rem;color:#64748b;font-size:.82rem;text-align:center;
  white-space:nowrap;border-bottom:1px solid #f1f5f9;">{dur}</td>
  <td style="padding:.4rem .7rem;color:#1e293b;font-size:.82rem;text-align:right;
  white-space:nowrap;border-bottom:1px solid #f1f5f9;">{fee}</td>
  <td style="padding:.4rem .7rem;color:#64748b;font-size:.8rem;
  border-bottom:1px solid #f1f5f9;">{note}</td>
</tr>"""

            tbl_html = ""
            if rows_html:
                tbl_html = f"""
<div style="overflow-x:auto;">
<table style="width:100%;border-collapse:collapse;font-size:.85rem;">
<thead><tr style="background:#f1f5f9;">
  <th style="padding:.45rem .7rem;text-align:left;color:#374151;font-weight:700;
  font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;
  border-bottom:2px solid #e2e8f0;white-space:nowrap;width:72px;">Time</th>
  <th style="padding:.45rem .7rem;text-align:left;color:#374151;font-weight:700;
  font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;
  border-bottom:2px solid #e2e8f0;">Activity / Place</th>
  <th style="padding:.45rem .7rem;text-align:center;color:#374151;font-weight:700;
  font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;
  border-bottom:2px solid #e2e8f0;white-space:nowrap;">Duration</th>
  <th style="padding:.45rem .7rem;text-align:right;color:#374151;font-weight:700;
  font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;
  border-bottom:2px solid #e2e8f0;white-space:nowrap;">Cost</th>
  <th style="padding:.45rem .7rem;text-align:left;color:#374151;font-weight:700;
  font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;
  border-bottom:2px solid #e2e8f0;">Notes</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table></div>"""

            meals = day.get("meals") or {}
            meal_bits = []
            if meals.get("breakfast"): meal_bits.append(f"🍳 <b>BF:</b> {meals['breakfast']}")
            if meals.get("lunch"):     meal_bits.append(f"🍽 <b>Lunch:</b> {meals['lunch']}")
            if meals.get("dinner"):    meal_bits.append(f"🌙 <b>Dinner:</b> {meals['dinner']}")
            meals_html = ""
            if meal_bits:
                meals_html = f"""
<div style="background:#f8fafc;padding:.45rem .75rem;border-top:1px solid #e2e8f0;
font-size:.82rem;color:#475569;">{" &nbsp;·&nbsp; ".join(meal_bits)}</div>"""

            cost_html = ""
            if dc:
                cost_html = f"""
<div style="background:{col}18;padding:.35rem .75rem;border-top:1px solid {col}33;
font-size:.82rem;font-weight:700;color:{col};text-align:right;">
  Day Budget: {dc} / person</div>"""

            parts.append(f"""
<div style="margin-bottom:1.1rem;border-radius:10px;overflow:hidden;
border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.06);">
  <div style="background:{col};padding:.6rem 1rem;display:flex;
  justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.4rem;">
    <span style="color:#fff;font-weight:700;font-size:.95rem;">{hdr}</span>
    {"<span style='color:rgba(255,255,255,.85);font-size:.82rem;font-weight:600;'>" + str(dc) + " / person</span>" if dc else ""}
  </div>
  {tbl_html}{meals_html}{cost_html}
</div>""")

        return "\n".join(parts)

    # ── JSON-stripping helper (used in itinerary tab render) ─────────────────
    # Matches a single line that is pure JSON syntax (never normal prose)
    _JSON_LINE_RE = re.compile(
        r'^\s*(?:'
        r'`+[a-z]*`*\s*'                              # ``` or ```json
        r'|json\s*'                                    # lone word "json" (fence artifact)
        r'|"[^"\\]*(?:\\.[^"\\]*)*"\s*:.*'            # "key": anything
        r'|"[^"\\]*(?:\\.[^"\\]*)*"\s*,?\s*$'         # "value", (lone string)
        r'|[{\}\[\]]\s*,?\s*$'                        # { } [ ] alone
        r'|null\s*,?\s*$'                             # null,
        r'|true\s*,?\s*$|false\s*,?\s*$'             # true, false
        r')\s*$',
        re.IGNORECASE
    )

    def _strip_json_from_prose(text: str, known_json: str = "") -> str:
        """Remove all JSON blocks and lines from prose text."""
        if not text:
            return ""
        # 1. Remove known JSON string verbatim
        if known_json and known_json in text:
            text = text.replace(known_json, "")
        # 2. Remove fenced code blocks — try closed first, then unclosed (truncated)
        text = re.sub(r'```[a-z]*[ \t]*\n[\s\S]*?```', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```[a-z]*[ \t]*\n[\s\S]*',     '', text, flags=re.IGNORECASE)
        # 3. Bracket-counting: strip {…} and […] JSON blocks
        #    If a block is UNCLOSED (truncated JSON), skip to END OF TEXT
        result, i, n = [], 0, len(text)
        while i < n:
            ch = text[i]
            if ch in ('{', '['):
                open_c, close_c = ch, ('}' if ch == '{' else ']')
                depth, j, in_str, closed = 0, i, False, False
                while j < n:
                    c = text[j]
                    if c == '"' and (j == 0 or text[j - 1] != '\\'):
                        in_str = not in_str
                    if not in_str:
                        if c == open_c:   depth += 1
                        elif c == close_c:
                            depth -= 1
                            if depth == 0:
                                j += 1; closed = True; break
                    j += 1
                peek = text[i:min(i + 300, n)]
                if not closed and ('": ' in peek or '":"' in peek):
                    # Truncated JSON — skip everything from here to end
                    i = n; continue
                if closed and ('": ' in text[i:j] or '":"' in text[i:j]):
                    i = j; continue
            result.append(text[i])
            i += 1
        text = ''.join(result)
        # 4. Line-by-line pass: drop any remaining lines that look like JSON
        kept = [ln for ln in text.split('\n') if not _JSON_LINE_RE.match(ln)]
        # 5. Clean up stray separator chars and extra blank lines
        cleaned = re.sub(r'^\s*[,\[\]\{\}]\s*$', '', '\n'.join(kept), flags=re.MULTILINE)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    _gen_final = st.session_state.get("generating_final", False)

    # ── Final itinerary generation — run ABOVE tabs so spinner is visible ──────
    if _gen_final:
        st.markdown("""
<style>
@keyframes _wdr_spin2{to{transform:rotate(360deg)}}
._wdr_spin2{display:inline-block;width:28px;height:28px;
border:3px solid rgba(16,185,129,0.25);border-top-color:#10b981;
border-radius:50%;animation:_wdr_spin2 0.9s linear infinite;
vertical-align:middle;margin-right:10px;}
</style>
<div style="background:linear-gradient(135deg,#052e16 0%,#14532d 100%);
border-radius:16px;padding:2rem 2.5rem;text-align:center;
margin:1.5rem 0;border:1px solid rgba(16,185,129,0.35);">
  <div style="font-size:2.4rem;margin-bottom:0.6rem;">📋</div>
  <div style="color:#fff;font-size:1.35rem;font-weight:700;margin-bottom:0.5rem;">
    <span class="_wdr_spin2"></span>Writing Your Day-by-Day Itinerary…
  </div>
  <div style="color:#86efac;font-size:0.92rem;margin-bottom:0.4rem;">
    AI is building your personalised travel plan with real prices &amp; timings
  </div>
  <div style="color:#4ade80;font-size:0.82rem;">
    ⏱️ &nbsp;This usually takes 20–40 seconds — please wait
  </div>
</div>""", unsafe_allow_html=True)
        _pc = st.session_state.pop("pending_choices", None)
        if _pc:
            with st.spinner(""):
                try:
                    _final_r = app.invoke(Command(resume=_pc), config=config)
                    st.session_state["latest_result"]        = _final_r
                    st.session_state["waiting_for_approval"] = False
                    st.session_state.pop("cached_pdf_bytes", None)   # invalidate PDF cache
                except Exception as _exc:
                    st.session_state["generating_final"] = False
                    st.error(f"Itinerary generation failed: {_exc}")
                    st.stop()
        st.session_state["generating_final"] = False
        st.session_state["open_itin_tab"]   = True   # trigger JS tab switch on next render
        st.rerun()

    # ── Auto-switch to Itinerary tab after generation ─────────────────────────
    if st.session_state.pop("open_itin_tab", False):
        import streamlit.components.v1 as _cmp
        _cmp.html("""
<script>
(function() {
  function clickItinTab() {
    var tabs = window.parent.document.querySelectorAll('[role="tab"]');
    if (tabs.length >= 6) {
      tabs[5].click();   // 6th tab = Itinerary (0-indexed: 5)
    } else if (tabs.length > 0) {
      tabs[tabs.length - 1].click();
    }
  }
  // Try immediately then retry after Streamlit finishes painting
  clickItinTab();
  setTimeout(clickItinTab, 300);
  setTimeout(clickItinTab, 700);
})();
</script>""", height=0)

    # ── Tabs (research results) ───────────────────────────────────────────────
    tab_tr, tab_ht, tab_wx, tab_nb, tab_bud, tab_itin = st.tabs(
        ["🚀  Transport", "🏨  Hotels", "🌤️  Weather", "📍  Nearby", "💰  Budget", "📋  Itinerary"]
    )
    with tab_tr:
        _c = _clean_tab(result.get("transport_results") or "")
        if _c:
            st.markdown(_c)
        else:
            st.markdown('<div class="tip-card"><div class="tip-card-text">Transport options — flights, trains, buses, and self-drive — appear here. Enter a departure city in the "From" field.</div></div>', unsafe_allow_html=True)
    with tab_ht:
        _c = _clean_tab(result.get("hotel_results") or "")
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
            st.markdown(_clean_tab(_c))
        else:
            st.markdown('<div class="tip-card"><div class="tip-card-text">Weather data will appear here once the destination is confirmed.</div></div>', unsafe_allow_html=True)
    with tab_nb:
        _c = _clean_tab(result.get("nearby_results") or "")
        if _c:
            st.markdown(_c)
        else:
            st.markdown('<div class="tip-card"><div class="tip-card-text">Nearby attractions, local food & markets will appear here.</div></div>', unsafe_allow_html=True)
    with tab_bud:
        _c = _clean_tab(result.get("budget_results") or "")
        if _c:
            st.markdown(_c)
        else:
            st.markdown('<div class="tip-card"><div class="tip-card-text">Budget breakdown will appear here. Mention your budget in the request for a cost estimate.</div></div>', unsafe_allow_html=True)

    # ── Itinerary tab — all plan states (approval / itinerary) ──────────────────
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
            # ── Parse itinerary_json + strip JSON from prose ─────────────────
            _itin_jstr  = _final.get("itinerary_json", "") or ""
            _itin_jdata = {}
            if _itin_jstr.strip():
                try:
                    _itin_jdata = json.loads(_itin_jstr)
                except Exception:
                    try:
                        _itin_jdata = json.loads(re.sub(r',(\s*[}\]])', r'\1', _itin_jstr))
                    except Exception:
                        pass

            _resp_text = _strip_json_from_prose(
                _final.get("final_response") or "", _itin_jstr
            )

            # ── Render: structured HTML tables (preferred) ───────────────────
            # Filter out null/empty day entries before rendering
            if _itin_jdata.get("days"):
                _itin_jdata["days"] = [d for d in _itin_jdata["days"] if d and isinstance(d, dict)]
            if _itin_jdata.get("days"):
                try:
                    st.markdown(_render_itin_html(_itin_jdata), unsafe_allow_html=True)
                except Exception:
                    pass  # fall through to prose fallback below
                # Show non-day prose sections (e.g. "What to Say & Do")
                _extra, _in_extra = [], False
                for _ln in _resp_text.split("\n"):
                    _ls = _ln.strip()
                    if _ls.startswith("## ") and "day" not in _ls.lower():
                        _in_extra = True
                    if _in_extra:
                        _extra.append(_ln)
                if _extra:
                    st.markdown("\n".join(_extra))

            elif _resp_text:
                # Fallback: clean prose/markdown — JSON already stripped above
                st.markdown(_resp_text)

            else:
                st.info("Itinerary content is being processed. If this persists, regenerate the plan.")
            st.divider()

            # Cache PDF bytes keyed by thread_id so each new run regenerates
            _pdf_cache_key = f"pdf_{st.session_state.get('thread_id', _dest_lbl)}"
            if st.session_state.get("_pdf_cache_key") != _pdf_cache_key or not st.session_state.get("cached_pdf_bytes"):
                st.session_state["cached_pdf_bytes"] = _build_pdf_bytes(_final)
                st.session_state["_pdf_cache_key"]   = _pdf_cache_key
            _pdf_bytes = st.session_state["cached_pdf_bytes"]

            _col_pdf, _col_share = st.columns([3, 2])
            with _col_pdf:
                st.download_button(
                    label="Download PDF",
                    data=_pdf_bytes,
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
            _itin_j2_str = _final.get("itinerary_json", "") or ""
            _itin_j2 = {}
            if _itin_j2_str.strip():
                try:
                    _itin_j2 = json.loads(_itin_j2_str)
                except Exception:
                    try:
                        _itin_j2 = json.loads(re.sub(r',(\s*[}\]])', r'\1', _itin_j2_str))
                    except Exception:
                        pass
            if _itin_j2.get("days"):
                _itin_j2["days"] = [d for d in _itin_j2["days"] if d and isinstance(d, dict)]
            if _itin_j2.get("days"):
                try:
                    st.markdown(_render_itin_html(_itin_j2), unsafe_allow_html=True)
                except Exception:
                    pass
            else:
                _raw_itin = _strip_json_from_prose(_final["itinerary"], _itin_j2_str)
                if _raw_itin:
                    st.markdown(_raw_itin)
        elif st.session_state.get("waiting_for_approval") and not _gen_final:
            # ── Approval form lives here so the user sees it immediately ──────
            _apr_res = st.session_state.get("latest_result", {}) or {}
            st.markdown(
                '<div class="approval-box">'
                '<div class="approval-title">Research complete — choose your preferences</div>'
                '<div class="approval-sub">'
                'Review findings in the other tabs, set your preferences below, then generate.'
                '</div></div>',
                unsafe_allow_html=True,
            )
            _tr2 = _apr_res.get("transport_results", "") or ""
            _ht2 = _apr_res.get("hotel_results", "") or ""
            _ex1, _ex2 = st.columns(2)
            with _ex1:
                if _tr2:
                    with st.expander("🚀 Transport options — review", expanded=False):
                        st.markdown(_tr2[:3000] + ("…" if len(_tr2) > 3000 else ""))
            with _ex2:
                if _ht2:
                    with st.expander("🏨 Hotel options — review", expanded=False):
                        st.markdown(_ht2[:2500] + ("…" if len(_ht2) > 2500 else ""))

            _transport_opts = ["Flight", "Train", "Bus", "Self-drive / Car rental", "No preference"]
            _ia1, _ia2 = st.columns(2)
            with _ia1:
                ch_transport = st.radio("Which transport mode do you prefer?",
                                        _transport_opts, key="apr_transport")
                ch_food = st.radio("Dietary preference?",
                                   ["Vegetarian", "Non-vegetarian", "No preference"],
                                   key="apr_food",
                                   help="Vegetarian filters hotels & meals to pure-veg only.")
            with _ia2:
                ch_hotel = st.radio("Hotel budget tier?",
                                    ["Budget  (< Rs.2,000/night)", "Mid-range  (Rs.2,000–5,000)", "Premium  (Rs.5,000+)"],
                                    key="apr_hotel")
                ch_style = st.radio("Travel style?",
                                    ["Explorer (lots of sightseeing)", "Relaxed (slow pace)",
                                     "Foodie (local cuisine focus)", "Cultural / Spiritual"],
                                    key="apr_style")
            ch_special = st.text_input("Any special requests? (optional)",
                                       placeholder="Wheelchair access, halal food, honeymoon couple…",
                                       key="apr_special")
            if st.button("Generate My Personalised Itinerary", type="primary",
                         use_container_width=True, key="apr_generate"):
                _choices = {
                    "transport": ch_transport, "hotel": ch_hotel,
                    "food": ch_food, "style": ch_style, "special_requests": ch_special,
                }
                st.session_state["user_choices_display"] = _choices
                st.session_state["pending_choices"]      = _choices
                st.session_state["generating_final"]     = True
                st.rerun()
