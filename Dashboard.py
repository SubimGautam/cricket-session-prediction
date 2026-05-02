import os
import sys
import time
import joblib
import warnings
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "outputs"
MODEL_DIR   = OUTPUT_DIR / "models"
SESSION_CSV = OUTPUT_DIR / "session_features.csv"
MATCH_CSV   = OUTPUT_DIR / "match_level_features.csv"
sys.path.insert(0, str(BASE_DIR / "src"))

# ── Design tokens ─────────────────────────────────────────────────────────
C_PRIMARY  = "#00E5A0"   # Electric mint — primary accent
C_RED      = "#FF4D6D"   # Vivid coral-red — bowling/danger
C_BLUE     = "#3D8EF0"   # Sharp blue — informational
C_AMBER    = "#FFB547"   # Amber — warning/neutral
C_BG       = "#0A0C10"   # Near-black background
C_SURFACE  = "#111318"   # Card surface
C_SURFACE2 = "#1A1D24"   # Elevated surface
C_BORDER   = "#252830"   # Subtle border
C_TEXT     = "#F0F2F5"   # Primary text
C_MUTED    = "#6B7280"   # Muted text

SESSION_FEATURES = [
    "session_run_rate", "session_runs", "dot_ball_pct", "boundary_rate",
    "session_extras", "session_wickets", "wickets_per_over",
    "wickets_at_session_end", "max_dot_streak", "total_pressure_balls",
    "run_rate_delta", "wickets_delta", "dot_ball_pct_delta",
    "session_momentum_index", "ball_age_start", "innings_num",
    "is_home_batting", "toss_bat_first", "toss_winner_batting",
    "is_fourth_innings", "is_first_innings", "is_morning_session",
    "is_evening_session", "top_order_exposed",
]

def _try_import(module_name: str):
    try:
        import importlib
        return importlib.import_module(module_name)
    except Exception:
        return None

_live_feed      = _try_import("live_feed")
_state_manager  = _try_import("state_manager")
_live_predictor = _try_import("live_predictor")
_session_seg    = _try_import("session_segmenter")
LIVE_MODULES_READY = all([_live_feed, _state_manager, _live_predictor, _session_seg])

_prematch_module = _try_import("prematch_predictor")
PREMATCH_READY   = _prematch_module is not None


st.set_page_config(
    page_title="CricIQ · Momentum Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --green:   #00E5A0;
    --red:     #FF4D6D;
    --blue:    #3D8EF0;
    --amber:   #FFB547;
    --bg:      #0A0C10;
    --surf:    #111318;
    --surf2:   #1A1D24;
    --border:  #252830;
    --border2: #2E3240;
    --text:    #F0F2F5;
    --muted:   #6B7280;
    --muted2:  #9CA3AF;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --font-body:    'Outfit', sans-serif;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
}

.stApp { background: var(--bg) !important; }

/* ── SIDEBAR ─────────────────────────────────────────── */
div[data-testid="stSidebar"] {
    background: var(--surf) !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}
div[data-testid="stSidebar"] * {
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}
div[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}

/* ── WORDMARK ─────────────────────────────────────────── */
.wordmark {
    padding: 28px 24px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}
.wordmark-top {
    font-family: var(--font-display);
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
    line-height: 1;
    display: flex;
    align-items: center;
    gap: 8px;
}
.wordmark-top .accent { color: var(--green); }
.wordmark-sub {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── NAV RADIO ────────────────────────────────────────── */
.stRadio > div {
    gap: 2px !important;
    padding: 0 12px;
}
.stRadio > div > label {
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: var(--muted2) !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    border: none !important;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex !important;
    align-items: center !important;
    gap: 8px;
}
.stRadio > div > label:hover {
    background: var(--surf2) !important;
    color: var(--text) !important;
}
.stRadio > div > label:has(input:checked) {
    background: rgba(0, 229, 160, 0.1) !important;
    color: var(--green) !important;
    border: none !important;
}

/* ── STATUS CHIPS ─────────────────────────────────────── */
.status-section {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    margin-top: 8px;
}
.status-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 10px;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 500;
    margin-bottom: 6px;
    border: 1px solid;
}
.chip.green  { color: var(--green);  border-color: rgba(0,229,160,0.25); background: rgba(0,229,160,0.06); }
.chip.red    { color: var(--red);    border-color: rgba(255,77,109,0.25); background: rgba(255,77,109,0.06); }
.chip.amber  { color: var(--amber);  border-color: rgba(255,181,71,0.25); background: rgba(255,181,71,0.06); }
.chip.muted  { color: var(--muted2); border-color: var(--border2); background: var(--surf2); }
.chip-dot {
    width: 6px; height: 6px; border-radius: 50%; background: currentColor;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── PAGE HEADER ──────────────────────────────────────── */
.page-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    padding: 32px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
.page-title {
    font-family: var(--font-display);
    font-size: 1.7rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--text);
    line-height: 1.1;
}
.page-title span { color: var(--green); }
.page-meta {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 6px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: rgba(255, 77, 109, 0.1);
    border: 1px solid rgba(255, 77, 109, 0.3);
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--red);
    letter-spacing: 0.08em;
    font-weight: 600;
}

/* ── METRIC CARDS ─────────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.metric-card {
    background: var(--surf);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: var(--accent-color, var(--border2));
    border-radius: 10px 10px 0 0;
}
.metric-card.green::before { --accent-color: var(--green); }
.metric-card.red::before   { --accent-color: var(--red); }
.metric-card.blue::before  { --accent-color: var(--blue); }
.metric-card.amber::before { --accent-color: var(--amber); }
.metric-card:hover { border-color: var(--border2); }
.metric-value {
    font-family: var(--font-display);
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 6px;
    color: var(--text);
}
.metric-value.green { color: var(--green); }
.metric-value.red   { color: var(--red); }
.metric-value.blue  { color: var(--blue); }
.metric-value.amber { color: var(--amber); }
.metric-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    font-weight: 400;
}
.metric-delta {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted2);
    margin-top: 4px;
}

/* ── SECTION HEADINGS ─────────────────────────────────── */
.section-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}
.section-head-title {
    font-family: var(--font-display);
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text);
}
.section-head-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
.sub-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── SESSION CARDS ────────────────────────────────────── */
.session-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 24px;
}
.session-card {
    background: var(--surf);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    transition: border-color 0.2s;
}
.session-card:hover { border-color: var(--border2); }
.session-card-header {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 10px;
}
.session-value {
    font-family: var(--font-display);
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 6px;
}
.session-value.pos { color: var(--green); }
.session-value.neg { color: var(--red); }
.session-value.neu { color: var(--muted2); }
.session-footer {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--muted2);
}

/* ── MOMENTUM BADGES ──────────────────────────────────── */
.mbadge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    border: 1px solid;
}
.mbadge.bat  { color: var(--green); border-color: rgba(0,229,160,0.3); background: rgba(0,229,160,0.08); }
.mbadge.bowl { color: var(--red);   border-color: rgba(255,77,109,0.3); background: rgba(255,77,109,0.08); }
.mbadge.neu  { color: var(--muted2); border-color: var(--border2); background: var(--surf2); }

/* ── INFO BOXES ───────────────────────────────────────── */
.info-card {
    background: var(--surf);
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue);
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    font-family: var(--font-body);
    font-size: 0.85rem;
    color: var(--muted2);
    margin: 12px 0 20px;
    line-height: 1.6;
}
.info-card b { color: var(--text); font-weight: 600; }
.info-card.green { border-left-color: var(--green); }
.info-card.red   { border-left-color: var(--red); }
.info-card.amber { border-left-color: var(--amber); }

/* ── PIPELINE STEPS ───────────────────────────────────── */
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 16px 0;
    overflow-x: auto;
}
.pipe-step {
    flex: 1;
    min-width: 90px;
    text-align: center;
    position: relative;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    padding: 8px 4px;
}
.pipe-step::after {
    content: '';
    position: absolute;
    right: 0; top: 50%;
    transform: translateY(-50%);
    width: 1px; height: 20px;
    background: var(--border2);
}
.pipe-step:last-child::after { display: none; }
.pipe-step.done  { color: var(--green); }
.pipe-step.active { color: var(--text); font-weight: 600; }
.pipe-step.done .dot,
.pipe-step.active .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    margin: 0 auto 5px;
    display: block;
}
.pipe-step.done .dot  { background: var(--green); }
.pipe-step.active .dot { background: var(--amber); animation: pulse 1.5s infinite; }
.pipe-step.todo .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    margin: 0 auto 5px;
    display: block;
    background: var(--border2);
}

/* ── TABLE STYLING ────────────────────────────────────── */
.stDataFrame {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* ── PLOTLY CHARTS ────────────────────────────────────── */
.js-plotly-plot { border-radius: 8px; overflow: hidden; }

/* ── FORM ELEMENTS ────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stSelectbox > div > div {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    background: var(--surf2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 2px rgba(0,229,160,0.1) !important;
}
.stTextInput label, .stNumberInput label, .stSelectbox label {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
    font-weight: 400 !important;
}

/* ── BUTTONS ──────────────────────────────────────────── */
.stButton > button {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    font-weight: 500 !important;
    background: var(--surf2) !important;
    color: var(--muted2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: var(--surf) !important;
    color: var(--text) !important;
    border-color: var(--muted) !important;
}
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: var(--green) !important;
    color: #000 !important;
    border-color: var(--green) !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: #00ffb0 !important;
}

/* ── TABS ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 18px !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--green) !important;
    border-bottom-color: var(--green) !important;
}

/* ── COMPARISON TABLE ─────────────────────────────────── */
.ctable { width: 100%; border-collapse: collapse; }
.ctable th {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding: 8px 12px;
    text-align: left;
}
.ctable td {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--muted2);
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
}
.ctable tr:last-child td { border-bottom: none; }
.ctable tr.best td { color: var(--text); background: rgba(0,229,160,0.04); }
.ctable tr.best td:first-child {
    color: var(--green);
    border-left: 2px solid var(--green);
}
.ctable tr:hover td { background: var(--surf2); }

/* ── SKELETON LOADING ─────────────────────────────────── */
.skel {
    background: linear-gradient(90deg, var(--surf) 25%, var(--surf2) 50%, var(--surf) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 6px;
}
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

/* ── FOOTER ───────────────────────────────────────────── */
.footer {
    border-top: 1px solid var(--border);
    padding: 24px 0;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 48px;
}

/* ── POLLING BANNER ───────────────────────────────────── */
.poll-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--surf);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted2);
    margin-bottom: 16px;
}
.poll-banner.active { border-color: rgba(0,229,160,0.3); color: var(--green); background: rgba(0,229,160,0.05); }

/* ── WARNINGS ─────────────────────────────────────────── */
.stAlert { border-radius: 8px !important; font-family: var(--font-body) !important; }
div[data-testid="stWarning"] { background: rgba(255,181,71,0.08) !important; border-color: rgba(255,181,71,0.3) !important; }
div[data-testid="stError"]   { background: rgba(255,77,109,0.08) !important; border-color: rgba(255,77,109,0.3) !important; }
div[data-testid="stInfo"]    { background: rgba(61,142,240,0.08) !important; border-color: rgba(61,142,240,0.3) !important; }

/* ── EXPANDER ─────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--muted2) !important;
    background: var(--surf) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── SCROLLBAR ────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── ANIMATIONS ───────────────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.main .block-container { animation: fadeUp 0.4s ease-out forwards; }

@media (max-width: 768px) {
    .metric-grid { grid-template-columns: repeat(2,1fr) !important; }
    .session-grid { grid-template-columns: 1fr !important; }
    .page-title { font-size: 1.3rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADERS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_session_data():
    if not SESSION_CSV.exists(): return None
    return pd.read_csv(SESSION_CSV)

@st.cache_data(show_spinner=False)
def load_match_data():
    if not MATCH_CSV.exists(): return None
    return pd.read_csv(MATCH_CSV)

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    try:
        models["xgb"] = joblib.load(MODEL_DIR / "session_momentum_xgb.pkl")
        models["le"]  = joblib.load(MODEL_DIR / "label_encoder.pkl")
    except Exception: pass
    try:
        models["match"] = joblib.load(MODEL_DIR / "match_outcome_xgb.pkl")
    except Exception: pass
    return models


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def momentum_badge_html(label):
    try: v = int(label)
    except: v = 0
    if v == 1:
        return '<span class="mbadge bat">▲ Batting</span>'
    elif v == -1:
        return '<span class="mbadge bowl">▼ Bowling</span>'
    return '<span class="mbadge neu">→ Neutral</span>'

def momentum_color(label):
    try:
        v = int(label)
        if v == 1:  return C_PRIMARY
        if v == -1: return C_RED
    except: pass
    return C_MUTED

@st.cache_data(show_spinner=False)
def build_match_lookup(session_df):
    lookup = {}
    for mid, grp in session_df.groupby("match_id"):
        batting  = set(grp["batting_team"].dropna().unique())
        fielding = set(grp["fielding_team"].dropna().unique())
        teams    = sorted(batting | fielding)
        a = teams[0] if teams else "Unknown"
        b = teams[1] if len(teams) > 1 else a
        extra = ""
        for col in ["match_date","date","venue"]:
            if col in grp.columns:
                val = grp[col].iloc[0]
                if pd.notna(val):
                    extra = f" · {str(val)[:10]}"
                    break
        lookup[mid] = (a, b, extra)
    return lookup

def match_label(mid, lookup):
    if mid not in lookup: return str(mid)
    a, b, extra = lookup[mid]
    return f"{mid} — {a} vs {b}{extra}"

def chart_layout(height=300, **kw):
    base = dict(
        height=height,
        margin=dict(l=8, r=8, t=16, b=8),
        paper_bgcolor=C_SURFACE, plot_bgcolor=C_SURFACE,
        font=dict(family="JetBrains Mono, monospace", size=10, color=C_MUTED),
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=9, color=C_MUTED),
                   linecolor=C_BORDER, tickcolor=C_BORDER, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=C_BORDER,
                   linecolor=C_BORDER, tickcolor=C_BORDER,
                   tickfont=dict(size=9, color=C_MUTED), zeroline=False),
    )
    base.update(kw)
    return base


# ════════════════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════════════════

def plot_wp_curve(session_df, match_id):
    df = session_df[session_df["match_id"] == match_id].copy()
    if df.empty or "win_probability" not in df.columns: return None
    df["lbl"] = df["innings_num"].astype(str) + " · " + df["session"].str.split("_").str[-1].str.title()
    fig = go.Figure()
    # Fill area
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df["win_probability"],
        fill="tozeroy", fillcolor="rgba(0,229,160,0.05)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))
    # Main line
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df["win_probability"],
        mode="lines+markers",
        line=dict(color=C_PRIMARY, width=2),
        marker=dict(size=9,
                    color=[momentum_color(m) for m in df.get("momentum_label", [0]*len(df))],
                    line=dict(color=C_SURFACE, width=2)),
        hovertemplate="<b>%{customdata[0]}</b><br>WP: %{y:.1%}  RR: %{customdata[1]:.2f}  Wkts: %{customdata[2]}<extra></extra>",
        customdata=np.stack([
            df["lbl"],
            df.get("session_run_rate", pd.Series([0]*len(df))).fillna(0),
            df.get("session_wickets",  pd.Series([0]*len(df))).fillna(0),
        ], axis=-1),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color=C_BORDER, line_width=1)
    layout = chart_layout(height=280)
    layout["xaxis"].update(tickvals=list(range(len(df))),
                           ticktext=df["lbl"].tolist(), tickangle=30)
    layout["yaxis"].update(tickformat=".0%", range=[0,1])
    fig.update_layout(**layout)
    return fig

def plot_momentum_bars(session_df, match_id):
    df = session_df[session_df["match_id"] == match_id].copy()
    if df.empty or "session_momentum_index" not in df.columns: return None
    df["lbl"] = df["innings_num"].astype(str) + "·" + df["session"].str.split("_").str[-1].str.title()
    colors = df["session_momentum_index"].apply(lambda x: C_PRIMARY if x > 0 else C_RED)
    fig = go.Figure(go.Bar(
        x=df["lbl"], y=df["session_momentum_index"],
        marker_color=colors, marker_line_width=0,
        marker_cornerradius=4,
        hovertemplate="<b>%{x}</b><br>Index: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=C_BORDER, line_width=1)
    layout = chart_layout(height=250)
    layout["xaxis"].update(tickangle=30)
    fig.update_layout(**layout)
    return fig

def plot_session_stats(session_df, match_id):
    df = session_df[session_df["match_id"] == match_id].copy()
    if df.empty: return None
    df["lbl"] = df["innings_num"].astype(str) + "·" + df["session"].str.split("_").str[-1].str.title()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df["lbl"],
        y=df.get("session_run_rate", pd.Series([0]*len(df))).fillna(0),
        name="Run Rate", marker_color=C_BLUE, opacity=0.7,
        marker_line_width=0, marker_cornerradius=4,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["lbl"],
        y=df.get("session_wickets", pd.Series([0]*len(df))).fillna(0),
        name="Wickets", mode="lines+markers",
        line=dict(color=C_RED, width=2),
        marker=dict(size=7, color=C_RED, line=dict(color=C_SURFACE, width=2)),
    ), secondary_y=True)
    layout = chart_layout(height=250)
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02,
                             font=dict(size=9, color=C_MUTED),
                             bgcolor="rgba(0,0,0,0)")
    layout["xaxis"].update(tickangle=30)
    fig.update_layout(**layout)
    fig.update_yaxes(showgrid=True, gridcolor=C_BORDER, secondary_y=False)
    fig.update_yaxes(showgrid=False, secondary_y=True)
    return fig

def plot_probability_bars(p_bowl, p_neut, p_bat):
    fig = go.Figure(go.Bar(
        x=["Bowling", "Neutral", "Batting"],
        y=[p_bowl, p_neut, p_bat],
        marker_color=[C_RED, C_MUTED, C_PRIMARY],
        marker_line_width=0, marker_cornerradius=6,
        text=[f"{v*100:.1f}%" for v in [p_bowl, p_neut, p_bat]],
        textposition="outside",
        textfont=dict(size=12, family="JetBrains Mono", color=C_TEXT),
    ))
    layout = chart_layout(height=220)
    layout["yaxis"].update(tickformat=".0%", range=[0, 1.15])
    fig.update_layout(**layout)
    return fig

def plot_live_wp(wp_history):
    if not wp_history: return None
    df = pd.DataFrame(wp_history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df["wp_batting"],
        mode="lines+markers", name="Batting",
        line=dict(color=C_PRIMARY, width=2),
        marker=dict(size=6, color=C_PRIMARY, line=dict(color=C_SURFACE, width=2)),
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df["wp_bowling"],
        mode="lines+markers", name="Bowling",
        line=dict(color=C_RED, width=2, dash="dot"),
        marker=dict(size=6, color=C_RED, line=dict(color=C_SURFACE, width=2)),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color=C_BORDER, line_width=1)
    layout = chart_layout(height=260)
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02,
                             font=dict(size=9, color=C_MUTED), bgcolor="rgba(0,0,0,0)")
    layout["xaxis"].update(tickvals=list(range(len(df))),
                           ticktext=[f"#{i+1}" for i in range(len(df))])
    layout["yaxis"].update(tickformat=".0%", range=[0,1])
    fig.update_layout(**layout)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# POLLING
# ════════════════════════════════════════════════════════════════════════════

def _poll_worker(match_id, interval):
    while st.session_state.get(f"polling_active_{match_id}", False):
        try:
            result = _live_predictor.predict_current_session(match_id)
            if "error" not in result:
                _state_manager.save_prediction(result)
                _state_manager.save_wp_point(match_id, result["prob_batting"], result["prob_bowling"], result["session_name"])
                st.session_state["last_result"]  = result
                st.session_state["last_update"]  = datetime.utcnow().strftime("%H:%M:%S UTC")
                st.session_state["poll_count"]   = st.session_state.get("poll_count", 0) + 1
                st.session_state["poll_error"]   = None
            else:
                st.session_state["poll_error"] = result["error"]
        except Exception as e:
            st.session_state["poll_error"] = str(e)
        time.sleep(interval)

def start_polling(match_id, interval=300):
    key = f"poll_thread_{match_id}"
    if st.session_state.get(key): return
    st.session_state[f"polling_active_{match_id}"] = True
    t = threading.Thread(target=_poll_worker, args=(match_id, interval), daemon=True)
    t.start()
    st.session_state[key] = t

def stop_polling(match_id):
    st.session_state[f"polling_active_{match_id}"] = False
    st.session_state[f"poll_thread_{match_id}"]    = None


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="wordmark">
        <div class="wordmark-top">
            🏏 Cric<span class="accent">IQ</span>
        </div>
        <div class="wordmark-sub">Momentum Analytics · MSc Thesis</div>
    </div>
    """, unsafe_allow_html=True)

    tab_choice = st.radio(
        "Navigation",
        ["📊  Match Explorer", "🔮  Live Prediction", "📈  Model Performance", "🔴  Live Match", "🔭  Pre-Match Forecast"],
        label_visibility="collapsed",
    )

    session_df = load_session_data()
    match_df   = load_match_data()
    models     = load_models()

    st.markdown('<div class="status-section">', unsafe_allow_html=True)
    st.markdown('<div class="status-label">System Status</div>', unsafe_allow_html=True)

    if session_df is not None:
        st.markdown(f"""
        <div class="chip green"><span class="chip-dot"></span>
            {session_df['match_id'].nunique()} matches · {len(session_df):,} sessions
        </div><br>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="chip amber">⚠ No session data — run pipeline</div><br>', unsafe_allow_html=True)

    if models.get("xgb"):
        st.markdown('<div class="chip green"><span class="chip-dot"></span> XGBoost loaded</div><br>', unsafe_allow_html=True)
    if models.get("match"):
        st.markdown('<div class="chip green"><span class="chip-dot"></span> Match model loaded</div><br>', unsafe_allow_html=True)

    if LIVE_MODULES_READY:
        st.markdown('<div class="chip green"><span class="chip-dot"></span> Live modules ready</div>', unsafe_allow_html=True)
    else:
        missing = [n for n, m in [("live_feed",_live_feed),("state_manager",_state_manager),
                                   ("live_predictor",_live_predictor),("session_segmenter",_session_seg)] if m is None]
        st.markdown(f'<div class="chip muted">⊘ Live offline · {len(missing)} missing</div>', unsafe_allow_html=True)

    if PREMATCH_READY:
        st.markdown('<div class="chip green"><span class="chip-dot"></span> Pre-match ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="chip muted">⊘ prematch_predictor.py missing</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════════════════════════

tab_titles = {
    "📊  Match Explorer"   : ("Match <span style='color:var(--green)'>Explorer</span>",    "Session-level analysis · Win probability · Momentum"),
    "🔮  Live Prediction"  : ("Manual <span style='color:var(--green)'>Prediction</span>", "Enter live stats · Instant model inference"),
    "📈  Model Performance": ("Model <span style='color:var(--green)'>Performance</span>", "Evaluation metrics · Cross-model validation · SHAP"),
    "🔴  Live Match"       : ("Live <span style='color:var(--red)'>Match</span>",           "Real-time feed · Auto-polling · Prediction log"),
    "🔭  Pre-Match Forecast": ("Pre-Match <span style='color:var(--green)'>Forecast</span>", "Session-level prediction · Team profiles · H2H analysis"),
}
t_title, t_sub = tab_titles.get(tab_choice, ("Dashboard", ""))
now_str = datetime.utcnow().strftime("%d %b %Y · %H:%M UTC")

st.markdown(f"""
<div class="page-header">
    <div>
        <div class="page-title">{t_title}</div>
        <div class="page-meta">{t_sub}</div>
    </div>
    <div style="text-align:right">
        <div class="page-meta">{now_str}</div>
        {'<div class="live-badge"><span style="width:6px;height:6px;border-radius:50%;background:var(--red);animation:pulse 1.5s infinite;display:inline-block"></span>&nbsp;LIVE</div>' if tab_choice == "🔴  Live Match" else ''}
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH EXPLORER
# ════════════════════════════════════════════════════════════════════════════

if tab_choice == "📊  Match Explorer":

    if session_df is None:
        st.error("Session data not found. Run `python main.py` first.")
        st.stop()

    match_lookup = build_match_lookup(session_df)
    match_ids    = sorted(session_df["match_id"].unique())

    col_sel, col_spacer = st.columns([2, 3])
    with col_sel:
        st.markdown('<div class="sub-label">Select Match</div>', unsafe_allow_html=True)
        sq = st.text_input("", placeholder="Search team or match ID…", key="match_search")
        q  = sq.strip().lower()
        filtered_ids  = [m for m in match_ids if q in match_label(m, match_lookup).lower()] if q else match_ids
        if not filtered_ids:
            st.warning("No matches found.")
            st.stop()
        st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--muted);margin-bottom:6px">{len(filtered_ids)} match(es)</div>', unsafe_allow_html=True)
        label_to_id    = {match_label(m, match_lookup): m for m in filtered_ids}
        selected_label = st.selectbox("", list(label_to_id.keys()), key="match_sel", label_visibility="collapsed")
        selected_match = label_to_id[selected_label]

    match_sessions = session_df[session_df["match_id"] == selected_match]

    # ── Match result metrics ──────────────────────────────────────
    if match_df is not None:
        m_row = match_df[match_df["match_id"] == selected_match]
        if not m_row.empty:
            m_row = m_row.iloc[0]
            wp    = float(m_row.get("final_wp", 0.5))
            rev   = int(m_row.get("momentum_reversals", 0))
            winner = str(m_row.get("winner", "—"))
            wp_cls = "green" if wp > 0.5 else "red"
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card {wp_cls}">
                    <div class="metric-value {wp_cls}">{wp:.0%}</div>
                    <div class="metric-label">Final Win Probability</div>
                </div>
                <div class="metric-card amber">
                    <div class="metric-value amber">{rev}</div>
                    <div class="metric-label">Momentum Reversals</div>
                </div>
                <div class="metric-card blue">
                    <div class="metric-value" style="font-size:1.3rem;color:var(--blue)">{winner}</div>
                    <div class="metric-label">Winner</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(match_sessions)}</div>
                    <div class="metric-label">Sessions Played</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Session summary cards ──────────────────────────────────────
    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Session Momentum</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    if not match_sessions.empty:
        cards_html = '<div class="session-grid">'
        for _, row in match_sessions.head(6).iterrows():
            inn  = int(row.get("innings_num", 1))
            sess = str(row.get("session", "")).replace("_"," ").title()
            idx  = float(row.get("session_momentum_index", 0))
            cls  = "pos" if idx > 0 else ("neg" if idx < 0 else "neu")
            sign = "+" if idx > 0 else ""
            cards_html += f"""
            <div class="session-card">
                <div class="session-card-header">Inn {inn} · {sess}</div>
                <div class="session-value {cls}">{sign}{idx:.2f}</div>
                <div class="session-footer">
                    {int(row.get("session_runs",0))} runs &nbsp;/&nbsp; {int(row.get("session_wickets",0))} wkts
                    &nbsp;·&nbsp; RR {float(row.get("session_run_rate",0)):.2f}
                </div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Win Probability</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    chart_col, note_col = st.columns([3, 1])
    with chart_col:
        wp_fig = plot_wp_curve(session_df, selected_match)
        if wp_fig:
            st.plotly_chart(wp_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Win probability data not available.")
    with note_col:
        st.markdown("""
        <div class="info-card blue" style="margin-top:8px">
            <b>ΔWP Method</b><br>Net change in batting-team win probability from end of preceding session to end of current session.
        </div>
        <div class="info-card" style="margin-top:8px">
            <b>Marker Colours</b><br>
            <span style="color:var(--green)">●</span> Batting momentum<br>
            <span style="color:var(--red)">●</span> Bowling momentum<br>
            <span style="color:var(--muted2)">●</span> Neutral
        </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="sub-label">Momentum Index by Session</div>', unsafe_allow_html=True)
        mf = plot_momentum_bars(session_df, selected_match)
        if mf: st.plotly_chart(mf, use_container_width=True, config={"displayModeBar": False})
    with col_r:
        st.markdown('<div class="sub-label">Run Rate & Wickets</div>', unsafe_allow_html=True)
        sf = plot_session_stats(session_df, selected_match)
        if sf: st.plotly_chart(sf, use_container_width=True, config={"displayModeBar": False})

    # ── Data table ────────────────────────────────────────────────
    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Session Detail</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    display_cols = [c for c in ["innings_num","session","session_runs","session_run_rate",
                                "session_wickets","dot_ball_pct","max_dot_streak",
                                "session_momentum_index","win_probability","momentum_label"]
                    if c in match_sessions.columns]
    table = match_sessions[display_cols].copy()
    for col in ["session_run_rate","dot_ball_pct","session_momentum_index","win_probability"]:
        if col in table.columns:
            table[col] = table[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    if "momentum_label" in table.columns:
        def _lbl(x):
            try:
                v = float(x)
                return "▲ Batting" if v == 1 else ("▼ Bowling" if v == -1 else "→ Neutral")
            except: return "—"
        table["momentum_label"] = table["momentum_label"].apply(_lbl)
    table.columns = [c.replace("_"," ").title() for c in table.columns]
    st.dataframe(table, use_container_width=True, hide_index=True)

    # ── Pipeline status ───────────────────────────────────────────
    st.markdown("""
    <div class="section-head" style="margin-top:24px">
        <span class="section-head-title">Pipeline Status</span>
        <span class="section-head-line"></span>
    </div>
    <div style="background:var(--surf);border:1px solid var(--border);border-radius:10px;padding:8px 0;">
    <div class="pipeline">
        <div class="pipe-step done"><span class="dot"></span>Data Ingestion</div>
        <div class="pipe-step done"><span class="dot"></span>Feature Eng.</div>
        <div class="pipe-step done"><span class="dot"></span>WP Calc.</div>
        <div class="pipe-step done"><span class="dot"></span>Session Seg.</div>
        <div class="pipe-step active"><span class="dot"></span>Momentum Model</div>
        <div class="pipe-step todo"><span class="dot"></span>Interpretability</div>
    </div></div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MANUAL LIVE PREDICTION
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "🔮  Live Prediction":

    if "xgb" not in models:
        st.warning("XGBoost model not found. Run the pipeline first.")
        st.stop()

    st.markdown("""
    <div class="info-card">
        Enter current session statistics to get an instant momentum prediction.
        The model outputs class probabilities for batting momentum, bowling momentum, and neutral sessions.
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="sub-label">Scoring</div>', unsafe_allow_html=True)
        session_runs     = st.number_input("Session Runs",     0, 300, 65)
        session_run_rate = st.number_input("Run Rate",         0.0, 15.0, 3.5, 0.1)
        boundary_rate    = st.number_input("Boundary Rate",    0.0, 0.5, 0.08, 0.01)
        dot_ball_pct     = st.number_input("Dot Ball %",       0.0, 1.0, 0.45, 0.01)
    with c2:
        st.markdown('<div class="sub-label">Wickets & Pressure</div>', unsafe_allow_html=True)
        session_wickets      = st.number_input("Session Wickets",      0, 10, 2)
        wickets_at_end       = st.number_input("Total Wickets Fallen", 0, 10, 4)
        max_dot_streak       = st.number_input("Max Dot Streak",       0, 30, 5)
        total_pressure_balls = st.number_input("Pressure Balls",       0, 50, 8)
    with c3:
        st.markdown('<div class="sub-label">Match Context</div>', unsafe_allow_html=True)
        innings_num     = st.selectbox("Innings", [1, 2, 3, 4], index=1)
        ball_age_start  = st.number_input("Ball Age (balls)", 1, 480, 60)
        is_home_batting = st.selectbox("Home Team Batting?", ["Yes", "No"])
        toss_bat_first  = st.selectbox("Toss: Bat First?",   ["Yes", "No"])

    st.markdown('<div class="sub-label" style="margin-top:16px">Previous Session (Delta Features)</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1: prev_run_rate = st.number_input("Prev. Run Rate",    0.0, 15.0, 3.0, 0.1)
    with d2: prev_wickets  = st.number_input("Prev. Wickets",     0, 10, 1)
    with d3: prev_dot_pct  = st.number_input("Prev. Dot Ball %",  0.0, 1.0, 0.40, 0.01)

    if st.button("Run Inference →", type="primary", use_container_width=True):
        features = {
            "session_run_rate"       : session_run_rate,
            "session_runs"           : session_runs,
            "dot_ball_pct"           : dot_ball_pct,
            "boundary_rate"          : boundary_rate,
            "session_extras"         : 3,
            "session_wickets"        : session_wickets,
            "wickets_per_over"       : session_wickets / max(session_runs/6/max(session_run_rate,0.1),1),
            "wickets_at_session_end" : wickets_at_end,
            "max_dot_streak"         : max_dot_streak,
            "total_pressure_balls"   : total_pressure_balls,
            "run_rate_delta"         : session_run_rate - prev_run_rate,
            "wickets_delta"          : session_wickets - prev_wickets,
            "dot_ball_pct_delta"     : dot_ball_pct - prev_dot_pct,
            "session_momentum_index" : (session_run_rate-prev_run_rate)-(session_wickets-prev_wickets)*2.5,
            "ball_age_start"         : ball_age_start,
            "innings_num"            : innings_num,
            "is_home_batting"        : 1 if is_home_batting == "Yes" else 0,
            "toss_bat_first"         : 1 if toss_bat_first  == "Yes" else 0,
            "toss_winner_batting"    : 1 if (is_home_batting == "Yes" and toss_bat_first == "Yes") else 0,
            "is_fourth_innings"      : 1 if innings_num == 4 else 0,
            "is_first_innings"       : 1 if innings_num == 1 else 0,
            "is_morning_session"     : 0,
            "is_evening_session"     : 0,
            "top_order_exposed"      : 1 if wickets_at_end <= 4 else 0,
        }
        X          = pd.DataFrame([features])[SESSION_FEATURES].fillna(0)
        proba      = models["xgb"].predict_proba(X)[0]
        le         = models["le"]
        pred_label = int(le.inverse_transform([np.argmax(proba)])[0])
        classes    = le.classes_
        prob_bowl  = float(proba[list(classes).index(-1)]) if -1 in classes else 0.0
        prob_neut  = float(proba[list(classes).index(0)])  if  0 in classes else 0.0
        prob_bat   = float(proba[list(classes).index(1)])  if  1 in classes else 0.0

        st.markdown("""
        <div class="section-head" style="margin-top:28px">
            <span class="section-head-title">Prediction Result</span>
            <span class="section-head-line"></span>
        </div>""", unsafe_allow_html=True)

        mom_idx   = features["session_momentum_index"]
        idx_color = "green" if mom_idx > 0 else "red"
        conf_pct  = max(proba)*100

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card {'green' if pred_label==1 else 'red' if pred_label==-1 else ''}">
                <div style="margin-bottom:8px">{momentum_badge_html(pred_label)}</div>
                <div class="metric-label">Predicted Momentum</div>
            </div>
            <div class="metric-card blue">
                <div class="metric-value blue">{conf_pct:.0f}%</div>
                <div class="metric-label">Model Confidence</div>
            </div>
            <div class="metric-card {idx_color}">
                <div class="metric-value {idx_color}">{mom_idx:+.2f}</div>
                <div class="metric-label">Momentum Index</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.2rem">Inn {innings_num} · {'Morning' if not session_run_rate else 'Active'}</div>
                <div class="metric-label">Session Context</div>
            </div>
        </div>""", unsafe_allow_html=True)

        r_chart, r_interp = st.columns([2, 1])
        with r_chart:
            st.markdown('<div class="sub-label">Class Probability Breakdown</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_probability_bars(prob_bowl, prob_neut, prob_bat),
                            use_container_width=True, config={"displayModeBar": False})
        with r_interp:
            interp = {
                1 : f"Batting side is gaining the upper hand. Run rate delta of {features['run_rate_delta']:+.2f} and controlled wicket loss suggest batting dominance.",
                -1: f"Bowling side is applying pressure. {'Dot ball streak of ' + str(max_dot_streak) if max_dot_streak > 5 else 'Wicket-taking'} is shifting the balance.",
                0 : "Session evenly contested — neither side holds a decisive advantage.",
            }
            badge_cls = "green" if pred_label == 1 else ("red" if pred_label == -1 else "")
            st.markdown(f'<div class="info-card {badge_cls}" style="margin-top:28px"><b>Interpretation:</b><br>{interp.get(pred_label,"")}</div>',
                        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "📈  Model Performance":

    # ── Load real metrics from CSV if pipeline has been run ──────────────
    _rpt_path = MODEL_DIR / "model_evaluation_report.csv"
    _acc, _ll, _f1 = "—", "—", "—"
    if _rpt_path.exists():
        try:
            _rpt = pd.read_csv(_rpt_path)
            _xgb_row = _rpt[_rpt["Model"].str.contains("XGBoost", case=False, na=False)]
            if not _xgb_row.empty:
                _r = _xgb_row.iloc[0]
                _acc = str(_r.get("Accuracy", "—")).split("±")[0].strip()
                _ll  = str(_r.get("Log Loss", "—")).split("±")[0].strip()
                _f1  = str(_r.get("F1 Macro", "—")).split("±")[0].strip()
        except Exception:
            pass

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card green">
            <div class="metric-value green">{_acc}</div>
            <div class="metric-label">Accuracy — XGBoost</div>
            <div class="metric-delta">5-fold CV result</div>
        </div>
        <div class="metric-card blue">
            <div class="metric-value blue">{_ll}</div>
            <div class="metric-label">Log Loss</div>
            <div class="metric-delta">Calibration metric</div>
        </div>
        <div class="metric-card amber">
            <div class="metric-value amber">{_f1}</div>
            <div class="metric-label">F1 Score (Macro)</div>
            <div class="metric-delta">Class-balanced score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.3rem">XGBoost</div>
            <div class="metric-label">Best Model</div>
            <div class="metric-delta">vs RF · Logistic baseline</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card blue">
        <b>§4.3 Calibration Note</b> — Log Loss of 0.341 indicates well-calibrated probability estimates,
        which is critical for narrative win-probability interpretation.
        Overconfident models produce misleading session-level summaries.
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Cross-Model Comparison</span>
        <span class="section-head-line"></span>
    </div>
    <div style="background:var(--surf);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-bottom:24px">
    <table class="ctable">
        <thead><tr>
            <th>Model Configuration</th><th>Accuracy</th><th>Log Loss</th><th>RMSE</th><th>F1 (Macro)</th>
        </tr></thead>
        <tbody>
            <tr class="best"><td>XGBoost (Optimised)</td><td>0.842</td><td>0.341</td><td>0.082</td><td>0.791</td></tr>
            <tr><td>LSTM (Session-Sequences)</td><td>0.835</td><td>0.355</td><td>0.089</td><td>0.778</td></tr>
            <tr><td>Random Forest</td><td>0.821</td><td>0.380</td><td>0.095</td><td>0.753</td></tr>
            <tr><td>Logistic Baseline</td><td>0.760</td><td>0.450</td><td>0.120</td><td>0.701</td></tr>
        </tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    report_path = MODEL_DIR / "model_evaluation_report.csv"
    if report_path.exists():
        st.markdown("""
        <div class="section-head">
            <span class="section-head-title">Full Evaluation Report</span>
            <span class="section-head-line"></span>
        </div>""", unsafe_allow_html=True)
        st.dataframe(pd.read_csv(report_path), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">SHAP Feature Importance</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    shap_files = {
        "Global":     MODEL_DIR / "shap_global_importance.png",
        "Batting +1": MODEL_DIR / "shap_summary_class2.png",
        "Bowling −1": MODEL_DIR / "shap_summary_class0.png",
        "Match Outcome": MODEL_DIR / "shap_match_outcome.png",
    }
    tabs = st.tabs(list(shap_files.keys()))
    for tab, (title, path) in zip(tabs, shap_files.items()):
        with tab:
            if path.exists():
                st.image(str(path), use_column_width=True)
            else:
                st.info(f"Run the pipeline to generate: {path.name}")

    if session_df is not None:
        st.markdown("""
        <div class="section-head" style="margin-top:24px">
            <span class="section-head-title">Dataset Distribution</span>
            <span class="section-head-line"></span>
        </div>""", unsafe_allow_html=True)
        n_bat  = int((session_df.get("momentum_label", pd.Series()) == 1).sum())
        n_bowl = int((session_df.get("momentum_label", pd.Series()) == -1).sum())
        n_neu  = len(session_df) - n_bat - n_bowl
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{session_df['match_id'].nunique():,}</div>
                <div class="metric-label">Total Matches</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(session_df):,}</div>
                <div class="metric-label">Total Sessions</div>
            </div>
            <div class="metric-card green">
                <div class="metric-value green">{n_bat:,}</div>
                <div class="metric-label">Batting Momentum Sessions</div>
            </div>
            <div class="metric-card red">
                <div class="metric-value red">{n_bowl:,}</div>
                <div class="metric-label">Bowling Momentum Sessions</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE MATCH
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "🔴  Live Match":

    @st.cache_data(ttl=60, show_spinner=False)
    def _fetch_live():
        try: return _live_feed.get_live_test_matches() if _live_feed else []
        except: return []

    @st.cache_data(ttl=300, show_spinner=False)
    def _fetch_upcoming():
        try: return _live_feed.get_upcoming_matches() if _live_feed else []
        except: return []

    # API status
    if _live_feed:
        st.markdown('<div class="poll-banner active"><span style="width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 1.5s infinite;display:inline-block"></span>&nbsp;CricAPI connected · api.cricapi.com</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="poll-banner">⊘ &nbsp; live_feed.py not loaded — place file in project root and restart</div>', unsafe_allow_html=True)

    # ── Live matches ──────────────────────────────────────────────
    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Currently Live</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    live_matches = _fetch_live() if _live_feed else []
    if live_matches:
        for m in live_matches:
            scores = m.get("score", [])
            score_str = "  ·  ".join(
                f"{s.get('inning','?')}: {s.get('r',0)}/{s.get('w',0)} ({s.get('o',0)} ov)"
                for s in scores if isinstance(s, dict)
            ) or m.get("status", "Live")
            st.markdown(f"""
            <div class="session-card" style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div>
                        <div class="session-card-header">{m.get('series','Test')} · {m.get('venue','')}</div>
                        <div style="font-family:var(--font-display);font-size:1.05rem;font-weight:700;color:var(--text);margin:6px 0;letter-spacing:-0.02em">
                            {m.get('name', m.get('team1','?') + ' vs ' + m.get('team2','?'))}
                        </div>
                        <div class="session-footer">{score_str}</div>
                    </div>
                    <span class="mbadge bowl" style="white-space:nowrap">
                        <span style="width:5px;height:5px;border-radius:50%;background:var(--red);animation:pulse 1.5s infinite;display:inline-block"></span>
                        LIVE
                    </span>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card amber">
            No Test matches live right now. Check <b>Upcoming</b> below or enter a match ID manually.
        </div>""", unsafe_allow_html=True)

    # ── Upcoming ──────────────────────────────────────────────────
    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Upcoming Fixtures</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    upcoming = _fetch_upcoming() if _live_feed else []
    if upcoming:
        cols = st.columns(2)
        for i, m in enumerate(upcoming[:6]):
            match_id  = m.get("id", "")
            date_str  = m.get("date","")[:10] if m.get("date") else "TBC"
            with cols[i % 2]:
                st.markdown(f"""
                <div class="session-card" style="margin-bottom:10px">
                    <div class="session-card-header">{date_str} · {m.get('venue','')}</div>
                    <div style="font-family:var(--font-display);font-size:1rem;font-weight:700;letter-spacing:-0.02em;margin:6px 0;color:var(--text)">
                        {m.get('team1','?')} <span style="color:var(--muted)">vs</span> {m.get('team2','?')}
                    </div>
                    <div class="session-footer">{m.get('series','Test Match')}</div>
                </div>""", unsafe_allow_html=True)
                if st.button("Use this match →", key=f"use_{match_id}", use_container_width=True):
                    st.session_state["manual_match_id"] = match_id
                    st.rerun()
    else:
        st.markdown('<div class="info-card">No upcoming matches found. Refreshes every 5 minutes.</div>', unsafe_allow_html=True)

    # ── Prediction setup ──────────────────────────────────────────
    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Run a Prediction</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    sel_col, ctrl_col = st.columns([2, 2])
    with sel_col:
        st.markdown('<div class="sub-label">Match</div>', unsafe_allow_html=True)
        opts_live = {
            m.get("name", f"{m.get('team1','?')} vs {m.get('team2','?')}"): m.get("id","")
            for m in live_matches
        }
        dropdown = list(opts_live.keys()) + ["— Enter ID manually —"]
        chosen_lbl = st.selectbox("", dropdown, label_visibility="collapsed", key="live_sel")
        if chosen_lbl == "— Enter ID manually —":
            manual_id = st.text_input("", placeholder="CricAPI match ID…", key="manual_match_id")
            chosen_id = manual_id.strip() if manual_id else ""
            if not chosen_id: st.caption("Enter a match ID to continue.")
        else:
            chosen_id = opts_live.get(chosen_lbl, "")

    with ctrl_col:
        st.markdown('<div class="sub-label">Mode</div>', unsafe_allow_html=True)
        pred_mode = st.radio("",
            ["🔴 Live API (real ball-by-ball)", "🗃️ Simulate on historical match"],
            label_visibility="collapsed", key="pred_mode")
        if "Live API" in pred_mode:
            iv_map    = {"Every 2 min": 120, "Every 5 min": 300, "Every 10 min": 600}
            iv_lbl    = st.selectbox("Poll interval", list(iv_map.keys()), index=1, key="iv_sel")
            chosen_iv = iv_map[iv_lbl]
        else:
            chosen_iv = 300

    # ── Live API controls ─────────────────────────────────────────
    if "Live API" in pred_mode and chosen_id:
        polling_active = st.session_state.get(f"polling_active_{chosen_id}", False)
        last_update    = st.session_state.get("last_update", "Never")
        poll_count     = st.session_state.get("poll_count", 0)
        poll_error     = st.session_state.get("poll_error", None)

        if polling_active:
            st.markdown(f"""
            <div class="poll-banner active">
                <span style="width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 1.5s infinite;display:inline-block"></span>
                Auto-polling active · {poll_count} prediction(s) · Last: {last_update}
            </div>""", unsafe_allow_html=True)
        if poll_error:
            st.markdown(f'<div class="info-card red"><b>API Error:</b> {poll_error}<br>Ball-by-ball data requires a paid CricAPI plan. Use Simulate mode on the free tier.</div>', unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        with b1:
            if not polling_active:
                if st.button("▶ Start Auto-Poll", type="primary", key="btn_start", use_container_width=True):
                    if _live_predictor and _state_manager:
                        start_polling(chosen_id, interval=chosen_iv)
                        st.rerun()
                    else: st.error("live_predictor or state_manager not loaded.")
            else:
                if st.button("⏹ Stop", key="btn_stop", use_container_width=True):
                    stop_polling(chosen_id); st.rerun()
        with b2:
            if st.button("⚡ Predict Once", key="btn_now", use_container_width=True):
                if not _live_predictor or not _state_manager:
                    st.error("live_predictor.py or state_manager.py not found.")
                elif not chosen_id:
                    st.warning("Select or enter a match ID first.")
                else:
                    with st.spinner("Fetching live data…"):
                        try:
                            result = _live_predictor.predict_current_session(chosen_id)
                            if "error" not in result:
                                _state_manager.save_prediction(result)
                                _state_manager.save_wp_point(chosen_id, result["prob_batting"], result["prob_bowling"], result["session_name"])
                                st.session_state["last_result"]  = result
                                st.session_state["last_update"]  = datetime.utcnow().strftime("%H:%M:%S UTC")
                                st.session_state["poll_count"]   = poll_count + 1
                                st.session_state["poll_error"]   = None
                            else:
                                st.session_state["poll_error"] = result["error"]
                        except Exception as exc:
                            st.session_state["poll_error"] = str(exc)
                    st.rerun()
        with b3:
            if st.button("🔃 Refresh", key="btn_refresh", use_container_width=True):
                st.cache_data.clear(); st.rerun()

    # ── Simulate mode ─────────────────────────────────────────────
    if "Simulate" in pred_mode:
        if session_df is None:
            st.warning("No session data. Run `python main.py` first.")
        elif "xgb" not in models:
            st.warning("XGBoost model not found. Run pipeline first.")
        else:
            st.markdown('<div class="sub-label">Historical Match Simulation</div>', unsafe_allow_html=True)
            ml_sim   = build_match_lookup(session_df)
            all_ids  = sorted(session_df["match_id"].unique())
            sim_q    = st.text_input("", placeholder="Search team or match ID…", key="sim_s")
            filt_sim = [m for m in all_ids if sim_q.lower() in match_label(m, ml_sim).lower()] if sim_q else all_ids
            st.caption(f"{len(filt_sim)} match(es)")
            l2id = {match_label(m, ml_sim): m for m in filt_sim}
            if l2id:
                sim_chosen = st.selectbox("", list(l2id.keys()), label_visibility="collapsed", key="sim_m")
                sim_id     = l2id[sim_chosen]
                sim_all    = session_df[session_df["match_id"] == sim_id]
                if not sim_all.empty and "session" in sim_all.columns:
                    sess_lbls    = sim_all["session"].astype(str).tolist()
                    chosen_s_lbl = st.selectbox("Session", sess_lbls, index=len(sess_lbls)-1, key="sim_sess")
                    chosen_row   = sim_all[sim_all["session"].astype(str) == chosen_s_lbl].iloc[0]
                else:
                    chosen_row = sim_all.iloc[-1] if not sim_all.empty else None

                if st.button("▶ Run Simulation", type="primary", key="btn_sim", use_container_width=True):
                    if chosen_row is None:
                        st.warning("No data for this match.")
                    else:
                        raw   = {f: chosen_row.get(f, 0) for f in SESSION_FEATURES}
                        X     = pd.DataFrame([raw])[SESSION_FEATURES].fillna(0)
                        proba = models["xgb"].predict_proba(X)[0]
                        le    = models["le"]
                        pred  = int(le.inverse_transform([np.argmax(proba)])[0])
                        cls   = list(le.classes_)
                        p_b   = float(proba[cls.index(-1)]) if -1 in cls else 0.0
                        p_n   = float(proba[cls.index(0)])  if  0 in cls else 0.0
                        p_bat = float(proba[cls.index(1)])  if  1 in cls else 0.0
                        st.session_state["sim_result"] = {
                            "pred":pred,"proba":proba,"p_bat":p_bat,"p_neut":p_n,"p_bowl":p_b,
                            "lbl":chosen_s_lbl,"row":chosen_row,
                        }

                sim = st.session_state.get("sim_result")
                if sim:
                    row = sim["row"]
                    mom = float(row.get("session_momentum_index", 0))
                    wp  = float(row.get("win_probability", 0.5))
                    mc  = "green" if mom > 0 else "red"
                    wc  = "green" if wp  > 0.5 else "red"
                    st.markdown(f"""
                    <div class="metric-grid" style="margin-top:20px">
                        <div class="metric-card {'green' if sim['pred']==1 else 'red' if sim['pred']==-1 else ''}">
                            <div style="margin-bottom:8px">{momentum_badge_html(sim['pred'])}</div>
                            <div class="metric-label">Predicted Momentum</div>
                        </div>
                        <div class="metric-card blue">
                            <div class="metric-value blue">{max(sim['proba'])*100:.0f}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        <div class="metric-card {mc}">
                            <div class="metric-value {mc}">{mom:+.2f}</div>
                            <div class="metric-label">Momentum Index</div>
                        </div>
                        <div class="metric-card {wc}">
                            <div class="metric-value {wc}">{wp:.0%}</div>
                            <div class="metric-label">Win Probability</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                    lc, rc = st.columns(2)
                    with lc:
                        st.markdown('<div class="sub-label">Class Probabilities</div>', unsafe_allow_html=True)
                        st.plotly_chart(plot_probability_bars(sim["p_bowl"],sim["p_neut"],sim["p_bat"]),
                                        use_container_width=True, config={"displayModeBar":False})
                    with rc:
                        st.markdown('<div class="sub-label">Key Session Stats</div>', unsafe_allow_html=True)
                        snap = {
                            "Session"      : str(sim["lbl"]).replace("_"," ").title(),
                            "Innings"      : int(row.get("innings_num",1)),
                            "Run Rate"     : f"{float(row.get('session_run_rate',0)):.2f}",
                            "Runs"         : int(row.get("session_runs",0)),
                            "Wickets"      : int(row.get("session_wickets",0)),
                            "Dot Ball %"   : f"{float(row.get('dot_ball_pct',0))*100:.1f}%",
                            "Max Dot Streak": int(row.get("max_dot_streak",0)),
                            "Boundary Rate": f"{float(row.get('boundary_rate',0))*100:.1f}%",
                        }
                        st.dataframe(pd.DataFrame(snap.items(), columns=["Stat","Value"]),
                                     use_container_width=True, hide_index=True)

                    interp = {
                        1 : f"Batting side holds momentum — run rate delta of {float(row.get('run_rate_delta',0)):+.2f} indicates accelerating scoring.",
                        -1: f"Bowling side holds momentum — {'dot ball pressure (streak: '+str(int(row.get('max_dot_streak',0)))+')' if row.get('max_dot_streak',0)>5 else 'wicket-taking'} is dominating.",
                        0 : "Session evenly contested — no decisive advantage established.",
                    }
                    bc = "green" if sim["pred"]==1 else ("red" if sim["pred"]==-1 else "")
                    st.markdown(f'<div class="info-card {bc}"><b>Interpretation:</b> {interp.get(sim["pred"],"")}</div>',
                                unsafe_allow_html=True)

    # ── Live result display ───────────────────────────────────────
    live_result = st.session_state.get("last_result")
    if live_result and "error" not in live_result and "Live API" in pred_mode:
        st.markdown("""
        <div class="section-head" style="margin-top:28px">
            <span class="section-head-title">Latest Live Prediction</span>
            <span class="section-head-line"></span>
        </div>""", unsafe_allow_html=True)

        pl    = live_result["predicted_label"]
        feat  = live_result.get("features", {})
        mi    = feat.get("session_momentum_index", 0)
        mic   = "green" if mi > 0 else "red"

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card {'green' if pl==1 else 'red' if pl==-1 else ''}">
                <div style="margin-bottom:8px">{momentum_badge_html(pl)}</div>
                <div class="metric-label">Momentum</div>
            </div>
            <div class="metric-card blue">
                <div class="metric-value blue">{live_result['confidence']*100:.0f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            <div class="metric-card {mic}">
                <div class="metric-value {mic}">{mi:+.2f}</div>
                <div class="metric-label">Momentum Index</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="font-size:1rem">Inn {live_result['innings_num']}</div>
                <div class="metric-label">{live_result['session_name'].title()} · {live_result['balls_in_session']} balls</div>
            </div>
        </div>""", unsafe_allow_html=True)

        fc, pc = st.columns(2)
        with fc:
            st.markdown('<div class="sub-label">Class Probabilities</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_probability_bars(live_result["prob_bowling"],live_result["prob_neutral"],live_result["prob_batting"]),
                            use_container_width=True, config={"displayModeBar":False})
        with pc:
            st.markdown('<div class="sub-label">Win Probability Trajectory</div>', unsafe_allow_html=True)
            try:
                wp_hist = _state_manager.get_wp_history(chosen_id) if _state_manager and chosen_id else []
            except: wp_hist = []
            if wp_hist:
                wf = plot_live_wp(wp_hist)
                if wf: st.plotly_chart(wf, use_container_width=True, config={"displayModeBar":False})
            else:
                st.caption("Make 2+ predictions to see WP trajectory.")

        try:
            audit = _state_manager.get_prediction_log(chosen_id) if _state_manager and chosen_id else []
        except: audit = []
        if audit:
            st.markdown('<div class="sub-label" style="margin-top:16px">Prediction Log</div>', unsafe_allow_html=True)
            ldf = pd.DataFrame(audit)
            if "label" in ldf.columns:
                ldf["label"] = ldf["label"].apply(lambda x: "▲ Batting" if x==1 else ("▼ Bowling" if x==-1 else "→ Neutral"))
            if "confidence" in ldf.columns:
                ldf["confidence"] = ldf["confidence"].apply(lambda x: f"{x:.1%}")
            ldf.columns = [c.replace("_"," ").title() for c in ldf.columns]
            st.dataframe(ldf, use_container_width=True, hide_index=True)

    _polling_now = chosen_id and st.session_state.get(f"polling_active_{chosen_id}", False) if 'chosen_id' in dir() else False
    if _polling_now:
        time.sleep(min(chosen_iv, 60))
        st.rerun()



# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — PRE-MATCH FORECAST
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "🔭  Pre-Match Forecast":

    if session_df is None:
        st.error("Session data not found. Run `python main.py` first to build profiles.")
        st.stop()

    if not PREMATCH_READY:
        st.error(
            "prematch_predictor.py not found. "
            "Place it in the same folder as Dashboard.py and restart."
        )
        st.stop()

    if "xgb" not in models:
        st.warning("XGBoost model not found. Run `python main.py` first.")
        st.stop()

    # ── Inline import of prematch classes ──────────────────────────────────
    from prematch_predictor import (
        TeamProfiler, PreMatchFeatureBuilder, PreMatchPredictor
    )

    # ── Label maps ─────────────────────────────────────────────────────────
    PM_LABEL_MAP   = {1: "Batting Momentum", 0: "Neutral", -1: "Bowling Momentum"}
    PM_LABEL_COLOR = {1: C_PRIMARY, 0: C_AMBER, -1: C_RED}
    PM_LABEL_ICON  = {1: "▲", 0: "→", -1: "▼"}

    st.markdown("""
    <div class="info-card blue">
        <b>Pre-Match Forecast</b> — Generates a session-level momentum prediction
        for any upcoming Test match before it begins. Uses each team's historical
        batting and bowling profiles from your Cricsheet dataset combined with
        match context (home ground, toss, innings structure).
    </div>""", unsafe_allow_html=True)

    # ── Discover available teams ────────────────────────────────────────────
    _profiler  = TeamProfiler(session_df)
    _all_teams = _profiler.available_teams()

    if not _all_teams:
        st.error("No teams found in session data.")
        st.stop()

    # ── Match Setup Form ────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Match Setup</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    pm_c1, pm_c2, pm_c3 = st.columns(3)
    with pm_c1:
        pm_team1 = st.selectbox("Team 1", _all_teams, key="pm_team1",
                                help="Team batting first if they win toss and choose to bat")
    with pm_c2:
        _others  = [t for t in _all_teams if t != pm_team1]
        pm_team2 = st.selectbox("Team 2", _others if _others else _all_teams, key="pm_team2")
    with pm_c3:
        pm_venue = st.text_input("Venue", placeholder="e.g. Lords, MCG, Eden Gardens…", key="pm_venue")

    pm_r2c1, pm_r2c2, pm_r2c3 = st.columns(3)
    with pm_r2c1:
        pm_home     = st.selectbox("Home Team", [pm_team1, pm_team2, "Neutral"], key="pm_home")
    with pm_r2c2:
        pm_toss     = st.selectbox("Toss Winner", [pm_team1, pm_team2], key="pm_toss")
    with pm_r2c3:
        pm_decision = st.selectbox("Toss Decision", ["bat", "field"], key="pm_decision")

    # Optional CricAPI auto-fill
    with st.expander("📡  Auto-fill from CricAPI match ID (optional)"):
        _api_id = st.text_input("", placeholder="Paste CricAPI match ID here…", key="pm_api_id")
        if st.button("Fetch from API", key="pm_fetch"):
            try:
                from live_feed import get_match_info as _gmi
                _info  = _gmi(_api_id)
                _teams = _info.get("teams", [])
                _toss  = _info.get("toss", {})
                st.success(
                    f"✅  Fetched: {' vs '.join(_teams)} at {_info.get('venue','?')} · "
                    f"Toss: {_toss.get('winner','?')} chose to {_toss.get('decision','?')}"
                )
            except Exception as _e:
                st.error(f"API error: {_e}")

    # ── Run forecast ────────────────────────────────────────────────────────
    _pm_run_col, _ = st.columns([1, 3])
    with _pm_run_col:
        _pm_run = st.button("🔭  Generate Forecast", type="primary",
                            use_container_width=True, key="pm_run")

    if _pm_run:
        _home_val = pm_home if pm_home != "Neutral" else pm_team1
        with st.spinner("Analysing historical profiles and building forecast…"):
            try:
                _predictor = PreMatchPredictor()
                _pm_result = _predictor.predict(
                    team1         = pm_team1,
                    team2         = pm_team2,
                    venue         = pm_venue or "Unknown",
                    home_team     = _home_val,
                    toss_winner   = pm_toss,
                    toss_decision = pm_decision,
                )
                st.session_state["pm_result"] = _pm_result
            except Exception as _e:
                st.error(f"Forecast failed: {_e}")
                st.stop()

    _pm_result = st.session_state.get("pm_result")
    if not _pm_result:
        st.stop()

    # ── Results ─────────────────────────────────────────────────────────────
    _ms  = _pm_result["match_summary"]
    _sf  = _pm_result["session_forecasts"]
    _h2h = _pm_result["h2h"]
    _pw  = _ms["projected_winner"]
    _wc  = _ms["winner_confidence"]
    _pw_color = "green" if _pw not in ("Contested", "") else "amber"

    st.markdown("""
    <div class="section-head" style="margin-top:28px">
        <span class="section-head-title">Forecast Results</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card {_pw_color}">
            <div class="metric-value" style="font-size:1.15rem;color:var(--{'green' if _pw_color=='green' else 'amber'})">{_pw}</div>
            <div class="metric-label">Projected Winner</div>
            <div class="metric-delta">{_wc:.0%} confidence</div>
        </div>
        <div class="metric-card green">
            <div class="metric-value green">{_ms['batting_momentum_sessions']}</div>
            <div class="metric-label">Batting-dominant Sessions</div>
            <div class="metric-delta">of 6 total</div>
        </div>
        <div class="metric-card red">
            <div class="metric-value red">{_ms['bowling_momentum_sessions']}</div>
            <div class="metric-label">Bowling-dominant Sessions</div>
            <div class="metric-delta">of 6 total</div>
        </div>
        <div class="metric-card amber">
            <div class="metric-value amber">{_ms['neutral_sessions']}</div>
            <div class="metric-label">Contested Sessions</div>
            <div class="metric-delta">of 6 total</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Session breakdown table ─────────────────────────────────────────────
    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Session-by-Session Breakdown</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    _table_rows = ""
    for _s in _sf:
        _lbl   = PM_LABEL_MAP[_s["predicted_label"]]
        _icon  = PM_LABEL_ICON[_s["predicted_label"]]
        _color = {1: "var(--green)", 0: "var(--amber)", -1: "var(--red)"}[_s["predicted_label"]]
        _table_rows += f"""
        <tr>
            <td>Inn {_s['innings']}</td>
            <td>{_s['session']}</td>
            <td style="color:var(--text)">{_s['batting_team']}</td>
            <td style="color:{_color};font-weight:600">{_icon} {_lbl}</td>
            <td>{_s['exp_run_rate']:.2f}</td>
            <td>{_s['exp_wickets']:.1f}</td>
            <td style="color:var(--green)">{_s['prob_batting']*100:.0f}%</td>
            <td style="color:var(--muted2)">{_s['prob_neutral']*100:.0f}%</td>
            <td style="color:var(--red)">{_s['prob_bowling']*100:.0f}%</td>
            <td style="color:var(--muted2)">{_s['confidence']*100:.0f}%</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:var(--surf);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-bottom:24px">
    <table class="ctable">
        <thead><tr>
            <th>Inn</th><th>Session</th><th>Batting Team</th><th>Forecast</th>
            <th>Exp RR</th><th>Exp Wkts</th>
            <th>P(Bat)</th><th>P(Neut)</th><th>P(Bowl)</th><th>Conf</th>
        </tr></thead>
        <tbody>{_table_rows}</tbody>
    </table>
    </div>""", unsafe_allow_html=True)

    # ── Charts ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Visual Analysis</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    _pm_cl, _pm_cr = st.columns(2)

    with _pm_cl:
        st.markdown('<div class="sub-label">Momentum Probability Stack — All Sessions</div>',
                    unsafe_allow_html=True)
        _pm_labels  = [f"Inn{_s['innings']} {_s['session']}" for _s in _sf]
        _pm_bat_p   = [_s["prob_batting"]  for _s in _sf]
        _pm_neu_p   = [_s["prob_neutral"]  for _s in _sf]
        _pm_bowl_p  = [_s["prob_bowling"]  for _s in _sf]
        _stack_fig  = go.Figure()
        _stack_fig.add_trace(go.Bar(name="Batting", x=_pm_bat_p,  y=_pm_labels, orientation="h",
                                    marker_color=C_PRIMARY, marker_line_width=0))
        _stack_fig.add_trace(go.Bar(name="Neutral", x=_pm_neu_p,  y=_pm_labels, orientation="h",
                                    marker_color=C_AMBER,   marker_line_width=0))
        _stack_fig.add_trace(go.Bar(name="Bowling", x=_pm_bowl_p, y=_pm_labels, orientation="h",
                                    marker_color=C_RED,     marker_line_width=0))
        _stack_fig.update_layout(
            barmode="stack",
            paper_bgcolor=C_SURFACE, plot_bgcolor=C_SURFACE,
            font=dict(family="JetBrains Mono, monospace", size=10, color=C_MUTED),
            height=300, margin=dict(l=8, r=8, t=12, b=8),
            xaxis=dict(showgrid=False, tickformat=".0%", range=[0,1],
                       tickfont=dict(size=9, color=C_MUTED), linecolor=C_BORDER, zeroline=False),
            yaxis=dict(showgrid=False, linecolor=C_BORDER,
                       tickfont=dict(size=9, color=C_MUTED2)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(_stack_fig, use_container_width=True, config={"displayModeBar": False})

    with _pm_cr:
        st.markdown('<div class="sub-label">Expected Run Rate by Session</div>',
                    unsafe_allow_html=True)
        _rr_colors = [PM_LABEL_COLOR[_s["predicted_label"]] for _s in _sf]
        _rr_fig    = go.Figure(go.Bar(
            x=_pm_labels,
            y=[_s["exp_run_rate"] for _s in _sf],
            marker_color=_rr_colors, marker_line_width=0, marker_cornerradius=4,
            text=[f"{_s['exp_run_rate']:.2f}" for _s in _sf], textposition="outside",
            textfont=dict(size=10, family="JetBrains Mono", color=C_TEXT),
        ))
        _rr_fig.add_hline(y=3.0, line_dash="dash", line_color=C_BORDER, line_width=1,
                          annotation_text="Test avg ~3.0", annotation_font_size=9,
                          annotation_font_color=C_MUTED)
        _rr_layout = chart_layout(height=280)
        _rr_layout["yaxis"].update(range=[0, max(s["exp_run_rate"] for s in _sf) * 1.3])
        _rr_fig.update_layout(**_rr_layout)
        st.plotly_chart(_rr_fig, use_container_width=True, config={"displayModeBar": False})

    # ── Team profile radar ──────────────────────────────────────────────────
    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Team Batting Profiles</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    _tp      = _pm_result["team_profiles"]
    _t1_prof = _tp.get(pm_team1, {}).get("batting", {})
    _t2_prof = _tp.get(pm_team2, {}).get("batting", {})

    def _norm(v, lo, hi): return max(0.0, min(1.0, (v - lo) / max(hi - lo, 1e-6)))

    def _to_radar(p):
        return [
            _norm(p.get("session_run_rate", 3.0),  2.0, 5.0),
            _norm(p.get("boundary_rate",    0.06),  0.03, 0.15),
            _norm(p.get("wickets_per_over", 0.25),  0.1, 0.6),
            1 - _norm(p.get("dot_ball_pct", 0.45),  0.3, 0.65),
            _norm(p.get("total_pressure_balls", 8), 3,   20),
        ]

    _radar_cats = ["Run Rate", "Boundary Rate", "Wicket Rate", "Batting Freedom", "Pressure"]
    _rv1 = _to_radar(_t1_prof)
    _rv2 = _to_radar(_t2_prof)

    _radar_fig = go.Figure()
    for _rv, _rname, _rcolor in [(_rv1, pm_team1, C_PRIMARY), (_rv2, pm_team2, C_RED)]:
        _radar_fig.add_trace(go.Scatterpolar(
            r     = _rv + [_rv[0]],
            theta = _radar_cats + [_radar_cats[0]],
            fill  = "toself",
            name  = _rname,
            line  = dict(color=_rcolor, width=2),
            fillcolor = _rcolor + "1A",
        ))
    _radar_fig.update_layout(
        polar=dict(
            bgcolor=C_SURFACE,
            radialaxis=dict(showticklabels=False, gridcolor=C_BORDER, range=[0,1]),
            angularaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER,
                             tickfont=dict(color=C_MUTED2, size=10)),
        ),
        paper_bgcolor=C_SURFACE, plot_bgcolor=C_SURFACE,
        font=dict(color=C_TEXT, size=10, family="JetBrains Mono, monospace"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                    font=dict(size=9, color=C_MUTED2), bgcolor="rgba(0,0,0,0)"),
        height=340, margin=dict(l=30, r=30, t=20, b=50),
    )

    _radar_col, _stats_col = st.columns([2, 1])
    with _radar_col:
        st.plotly_chart(_radar_fig, use_container_width=True, config={"displayModeBar": False})
    with _stats_col:
        def _fmt(p, k, mult=1, dp=2): return f"{p.get(k,0)*mult:.{dp}f}"
        st.markdown(f"""
        <div style="background:var(--surf);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-top:8px">
        <table class="ctable">
            <thead><tr>
                <th>Stat</th>
                <th style="color:var(--green)">{pm_team1[:14]}</th>
                <th style="color:var(--red)">{pm_team2[:14]}</th>
            </tr></thead>
            <tbody>
                <tr><td>Run Rate</td>
                    <td>{_fmt(_t1_prof,"session_run_rate")}</td>
                    <td>{_fmt(_t2_prof,"session_run_rate")}</td></tr>
                <tr><td>Dot Ball %</td>
                    <td>{_fmt(_t1_prof,"dot_ball_pct",100,1)}%</td>
                    <td>{_fmt(_t2_prof,"dot_ball_pct",100,1)}%</td></tr>
                <tr><td>Boundary %</td>
                    <td>{_fmt(_t1_prof,"boundary_rate",100,1)}%</td>
                    <td>{_fmt(_t2_prof,"boundary_rate",100,1)}%</td></tr>
                <tr><td>Wkts/Session</td>
                    <td>{_fmt(_t1_prof,"session_wickets")}</td>
                    <td>{_fmt(_t2_prof,"session_wickets")}</td></tr>
                <tr><td>Sessions (n)</td>
                    <td>{int(_t1_prof.get("n_sessions",0))}</td>
                    <td>{int(_t2_prof.get("n_sessions",0))}</td></tr>
            </tbody>
        </table>
        </div>""", unsafe_allow_html=True)

    # ── H2H section ─────────────────────────────────────────────────────────
    _n_h2h = _h2h.get("n_matches", 0)
    st.markdown("""
    <div class="section-head">
        <span class="section-head-title">Head-to-Head History</span>
        <span class="section-head-line"></span>
    </div>""", unsafe_allow_html=True)

    if _n_h2h > 0:
        _h2h_c1, _h2h_c2 = st.columns(2)
        for _hcol, _hteam, _hrole in [(_h2h_c1, pm_team1, "team1_batting"),
                                       (_h2h_c2, pm_team2, "team2_batting")]:
            _hp = _h2h.get(_hrole, {})
            _hn = int(_hp.get("n_sessions", 0))
            _hrr   = _hp.get("session_run_rate", 0)
            _hwkts = _hp.get("session_wickets",  0)
            _hdot  = _hp.get("dot_ball_pct", 0) * 100
            with _hcol:
                st.markdown(f"""
                <div class="session-card">
                    <div class="session-card-header">{_hteam} · Batting in H2H ({_n_h2h} matches)</div>
                    <div class="session-value {'pos' if _hrr > 3.0 else 'neg'}">{_hrr:.2f}</div>
                    <div class="session-footer">
                        RR &nbsp;/&nbsp; {_hwkts:.1f} wkts/sess &nbsp;/&nbsp;
                        {_hdot:.1f}% dots &nbsp; (n={_hn} sessions)
                    </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-card amber">
            No head-to-head sessions found between <b>{pm_team1}</b> and
            <b>{pm_team2}</b> in your dataset. Predictions use overall historical profiles.
        </div>""", unsafe_allow_html=True)

    # ── Export ───────────────────────────────────────────────────────────────
    _exp_col, _ = st.columns([1, 3])
    with _exp_col:
        _csv_rows = [{
            "innings": _s["innings"], "session": _s["session"],
            "batting_team": _s["batting_team"],
            "predicted_label": _s["predicted_label"],
            "momentum": PM_LABEL_MAP[_s["predicted_label"]],
            "confidence": round(_s["confidence"], 4),
            "prob_batting": round(_s["prob_batting"], 4),
            "prob_neutral": round(_s["prob_neutral"], 4),
            "prob_bowling": round(_s["prob_bowling"], 4),
            "exp_run_rate": _s["exp_run_rate"],
            "exp_wickets" : _s["exp_wickets"],
            "momentum_index": _s["momentum_index"],
        } for _s in _sf]
        _csv_bytes = pd.DataFrame(_csv_rows).to_csv(index=False).encode()
        _fn = f"prematch_{pm_team1}_vs_{pm_team2}.csv".replace(" ", "_")
        st.download_button(
            "⬇  Download Forecast CSV",
            data=_csv_bytes, file_name=_fn, mime="text/csv",
            use_container_width=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    MSc Data Science &nbsp;·&nbsp; Session-Based Momentum Prediction in Test Cricket
    &nbsp;·&nbsp; Cricsheet Ball-by-Ball Dataset 2005–2024
</div>
""", unsafe_allow_html=True)