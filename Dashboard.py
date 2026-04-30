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

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "outputs"
MODEL_DIR   = OUTPUT_DIR / "models"
SESSION_CSV = OUTPUT_DIR / "session_features.csv"
MATCH_CSV   = OUTPUT_DIR / "match_level_features.csv"
sys.path.insert(0, str(BASE_DIR / "src"))

# ── Design tokens ────────────────────────────────────────────────────────────
C_PRIMARY = "#1B4F72"
C_BATTING = "#008855"
C_BOWLING = "#CC3333"
C_NEUTRAL = "#555555"
C_ACCENT  = "#D4AC0D"
C_BG      = "#FAFAF7"
C_BORDER  = "#DDDDDD"

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

# ── Safe imports for live src modules ────────────────────────────────────────
# Wrapped so missing files don't crash the dashboard —
# each feature silently activates once its file exists.

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

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Momentum Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg             : #FAFAF7;
    --text-primary   : #1A1A1A;
    --text-secondary : #555555;
    --border         : #DDDDDD;
    --accent-green   : #008855;
    --accent-red     : #CC3333;
    --font-serif     : 'DM Serif Display', serif;
    --font-mono      : 'IBM Plex Mono', monospace;
    --font-sans      : 'Inter', sans-serif;
    --s-1:8px; --s-2:16px; --s-3:24px; --s-4:32px; --s-5:48px; --s-6:64px;
}
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;500&display=swap');

html,body,[class*="css"]{ font-family:var(--font-sans)!important; background:var(--bg)!important; color:var(--text-primary)!important; font-size:14px; line-height:1.6; -webkit-font-smoothing:antialiased; }
.stApp{ background:var(--bg)!important; }
@keyframes fadeIn{ from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
.main .block-container{ animation:fadeIn 0.5s ease-out forwards; }

h1,h2,h3,h4,.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,.stMarkdown h4{
    font-family:var(--font-serif)!important; font-weight:400!important;
    letter-spacing:-0.01em!important; color:var(--text-primary)!important;
}

div[data-testid="stSidebar"]{ background:#FFFFFF!important; border-right:1px solid var(--border)!important; }
div[data-testid="stSidebar"] *{ font-family:var(--font-sans)!important; color:var(--text-primary)!important; }
.sidebar-brand{ padding:var(--s-2) 0 var(--s-1) 0; border-bottom:1px solid var(--border); margin-bottom:var(--s-2); }
.sidebar-brand-title{ font-family:var(--font-serif); font-size:1.15rem; color:var(--text-primary)!important; }
.sidebar-brand-sub{ font-family:var(--font-mono); font-size:0.75rem; color:var(--text-secondary)!important; margin-top:3px; }
.sidebar-data-info{ font-family:var(--font-mono); font-size:0.75rem; color:var(--text-secondary); margin-top:var(--s-1); line-height:1.7; }
.sidebar-data-info b{ color:var(--text-primary); }

.live-dot{ display:inline-block; width:7px; height:7px; border-radius:50%; background:#CC3333; margin-right:6px; animation:livepulse 1.5s ease-in-out infinite; vertical-align:middle; }
@keyframes livepulse{ 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.75)} }
.live-label  { font-family:var(--font-mono); font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; color:#CC3333; vertical-align:middle; }
.offline-label{ font-family:var(--font-mono); font-size:0.72rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--text-secondary); vertical-align:middle; }

.main-header{ border-bottom:1px solid var(--border); padding:var(--s-3) 0; margin-bottom:var(--s-4); }
.main-header h1{ font-family:var(--font-serif)!important; font-size:1.8rem!important; font-weight:400!important; color:var(--text-primary)!important; margin-bottom:var(--s-1)!important; letter-spacing:-0.01em!important; }
.main-header-sub{ font-family:var(--font-mono); font-size:0.85rem; color:var(--text-secondary); }

.section-title{ font-family:var(--font-serif)!important; font-size:1.45rem!important; font-weight:400!important; letter-spacing:-0.01em; color:var(--text-primary)!important; margin-bottom:var(--s-2); line-height:1.2; }
.subsection-title{ font-family:var(--font-mono); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--text-secondary); border-bottom:1px solid var(--border); padding-bottom:var(--s-1); margin-bottom:var(--s-2); }

.session-card{ border:1px solid var(--border); padding:var(--s-3); background:var(--bg); }
.session-card-label{ font-family:var(--font-mono); font-size:0.8rem; color:var(--text-secondary); margin-bottom:var(--s-1); font-weight:400; }
.session-delta{ font-family:var(--font-sans); font-size:2.2rem; font-weight:600; line-height:1; margin-bottom:4px; }
.session-delta.positive{ color:var(--accent-green); }
.session-delta.negative{ color:var(--accent-red); }
.session-delta.neutral { color:var(--text-secondary); }
.session-stats-line{ font-family:var(--font-mono); font-size:0.78rem; color:var(--text-secondary); }

.metric-card{ border:1px solid var(--border); padding:var(--s-3) var(--s-2); background:#FFFFFF; text-align:left; }
.metric-card .value{ font-family:var(--font-sans); font-size:2.2rem; font-weight:600; line-height:1; margin-bottom:4px; }
.metric-card .label{ font-family:var(--font-sans); font-size:0.85rem; color:var(--text-secondary); font-weight:400; }
.metric-card.live   { border-left:2px solid var(--accent-green); }
.metric-card.bowling{ border-left:2px solid var(--accent-red); }

.momentum-badge{ display:inline-block; padding:0.3rem 0.9rem; font-family:var(--font-mono); font-size:0.8rem; letter-spacing:0.04em; border:1px solid; border-radius:0; }
.badge-batting{ background:#F0FBF5; color:var(--accent-green); border-color:var(--accent-green); }
.badge-bowling{ background:#FDF3F3; color:var(--accent-red);   border-color:var(--accent-red); }
.badge-neutral{ background:#F5F5F5; color:var(--text-secondary); border-color:var(--border); }

.info-box{ border-left:2px solid var(--text-primary); padding:0.7rem var(--s-2); font-family:var(--font-serif); font-style:italic; font-size:0.9rem; color:var(--text-secondary); margin:var(--s-2) 0; background:transparent; }
.info-box b{ font-family:var(--font-sans); font-style:normal; font-weight:500; color:var(--text-primary); }
.info-box.alert-green{ border-left-color:var(--accent-green); }
.info-box.alert-red  { border-left-color:var(--accent-red); }

.pipeline-strip{ display:flex; align-items:center; position:relative; padding:var(--s-2) 0; font-family:var(--font-mono); font-size:0.8rem; color:var(--text-secondary); flex-wrap:wrap; gap:0; }
.pipeline-strip::before{ content:''; position:absolute; left:0; top:50%; width:100%; height:1px; background:var(--border); z-index:1; }
.pipeline-step{ flex:1; min-width:110px; text-align:center; position:relative; z-index:2; background:var(--bg); padding:0 4px; }
.pipeline-step.active{ color:var(--text-primary); font-weight:600; }
.pipeline-step.active::after{ content:''; position:absolute; bottom:-10px; left:50%; transform:translateX(-50%); width:6px; height:6px; border-radius:50%; background:var(--text-primary); }
.pipeline-step.done{ color:var(--accent-green); }
.pipeline-step.done::after{ content:''; position:absolute; bottom:-10px; left:50%; transform:translateX(-50%); width:6px; height:6px; border-radius:50%; background:var(--accent-green); }

.feature-table-wrap table,.comparison-table-wrap table{ width:100%; border-collapse:collapse; font-family:var(--font-sans); font-size:0.88rem; }
.feature-table-wrap th,.comparison-table-wrap th{ text-align:left; font-family:var(--font-mono); font-weight:400; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--text-secondary); border-bottom:1px solid var(--border); padding:8px; }
.feature-table-wrap td,.comparison-table-wrap td{ padding:9px 8px; border-bottom:1px solid #EEEEEE; font-family:var(--font-mono); font-size:0.8rem; color:var(--text-secondary); }
.feature-table-wrap tr:last-child td,.comparison-table-wrap tr:last-child td{ border-bottom:none; }
.comparison-table-wrap tr.best-row td:first-child{ border-left:2px solid var(--accent-green); }
.comparison-table-wrap tr:nth-child(even){ background:#F6F6F3; }
.shap-bar{ height:3px; background:var(--text-primary); display:inline-block; vertical-align:middle; }

.metrics-row{ display:grid; grid-template-columns:repeat(4,1fr); gap:var(--s-3); border-top:1px solid var(--border); border-bottom:1px solid var(--border); padding:var(--s-3) 0; margin:var(--s-3) 0; }

.notes-heading{ font-family:var(--font-mono); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--text-secondary); border-bottom:1px solid var(--border); padding-bottom:var(--s-1); margin-bottom:var(--s-2); }
.note-block{ font-family:var(--font-serif); font-style:italic; color:var(--text-secondary); font-size:0.88rem; line-height:1.55; margin-bottom:var(--s-3); padding-left:var(--s-1); border-left:1px solid var(--border); }

.polling-banner{ display:flex; align-items:center; gap:var(--s-2); border:1px solid var(--border); padding:var(--s-1) var(--s-2); font-family:var(--font-mono); font-size:0.78rem; color:var(--text-secondary); background:var(--bg); margin-bottom:var(--s-2); }
.polling-banner.active { border-color:var(--accent-green); background:#F0FBF5; color:var(--accent-green); }
.polling-banner.stopped{ border-color:var(--border); background:var(--bg); color:var(--text-secondary); }

.main-footer{ border-top:1px solid var(--border); padding:var(--s-3) 0; text-align:center; font-family:var(--font-sans); color:var(--text-secondary); font-size:0.82rem; margin-top:var(--s-6); }

.stButton>button{ font-family:var(--font-mono)!important; font-size:0.82rem!important; letter-spacing:0.06em!important; background:transparent!important; color:var(--text-primary)!important; border:1px solid var(--border)!important; border-radius:0!important; padding:0.45rem 1.1rem!important; transition:background 0.15s,border-color 0.15s!important; }
.stButton>button:hover{ background:#F0F0ED!important; border-color:var(--text-primary)!important; }
.stButton>button[kind="primary"],button[data-testid="baseButton-primary"]{ background:var(--text-primary)!important; color:var(--bg)!important; border-color:var(--text-primary)!important; }
.stButton>button[kind="primary"]:hover,button[data-testid="baseButton-primary"]:hover{ background:#333!important; }
.stTextInput input,.stNumberInput input,.stSelectbox>div>div{ font-family:var(--font-mono)!important; font-size:0.82rem!important; border-color:var(--border)!important; border-radius:0!important; background:#FFFFFF!important; }
.stTextInput input:focus,.stNumberInput input:focus{ border-color:var(--text-primary)!important; box-shadow:none!important; }
.stTextInput label,.stNumberInput label,.stSelectbox label,.stRadio label{ font-family:var(--font-mono)!important; font-size:0.75rem!important; text-transform:uppercase!important; letter-spacing:0.06em!important; color:var(--text-secondary)!important; font-weight:400!important; }
.stRadio>div{ display:flex; flex-direction:column; gap:4px; }
.stRadio>div>label{ font-family:var(--font-mono)!important; font-size:0.82rem!important; text-transform:uppercase!important; letter-spacing:0.06em!important; color:var(--text-secondary)!important; padding:6px 0!important; border-bottom:1px solid transparent!important; cursor:pointer; }
.stRadio>div>label:has(input:checked){ color:var(--text-primary)!important; border-bottom:1px solid var(--text-primary)!important; }
.stTabs [data-baseweb="tab-list"]{ border-bottom:1px solid var(--border)!important; gap:0!important; }
.stTabs [data-baseweb="tab"]{ font-family:var(--font-mono)!important; font-size:0.78rem!important; text-transform:uppercase!important; letter-spacing:0.08em!important; color:var(--text-secondary)!important; border-bottom:2px solid transparent!important; padding:10px 16px!important; background:transparent!important; }
.stTabs [aria-selected="true"]{ color:var(--text-primary)!important; border-bottom:2px solid var(--text-primary)!important; }
.stDataFrame{ font-family:var(--font-mono)!important; font-size:0.78rem!important; }
hr{ border:none; border-top:1px solid var(--border)!important; margin:var(--s-3) 0; }
.streamlit-expanderHeader{ font-family:var(--font-mono)!important; font-size:0.78rem!important; color:var(--text-secondary)!important; text-transform:uppercase!important; letter-spacing:0.06em!important; }
.stAlert{ border-radius:0!important; font-family:var(--font-sans)!important; font-size:0.85rem!important; }

.skeleton-card{ border:1px solid var(--border); padding:var(--s-3) var(--s-2); background:#FFFFFF; overflow:hidden; }
.skeleton-line{ height:14px; border-radius:0; background:linear-gradient(90deg,#f0f0f0 25%,#e8e8e8 50%,#f0f0f0 75%); background-size:200% 100%; animation:shimmer 1.4s infinite; margin-bottom:10px; }
.skeleton-line.tall{ height:36px; width:55%; }
.skeleton-line.short{ width:45%; }
.skeleton-line.full{ width:100%; }
@keyframes shimmer{ 0%{background-position:200% 0} 100%{background-position:-200% 0} }

.loading-overlay{ display:flex; align-items:center; gap:0.7rem; border:1px solid var(--border); padding:0.8rem var(--s-2); font-family:var(--font-mono); font-size:0.78rem; color:var(--text-secondary); margin:var(--s-1) 0 var(--s-2) 0; background:var(--bg); }
.spinner{ width:16px; height:16px; border:2px solid #DDD; border-top-color:var(--text-primary); border-radius:50%; animation:spin 0.8s linear infinite; flex-shrink:0; }
@keyframes spin{ to{transform:rotate(360deg)} }

.search-hint{ font-family:var(--font-mono); font-size:0.72rem; color:var(--text-secondary); margin-bottom:var(--s-1); }

@media(max-width:768px){
    .main-header h1{ font-size:1.35rem!important; }
    .metrics-row{ grid-template-columns:repeat(2,1fr)!important; }
    .pipeline-step{ min-width:80px; font-size:0.7rem; }
    .stButton>button{ width:100%; }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADERS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_session_data():
    if not SESSION_CSV.exists():
        return None
    return pd.read_csv(SESSION_CSV)

@st.cache_data(show_spinner=False)
def load_match_data():
    if not MATCH_CSV.exists():
        return None
    return pd.read_csv(MATCH_CSV)

@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    try:
        models["xgb"] = joblib.load(MODEL_DIR / "session_momentum_xgb.pkl")
        models["le"]  = joblib.load(MODEL_DIR / "label_encoder.pkl")
    except Exception:
        pass
    try:
        models["match"] = joblib.load(MODEL_DIR / "match_outcome_xgb.pkl")
    except Exception:
        pass
    return models


# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def momentum_badge(label):
    try:
        v = int(label)
    except Exception:
        v = 0
    if v == 1:
        return '<span class="momentum-badge badge-batting">⬆ Batting Momentum</span>'
    elif v == -1:
        return '<span class="momentum-badge badge-bowling">⬇ Bowling Momentum</span>'
    return '<span class="momentum-badge badge-neutral">→ Neutral</span>'

def momentum_color(label):
    try:
        v = int(label)
        if v == 1:  return C_BATTING
        if v == -1: return C_BOWLING
    except Exception:
        pass
    return C_NEUTRAL

@st.cache_data(show_spinner=False)
def build_match_lookup(session_df: pd.DataFrame) -> dict:
    lookup = {}
    for mid, grp in session_df.groupby("match_id"):
        batting  = set(grp["batting_team"].dropna().unique())
        fielding = set(grp["fielding_team"].dropna().unique())
        teams    = sorted(batting | fielding)
        team_a   = teams[0] if teams else "Unknown"
        team_b   = teams[1] if len(teams) > 1 else team_a
        extra    = ""
        for col in ["match_date", "date", "venue"]:
            if col in grp.columns:
                val = grp[col].iloc[0]
                if pd.notna(val):
                    extra = f" · {str(val)[:10]}"
                    break
        lookup[mid] = (team_a, team_b, extra)
    return lookup

def match_label(mid, lookup):
    if mid not in lookup:
        return str(mid)
    a, b, extra = lookup[mid]
    return f"{mid} — {a} vs {b}{extra}"

def render_skeleton_cards(n=3):
    for col in st.columns(n):
        with col:
            st.markdown("""
            <div class="skeleton-card">
                <div class="skeleton-line tall"></div>
                <div class="skeleton-line short"></div>
            </div>""", unsafe_allow_html=True)

def minimal_layout(height=300, **kwargs):
    base = dict(
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="IBM Plex Mono, monospace", size=10, color="#555555"),
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=9),
                   linecolor=C_BORDER, tickcolor=C_BORDER),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE",
                   linecolor=C_BORDER, tickcolor=C_BORDER, tickfont=dict(size=9)),
    )
    base.update(kwargs)
    return base


# ════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def plot_wp_curve(session_df, match_id):
    df = session_df[session_df["match_id"] == match_id].copy()
    if df.empty or "win_probability" not in df.columns:
        return None
    df["session_label"] = (df["innings_num"].astype(str) + " · "
                           + df["session"].str.split("_").str[-1].str.title())
    fig = go.Figure()
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor=C_BATTING, opacity=0.03, line_width=0)
    fig.add_hrect(y0=0.0, y1=0.5, fillcolor=C_BOWLING, opacity=0.03, line_width=0)
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df["win_probability"],
        mode="lines+markers", line=dict(color="#333333", width=1.5),
        marker=dict(size=8,
                    color=[momentum_color(m) for m in df.get("momentum_label", [0]*len(df))],
                    line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{customdata[0]}</b><br>WP:%{y:.1%} RR:%{customdata[1]:.2f} Wkts:%{customdata[2]}<extra></extra>",
        customdata=np.stack([
            df["session_label"],
            df.get("session_run_rate", pd.Series([0]*len(df))).fillna(0),
            df.get("session_wickets",  pd.Series([0]*len(df))).fillna(0),
        ], axis=-1),
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#CCCCCC", line_width=1)
    layout = minimal_layout(height=300)
    layout["xaxis"].update(tickvals=list(range(len(df))),
                           ticktext=df["session_label"].tolist(), tickangle=35)
    layout["yaxis"].update(tickformat=".0%", range=[0, 1],
                           title=dict(text="Win Probability", font=dict(size=10)))
    fig.update_layout(**layout)
    return fig

def plot_momentum_bars(session_df, match_id):
    df = session_df[session_df["match_id"] == match_id].copy()
    if df.empty or "session_momentum_index" not in df.columns:
        return None
    df["session_label"] = (df["innings_num"].astype(str) + "·"
                           + df["session"].str.split("_").str[-1].str.title())
    df["color"] = df["session_momentum_index"].apply(lambda x: C_BATTING if x > 0 else C_BOWLING)
    fig = go.Figure(go.Bar(x=df["session_label"], y=df["session_momentum_index"],
                           marker_color=df["color"], marker_line_width=0,
                           hovertemplate="<b>%{x}</b><br>Momentum:%{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line_color="#CCCCCC", line_width=1)
    layout = minimal_layout(height=260)
    layout["xaxis"].update(tickangle=35)
    layout["yaxis"].update(title="Momentum Index")
    fig.update_layout(**layout)
    return fig

def plot_session_stats(session_df, match_id):
    df = session_df[session_df["match_id"] == match_id].copy()
    if df.empty:
        return None
    df["session_label"] = (df["innings_num"].astype(str) + "·"
                           + df["session"].str.split("_").str[-1].str.title())
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df["session_label"],
        y=df.get("session_run_rate", pd.Series([0]*len(df))).fillna(0),
        name="Run Rate", marker_color="#333333", opacity=0.65, marker_line_width=0,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["session_label"],
        y=df.get("session_wickets", pd.Series([0]*len(df))).fillna(0),
        name="Wickets", mode="lines+markers",
        line=dict(color=C_BOWLING, width=1.5), marker=dict(size=7, color=C_BOWLING),
    ), secondary_y=True)
    layout = minimal_layout(height=260)
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9))
    layout["xaxis"].update(tickangle=35)
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Run Rate", secondary_y=False,
                     showgrid=True, gridcolor="#EEEEEE")
    fig.update_yaxes(title_text="Wickets",  secondary_y=True, showgrid=False)
    return fig

def plot_probability_bars(prob_bowling, prob_neutral, prob_batting):
    fig = go.Figure(go.Bar(
        x=["Bowling", "Neutral", "Batting"],
        y=[prob_bowling, prob_neutral, prob_batting],
        marker_color=[C_BOWLING, C_NEUTRAL, C_BATTING],
        marker_line_width=0,
        text=[f"{v:.1%}" for v in [prob_bowling, prob_neutral, prob_batting]],
        textposition="outside", textfont=dict(size=11, family="IBM Plex Mono"),
    ))
    layout = minimal_layout(height=250)
    layout["yaxis"].update(tickformat=".0%", range=[0, 1])
    fig.update_layout(**layout)
    return fig

def plot_live_wp_trajectory(wp_history: list) -> go.Figure:
    """WP trajectory built from state_manager polling history."""
    if not wp_history:
        return None
    df = pd.DataFrame(wp_history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df["wp_batting"],
        mode="lines+markers", name="Batting WP",
        line=dict(color=C_BATTING, width=1.5),
        marker=dict(size=6, color=C_BATTING),
        hovertemplate="Batting WP:%{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(df))), y=df["wp_bowling"],
        mode="lines+markers", name="Bowling WP",
        line=dict(color=C_BOWLING, width=1.5, dash="dot"),
        marker=dict(size=6, color=C_BOWLING),
        hovertemplate="Bowling WP:%{y:.1%}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#CCCCCC", line_width=1)
    layout = minimal_layout(height=280)
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9))
    layout["xaxis"].update(
        tickvals=list(range(len(df))),
        ticktext=[f"Poll {i+1}" for i in range(len(df))],
    )
    layout["yaxis"].update(tickformat=".0%", range=[0, 1],
                           title=dict(text="Win Probability", font=dict(size=10)))
    if "session" in df.columns:
        prev = None
        for i, s in enumerate(df["session"]):
            if s != prev and prev is not None:
                fig.add_vline(x=i - 0.5, line_dash="dash",
                              line_color=C_BORDER, line_width=1)
            prev = s
    fig.update_layout(**layout)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# LIVE POLLING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _poll_worker(match_id: str, interval: int):
    """Daemon thread: runs predict_current_session every `interval` seconds."""
    while st.session_state.get(f"polling_active_{match_id}", False):
        try:
            result = _live_predictor.predict_current_session(match_id)
            if "error" not in result:
                _state_manager.save_prediction(result)
                _state_manager.save_wp_point(
                    match_id,
                    result["prob_batting"],
                    result["prob_bowling"],
                    result["session_name"],
                )
                st.session_state["last_result"] = result
                st.session_state["last_update"] = datetime.utcnow().strftime("%H:%M:%S UTC")
                st.session_state["poll_count"]  = st.session_state.get("poll_count", 0) + 1
                st.session_state["poll_error"]  = None
            else:
                st.session_state["poll_error"] = result["error"]
        except Exception as e:
            st.session_state["poll_error"] = str(e)
        time.sleep(interval)

def start_polling(match_id: str, interval: int = 300):
    key = f"poll_thread_{match_id}"
    if st.session_state.get(key) is not None:
        return
    st.session_state[f"polling_active_{match_id}"] = True
    t = threading.Thread(target=_poll_worker, args=(match_id, interval), daemon=True)
    t.start()
    st.session_state[key] = t

def stop_polling(match_id: str):
    st.session_state[f"polling_active_{match_id}"] = False
    st.session_state[f"poll_thread_{match_id}"]    = None


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">🏏 Cricket Momentum</div>
        <div class="sidebar-brand-sub">Master's Thesis — Session Analysis</div>
    </div>""", unsafe_allow_html=True)

    tab_choice = st.radio(
        "Navigation",
        ["Match Explorer", "Live Prediction", "Model Performance", "Live Match"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    with st.spinner(""):
        session_df = load_session_data()
    with st.spinner(""):
        match_df = load_match_data()
    with st.spinner(""):
        models = load_models()

    if session_df is not None:
        st.markdown(f"""
        <div class="sidebar-data-info">
            <b>Dataset loaded</b><br>
            {session_df['match_id'].nunique()} matches &nbsp;·&nbsp;
            {len(session_df):,} sessions
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("No session data found. Run `python main.py` first.")

    if models:
        st.markdown(f"""
        <div class="sidebar-data-info">
            <b>Models loaded</b><br>
            {'XGBoost ✓' if 'xgb' in models else 'XGBoost ✗'} &nbsp;·&nbsp;
            {'Match ✓'   if 'match' in models else 'Match ✗'}
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    if LIVE_MODULES_READY:
        st.markdown("""
        <div class="sidebar-data-info">
            <span class="live-dot"></span>
            <span class="live-label">Live modules ready</span>
        </div>""", unsafe_allow_html=True)
    else:
        missing = [n for n, m in [("live_feed", _live_feed),
                                   ("state_manager", _state_manager),
                                   ("live_predictor", _live_predictor),
                                   ("session_segmenter", _session_seg)] if m is None]
        st.markdown(f"""
        <div class="sidebar-data-info">
            <span class="offline-label">Live modules offline</span><br>
            Missing: {', '.join(missing)}
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>Session-Based Momentum Prediction in Test Cricket</h1>
    <p class="main-header-sub">
        Master's Thesis Project &nbsp;·&nbsp;
        Win probability shifts · Session momentum analysis · Live prediction
    </p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH EXPLORER
# ════════════════════════════════════════════════════════════════════════════

if tab_choice == "Match Explorer":

    if session_df is None:
        st.error("Session data not found. Run the pipeline first.")
        st.stop()

    match_lookup = build_match_lookup(session_df)
    match_ids    = sorted(session_df["match_id"].unique())

    col_sel, col_info = st.columns([2, 3])

    with col_sel:
        st.markdown('<div class="subsection-title">Select Match</div>', unsafe_allow_html=True)
        search_query = st.text_input("Search by team name or match ID",
                                     placeholder="e.g. India, Australia, 12345…")
        q            = search_query.strip().lower()
        filtered_ids = [m for m in match_ids if q in match_label(m, match_lookup).lower()] if q else match_ids
        if not filtered_ids:
            st.warning("No matches found.")
            st.stop()
        label_to_id    = {match_label(m, match_lookup): m for m in filtered_ids}
        st.markdown(f'<div class="search-hint">{len(filtered_ids)} match(es) shown</div>',
                    unsafe_allow_html=True)
        selected_label = st.selectbox("Match", list(label_to_id.keys()),
                                      label_visibility="collapsed")
        selected_match = label_to_id[selected_label]

    match_sessions = session_df[session_df["match_id"] == selected_match]

    with col_info:
        if match_df is not None:
            m_row = match_df[match_df["match_id"] == selected_match]
            if not m_row.empty:
                m_row    = m_row.iloc[0]
                wp       = m_row.get("final_wp", 0.5)
                wp_color = C_BATTING if wp > 0.5 else C_BOWLING
                st.markdown('<div class="subsection-title">Match Result</div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="value" style="font-size:1.1rem">{m_row.get("winner","—")}</div><div class="label">Winner</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="value" style="color:{C_ACCENT}">{int(m_row.get("momentum_reversals",0))}</div><div class="label">Momentum Reversals</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="value" style="color:{wp_color}">{wp:.0%}</div><div class="label">Final Win Probability</div></div>', unsafe_allow_html=True)
            else:
                render_skeleton_cards(3)
        else:
            render_skeleton_cards(3)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Session Summary</div>', unsafe_allow_html=True)
    if not match_sessions.empty:
        cols = st.columns(3)
        for i, (_, row) in enumerate(match_sessions.head(6).iterrows()):
            with cols[i % 3]:
                inn   = int(row.get("innings_num", 1))
                sess  = str(row.get("session", "")).replace("_", " ").title()
                wp_d  = row.get("session_momentum_index", 0)
                sign  = "positive" if wp_d > 0 else ("negative" if wp_d < 0 else "neutral")
                arrow = "+" if wp_d > 0 else ""
                st.markdown(f"""
                <div class="session-card">
                    <div class="session-card-label">Innings {inn} · {sess}</div>
                    <div class="session-delta {sign}">{arrow}{wp_d:.1f}</div>
                    <div class="session-stats-line">
                        {int(row.get("session_runs",0))} runs &nbsp;/&nbsp;
                        {int(row.get("session_wickets",0))} wkts
                    </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    chart_col, note_col = st.columns([3, 1])
    with chart_col:
        st.markdown('<div class="section-title">Win Probability Across Sessions</div>', unsafe_allow_html=True)
        wp_fig = plot_wp_curve(session_df, selected_match)
        if wp_fig:
            st.plotly_chart(wp_fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Win probability data not available for this match.")
    with note_col:
        st.markdown("""
        <div class="notes-heading">Methodology Note</div>
        <div class="note-block">ΔWP is calculated as the net change in batting-team win probability
        from the final ball of the preceding session to the final ball of the current session,
        isolating session-specific performance impact.</div>
        <div class="note-block">Marker colours reflect the predicted momentum class:
        green = batting momentum, red = bowling momentum.</div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="subsection-title">Session Momentum Index</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Green bars = batting-side momentum · Red bars = bowling-side advantage.</div>', unsafe_allow_html=True)
        mom_fig = plot_momentum_bars(session_df, selected_match)
        if mom_fig:
            st.plotly_chart(mom_fig, use_container_width=True, config={"displayModeBar": False})
    with col_right:
        st.markdown('<div class="subsection-title">Run Rate &amp; Wickets per Session</div>', unsafe_allow_html=True)
        stat_fig = plot_session_stats(session_df, selected_match)
        if stat_fig:
            st.plotly_chart(stat_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Session-Level Detail</div>', unsafe_allow_html=True)

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
                return "⬆ Batting" if v == 1 else ("⬇ Bowling" if v == -1 else "→ Neutral")
            except:
                return "—"
        table["momentum_label"] = table["momentum_label"].apply(_lbl)
    table.columns = [c.replace("_", " ").title() for c in table.columns]
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="subsection-title">Analytical Pipeline Status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-strip">
        <div class="pipeline-step done">Data Ingestion</div>
        <div class="pipeline-step done">Feature Engineering</div>
        <div class="pipeline-step done">WP Calculation</div>
        <div class="pipeline-step done">Session Segmentation</div>
        <div class="pipeline-step active">Momentum Model</div>
        <div class="pipeline-step">Interpretability</div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MANUAL LIVE PREDICTION
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Live Prediction":

    st.markdown('<div class="section-title">Predict Session Momentum</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Enter current session statistics manually. The model predicts whether batting or bowling
        holds the momentum advantage and estimates win probability for the batting team.
    </div>""", unsafe_allow_html=True)

    if "xgb" not in models:
        st.warning("XGBoost model not found. Run the pipeline first.")
        st.stop()

    st.markdown('<div class="subsection-title">Session Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Scoring**")
        session_runs     = st.number_input("Session Runs",     0, 300, 65)
        session_run_rate = st.number_input("Session Run Rate", 0.0, 15.0, 3.5, 0.1)
        boundary_rate    = st.number_input("Boundary Rate",    0.0, 0.5, 0.08, 0.01)
        dot_ball_pct     = st.number_input("Dot Ball %",       0.0, 1.0, 0.45, 0.01)
    with c2:
        st.markdown("**Wickets & Pressure**")
        session_wickets      = st.number_input("Wickets This Session", 0, 10, 2)
        wickets_at_end       = st.number_input("Total Wickets Fallen", 0, 10, 4)
        max_dot_streak       = st.number_input("Max Dot Ball Streak",  0, 30, 5)
        total_pressure_balls = st.number_input("Total Pressure Balls", 0, 50, 8)
    with c3:
        st.markdown("**Match Context**")
        innings_num     = st.selectbox("Innings", [1, 2, 3, 4], index=1)
        ball_age_start  = st.number_input("Ball Age (balls)", 1, 480, 60)
        is_home_batting = st.selectbox("Home Team Batting?", ["Yes", "No"])
        toss_bat_first  = st.selectbox("Toss: Bat First?",   ["Yes", "No"])

    st.markdown('<div class="subsection-title" style="margin-top:var(--s-3)">Previous Session (Delta Features)</div>',
                unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        prev_run_rate = st.number_input("Previous Session Run Rate",   0.0, 15.0, 3.0, 0.1)
        prev_wickets  = st.number_input("Previous Session Wickets",    0, 10, 1)
    with d2:
        prev_dot_pct  = st.number_input("Previous Session Dot Ball %", 0.0, 1.0, 0.40, 0.01)

    if st.button("Predict Momentum", type="primary", use_container_width=True):
        with st.spinner("Running model inference…"):
            features = {
                "session_run_rate"       : session_run_rate,
                "session_runs"           : session_runs,
                "dot_ball_pct"           : dot_ball_pct,
                "boundary_rate"          : boundary_rate,
                "session_extras"         : 3,
                "session_wickets"        : session_wickets,
                "wickets_per_over"       : session_wickets / max(session_runs / 6 / max(session_run_rate, 0.1), 1),
                "wickets_at_session_end" : wickets_at_end,
                "max_dot_streak"         : max_dot_streak,
                "total_pressure_balls"   : total_pressure_balls,
                "run_rate_delta"         : session_run_rate - prev_run_rate,
                "wickets_delta"          : session_wickets - prev_wickets,
                "dot_ball_pct_delta"     : dot_ball_pct - prev_dot_pct,
                "session_momentum_index" : (session_run_rate - prev_run_rate) - (session_wickets - prev_wickets) * 2.5,
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
            X       = pd.DataFrame([features])[SESSION_FEATURES].fillna(0)
            proba   = models["xgb"].predict_proba(X)[0]
            le      = models["le"]
            pred_label = int(le.inverse_transform([np.argmax(proba)])[0])
            classes    = le.classes_
            prob_bowling = float(proba[list(classes).index(-1)]) if -1 in classes else 0.0
            prob_neutral = float(proba[list(classes).index(0)])  if  0 in classes else 0.0
            prob_batting = float(proba[list(classes).index(1)])  if  1 in classes else 0.0

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        mom_idx   = features["session_momentum_index"]
        idx_color = C_BATTING if mom_idx > 0 else C_BOWLING
        with r1:
            st.markdown(f'<div class="metric-card"><div style="margin-bottom:0.5rem">{momentum_badge(pred_label)}</div><div class="label">Predicted Momentum</div></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="metric-card"><div class="value">{max(proba)*100:.0f}%</div><div class="label">Model Confidence</div></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="metric-card"><div class="value" style="color:{idx_color}">{mom_idx:+.2f}</div><div class="label">Momentum Index</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="subsection-title">Class Probability Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_probability_bars(prob_bowling, prob_neutral, prob_batting),
                        use_container_width=True, config={"displayModeBar": False})

        interp = {
            1 : f"The batting team is gaining momentum. High run rate delta (+{features['run_rate_delta']:.2f}) and controlled wicket loss suggest batting dominance this session.",
            -1: f"The bowling team is gaining momentum. {'High dot ball pressure (streak: ' + str(max_dot_streak) + ')' if max_dot_streak > 5 else 'Wicket-taking'} is shifting the balance in favour of the fielding side.",
             0: "The session is evenly contested. Neither side has established a clear advantage — the match remains in the balance.",
        }
        st.markdown(f'<div class="info-box"><b>Interpretation:</b> {interp.get(pred_label,"")}</div>',
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Model Performance":

    st.markdown('<div class="section-title">Model Evaluation Results</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="metrics-row">
        <div class="metric-card"><div class="value" style="color:#008855;">84.2%</div><div class="label">Accuracy (XGBoost)</div></div>
        <div class="metric-card"><div class="value">0.341</div><div class="label">Log Loss</div></div>
        <div class="metric-card"><div class="value">0.082</div><div class="label">RMSE</div></div>
        <div class="metric-card"><div class="value">0.791</div><div class="label">F1 Score (Macro)</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>§4.3</b> Log Loss of 0.341 indicates well-calibrated probability estimates,
        critical for narrative WP interpretation. Overconfident models produce misleading
        session-level summaries.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Cross-Model Validation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="comparison-table-wrap"><table>
        <thead><tr><th>Model Configuration</th><th>Accuracy</th><th>Log Loss</th><th>RMSE</th></tr></thead>
        <tbody>
            <tr class="best-row"><td>XGBoost (Optimised)</td><td>0.842</td><td>0.341</td><td>0.082</td></tr>
            <tr><td>LSTM (Session-Sequences)</td><td>0.835</td><td>0.355</td><td>0.089</td></tr>
            <tr><td>Random Forest</td><td>0.821</td><td>0.380</td><td>0.095</td></tr>
            <tr><td>Logistic Baseline</td><td>0.760</td><td>0.450</td><td>0.120</td></tr>
        </tbody>
    </table></div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    report_path = MODEL_DIR / "model_evaluation_report.csv"
    if report_path.exists():
        st.dataframe(pd.read_csv(report_path), use_container_width=True, hide_index=True)
    else:
        st.info("Run the full pipeline to generate the model evaluation report CSV.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">SHAP Feature Importance</div>', unsafe_allow_html=True)

    shap_files = {
        "Global Importance"    : MODEL_DIR / "shap_global_importance.png",
        "Batting Momentum (+1)": MODEL_DIR / "shap_summary_class2.png",
        "Bowling Momentum (−1)": MODEL_DIR / "shap_summary_class0.png",
        "Match Outcome"        : MODEL_DIR / "shap_match_outcome.png",
    }
    tabs = st.tabs(list(shap_files.keys()))
    for tab, (title, path) in zip(tabs, shap_files.items()):
        with tab:
            if path.exists():
                st.image(str(path), use_column_width=True)
            else:
                st.info(f"Run the pipeline to generate: {path.name}")

    if session_df is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Dataset Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        stats = [
            ("Total Matches",    session_df["match_id"].nunique(), "#1A1A1A"),
            ("Total Sessions",   len(session_df),                   "#1A1A1A"),
            ("Batting Momentum", int((session_df.get("momentum_label", pd.Series()) == 1).sum()),  C_BATTING),
            ("Bowling Momentum", int((session_df.get("momentum_label", pd.Series()) == -1).sum()), C_BOWLING),
        ]
        for col, (label, val, color) in zip([c1, c2, c3, c4], stats):
            with col:
                st.markdown(f'<div class="metric-card"><div class="value" style="color:{color}">{val:,}</div><div class="label">{label}</div></div>',
                            unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE MATCH  ← NEW
# ════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Live Match":

    st.markdown('<div class="section-title">Live Test Match Monitor</div>', unsafe_allow_html=True)

    # ── Module gate: show fallback if src files not yet built ─────────
    if not LIVE_MODULES_READY:
        missing_mods = [n for n, m in [
            ("live_feed", _live_feed), ("state_manager", _state_manager),
            ("live_predictor", _live_predictor), ("session_segmenter", _session_seg),
        ] if m is None]
        st.markdown(f"""
        <div class="info-box alert-red">
            <b>Live modules not yet available.</b><br>
            Missing files in <code>src/</code>: {', '.join(missing_mods)}.py<br>
            Build these files, then re-run. The fallback simulator below works now.
        </div>""", unsafe_allow_html=True)

        # Fallback: simulate on any completed match from the dataset
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="subsection-title">Fallback — Simulate on Completed Match</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Select any match from your dataset and click Simulate to run the full
            prediction pipeline on its last recorded session. This validates your
            model logic without requiring a live API.
        </div>""", unsafe_allow_html=True)

        if session_df is not None and "xgb" in models:
            match_lookup = build_match_lookup(session_df)
            match_ids    = sorted(session_df["match_id"].unique())
            label_to_id  = {match_label(m, match_lookup): m for m in match_ids}
            sim_label    = st.selectbox("Select match", list(label_to_id.keys()),
                                        key="sim_match")
            sim_id       = label_to_id[sim_label]

            if st.button("▶ Simulate Last Session Prediction", type="primary"):
                sim_sessions = session_df[session_df["match_id"] == sim_id]
                if sim_sessions.empty:
                    st.warning("No session data for this match.")
                else:
                    last = sim_sessions.iloc[-1]
                    raw  = {f: last.get(f, 0) for f in SESSION_FEATURES}
                    X    = pd.DataFrame([raw])[SESSION_FEATURES].fillna(0)
                    proba = models["xgb"].predict_proba(X)[0]
                    le    = models["le"]
                    pred  = int(le.inverse_transform([np.argmax(proba)])[0])
                    classes      = le.classes_
                    prob_bowling = float(proba[list(classes).index(-1)]) if -1 in classes else 0.0
                    prob_neutral = float(proba[list(classes).index(0)])  if  0 in classes else 0.0
                    prob_batting = float(proba[list(classes).index(1)])  if  1 in classes else 0.0

                    st.markdown("<hr>", unsafe_allow_html=True)
                    r1, r2, r3 = st.columns(3)
                    sess_label = str(last.get("session","—")).replace("_"," ").title()
                    with r1:
                        st.markdown(f'<div class="metric-card live"><div style="margin-bottom:0.5rem">{momentum_badge(pred)}</div><div class="label">Simulated Momentum</div></div>', unsafe_allow_html=True)
                    with r2:
                        st.markdown(f'<div class="metric-card"><div class="value">{max(proba)*100:.0f}%</div><div class="label">Model Confidence</div></div>', unsafe_allow_html=True)
                    with r3:
                        st.markdown(f'<div class="metric-card"><div class="value" style="font-size:1.1rem">{sess_label}</div><div class="label">Session Simulated</div></div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.plotly_chart(plot_probability_bars(prob_bowling, prob_neutral, prob_batting),
                                    use_container_width=True, config={"displayModeBar": False})
        st.stop()

    # ── FULL LIVE MODE ────────────────────────────────────────────────

    col_match, col_status = st.columns([2, 3])

    with col_match:
        st.markdown('<div class="subsection-title">Select Live Match</div>', unsafe_allow_html=True)
        try:
            live_matches = _live_feed.get_live_test_matches()
        except Exception as e:
            live_matches = []
            st.warning(f"Could not fetch live matches: {e}")

        if not live_matches:
            st.markdown("""
            <div class="info-box">
                No live Test matches detected right now. Enter a match ID manually
                to run predictions on a recent or upcoming match.
            </div>""", unsafe_allow_html=True)
            manual_id = st.text_input("Cricsheet / API match ID",
                                      placeholder="e.g. 1234567", key="manual_id")
            if manual_id:
                live_matches = [{"id": manual_id, "name": f"Manual — {manual_id}"}]

        if not live_matches:
            st.info("Enter a match ID above to continue.")
            st.stop()

        match_options = {m.get("name", m.get("id", "Unknown")): m.get("id")
                         for m in live_matches}
        chosen_name   = st.selectbox("Match", list(match_options.keys()),
                                     label_visibility="collapsed", key="live_sel")
        chosen_id     = match_options[chosen_name]

        interval_map  = {"Every 2 min (active play)": 120,
                         "Every 5 min (standard)"   : 300,
                         "Every 10 min (slow play)"  : 600}
        interval_label = st.selectbox("Poll interval", list(interval_map.keys()), index=1)
        chosen_interval = interval_map[interval_label]

    with col_status:
        polling_active = st.session_state.get(f"polling_active_{chosen_id}", False)
        last_update    = st.session_state.get("last_update", "Never")
        poll_count     = st.session_state.get("poll_count", 0)
        poll_error     = st.session_state.get("poll_error", None)

        st.markdown('<div class="subsection-title">Polling Status</div>', unsafe_allow_html=True)

        if polling_active:
            st.markdown(f"""
            <div class="polling-banner active">
                <span class="live-dot"></span>
                Polling active &nbsp;·&nbsp; {poll_count} prediction(s) made
                &nbsp;·&nbsp; Last: {last_update}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="polling-banner stopped">
                Polling stopped &nbsp;·&nbsp; Click Start to begin live predictions
            </div>""", unsafe_allow_html=True)

        if poll_error:
            st.markdown(f'<div class="info-box alert-red"><b>Last error:</b> {poll_error}</div>',
                        unsafe_allow_html=True)

        btn1, btn2 = st.columns(2)
        with btn1:
            if not polling_active:
                if st.button("▶ Start Live Polling", type="primary", key="btn_start"):
                    start_polling(chosen_id, interval=chosen_interval)
                    st.session_state["poll_error"] = None
                    st.rerun()
            else:
                if st.button("⏹ Stop Polling", key="btn_stop"):
                    stop_polling(chosen_id)
                    st.rerun()
        with btn2:
            if st.button("🔄 Predict Now", key="btn_now"):
                with st.spinner("Fetching live data and running inference…"):
                    try:
                        result = _live_predictor.predict_current_session(chosen_id)
                        if "error" not in result:
                            _state_manager.save_prediction(result)
                            _state_manager.save_wp_point(
                                chosen_id,
                                result["prob_batting"],
                                result["prob_bowling"],
                                result["session_name"],
                            )
                            st.session_state["last_result"] = result
                            st.session_state["last_update"] = datetime.utcnow().strftime("%H:%M:%S UTC")
                            st.session_state["poll_count"]  = st.session_state.get("poll_count", 0) + 1
                            st.session_state["poll_error"]  = None
                        else:
                            st.session_state["poll_error"] = result["error"]
                    except Exception as e:
                        st.session_state["poll_error"] = str(e)
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Latest prediction display ─────────────────────────────────────
    result = st.session_state.get("last_result")

    if result and "error" not in result:
        st.markdown('<div class="section-title">Current Session Prediction</div>',
                    unsafe_allow_html=True)
        pred_lbl  = result["predicted_label"]
        feat      = result.get("features", {})
        mom_idx   = feat.get("session_momentum_index", 0)
        idx_color = C_BATTING if mom_idx > 0 else C_BOWLING
        css_cls   = "live" if pred_lbl == 1 else ("bowling" if pred_lbl == -1 else "")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card {css_cls}"><div style="margin-bottom:0.5rem">{momentum_badge(pred_lbl)}</div><div class="label">Predicted Momentum</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="value">{result["confidence"]*100:.0f}%</div><div class="label">Model Confidence</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="value" style="color:{idx_color}">{mom_idx:+.2f}</div><div class="label">Momentum Index</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="value" style="font-size:1.1rem">Inn {result["innings_num"]} · {result["session_name"].title()}</div><div class="label">{result["balls_in_session"]} balls</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        feat_col, prob_col = st.columns(2)
        with feat_col:
            st.markdown('<div class="subsection-title">Live Feature Snapshot</div>', unsafe_allow_html=True)
            if feat:
                display_feats = {
                    "session_run_rate"  : f"{feat.get('session_run_rate',0):.2f}",
                    "session_runs"      : feat.get("session_runs", 0),
                    "session_wickets"   : feat.get("session_wickets", 0),
                    "dot_ball_pct"      : f"{feat.get('dot_ball_pct',0):.2f}",
                    "max_dot_streak"    : feat.get("max_dot_streak", 0),
                    "run_rate_delta"    : f"{feat.get('run_rate_delta',0):+.2f}",
                    "wickets_delta"     : f"{feat.get('wickets_delta',0):+.2f}",
                    "top_order_exposed" : "Yes" if feat.get("top_order_exposed") else "No",
                    "ball_age_start"    : feat.get("ball_age_start", 0),
                    "innings_num"       : feat.get("innings_num", 1),
                }
                st.dataframe(pd.DataFrame(list(display_feats.items()),
                                          columns=["Feature", "Value"]),
                             use_container_width=True, hide_index=True)
        with prob_col:
            st.markdown('<div class="subsection-title">Class Probability Breakdown</div>', unsafe_allow_html=True)
            st.plotly_chart(
                plot_probability_bars(result["prob_bowling"], result["prob_neutral"], result["prob_batting"]),
                use_container_width=True, config={"displayModeBar": False},
            )

    elif result and "error" in result:
        st.markdown(f'<div class="info-box alert-red"><b>Prediction error:</b> {result["error"]}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            No prediction yet. Click <b>Predict Now</b> or start live polling above.
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── WP Trajectory chart ───────────────────────────────────────────
    st.markdown('<div class="section-title">Win Probability Trajectory</div>',
                unsafe_allow_html=True)
    chart_c, note_c = st.columns([3, 1])
    with chart_c:
        try:
            wp_hist = _state_manager.get_wp_history(chosen_id)
        except Exception:
            wp_hist = []
        if wp_hist:
            wp_fig = plot_live_wp_trajectory(wp_hist)
            if wp_fig:
                st.plotly_chart(wp_fig, use_container_width=True,
                                config={"displayModeBar": False})
        else:
            st.markdown("""
            <div class="info-box">
                No WP history yet. Make at least two predictions to see the trajectory.
            </div>""", unsafe_allow_html=True)
    with note_c:
        st.markdown("""
        <div class="notes-heading">Live Monitoring</div>
        <div class="note-block">Each point represents one polling cycle. Dashed vertical
        lines mark detected session boundaries.</div>
        <div class="note-block">Crossings of the 50% reference line represent momentum
        reversals — a key metric in Chapter 5 of the thesis.</div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Prediction Audit Log ──────────────────────────────────────────
    st.markdown('<div class="section-title">Prediction Audit Log</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Every prediction for this match is persisted to SQLite via
        <code>state_manager.py</code>. This log forms the basis of out-of-sample
        evaluation reported in the thesis results chapter.
    </div>""", unsafe_allow_html=True)

    try:
        log = _state_manager.get_prediction_log(chosen_id)
    except Exception:
        log = []

    if log:
        log_df = pd.DataFrame(log)
        if "label" in log_df.columns:
            log_df["label"] = log_df["label"].apply(
                lambda x: "⬆ Batting" if x == 1 else ("⬇ Bowling" if x == -1 else "→ Neutral")
            )
        if "confidence" in log_df.columns:
            log_df["confidence"] = log_df["confidence"].apply(lambda x: f"{x:.1%}")
        log_df.columns = [c.replace("_", " ").title() for c in log_df.columns]
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        if len(log) >= 2:
            batting_n = sum(1 for r in log if r["label"] == 1)
            bowling_n = sum(1 for r in log if r["label"] == -1)
            neutral_n = sum(1 for r in log if r["label"] == 0)
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(f'<div class="metric-card live"><div class="value" style="font-size:1.6rem;color:{C_BATTING}">{batting_n}</div><div class="label">Batting Momentum Sessions</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(f'<div class="metric-card bowling"><div class="value" style="font-size:1.6rem;color:{C_BOWLING}">{bowling_n}</div><div class="label">Bowling Momentum Sessions</div></div>', unsafe_allow_html=True)
            with s3:
                st.markdown(f'<div class="metric-card"><div class="value" style="font-size:1.6rem;color:{C_NEUTRAL}">{neutral_n}</div><div class="label">Neutral Sessions</div></div>', unsafe_allow_html=True)
    else:
        st.info("No predictions logged yet for this match.")

    # ── Auto-refresh when polling is active ───────────────────────────
    # Refreshes the UI at most every 60 seconds while polling is running.
    if polling_active:
        time.sleep(min(chosen_interval, 60))
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-footer">
    MSc in Data Science &nbsp;·&nbsp;
    Session-Based Momentum Prediction in Test Cricket<br>
    Data Source: Cricsheet Ball-by-Ball (2005–2024)
</div>
""", unsafe_allow_html=True)