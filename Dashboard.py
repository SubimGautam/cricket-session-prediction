import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

warnings.filterwarnings("ignore")

# Path setup
BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "outputs"
MODEL_DIR   = OUTPUT_DIR / "models"
SESSION_CSV = OUTPUT_DIR / "session_features.csv"
MATCH_CSV   = OUTPUT_DIR / "match_level_features.csv"
sys.path.insert(0, str(BASE_DIR / "src"))

# Color palette (clean academic)
C_PRIMARY   = "#1B4F72"   # Deep navy
C_BATTING   = "#1E8449"   # Cricket green
C_BOWLING   = "#C0392B"   # Alert red
C_NEUTRAL   = "#7F8C8D"   # Neutral grey
C_ACCENT    = "#D4AC0D"   # Gold
C_BG        = "#FAFAFA"
C_BORDER    = "#E8E8E8"

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

# PAGE CONFIG

st.set_page_config(
    page_title="Cricket Momentum Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Global CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #FAFAFA;
    }
    h1, h2, h3 {
        font-family: 'Source Serif 4', serif;
        color: #1B4F72;
    }
    .main-header {
        background: linear-gradient(135deg, #1B4F72 0%, #2E86AB 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        font-weight: 300;
    }
    .metric-card {
        background: white;
        border: 1px solid #E8E8E8;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Source Serif 4', serif;
        line-height: 1;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }
    .momentum-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.04em;
    }
    .badge-batting  { background: #E8F8F0; color: #1E8449; border: 1px solid #A9DFBF; }
    .badge-bowling  { background: #FDEDEC; color: #C0392B; border: 1px solid #F1948A; }
    .badge-neutral  { background: #F4F6F7; color: #5D6D7E; border: 1px solid #D5D8DC; }
    .section-title {
        font-family: 'Source Serif 4', serif;
        font-size: 1.15rem;
        color: #1B4F72;
        border-bottom: 2px solid #E8E8E8;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
    }
    .info-box {
        background: #EBF5FB;
        border-left: 4px solid #2E86AB;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #1B4F72;
        margin: 0.8rem 0;
    }
    div[data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #E8E8E8;
    }

    /* ── Loading spinner overlay ──────────────────────────────────────── */
    .loading-overlay {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        background: #EBF5FB;
        border: 1px solid #AED6F1;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        font-size: 0.9rem;
        color: #1B4F72;
        margin: 0.5rem 0 1rem 0;
    }
    .spinner {
        width: 20px; height: 20px;
        border: 3px solid #AED6F1;
        border-top-color: #1B4F72;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        flex-shrink: 0;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Skeleton shimmer cards ───────────────────────────────────────── */
    .skeleton-card {
        background: white;
        border: 1px solid #E8E8E8;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        overflow: hidden;
        position: relative;
    }
    .skeleton-line {
        height: 14px;
        border-radius: 6px;
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: shimmer 1.4s infinite;
        margin-bottom: 10px;
    }
    .skeleton-line.tall  { height: 36px; width: 60%; }
    .skeleton-line.short { width: 50%; }
    .skeleton-line.full  { width: 100%; }
    @keyframes shimmer {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* ── Responsive: stack columns on narrow screens ──────────────────── */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.35rem; }
        .main-header p  { font-size: 0.82rem; }
        .metric-card .value { font-size: 1.5rem; }
        .section-title { font-size: 1rem; }
        /* Streamlit columns collapse naturally; these reinforce touch targets */
        .stSelectbox > div { font-size: 0.9rem; }
        .stNumberInput input { font-size: 0.9rem; }
        .stButton > button  { width: 100%; font-size: 0.9rem; }
    }

    /* ── Search-style text filter above selectbox ─────────────────────── */
    .search-hint {
        font-size: 0.78rem;
        color: #999;
        margin-bottom: 0.3rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# DATA & MODEL LOADERS

@st.cache_data(show_spinner=False)
def load_session_data():
    if not SESSION_CSV.exists():
        return None
    df = pd.read_csv(SESSION_CSV)
    return df

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

# HELPER FUNCTIONS

def momentum_badge(label):
    if label == 1 or label == "1.0":
        return '<span class="momentum-badge badge-batting">⬆ Batting Momentum</span>'
    elif label == -1 or label == "-1.0":
        return '<span class="momentum-badge badge-bowling">⬇ Bowling Momentum</span>'
    else:
        return '<span class="momentum-badge badge-neutral">→ Neutral</span>'

def momentum_color(label):
    if label == 1:   return C_BATTING
    elif label == -1: return C_BOWLING
    else:             return C_NEUTRAL


#BUG FIX: resolve both team names correctly 
@st.cache_data(show_spinner=False)
def build_match_lookup(session_df: pd.DataFrame) -> dict:
    """
    For each match_id, extract the two unique team names from ALL rows
    (not just the first row). In Test cricket the batting and fielding teams
    swap across innings, so we union both columns across every session of
    the match to reliably get both teams.

    Returns: {match_id: (team_a, team_b, venue_or_date_str)}
    """
    lookup = {}
    for mid, grp in session_df.groupby("match_id"):
        batting_teams  = set(grp["batting_team"].dropna().unique())
        fielding_teams = set(grp["fielding_team"].dropna().unique())
        all_teams = sorted(batting_teams | fielding_teams)

        if len(all_teams) >= 2:
            team_a, team_b = all_teams[0], all_teams[1]
        elif len(all_teams) == 1:
            team_a = team_b = all_teams[0]   # edge-case guard
        else:
            team_a = team_b = "Unknown"

        # Optional: pull date/venue for richer label if column exists
        extra = ""
        for col in ["match_date", "date", "venue"]:
            if col in grp.columns:
                val = grp[col].iloc[0]
                if pd.notna(val):
                    extra = f" · {str(val)[:10]}"
                    break

        lookup[mid] = (team_a, team_b, extra)
    return lookup


def match_label(mid: str, lookup: dict) -> str:
    """Human-readable match label: 'ID — TeamA vs TeamB · date'"""
    if mid not in lookup:
        return str(mid)
    team_a, team_b, extra = lookup[mid]
    return f"{mid} — {team_a} vs {team_b}{extra}"


def render_skeleton_cards(n: int = 3):
    """Render n shimmer placeholder cards while data loads."""
    cols = st.columns(n)
    for col in cols:
        with col:
            st.markdown("""
            <div class="skeleton-card">
                <div class="skeleton-line tall"></div>
                <div class="skeleton-line short"></div>
            </div>""", unsafe_allow_html=True)


def render_loading(message: str = "Loading data…"):
    """Inline spinner with message."""
    st.markdown(f"""
    <div class="loading-overlay">
        <div class="spinner"></div>
        <span>{message}</span>
    </div>""", unsafe_allow_html=True)


def plot_wp_curve(session_df: pd.DataFrame, match_id: str):
    """Win probability curve across all sessions of a match."""
    df = session_df[session_df["match_id"] == match_id].copy()
    if df.empty or "win_probability" not in df.columns:
        return None

    df["session_label"] = df["innings_num"].astype(str) + " · " + df["session"].str.split("_").str[-1].str.title()

    fig = go.Figure()

    # Shaded regions
    fig.add_hrect(y0=0.5, y1=1.0, fillcolor=C_BATTING, opacity=0.04, line_width=0)
    fig.add_hrect(y0=0.0, y1=0.5, fillcolor=C_BOWLING, opacity=0.04, line_width=0)

    # WP line
    fig.add_trace(go.Scatter(
        x=list(range(len(df))),
        y=df["win_probability"],
        mode="lines+markers",
        line=dict(color=C_PRIMARY, width=2.5),
        marker=dict(
            size=9,
            color=[momentum_color(int(m)) if not pd.isna(m) else C_NEUTRAL
                   for m in df.get("momentum_label", [0]*len(df))],
            line=dict(color="white", width=1.5)
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Win Probability: %{y:.1%}<br>"
            "Run Rate: %{customdata[1]:.2f}<br>"
            "Wickets: %{customdata[2]}<extra></extra>"
        ),
        customdata=np.stack([
            df["session_label"],
            df.get("session_run_rate", pd.Series([0]*len(df))).fillna(0),
            df.get("session_wickets",  pd.Series([0]*len(df))).fillna(0),
        ], axis=-1),
    ))

    # 50% reference line
    fig.add_hline(y=0.5, line_dash="dot", line_color="#AAAAAA", line_width=1.2)

    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            tickvals=list(range(len(df))),
            ticktext=df["session_label"].tolist(),
            tickangle=35,
            tickfont=dict(size=10),
            showgrid=False,
            title=None,
        ),
        yaxis=dict(
            tickformat=".0%",
            range=[0, 1],
            showgrid=True,
            gridcolor="#F0F0F0",
            title=dict(text="Win Probability", font=dict(size=11)),
        ),
        showlegend=False,
        font=dict(family="DM Sans"),
    )
    return fig

def plot_momentum_bars(session_df: pd.DataFrame, match_id: str):
    """Session-by-session momentum index bar chart."""
    df = session_df[session_df["match_id"] == match_id].copy()
    if df.empty or "session_momentum_index" not in df.columns:
        return None

    df["session_label"] = df["innings_num"].astype(str) + "·" + df["session"].str.split("_").str[-1].str.title()
    df["color"] = df["session_momentum_index"].apply(
        lambda x: C_BATTING if x > 0 else C_BOWLING
    )

    fig = go.Figure(go.Bar(
        x=df["session_label"],
        y=df["session_momentum_index"],
        marker_color=df["color"],
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Momentum Index: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_color="#CCCCCC", line_width=1)

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, tickangle=35, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0", title="Momentum Index"),
        font=dict(family="DM Sans"),
        showlegend=False,
    )
    return fig

def plot_session_stats(session_df: pd.DataFrame, match_id: str):
    """Run rate and wickets per session."""
    df = session_df[session_df["match_id"] == match_id].copy()
    if df.empty:
        return None

    df["session_label"] = df["innings_num"].astype(str) + "·" + df["session"].str.split("_").str[-1].str.title()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=df["session_label"],
        y=df.get("session_run_rate", pd.Series([0]*len(df))).fillna(0),
        name="Run Rate",
        marker_color=C_PRIMARY,
        opacity=0.75,
        marker_line_width=0,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["session_label"],
        y=df.get("session_wickets", pd.Series([0]*len(df))).fillna(0),
        name="Wickets",
        mode="lines+markers",
        line=dict(color=C_BOWLING, width=2),
        marker=dict(size=8, color=C_BOWLING),
    ), secondary_y=True)

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        font=dict(family="DM Sans"),
        xaxis=dict(showgrid=False, tickangle=35, tickfont=dict(size=10)),
    )
    fig.update_yaxes(title_text="Run Rate", secondary_y=False,
                     showgrid=True, gridcolor="#F0F0F0")
    fig.update_yaxes(title_text="Wickets", secondary_y=True, showgrid=False)
    return fig

# SIDEBAR

with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem 0;">
        <div style="font-family: 'Source Serif 4', serif; font-size: 1.1rem;
                    color: #1B4F72; font-weight: 600;">🏏 Cricket Momentum</div>
        <div style="font-size: 0.78rem; color: #999; margin-top: 0.2rem;">
            Master's Thesis — Session Analysis
        </div>
    </div>
    <hr style="border: none; border-top: 1px solid #EEE; margin: 0.5rem 0 1rem 0;">
    """, unsafe_allow_html=True)

    tab_choice = st.radio(
        "Navigation",
        ["📊 Match Explorer", "🔮 Live Prediction", "📈 Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border:none;border-top:1px solid #EEE;margin:1rem 0'>",
                unsafe_allow_html=True)

    #Load data with spinner feedback
    with st.spinner("Loading session data…"):
        session_df = load_session_data()

    with st.spinner("Loading match data…"):
        match_df = load_match_data()

    with st.spinner("Loading models…"):
        models = load_models()

    if session_df is not None:
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#888;">
            <b style="color:#1B4F72">Dataset loaded</b><br>
            {session_df['match_id'].nunique()} matches &nbsp;·&nbsp;
            {len(session_df):,} sessions
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No session data found.\nRun `python main.py --skip-charts` first.")

    if models:
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#888; margin-top:0.8rem;">
            <b style="color:#1B4F72">Models loaded</b><br>
            {'XGBoost ✓' if 'xgb' in models else 'XGBoost ✗'} &nbsp;·&nbsp;
            {'Match ✓' if 'match' in models else 'Match ✗'}
        </div>
        """, unsafe_allow_html=True)

# MAIN HEADER

st.markdown("""
<div class="main-header">
    <h1>Session-Based Momentum Prediction in Test Cricket</h1>
    <p>Win probability shifts · Session momentum analysis · Match outcome prediction</p>
</div>
""", unsafe_allow_html=True)

# TAB 1: MATCH EXPLORER

if tab_choice == "📊 Match Explorer":

    if session_df is None:
        st.error("Session data not found. Run the pipeline first.")
        st.stop()

    #Build team-name lookup once (cached)
    match_lookup = build_match_lookup(session_df)
    match_ids    = sorted(session_df["match_id"].unique())

    # ── Match selector with search/filter
    col_sel, col_info = st.columns([2, 3])

    with col_sel:
        st.markdown('<div class="section-title">Select Match</div>', unsafe_allow_html=True)

        # Team name search filter — narrows the dropdown list
        search_query = st.text_input(
            "Search by team name or match ID",
            placeholder="e.g. India, South Africa, 12345…",
            label_visibility="visible",
        )

        # Filter match list based on search query
        if search_query.strip():
            q = search_query.strip().lower()
            filtered_ids = [
                mid for mid in match_ids
                if q in match_label(mid, match_lookup).lower()
            ]
        else:
            filtered_ids = match_ids

        if not filtered_ids:
            st.warning("No matches found for that search term.")
            st.stop()

        # Build labels for filtered list
        label_to_id = {match_label(mid, match_lookup): mid for mid in filtered_ids}
        labels      = list(label_to_id.keys())

        st.markdown(
            f'<div class="search-hint">{len(filtered_ids)} match(es) shown</div>',
            unsafe_allow_html=True,
        )

        selected_label = st.selectbox(
            "Match",
            labels,
            label_visibility="collapsed",
        )
        selected_match = label_to_id[selected_label]

    #Load match sessions with loading feedback
    match_sessions = session_df[session_df["match_id"] == selected_match]

    with col_info:
        if match_df is not None:
            m_row = match_df[match_df["match_id"] == selected_match]
            if not m_row.empty:
                m_row = m_row.iloc[0]
                st.markdown('<div class="section-title">Match Result</div>',
                            unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="value" style="color:{C_PRIMARY};font-size:1.1rem;">
                            {m_row.get('winner','—')}
                        </div>
                        <div class="label">Winner</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    rev = int(m_row.get("momentum_reversals", 0))
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="value" style="color:{C_ACCENT};">{rev}</div>
                        <div class="label">Momentum Reversals</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    wp = m_row.get("final_wp", 0.5)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="value" style="color:{C_BATTING if wp>0.5 else C_BOWLING};">
                            {wp:.0%}
                        </div>
                        <div class="label">Final Win Probability</div>
                    </div>""", unsafe_allow_html=True)
            else:
                # Skeleton placeholder if match row not available yet
                render_skeleton_cards(3)
        else:
            render_skeleton_cards(3)

    st.markdown("<br>", unsafe_allow_html=True)

    #Win Probability Curve
    st.markdown('<div class="section-title">Win Probability Across Sessions</div>',
                unsafe_allow_html=True)

    with st.spinner("Rendering win probability curve…"):
        wp_fig = plot_wp_curve(session_df, selected_match)

    if wp_fig:
        st.plotly_chart(wp_fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Win probability data not available for this match.")

    #Momentum Index + Session Stats
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-title">Session Momentum Index</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            Green bars = batting momentum gaining · Red bars = bowling momentum gaining
        </div>""", unsafe_allow_html=True)
        with st.spinner("Rendering momentum index…"):
            mom_fig = plot_momentum_bars(session_df, selected_match)
        if mom_fig:
            st.plotly_chart(mom_fig, use_container_width=True,
                            config={"displayModeBar": False})

    with col_right:
        st.markdown('<div class="section-title">Run Rate & Wickets per Session</div>',
                    unsafe_allow_html=True)
        with st.spinner("Rendering session statistics…"):
            stat_fig = plot_session_stats(session_df, selected_match)
        if stat_fig:
            st.plotly_chart(stat_fig, use_container_width=True,
                            config={"displayModeBar": False})

    # ── Session Table
    st.markdown('<div class="section-title">Session-Level Detail</div>',
                unsafe_allow_html=True)

    display_cols = [c for c in [
        "innings_num", "session", "session_runs", "session_run_rate",
        "session_wickets", "dot_ball_pct", "max_dot_streak",
        "session_momentum_index", "win_probability", "momentum_label"
    ] if c in match_sessions.columns]

    table = match_sessions[display_cols].copy()

    # Format numeric columns
    for col in ["session_run_rate", "dot_ball_pct", "session_momentum_index", "win_probability"]:
        if col in table.columns:
            table[col] = table[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

    # Color momentum label
    if "momentum_label" in table.columns:
        def label_text(x):
            try:
                v = float(x)
                if v == 1:   return "⬆ Batting"
                elif v == -1: return "⬇ Bowling"
                else:         return "→ Neutral"
            except:
                return "—"
        table["momentum_label"] = table["momentum_label"].apply(label_text)

    table.columns = [c.replace("_", " ").title() for c in table.columns]
    st.dataframe(table, use_container_width=True, hide_index=True)

# TAB 2: LIVE PREDICTION

elif tab_choice == "🔮 Live Prediction":

    st.markdown('<div class="section-title">Predict Session Momentum</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Enter the current session statistics below. The model will predict
        whether batting or bowling has the momentum advantage, and estimate
        the win probability for the batting team.
    </div>
    """, unsafe_allow_html=True)

    if "xgb" not in models:
        st.warning("XGBoost model not found. Run the pipeline first.")
        st.stop()

    #Input Form 
    st.markdown("#### Session Statistics")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Scoring**")
        session_runs     = st.number_input("Session Runs",       0, 300, 65)
        session_run_rate = st.number_input("Session Run Rate",   0.0, 15.0, 3.5, 0.1)
        boundary_rate    = st.number_input("Boundary Rate",      0.0, 0.5, 0.08, 0.01)
        dot_ball_pct     = st.number_input("Dot Ball %",         0.0, 1.0, 0.45, 0.01)

    with c2:
        st.markdown("**Wickets & Pressure**")
        session_wickets      = st.number_input("Wickets This Session", 0, 10, 2)
        wickets_at_end       = st.number_input("Total Wickets Fallen", 0, 10, 4)
        max_dot_streak       = st.number_input("Max Dot Ball Streak",  0, 30, 5)
        total_pressure_balls = st.number_input("Total Pressure Balls", 0, 50, 8)

    with c3:
        st.markdown("**Match Context**")
        innings_num      = st.selectbox("Innings",         [1, 2, 3, 4], index=1)
        ball_age_start   = st.number_input("Ball Age (balls)", 1, 480, 60)
        is_home_batting  = st.selectbox("Home Team Batting?", ["Yes", "No"])
        toss_bat_first   = st.selectbox("Toss: Bat First?",   ["Yes", "No"])

    st.markdown("#### Previous Session (for Delta Features)")
    d1, d2 = st.columns(2)
    with d1:
        prev_run_rate  = st.number_input("Previous Session Run Rate",  0.0, 15.0, 3.0, 0.1)
        prev_wickets   = st.number_input("Previous Session Wickets",   0, 10, 1)
    with d2:
        prev_dot_pct   = st.number_input("Previous Session Dot Ball %", 0.0, 1.0, 0.40, 0.01)

    # ── Prediction
    if st.button("🔮 Predict Momentum", type="primary", use_container_width=True):

        with st.spinner("Running model inference…"):
            # Build feature vector
            features = {
                "session_run_rate"      : session_run_rate,
                "session_runs"          : session_runs,
                "dot_ball_pct"          : dot_ball_pct,
                "boundary_rate"         : boundary_rate,
                "session_extras"        : 3,
                "session_wickets"       : session_wickets,
                "wickets_per_over"      : session_wickets / max(session_runs / 6 / max(session_run_rate, 0.1), 1),
                "wickets_at_session_end": wickets_at_end,
                "max_dot_streak"        : max_dot_streak,
                "total_pressure_balls"  : total_pressure_balls,
                "run_rate_delta"        : session_run_rate - prev_run_rate,
                "wickets_delta"         : session_wickets - prev_wickets,
                "dot_ball_pct_delta"    : dot_ball_pct - prev_dot_pct,
                "session_momentum_index": (session_run_rate - prev_run_rate) - (session_wickets - prev_wickets) * 2.5,
                "ball_age_start"        : ball_age_start,
                "innings_num"           : innings_num,
                "is_home_batting"       : 1 if is_home_batting == "Yes" else 0,
                "toss_bat_first"        : 1 if toss_bat_first == "Yes" else 0,
                "toss_winner_batting"   : 1 if is_home_batting == "Yes" and toss_bat_first == "Yes" else 0,
                "is_fourth_innings"     : 1 if innings_num == 4 else 0,
                "is_first_innings"      : 1 if innings_num == 1 else 0,
                "is_morning_session"    : 0,
                "is_evening_session"    : 0,
                "top_order_exposed"     : 1 if wickets_at_end <= 4 else 0,
            }

            X_input = pd.DataFrame([features])[SESSION_FEATURES].fillna(0)
            proba   = models["xgb"].predict_proba(X_input)[0]
            le      = models["le"]
            pred_class_encoded = np.argmax(proba)
            pred_label = le.inverse_transform([pred_class_encoded])[0]

            class_order = le.classes_
            prob_bowling = proba[list(class_order).index(-1)] if -1 in class_order else proba[0]
            prob_neutral = proba[list(class_order).index(0)]  if  0 in class_order else proba[1]
            prob_batting = proba[list(class_order).index(1)]  if  1 in class_order else proba[2]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Prediction Result")

        res1, res2, res3 = st.columns(3)

        with res1:
            badge = momentum_badge(pred_label)
            st.markdown(f"""
            <div class="metric-card">
                <div style="margin-bottom:0.5rem">{badge}</div>
                <div class="label">Predicted Momentum</div>
            </div>""", unsafe_allow_html=True)

        with res2:
            confidence = max(proba) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="value" style="color:{C_PRIMARY};">{confidence:.0f}%</div>
                <div class="label">Model Confidence</div>
            </div>""", unsafe_allow_html=True)

        with res3:
            mom_idx = features["session_momentum_index"]
            idx_color = C_BATTING if mom_idx > 0 else C_BOWLING
            st.markdown(f"""
            <div class="metric-card">
                <div class="value" style="color:{idx_color};">{mom_idx:+.2f}</div>
                <div class="label">Momentum Index</div>
            </div>""", unsafe_allow_html=True)

        # Probability bar chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Class Probability Breakdown")

        fig_prob = go.Figure(go.Bar(
            x=["Bowling Momentum", "Neutral", "Batting Momentum"],
            y=[prob_bowling, prob_neutral, prob_batting],
            marker_color=[C_BOWLING, C_NEUTRAL, C_BATTING],
            marker_line_width=0,
            text=[f"{v:.1%}" for v in [prob_bowling, prob_neutral, prob_batting]],
            textposition="outside",
            textfont=dict(size=13, family="DM Sans"),
        ))
        fig_prob.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            yaxis=dict(tickformat=".0%", range=[0, 1],
                       showgrid=True, gridcolor="#F0F0F0"),
            xaxis=dict(showgrid=False),
            font=dict(family="DM Sans"),
            showlegend=False,
        )
        st.plotly_chart(fig_prob, use_container_width=True,
                        config={"displayModeBar": False})

        # Interpretation
        interp = {
            1 : f"The batting team is gaining momentum. High run rate delta (+{features['run_rate_delta']:.2f}) and controlled wicket loss suggest batting dominance this session.",
            -1: f"The bowling team is gaining momentum. {'High dot ball pressure (streak: ' + str(max_dot_streak) + ')' if max_dot_streak > 5 else 'Wicket taking'} is shifting the balance in favour of the fielding side.",
             0: "The session is evenly contested. Neither side has established a clear advantage — the match is in the balance.",
        }
        st.markdown(f"""
        <div class="info-box">
            <b>Interpretation:</b> {interp.get(pred_label, '')}
        </div>""", unsafe_allow_html=True)


# TAB 3: MODEL PERFORMANCE

elif tab_choice == "📈 Model Performance":

    st.markdown('<div class="section-title">Model Evaluation Results</div>',
                unsafe_allow_html=True)

    report_path = MODEL_DIR / "model_evaluation_report.csv"

    if report_path.exists():
        report = pd.read_csv(report_path)
        st.dataframe(report, use_container_width=True, hide_index=True)
    else:
        st.info("Run the full pipeline to generate model evaluation report.")

    # SHAP plots
    st.markdown('<div class="section-title">SHAP Feature Importance</div>',
                unsafe_allow_html=True)

    shap_files = {
        "Global Importance (All Classes)" : MODEL_DIR / "shap_global_importance.png",
        "Batting Momentum (+1)"            : MODEL_DIR / "shap_summary_class2.png",
        "Bowling Momentum (−1)"            : MODEL_DIR / "shap_summary_class0.png",
        "Match Outcome"                    : MODEL_DIR / "shap_match_outcome.png",
    }

    tabs = st.tabs(list(shap_files.keys()))
    for tab, (title, path) in zip(tabs, shap_files.items()):
        with tab:
            if path.exists():
                st.image(str(path), use_column_width=True)
            else:
                st.info(f"Run the pipeline to generate: {path.name}")

    # Dataset summary
    if session_df is not None:
        st.markdown('<div class="section-title">Dataset Summary</div>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        stats = [
            ("Total Matches",  session_df["match_id"].nunique(), C_PRIMARY),
            ("Total Sessions", len(session_df),                  C_PRIMARY),
            ("Batting Momentum Sessions",
             int((session_df.get("momentum_label", pd.Series()) == 1).sum()),  C_BATTING),
            ("Bowling Momentum Sessions",
             int((session_df.get("momentum_label", pd.Series()) == -1).sum()), C_BOWLING),
        ]
        for col, (label, val, color) in zip([c1, c2, c3, c4], stats):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value" style="color:{color};">{val:,}</div>
                    <div class="label">{label}</div>
                </div>""", unsafe_allow_html=True)