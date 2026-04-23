"""
src/visualize.py
Generates exploratory data visualizations for the cricket momentum project.

Usage:
    python -m src.visualize
    
    Or from a notebook:
    from src.visualize import run_all_charts
    run_all_charts(df)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from src.config import OUTPUTS

# ── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"  : "#0d1117",
    "axes.facecolor"    : "#161b22",
    "axes.edgecolor"    : "#30363d",
    "axes.labelcolor"   : "#c9d1d9",
    "xtick.color"       : "#8b949e",
    "ytick.color"       : "#8b949e",
    "text.color"        : "#c9d1d9",
    "grid.color"        : "#21262d",
    "grid.linewidth"    : 0.6,
    "font.family"       : "monospace",
})

ACCENT  = "#58a6ff"
RED     = "#f85149"
GREEN   = "#3fb950"
YELLOW  = "#d29922"
PURPLE  = "#bc8cff"


def _save(fig, name):
    os.makedirs(OUTPUTS, exist_ok=True)
    path = os.path.join(OUTPUTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  💾 Saved → {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1: Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────
def chart_overview(df: pd.DataFrame):
    """4-panel overview of the dataset."""
    print("📊 Generating Chart 1: Dataset Overview...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("CRICKET MOMENTUM — Dataset Overview", fontsize=16,
                 fontweight="bold", color="#e6edf3", y=1.01)

    # 1a — Runs distribution
    ax = axes[0, 0]
    counts = df["batter_runs"].value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values, color=ACCENT, alpha=0.85)
    ax.set_title("Batter Runs Distribution", color="#e6edf3")
    ax.set_xlabel("Runs per Ball"); ax.set_ylabel("Count")
    ax.grid(axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 500,
                f"{h/1000:.0f}k", ha="center", fontsize=7, color="#8b949e")

    # 1b — Wicket types
    ax = axes[0, 1]
    wkt = df[df["is_wicket"] == 1]["wicket_kind"].value_counts().head(8)
    colors = [ACCENT, GREEN, YELLOW, RED, PURPLE,
              "#ffa657", "#79c0ff", "#56d364"][:len(wkt)]
    wedges, texts, autotexts = ax.pie(
        wkt.values, labels=wkt.index, autopct="%1.1f%%",
        colors=colors, startangle=140,
        textprops={"color": "#c9d1d9", "fontsize": 8}
    )
    for at in autotexts:
        at.set_color("#e6edf3"); at.set_fontsize(7)
    ax.set_title("Wicket Types", color="#e6edf3")

    # 1c — Runs per over (overall run rate curve)
    ax = axes[1, 0]
    over_runs = df.groupby("over")["total_runs"].mean()
    ax.plot(over_runs.index, over_runs.values, color=GREEN, linewidth=1.8)
    ax.fill_between(over_runs.index, over_runs.values, alpha=0.15, color=GREEN)
    ax.axvline(30, color=YELLOW, linestyle="--", linewidth=0.8, label="Session 1/2")
    ax.axvline(60, color=RED,    linestyle="--", linewidth=0.8, label="Session 2/3")
    ax.legend(fontsize=8); ax.grid(axis="y")
    ax.set_title("Avg Runs/Ball by Over Number", color="#e6edf3")
    ax.set_xlabel("Over"); ax.set_ylabel("Avg Runs/Ball")

    # 1d — Wickets per over
    ax = axes[1, 1]
    over_wkts = df.groupby("over")["is_wicket"].mean() * 100
    ax.bar(over_wkts.index, over_wkts.values, color=RED, alpha=0.7, width=0.8)
    ax.axvline(30, color=YELLOW, linestyle="--", linewidth=0.8)
    ax.axvline(60, color=ACCENT, linestyle="--", linewidth=0.8)
    ax.set_title("Wicket % by Over Number", color="#e6edf3")
    ax.set_xlabel("Over"); ax.set_ylabel("Wicket %")
    ax.grid(axis="y")

    plt.tight_layout()
    _save(fig, "01_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2: Session-Level Analysis
# ─────────────────────────────────────────────────────────────────────────────
def chart_session_analysis(df: pd.DataFrame):
    """Compare runs, wickets, dot balls across the 3 Test sessions."""
    print("📊 Generating Chart 2: Session Analysis...")

    def assign_session(over):
        if over <= 30: return "Session 1\n(Overs 1-30)"
        elif over <= 60: return "Session 2\n(Overs 31-60)"
        else: return "Session 3\n(Overs 61-90)"

    df = df.copy()
    df["session"] = df["over"].apply(assign_session)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("SESSION-BY-SESSION COMPARISON", fontsize=15,
                 fontweight="bold", color="#e6edf3")

    sessions = ["Session 1\n(Overs 1-30)", "Session 2\n(Overs 31-60)", "Session 3\n(Overs 61-90)"]
    sess_colors = [ACCENT, GREEN, YELLOW]

    # Avg run rate per session
    ax = axes[0]
    rr = df.groupby("session")["batter_runs"].mean().reindex(sessions)
    bars = ax.bar(["S1", "S2", "S3"], rr.values, color=sess_colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, rr.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.002,
                f"{v:.3f}", ha="center", fontsize=10, color="#e6edf3", fontweight="bold")
    ax.set_title("Avg Runs/Ball", color="#e6edf3", fontsize=12)
    ax.set_ylabel("Runs per Ball"); ax.grid(axis="y")

    # Wicket % per session
    ax = axes[1]
    wr = df.groupby("session")["is_wicket"].mean().reindex(sessions) * 100
    bars = ax.bar(["S1", "S2", "S3"], wr.values, color=sess_colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, wr.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.2f}%", ha="center", fontsize=10, color="#e6edf3", fontweight="bold")
    ax.set_title("Wicket % per Ball", color="#e6edf3", fontsize=12)
    ax.set_ylabel("Wicket %"); ax.grid(axis="y")

    # Dot ball % per session
    ax = axes[2]
    dr = df.groupby("session")["is_dot"].mean().reindex(sessions) * 100
    bars = ax.bar(["S1", "S2", "S3"], dr.values, color=sess_colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, dr.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                f"{v:.1f}%", ha="center", fontsize=10, color="#e6edf3", fontweight="bold")
    ax.set_title("Dot Ball %", color="#e6edf3", fontsize=12)
    ax.set_ylabel("Dot Ball %"); ax.grid(axis="y")

    plt.tight_layout()
    _save(fig, "02_session_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3: Momentum Index Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def chart_momentum_heatmap(df: pd.DataFrame):
    """Momentum index heatmap: over number vs innings."""
    print("📊 Generating Chart 3: Momentum Heatmap...")

    df = df.copy()
    df["momentum"] = (
        df["batter_runs"] * 0.4
        + df["is_boundary_4"] * 1.5
        + df["is_boundary_6"] * 2.5
        - df["is_wicket"] * 4.0
        - df["is_dot"] * 0.3
    )

    pivot = df.groupby(["innings", "over"])["momentum"].mean().unstack(level=0)
    pivot.columns = [f"Innings {c}" for c in pivot.columns]
    pivot = pivot[(pivot.index >= 1) & (pivot.index <= 90)]

    fig, ax = plt.subplots(figsize=(18, 5))
    sns.heatmap(
        pivot.T, ax=ax,
        cmap="RdYlGn", center=0,
        linewidths=0, xticklabels=10,
        cbar_kws={"label": "Momentum Index", "shrink": 0.6}
    )
    ax.set_title("MOMENTUM INDEX HEATMAP — By Over & Innings",
                 fontsize=14, fontweight="bold", color="#e6edf3", pad=12)
    ax.set_xlabel("Over Number"); ax.set_ylabel("Innings")
    plt.tight_layout()
    _save(fig, "03_momentum_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4: Single Match Momentum Wave
# ─────────────────────────────────────────────────────────────────────────────
def chart_match_momentum_wave(df: pd.DataFrame, match_id: str = None):
    """Rolling momentum wave for one match — the 'hero chart'."""
    print("📊 Generating Chart 4: Match Momentum Wave...")

    if match_id is None:
        # Pick a match with enough data
        counts = df.groupby("match_id")["over"].max()
        good = counts[counts >= 80].index
        match_id = good[len(good)//2] if len(good) > 0 else df["match_id"].iloc[0]

    m = df[df["match_id"] == match_id].copy()
    m["momentum"] = (
        m["batter_runs"] * 0.4
        + m["is_boundary_4"] * 1.5
        + m["is_boundary_6"] * 2.5
        - m["is_wicket"] * 4.0
        - m["is_dot"] * 0.3
    )
    m["ball_num"] = range(len(m))
    m["momentum_smooth"] = m["momentum"].rolling(12, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    innings_colors = {1: ACCENT, 2: GREEN, 3: YELLOW, 4: RED}
    for inn in sorted(m["innings"].unique()):
        sub = m[m["innings"] == inn]
        color = innings_colors.get(inn, ACCENT)
        ax.plot(sub["ball_num"], sub["momentum_smooth"],
                color=color, linewidth=2, label=f"Innings {inn}")
        ax.fill_between(sub["ball_num"], sub["momentum_smooth"],
                        alpha=0.08, color=color)

        # Mark wickets
        wkts = sub[sub["is_wicket"] == 1]
        ax.scatter(wkts["ball_num"], wkts["momentum_smooth"],
                   color=RED, zorder=5, s=40, marker="v", alpha=0.9)

    ax.axhline(0, color="#30363d", linewidth=1, linestyle="--")

    # Session dividers — rough estimation
    first_inn = m[m["innings"] == 1]
    if len(first_inn) > 0:
        balls_30 = first_inn[first_inn["over"] == 30]
        balls_60 = first_inn[first_inn["over"] == 60]
        if len(balls_30):
            ax.axvline(balls_30["ball_num"].iloc[-1], color=YELLOW,
                       linestyle=":", alpha=0.5, linewidth=1.2)
            ax.text(balls_30["ball_num"].iloc[-1], ax.get_ylim()[1] * 0.9,
                    " S1|S2", color=YELLOW, fontsize=8, alpha=0.7)
        if len(balls_60):
            ax.axvline(balls_60["ball_num"].iloc[-1], color=YELLOW,
                       linestyle=":", alpha=0.5, linewidth=1.2)
            ax.text(balls_60["ball_num"].iloc[-1], ax.get_ylim()[1] * 0.9,
                    " S2|S3", color=YELLOW, fontsize=8, alpha=0.7)

    teams = m["teams"].iloc[0] if "teams" in m.columns else match_id
    ax.set_title(f"MOMENTUM WAVE  ·  {teams}",
                 fontsize=15, fontweight="bold", color="#e6edf3", pad=14)
    ax.set_xlabel("Ball Number (all innings)", fontsize=11)
    ax.set_ylabel("Momentum Index (12-ball rolling avg)", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.2, fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Annotate red dots
    ax.scatter([], [], color=RED, marker="v", s=40, label="Wicket")
    ax.legend(loc="upper right", framealpha=0.2, fontsize=10)

    plt.tight_layout()
    _save(fig, "04_momentum_wave.png")
    print(f"   (Match: {match_id} | {teams})")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 5: Top Bowlers & Batters by Momentum Impact
# ─────────────────────────────────────────────────────────────────────────────
def chart_top_players(df: pd.DataFrame, top_n: int = 12):
    """Bar charts for top run-scorers and most economical bowlers."""
    print("📊 Generating Chart 5: Top Players...")

    df = df.copy()
    MIN_BALLS = 200

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("PLAYER IMPACT — Top Batters & Bowlers",
                 fontsize=14, fontweight="bold", color="#e6edf3")

    # Top batters by strike rate
    ax = axes[0]
    batter_stats = df.groupby("batter").agg(
        runs=("batter_runs", "sum"),
        balls=("batter_runs", "count")
    )
    batter_stats = batter_stats[batter_stats["balls"] >= MIN_BALLS].copy()
    batter_stats["sr"] = batter_stats["runs"] / batter_stats["balls"] * 100
    top_bat = batter_stats.nlargest(top_n, "sr")

    bars = ax.barh(top_bat.index, top_bat["sr"], color=ACCENT, alpha=0.85)
    ax.set_title(f"Top {top_n} Batters by Strike Rate\n(min {MIN_BALLS} balls)",
                 color="#e6edf3", fontsize=11)
    ax.set_xlabel("Strike Rate")
    for bar, v in zip(bars, top_bat["sr"]):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}", va="center", fontsize=8, color="#e6edf3")
    ax.grid(axis="x", alpha=0.4)
    ax.invert_yaxis()

    # Top bowlers by wickets
    ax = axes[1]
    bowler_stats = df.groupby("bowler").agg(
        wickets=("is_wicket", "sum"),
        runs_given=("total_runs", "sum"),
        balls=("total_runs", "count")
    )
    bowler_stats = bowler_stats[bowler_stats["balls"] >= MIN_BALLS].copy()
    bowler_stats["economy"] = bowler_stats["runs_given"] / bowler_stats["balls"] * 6
    top_bowl = bowler_stats.nlargest(top_n, "wickets")

    bars = ax.barh(top_bowl.index, top_bowl["wickets"], color=GREEN, alpha=0.85)
    ax.set_title(f"Top {top_n} Bowlers by Wickets\n(min {MIN_BALLS} balls bowled)",
                 color="#e6edf3", fontsize=11)
    ax.set_xlabel("Total Wickets")
    for bar, v in zip(bars, top_bowl["wickets"]):
        ax.text(v + 0.2, bar.get_y() + bar.get_height()/2,
                f"{int(v)}", va="center", fontsize=8, color="#e6edf3")
    ax.grid(axis="x", alpha=0.4)
    ax.invert_yaxis()

    plt.tight_layout()
    _save(fig, "05_top_players.png")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 6: Wicket Cluster Analysis
# ─────────────────────────────────────────────────────────────────────────────
def chart_wicket_clusters(df: pd.DataFrame):
    """Shows how often wickets cluster (momentum-killing sequences)."""
    print("📊 Generating Chart 6: Wicket Clusters...")

    # Flag balls within 10 balls of a wicket
    df = df.copy().sort_values(["match_id", "innings", "over", "ball"])
    df["ball_idx"] = df.groupby(["match_id", "innings"]).cumcount()

    # For each wicket, mark ±5 balls as a "cluster zone"
    df["cluster_zone"] = 0
    wicket_indices = df.index[df["is_wicket"] == 1].tolist()

    # Runs scored in 5-over blocks after wickets vs normal play
    after_wicket_runs = []
    normal_runs = []

    for match in df["match_id"].unique():
        for inn in [1, 2]:
            sub = df[(df["match_id"] == match) & (df["innings"] == inn)].reset_index()
            wkt_positions = sub.index[sub["is_wicket"] == 1].tolist()
            for pos in wkt_positions:
                after = sub.iloc[pos+1:pos+31]["batter_runs"].sum()
                after_wicket_runs.append(after)
            # Random normal windows
            if len(sub) > 60:
                for _ in range(len(wkt_positions)):
                    rand_pos = np.random.randint(5, len(sub)-31)
                    normal = sub.iloc[rand_pos:rand_pos+30]["batter_runs"].sum()
                    normal_runs.append(normal)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("WICKET CLUSTER ANALYSIS", fontsize=14,
                 fontweight="bold", color="#e6edf3")

    # Distribution of runs in 30 balls after wicket vs normal
    ax = axes[0]
    ax.hist(after_wicket_runs, bins=30, color=RED, alpha=0.6,
            label="After Wicket (30 balls)", density=True)
    ax.hist(normal_runs, bins=30, color=GREEN, alpha=0.6,
            label="Normal Play (30 balls)", density=True)
    ax.axvline(np.mean(after_wicket_runs), color=RED, linestyle="--",
               linewidth=1.5, label=f"Mean after: {np.mean(after_wicket_runs):.1f}")
    ax.axvline(np.mean(normal_runs), color=GREEN, linestyle="--",
               linewidth=1.5, label=f"Mean normal: {np.mean(normal_runs):.1f}")
    ax.set_title("Runs Scored in 30 Balls:\nAfter Wicket vs Normal Play", color="#e6edf3")
    ax.set_xlabel("Runs"); ax.legend(fontsize=8); ax.grid(axis="y")

    # Wickets per over (frequency)
    ax = axes[1]
    wkt_by_over = df.groupby("over")["is_wicket"].sum()
    wkt_by_over = wkt_by_over[wkt_by_over.index <= 90]
    ax.bar(wkt_by_over.index, wkt_by_over.values, color=RED, alpha=0.75, width=0.8)
    ax.axvline(30, color=YELLOW, linestyle="--", linewidth=1, label="Session boundaries")
    ax.axvline(60, color=YELLOW, linestyle="--", linewidth=1)
    ax.set_title("Total Wickets by Over Number\n(all matches)", color="#e6edf3")
    ax.set_xlabel("Over"); ax.set_ylabel("Total Wickets")
    ax.legend(fontsize=8); ax.grid(axis="y")

    plt.tight_layout()
    _save(fig, "06_wicket_clusters.png")


# ─────────────────────────────────────────────────────────────────────────────
# Run everything
# ─────────────────────────────────────────────────────────────────────────────
def run_all_charts(df: pd.DataFrame):
    print("\n🏏 Generating all visualizations...\n")
    chart_overview(df)
    chart_session_analysis(df)
    chart_momentum_heatmap(df)
    chart_match_momentum_wave(df)
    chart_top_players(df)
    chart_wicket_clusters(df)
    print(f"\n✅ All charts saved to: {OUTPUTS}/")


if __name__ == "__main__":
    from src.load_data import load_all_matches
    df = load_all_matches()
    run_all_charts(df)
