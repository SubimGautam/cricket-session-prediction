"""
main.py — Run the full Cricket Momentum pipeline.

Steps:
    1. Load all YAML/JSON files
    2. Save processed CSV
    3. Generate visualizations          ← already working
    4. Feature engineering (sessions)   ← NEW
    5. Win probability computation      ← NEW
    6. XGBoost + RF modeling            ← NEW
    7. SHAP interpretability            ← NEW
    8. Match outcome prediction         ← NEW

Usage:
    python main.py                  # full run (all files)
    python main.py --sample 20      # quick test with 20 files
    python main.py --charts-only    # skip loading, use existing CSV
    python main.py --skip-charts    # skip charts, go straight to modeling
    python main.py --sample 20 --skip-charts   # fastest dev loop
"""

import argparse
import os
import sys
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
# These match YOUR project's existing config. Adjust if different.
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PROC   = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
CHARTS_DIR  = os.path.join(OUTPUT_DIR, "charts")
MODEL_DIR   = os.path.join(OUTPUT_DIR, "models")

# Processed file paths (written by Phase 1/2, read by Phase 3)
BALL_CSV    = os.path.join(DATA_PROC,  "ball_by_ball.csv")
SESSION_CSV = os.path.join(OUTPUT_DIR, "session_features.csv")
MATCH_CSV   = os.path.join(OUTPUT_DIR, "match_level_features.csv")

os.makedirs(DATA_PROC,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 BRIDGE: Convert your existing ball-by-ball DataFrame into the format
# expected by our SessionFeatureEngineer.
#
# WHY THIS IS NEEDED: Your load_data.py produces a DataFrame from YAML files.
# Our feature_engineering.py expects specific column names (e.g. 'total_runs',
# 'is_wicket', 'innings_num'). This bridge maps your columns → our columns
# without touching your existing src/ code.
# ─────────────────────────────────────────────────────────────────────────────

def bridge_to_feature_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map your existing ball_by_ball DataFrame columns to the schema expected
    by SessionFeatureEngineer.

    EDIT THIS FUNCTION if your column names differ — run `print(df.columns.tolist())`
    to see your exact column names and map them below.
    """
    col_map = {}

    # ── Detect and remap common column name variants ──────────────────
    # Runs
    for candidate in ["runs_off_bat", "batter_runs", "runs"]:
        if candidate in df.columns:
            col_map[candidate] = "batter_runs"
            break

    for candidate in ["total_runs", "runs_total", "run_total"]:
        if candidate in df.columns:
            col_map[candidate] = "total_runs"
            break

    for candidate in ["extras", "extra_runs"]:
        if candidate in df.columns:
            col_map[candidate] = "extras_runs"
            break

    # Wickets
    for candidate in ["wicket", "is_wicket", "wicket_fell"]:
        if candidate in df.columns:
            col_map[candidate] = "is_wicket"
            break

    # Innings
    for candidate in ["innings", "innings_num", "inning"]:
        if candidate in df.columns:
            col_map[candidate] = "innings_num"
            break

    # Over
    for candidate in ["over", "over_num", "over_number"]:
        if candidate in df.columns:
            col_map[candidate] = "over_num"
            break

    # Ball
    for candidate in ["ball", "ball_num", "delivery_num", "ball_in_over"]:
        if candidate in df.columns:
            col_map[candidate] = "ball_in_over"
            break

    # Teams
    for candidate in ["batting_team", "bat_team", "batting"]:
        if candidate in df.columns:
            col_map[candidate] = "batting_team"
            break

    for candidate in ["bowling_team", "bowl_team", "fielding_team", "fielding"]:
        if candidate in df.columns:
            col_map[candidate] = "fielding_team"
            break

    # Match ID
    for candidate in ["match_id", "matchid", "match"]:
        if candidate in df.columns:
            col_map[candidate] = "match_id"
            break

    df = df.rename(columns=col_map)

    # ── Synthesize missing columns with safe defaults ─────────────────

    # total_runs = batter_runs + extras if not already present
    if "total_runs" not in df.columns:
        batter = df.get("batter_runs", pd.Series(0, index=df.index))
        extras = df.get("extras_runs",  pd.Series(0, index=df.index))
        df["total_runs"] = batter + extras

    if "extras_runs" not in df.columns:
        df["extras_runs"] = 0

    if "batter_runs" not in df.columns:
        df["batter_runs"] = df["total_runs"]

    # Wicket flag — coerce to int
    if "is_wicket" not in df.columns:
        df["is_wicket"] = 0
    df["is_wicket"] = df["is_wicket"].fillna(0).astype(int)

    # Dot ball — 0 runs off the bat, no wide, no no-ball
    if "is_dot_ball" not in df.columns:
        is_wide   = df.get("is_wide",   pd.Series(0, index=df.index)).fillna(0).astype(int)
        is_noball = df.get("is_noball", pd.Series(0, index=df.index)).fillna(0).astype(int)
        df["is_dot_ball"] = ((df["batter_runs"] == 0) & (is_wide == 0) & (is_noball == 0)).astype(int)

    # Boundary flags
    if "is_boundary_4" not in df.columns:
        df["is_boundary_4"] = (df["batter_runs"] == 4).astype(int)
    if "is_boundary_6" not in df.columns:
        df["is_boundary_6"] = (df["batter_runs"] == 6).astype(int)

    # Extras breakdown
    for col in ["is_wide", "is_noball", "is_bye", "is_legbye"]:
        if col not in df.columns:
            df[col] = 0

    # Innings number — default to 1 if missing
    if "innings_num" not in df.columns:
        df["innings_num"] = 1
    df["innings_num"] = df["innings_num"].fillna(1).astype(int)

    # Over number
    if "over_num" not in df.columns:
        df["over_num"] = 0

    # Legal ball number within innings (absolute ball counter)
    if "legal_ball_num" not in df.columns:
        df["legal_ball_num"] = df.groupby(
            ["match_id", "innings_num"]
        ).cumcount() + 1

    # Team names
    if "batting_team"  not in df.columns: df["batting_team"]  = "Team_A"
    if "fielding_team" not in df.columns: df["fielding_team"] = "Team_B"

    # Context defaults
    if "home_team"       not in df.columns: df["home_team"]       = "Unknown"
    if "is_home_batting" not in df.columns: df["is_home_batting"] = 0
    if "toss_winner"     not in df.columns: df["toss_winner"]     = "Unknown"
    if "toss_decision"   not in df.columns: df["toss_decision"]   = "bat"
    if "winner"          not in df.columns: df["winner"]          = "Unknown"
    if "result"          not in df.columns: df["result"]          = "normal"

    if "batting_team_won" not in df.columns:
        df["batting_team_won"] = (df["batting_team"] == df["winner"]).astype(int)

    # Session label — assign if missing
    if "session" not in df.columns:
        def label_session(row):
            s = int(row["over_num"]) // 30
            names = ["morning", "afternoon", "evening"]
            return f"inn{int(row['innings_num'])}_{names[min(s,2)]}"
        df["session"] = df.apply(label_session, axis=1)

    # Venue defaults
    for col in ["venue", "city", "match_date"]:
        if col not in df.columns:
            df[col] = "Unknown"

    print(f"\n[Bridge] Schema mapped. Final columns: {sorted(df.columns.tolist())}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────────────────────────────────────

def step_load(args) -> pd.DataFrame:
    """Step 1: Load raw data using your existing src/load_data.py."""
    from src.config import DATA_PROC as SRC_PROC

    if args.charts_only or args.skip_charts:
        # Use existing processed CSV if available
        if os.path.exists(BALL_CSV):
            print(f"📂 Loading from existing CSV: {BALL_CSV}")
            df = pd.read_csv(BALL_CSV)
            print(f"✅ Loaded {len(df):,} rows.")
            return df

    from src.load_data import load_all_matches, save_processed
    df = load_all_matches(max_files=args.sample)
    save_processed(df)
    return df


def step_visualize(df: pd.DataFrame):
    """Step 2: Generate visualizations (your existing code)."""
    from src.visualize import run_all_charts
    run_all_charts(df)


def step_feature_engineering(df: pd.DataFrame):
    """
    Step 3: Run SessionFeatureEngineer on the loaded ball-by-ball DataFrame.
    Outputs: session_features.csv, match_level_features.csv
    """
    print("\n" + "="*60)
    print("⚙️  PHASE 1 & 2 — Feature Engineering + Win Probability")
    print("="*60)

    # Import our feature engineering module
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    from feature_engineering import (
        SessionFeatureEngineer,
        WinProbabilityComputer,
        MatchOutcomeAggregator,
    )

    # Bridge column names from your schema → our schema
    df_mapped = bridge_to_feature_schema(df.copy())

    # 1. Engineer session features
    engineer   = SessionFeatureEngineer()
    session_df = engineer.engineer_session_features(df_mapped)

    # 2. Compute Win Probability (logistic baseline)
    wp = WinProbabilityComputer()

    # Need at least 2 classes to fit — check before fitting
    unique_outcomes = session_df["batting_team_won"].nunique()
    if unique_outcomes < 2:
        print("⚠️  Only one outcome class in this sample — skipping WP model fit.")
        print("   Tip: Use --sample 50+ or run on the full dataset for WP training.")
        session_df["win_probability"]     = 0.5
        session_df["wp_shift"]            = 0.0
        session_df["momentum_label"]      = 0
        session_df["prev_wp"]             = 0.5
    else:
        wp.fit(session_df)
        session_df = wp.predict_wp(session_df)
        wp.save(MODEL_DIR)

    # 3. Aggregate to match level
    aggregator = MatchOutcomeAggregator()
    match_df   = aggregator.aggregate_to_match_level(session_df)

    # 4. Save
    session_df.to_csv(SESSION_CSV, index=False)
    match_df.to_csv(  MATCH_CSV,   index=False)

    print(f"\n✅ Feature engineering complete.")
    print(f"   session_features.csv    → {len(session_df):,} session rows")
    print(f"   match_level_features.csv→ {len(match_df):,} match rows")

    return session_df, match_df


def step_modeling(session_df: pd.DataFrame, match_df: pd.DataFrame):
    """
    Step 4: Train XGBoost + RF models and compute SHAP values.
    Requires enough samples — skips gracefully on tiny datasets.
    """
    print("\n" + "="*60)
    print("🤖  PHASE 3 — XGBoost Modeling + SHAP Interpretability")
    print("="*60)

    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    from modeling import SessionMomentumModel, MatchOutcomeModel, EvaluationReporter

    MIN_SESSIONS_FOR_MODELING = 30   # Need at least 30 sessions for 5-fold CV

    session_metrics = {}
    match_metrics   = {}

    # ── A. Session Momentum Model ─────────────────────────────────────
    valid_sessions = session_df.dropna(subset=["momentum_label"])
    if len(valid_sessions) < MIN_SESSIONS_FOR_MODELING:
        print(f"\n⚠️  Only {len(valid_sessions)} labeled sessions found.")
        print(f"   Need {MIN_SESSIONS_FOR_MODELING}+ for modeling. Use --sample 50+")
    else:
        session_model  = SessionMomentumModel(output_dir=MODEL_DIR)
        session_metrics = session_model.train_and_evaluate(session_df)

        print("\n🔍 Computing SHAP values (this may take ~30s)...")
        session_model.compute_shap(session_df)
        print(f"   SHAP plots saved → {MODEL_DIR}/")

    # ── B. Match Outcome Model ────────────────────────────────────────
    decidable_matches = match_df[match_df["result"] == "normal"]
    if len(decidable_matches) < 10:
        print(f"\n⚠️  Only {len(decidable_matches)} decided matches — skipping match model.")
        print(f"   Use --sample 100+ for the match outcome model.")
    else:
        match_model   = MatchOutcomeModel(output_dir=MODEL_DIR)
        match_metrics = match_model.train_and_evaluate(match_df)

    # ── C. Evaluation Report ──────────────────────────────────────────
    if session_metrics or match_metrics:
        EvaluationReporter.generate_report(
            session_metrics, match_metrics, MODEL_DIR
        )

    return session_metrics, match_metrics


def step_print_summary(session_df: pd.DataFrame, match_df: pd.DataFrame):
    """Print a readable session + match summary to terminal."""
    print("\n" + "="*60)
    print("📋  SESSION MOMENTUM SUMMARY (last 10 sessions)")
    print("="*60)

    display_cols = [
        c for c in [
            "match_id", "innings_num", "session",
            "session_run_rate", "session_wickets",
            "session_momentum_index", "win_probability",
            "wp_shift", "momentum_label"
        ] if c in session_df.columns
    ]
    print(session_df[display_cols].tail(10).to_string(index=False))

    print("\n" + "="*60)
    print("🏆  MATCH OUTCOME SUMMARY (all matches)")
    print("="*60)

    match_display = [
        c for c in [
            "match_id", "winner", "home_team",
            "final_wp", "wp_volatility",
            "momentum_reversals", "avg_run_rate"
        ] if c in match_df.columns
    ]
    print(match_df[match_display].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cricket Momentum System")
    parser.add_argument("--sample",      type=int,  default=None,
                        help="Load only N files (for quick testing)")
    parser.add_argument("--charts-only", action="store_true",
                        help="Skip loading, use existing CSV, run charts only")
    parser.add_argument("--skip-charts", action="store_true",
                        help="Skip chart generation, go straight to modeling")
    parser.add_argument("--skip-model",  action="store_true",
                        help="Run feature engineering but skip model training")
    args = parser.parse_args()

    print("\n🏏 Cricket Momentum — Full Pipeline")
    print(f"   Sample size : {args.sample or 'ALL'}")
    print(f"   Charts      : {'SKIP' if args.skip_charts else 'YES'}")
    print(f"   Modeling    : {'SKIP' if args.skip_model  else 'YES'}")

    # ── Step 1: Load data ─────────────────────────────────────────────
    df = step_load(args)

    # ── Step 2: Visualizations (your existing charts) ─────────────────
    if not args.skip_charts:
        print("\n🏏 Generating visualizations...")
        step_visualize(df)

    # ── Step 3: Feature Engineering + Win Probability ─────────────────
    session_df, match_df = step_feature_engineering(df)

    # Print readable summary to terminal
    step_print_summary(session_df, match_df)

    # ── Step 4: Modeling + SHAP ───────────────────────────────────────
    if not args.skip_model:
        step_modeling(session_df, match_df)

    # ── Done ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("🏁 Pipeline complete! Outputs:")
    print(f"   📊 Charts   → {CHARTS_DIR}/")
    print(f"   📄 Sessions → {SESSION_CSV}")
    print(f"   📄 Matches  → {MATCH_CSV}")
    print(f"   🤖 Models   → {MODEL_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()