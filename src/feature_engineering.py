"""
=============================================================================
SESSION-BASED MOMENTUM PREDICTION IN TEST CRICKET
Phase 1 & 2: Feature Engineering + Win Probability Computation
=============================================================================
Author      : [Your Name] — Master's Thesis
Data Source : Cricsheet ball-by-ball JSON/CSV
Description : This module engineers session-level momentum features from
              raw Cricsheet data and computes a proxy Win Probability (WP)
              at the end of each session using a logistic regression baseline.
              All features are documented with their statistical/cricket rationale.
=============================================================================
"""

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS — Test cricket domain knowledge encoded here
# ---------------------------------------------------------------------------

# Sessions in Test cricket: each day has 3 sessions
SESSION_MAP = {
    "morning": (0, 30),    # overs 0–29 approximate
    "afternoon": (30, 60),
    "evening": (60, 90),
}

# Batting position thresholds — determines "top-order impact"
TOP_ORDER_POSITIONS     = [1, 2, 3, 4]   # Openers + top middle-order
MIDDLE_ORDER_POSITIONS  = [5, 6, 7]
TAIL_POSITIONS          = [8, 9, 10, 11]

# Ball age thresholds — new ball swings; old ball reverse-swings
NEW_BALL_THRESHOLD      = 20   # first 20 overs (120 balls)
REVERSE_SWING_THRESHOLD = 150  # reverse swing kicks in around ball 150+

# Pressure escalation: consecutive dot balls trigger field pressure
DOT_BALL_PRESSURE_THRESHOLD = 4   # 4+ consecutive dots = pressure building
DANGER_DOT_STREAK           = 8   # 8+ consecutive dots = crisis

# Win probability clipping: avoid extreme predictions in early overs
WP_CLIP_LOW  = 0.05
WP_CLIP_HIGH = 0.95


# =============================================================================
# MODULE 1: CRICSHEET DATA LOADER
# =============================================================================

class CricsheetLoader:
    """
    Loads and parses Cricsheet JSON match files into a unified ball-by-ball
    DataFrame. Handles both the older 'overs' structure and the newer
    'innings' → 'overs' → 'deliveries' nested format.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_all_matches(self, match_type: str = "test") -> pd.DataFrame:
        """
        Iterates all JSON files in data_dir, filters by match_type,
        and concatenates into one master DataFrame.

        Parameters
        ----------
        match_type : str, default 'test'
            Filter string matched against 'info.match_type' in Cricsheet JSON.

        Returns
        -------
        pd.DataFrame : ball-by-ball records across all matches.
        """
        all_records = []
        json_files = list(self.data_dir.glob("*.json"))
        print(f"[Loader] Found {len(json_files)} JSON files in {self.data_dir}")

        for filepath in json_files:
            try:
                with open(filepath, "r") as f:
                    raw = json.load(f)

                # Filter: only process Test matches
                if raw.get("info", {}).get("match_type", "").lower() != match_type:
                    continue

                records = self._parse_match(raw, filepath.stem)
                all_records.extend(records)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"[Warning] Skipping {filepath.name}: {e}")
                continue

        df = pd.DataFrame(all_records)
        print(f"[Loader] Loaded {len(df):,} deliveries from {df['match_id'].nunique()} Test matches.")
        return df

    def _parse_match(self, raw: dict, match_id: str) -> list:
        """Parse a single Cricsheet JSON match into a list of delivery records."""
        info        = raw.get("info", {})
        teams       = info.get("teams", ["TeamA", "TeamB"])
        venue       = info.get("venue", "Unknown")
        city        = info.get("city", "Unknown")
        match_dates = info.get("dates", ["1900-01-01"])
        toss_winner = info.get("toss", {}).get("winner", "Unknown")
        toss_decision = info.get("toss", {}).get("decision", "Unknown")

        # Outcome
        outcome      = info.get("outcome", {})
        winner       = outcome.get("winner", "draw")
        result       = outcome.get("result", "normal")  # 'draw', 'tie', 'no result'

        records = []

        for innings_idx, innings in enumerate(raw.get("innings", [])):
            batting_team  = innings.get("team", "Unknown")
            fielding_team = [t for t in teams if t != batting_team]
            fielding_team = fielding_team[0] if fielding_team else "Unknown"

            # Determine innings number (1st, 2nd, 3rd, 4th)
            innings_num = innings_idx + 1

            for over_data in innings.get("overs", []):
                over_num = over_data.get("over", 0)  # 0-indexed in Cricsheet

                for ball_idx, delivery in enumerate(over_data.get("deliveries", [])):
                    record = self._parse_delivery(
                        delivery, match_id, innings_num,
                        batting_team, fielding_team,
                        over_num, ball_idx,
                        venue, city, match_dates[0],
                        toss_winner, toss_decision,
                        winner, result, teams
                    )
                    records.append(record)

        return records

    def _parse_delivery(self, delivery: dict, match_id: str, innings_num: int,
                        batting_team: str, fielding_team: str,
                        over_num: int, ball_idx: int,
                        venue: str, city: str, match_date: str,
                        toss_winner: str, toss_decision: str,
                        winner: str, result: str, teams: list) -> dict:
        """Extract all fields from a single delivery dict."""

        runs_data    = delivery.get("runs", {})
        batter_runs  = runs_data.get("batter", 0)
        extras_runs  = runs_data.get("extras", 0)
        total_runs   = runs_data.get("total", 0)

        # Wicket info
        wickets      = delivery.get("wickets", [])
        is_wicket    = 1 if wickets else 0
        wicket_type  = wickets[0].get("kind", None) if wickets else None
        dismissed_batter = wickets[0].get("player_out", None) if wickets else None

        # Extras breakdown
        extras_detail = delivery.get("extras", {})
        is_wide  = 1 if "wides"   in extras_detail else 0
        is_noball= 1 if "noballs" in extras_detail else 0
        is_bye   = 1 if "byes"    in extras_detail else 0
        is_legbye= 1 if "legbyes" in extras_detail else 0

        # Ball number within innings (absolute)
        # Each over = 6 legal deliveries; extras don't advance the over counter
        legal_ball_num = over_num * 6 + ball_idx + 1

        # Home ground advantage
        home_team = self._infer_home_team(teams, venue, city)

        # Session label based on over number
        session = self._assign_session(over_num, innings_num)

        return {
            "match_id"          : match_id,
            "match_date"        : match_date,
            "venue"             : venue,
            "city"              : city,
            "innings_num"       : innings_num,
            "batting_team"      : batting_team,
            "fielding_team"     : fielding_team,
            "home_team"         : home_team,
            "is_home_batting"   : int(batting_team == home_team),
            "over_num"          : over_num,
            "ball_in_over"      : ball_idx + 1,
            "legal_ball_num"    : legal_ball_num,
            "session"           : session,
            "batter_runs"       : batter_runs,
            "extras_runs"       : extras_runs,
            "total_runs"        : total_runs,
            "is_dot_ball"       : int(batter_runs == 0 and is_wide == 0 and is_noball == 0),
            "is_boundary_4"     : int(batter_runs == 4),
            "is_boundary_6"     : int(batter_runs == 6),
            "is_wide"           : is_wide,
            "is_noball"         : is_noball,
            "is_bye"            : is_bye,
            "is_legbye"         : is_legbye,
            "is_wicket"         : is_wicket,
            "wicket_type"       : wicket_type,
            "dismissed_batter"  : dismissed_batter,
            "toss_winner"       : toss_winner,
            "toss_decision"     : toss_decision,
            "winner"            : winner,
            "result"            : result,
            # Label: did the batting team win?
            "batting_team_won"  : int(batting_team == winner),
        }

    @staticmethod
    def _infer_home_team(teams: list, venue: str, city: str) -> str:
        """
        Heuristic: map venue/city to home team.
        In production, replace with a lookup CSV of ground–country mappings.
        """
        venue_lower = (venue + " " + city).lower()
        country_keywords = {
            "india"      : ["eden gardens", "wankhede", "chepauk", "mohali", "mumbai",
                           "kolkata", "delhi", "bangalore", "hyderabad", "nagpur", "pune"],
            "australia"  : ["mcg", "scg", "waca", "gabba", "adelaide", "perth",
                           "melbourne", "sydney", "brisbane"],
            "england"    : ["lords", "headingley", "edgbaston", "the oval", "old trafford",
                           "trent bridge", "london", "leeds", "birmingham", "manchester"],
            "pakistan"   : ["karachi", "lahore", "rawalpindi", "multan", "faisalabad"],
            "south africa": ["newlands", "wanderers", "centurion", "cape town",
                             "johannesburg", "durban", "port elizabeth"],
            "new zealand": ["eden park", "basin reserve", "hagley", "auckland",
                            "wellington", "christchurch"],
            "west indies": ["kensington", "sabina", "queen's park", "barbados",
                            "bridgetown", "kingston"],
            "sri lanka"  : ["galle", "colombo", "kandy", "pallekele", "p sara"],
            "bangladesh" : ["mirpur", "chittagong", "dhaka", "shere bangla"],
            "zimbabwe"   : ["harare", "bulawayo", "queens sports"],
        }

        for country, keywords in country_keywords.items():
            if any(kw in venue_lower for kw in keywords):
                for team in teams:
                    if country in team.lower():
                        return team
        return "Unknown"

    @staticmethod
    def _assign_session(over_num: int, innings_num: int) -> str:
        """
        Assign session label. In Test cricket, each day has 3 sessions of
        ~30 overs each. Session labeling also encodes innings context.
        """
        session_in_day = over_num // 30  # 0 = morning, 1 = afternoon, 2 = evening
        session_names  = ["morning", "afternoon", "evening"]
        session_label  = session_names[min(session_in_day, 2)]
        return f"inn{innings_num}_{session_label}"


# =============================================================================
# MODULE 2: SESSION-LEVEL FEATURE ENGINEERING
# =============================================================================

class SessionFeatureEngineer:
    """
    Aggregates ball-by-ball data into session-level feature vectors.
    Each row = one session within one innings of one match.

    Statistical Rationale for Each Feature Group:
    -----------------------------------------------
    1. SCORING RATE FEATURES: RPO (runs per over) and boundary rate capture
       the *pace* of batting dominance within a session.

    2. WICKET IMPACT FEATURES: Wickets taken are not equal — losing a top-order
       batsman collapses the batting lineup disproportionately. We weight wickets
       by batting position to compute a 'weighted wicket impact' score.

    3. PRESSURE FEATURES: Consecutive dot balls create psychological and tactical
       pressure. A dot-ball streak of 8+ is a 'crisis' indicator for batting teams.

    4. BALL AGE FEATURES: The age of the ball governs swing conditions. New-ball
       overs (1–20) and reverse-swing zones (80+) have asymmetric risk profiles.

    5. MOMENTUM SHIFT DELTA: Change in run rate and wickets between consecutive
       sessions — captures momentum *direction*, not just magnitude.

    6. CONTEXTUAL FEATURES: Home ground advantage, toss outcome, innings number —
       encode match context that influences win probability independently of play.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def engineer_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline: ball-by-ball DataFrame → session-level feature DataFrame.
        """
        print("[Feature Engineer] Computing session-level aggregations...")

        # Step 1: Compute cumulative innings state (runs, wickets) at each delivery
        df = self._compute_innings_state(df)

        # Step 2: Compute consecutive dot ball streaks (pressure metric)
        df = self._compute_dot_ball_streaks(df)

        # Step 3: Aggregate to session level
        session_df = self._aggregate_by_session(df)

        # Step 4: Compute momentum delta (session-over-session change)
        session_df = self._compute_momentum_delta(session_df)

        # Step 5: Encode contextual features
        session_df = self._encode_context(session_df)

        print(f"[Feature Engineer] Generated {len(session_df):,} session records "
              f"with {len(session_df.columns)} features.")
        return session_df

    # ------------------------------------------------------------------
    # STEP 1: Cumulative Innings State
    # ------------------------------------------------------------------

    def _compute_innings_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each delivery, compute the cumulative state of the innings:
        - cumulative runs scored
        - cumulative wickets fallen
        - current run rate (runs / overs)
        - required run rate (only relevant in 4th innings)
        """
        group_keys = ["match_id", "innings_num", "batting_team"]

        df = df.sort_values(["match_id", "innings_num", "legal_ball_num"])

        # Cumulative sums within each innings
        df["cum_runs"]    = df.groupby(group_keys)["total_runs"].cumsum()
        df["cum_wickets"] = df.groupby(group_keys)["is_wicket"].cumsum()

        # Run rate = runs / overs completed (avoid divide-by-zero)
        df["overs_completed"] = df["legal_ball_num"] / 6.0
        df["current_run_rate"] = np.where(
            df["overs_completed"] > 0,
            df["cum_runs"] / df["overs_completed"],
            0.0
        )

        return df

    # ------------------------------------------------------------------
    # STEP 2: Dot Ball Streak (Pressure Metric)
    # ------------------------------------------------------------------

    def _compute_dot_ball_streaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute consecutive dot ball streaks at each delivery.

        Statistical Logic: A sequence of n consecutive dot balls represents
        escalating psychological pressure on the batting side. We model this
        as a geometric pressure accumulation — the longer the streak, the
        higher the probability of a reckless shot or bowling breakthrough.
        This is analogous to a 'runs of successes' in binary sequence analysis.
        """
        df = df.sort_values(["match_id", "innings_num", "legal_ball_num"])

        def streak_count(series: pd.Series) -> pd.Series:
            """Vectorized consecutive dot ball streak counter."""
            streaks = []
            current_streak = 0
            for val in series:
                if val == 1:
                    current_streak += 1
                else:
                    current_streak = 0
                streaks.append(current_streak)
            return pd.Series(streaks, index=series.index)

        df["dot_streak"] = (
            df.groupby(["match_id", "innings_num"])["is_dot_ball"]
              .transform(streak_count)
        )

        # Binary flag: is the batting side currently in a pressure crisis?
        df["is_pressure_crisis"] = (df["dot_streak"] >= DANGER_DOT_STREAK).astype(int)

        return df

    # ------------------------------------------------------------------
    # STEP 3: Aggregate to Session Level
    # ------------------------------------------------------------------

    def _aggregate_by_session(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapse ball-by-ball data into one row per session per innings.
        Each session row captures the *net state and flow* of that session.
        """
        group_keys = ["match_id", "innings_num", "batting_team",
                      "fielding_team", "session", "home_team",
                      "toss_winner", "toss_decision", "winner", "result",
                      "batting_team_won"]

        agg = df.groupby(group_keys).agg(

            # --- Scoring features ---
            session_runs        = ("total_runs",       "sum"),
            session_balls       = ("is_dot_ball",      "count"),   # total legal balls
            session_dot_balls   = ("is_dot_ball",      "sum"),
            session_fours       = ("is_boundary_4",    "sum"),
            session_sixes       = ("is_boundary_6",    "sum"),
            session_extras      = ("extras_runs",      "sum"),
            session_wides       = ("is_wide",          "sum"),
            session_noballs     = ("is_noball",        "sum"),

            # --- Wicket features ---
            session_wickets     = ("is_wicket",        "sum"),

            # --- Pressure features ---
            max_dot_streak      = ("dot_streak",       "max"),
            total_pressure_balls= ("is_pressure_crisis","sum"),

            # --- Ball age (use first ball num of session as proxy for ball age) ---
            ball_age_start      = ("legal_ball_num",   "min"),
            ball_age_end        = ("legal_ball_num",   "max"),

            # --- Innings state at session END (last value = end-of-session state) ---
            runs_at_session_end     = ("cum_runs",         "last"),
            wickets_at_session_end  = ("cum_wickets",      "last"),
            run_rate_at_session_end = ("current_run_rate", "last"),

            # --- Home ground context ---
            is_home_batting     = ("is_home_batting",  "first"),

        ).reset_index()

        # --- Derived rate features ---
        # Run rate within this session only (session runs / session overs)
        agg["session_run_rate"] = np.where(
            agg["session_balls"] > 0,
            agg["session_runs"] / (agg["session_balls"] / 6.0),
            0.0
        )

        # Dot ball percentage — measures batting freedom
        agg["dot_ball_pct"] = np.where(
            agg["session_balls"] > 0,
            agg["session_dot_balls"] / agg["session_balls"],
            0.0
        )

        # Boundary rate — aggressive scoring signal
        agg["boundary_rate"] = np.where(
            agg["session_balls"] > 0,
            (agg["session_fours"] + agg["session_sixes"]) / agg["session_balls"],
            0.0
        )

        # Wickets per over — bowling efficiency within session
        agg["wickets_per_over"] = np.where(
            agg["session_balls"] >= 6,
            agg["session_wickets"] / (agg["session_balls"] / 6.0),
            agg["session_wickets"]
        )

        # Ball age indicator: 'new_ball_phase', 'mid_innings', 'reverse_swing_zone'
        agg["ball_age_zone"] = pd.cut(
        agg["ball_age_start"],
        bins=[0, NEW_BALL_THRESHOLD * 6, REVERSE_SWING_THRESHOLD, 9999],
        labels=["new_ball", "mid_innings", "reverse_swing"],
        duplicates="drop"          # ← this is the fix
        ).astype(str).replace("nan", "mid_innings")

        # Wicket weight: top-order wickets are more damaging than tail wickets.
        # We approximate this using innings_wickets_count (proxy for batting position).
        # Refined approach: join with player batting order data from Cricsheet roster.
        agg["top_order_exposed"] = (agg["wickets_at_session_end"] <= 4).astype(int)

        return agg

    # ------------------------------------------------------------------
    # STEP 4: Momentum Delta (Session-over-Session Change)
    # ------------------------------------------------------------------

    def _compute_momentum_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the change in run rate and wickets between consecutive sessions
        within the same innings. This 'delta' captures DIRECTION of momentum —
        a team accelerating from 3.0 to 5.0 RPO is gaining momentum even if
        their absolute run rate is still modest.

        Statistical Note: These delta features are first-order finite differences
        in the session time-series. They serve as the primary 'momentum signal'
        in the subsequent XGBoost model.
        """
        df = df.sort_values(["match_id", "innings_num", "batting_team", "session"])

        inn_group = ["match_id", "innings_num", "batting_team"]

        df["prev_session_run_rate"] = df.groupby(inn_group)["session_run_rate"].shift(1)
        df["prev_session_wickets"]  = df.groupby(inn_group)["session_wickets"].shift(1)
        df["prev_dot_ball_pct"]     = df.groupby(inn_group)["dot_ball_pct"].shift(1)

        # Delta features (current − previous)
        df["run_rate_delta"]   = df["session_run_rate"] - df["prev_session_run_rate"].fillna(0)
        df["wickets_delta"]    = df["session_wickets"]  - df["prev_session_wickets"].fillna(0)
        df["dot_ball_pct_delta"] = df["dot_ball_pct"]  - df["prev_dot_ball_pct"].fillna(0)

        # Momentum index: run_rate_delta − wickets_delta (scaled)
        # Positive = batting momentum gaining; Negative = bowling momentum gaining
        df["session_momentum_index"] = df["run_rate_delta"] - (df["wickets_delta"] * 2.5)

        return df

    # ------------------------------------------------------------------
    # STEP 5: Encode Contextual Features
    # ------------------------------------------------------------------

    def _encode_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical context variables as numeric flags."""

        # Toss impact: batting first vs. fielding first
        df["toss_bat_first"] = (df["toss_decision"] == "bat").astype(int)

        # Did the toss winner bat in THIS innings?
        df["toss_winner_batting"] = (df["batting_team"] == df["toss_winner"]).astype(int)

        # Innings number (1st = most runs context; 4th = highest pressure)
        df["is_fourth_innings"] = (df["innings_num"] == 4).astype(int)
        df["is_first_innings"]  = (df["innings_num"] == 1).astype(int)

        # Session time-of-day encoding (morning = freshest pitch/ball)
        df["is_morning_session"] = df["session"].str.contains("morning").astype(int)
        df["is_evening_session"] = df["session"].str.contains("evening").astype(int)

        return df


# =============================================================================
# MODULE 3: WIN PROBABILITY COMPUTATION (LOGISTIC BASELINE)
# =============================================================================

class WinProbabilityComputer:
    """
    Computes a session-level Win Probability (WP) proxy using logistic regression.

    Methodology:
    -----------
    We train a logistic regression on historical sessions with known match outcomes.
    The output probability P(batting_team_wins | session_state) serves as our
    Win Probability estimate.

    This is intentionally a *baseline* WP — the XGBoost model in Phase 3 will
    replace/augment this with a more powerful estimator. For the thesis, the
    logistic baseline provides an interpretable benchmark (AUC, coefficients).

    Feature Set for WP:
    - Current run rate
    - Wickets in hand (10 - wickets fallen)
    - Session momentum index
    - Innings number
    - Home advantage
    - Dot ball percentage
    """

    # Features used to estimate win probability at session level
    WP_FEATURES = [
        "run_rate_at_session_end",
        "wickets_at_session_end",
        "session_run_rate",
        "session_wickets",
        "dot_ball_pct",
        "boundary_rate",
        "max_dot_streak",
        "session_momentum_index",
        "run_rate_delta",
        "wickets_delta",
        "is_home_batting",
        "toss_bat_first",
        "toss_winner_batting",
        "innings_num",
        "is_fourth_innings",
        "is_first_innings",
        "is_morning_session",
    ]

    def __init__(self):
        self.model  = LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, session_df: pd.DataFrame) -> "WinProbabilityComputer":
        """
        Fit the logistic WP model on historical session data.
        Only uses sessions where the match result is known (excludes draws for binary WP).
        """
        # Exclude draws for binary WP model (batting_team_won = 0 can be loss or draw)
        train_df = session_df[session_df["result"] != "draw"].copy()

        X = train_df[self.WP_FEATURES].fillna(0)
        y = train_df["batting_team_won"]

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        # Cross-validated AUC for thesis reporting
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring="roc_auc")
        print(f"\n[WP Model] Logistic Regression Baseline (5-fold CV AUC):")
        print(f"  Mean AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        self.is_fitted = True
        return self

    def predict_wp(self, session_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'win_probability' column to session_df.
        WP is clipped to [WP_CLIP_LOW, WP_CLIP_HIGH] to avoid overconfident early predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("Call .fit() before .predict_wp()")

        X = session_df[self.WP_FEATURES].fillna(0)
        X_scaled = self.scaler.transform(X)
        raw_probs = self.model.predict_proba(X_scaled)[:, 1]

        # Clip extreme probabilities — early innings WP is inherently uncertain
        session_df = session_df.copy()
        session_df["win_probability"] = np.clip(raw_probs, WP_CLIP_LOW, WP_CLIP_HIGH)

        # WP shift (delta WP between sessions) — THIS IS THE TARGET for the XGBoost model
        inn_group = ["match_id", "innings_num", "batting_team"]
        session_df["prev_wp"] = session_df.groupby(inn_group)["win_probability"].shift(1)
        session_df["wp_shift"] = session_df["win_probability"] - session_df["prev_wp"].fillna(0.5)

        # Label: momentum shift direction
        # wp_shift > +0.05 → batting team gaining momentum (label = 1)
        # wp_shift < −0.05 → bowling team gaining momentum (label = -1)
        # else             → neutral (label = 0)
        session_df["momentum_label"] = pd.cut(
            session_df["wp_shift"],
            bins=[-1.0, -0.05, 0.05, 1.0],
            labels=[-1, 0, 1]
        ).astype(float)

        print(f"\n[WP Model] Momentum label distribution:")
        print(session_df["momentum_label"].value_counts().to_string())

        return session_df

    def save(self, output_dir: str):
        """Persist scaler + model for reproducibility (thesis appendix)."""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.model,  os.path.join(output_dir, "wp_logistic_model.pkl"))
        joblib.dump(self.scaler, os.path.join(output_dir, "wp_scaler.pkl"))
        print(f"[WP Model] Saved to {output_dir}/")


# =============================================================================
# MODULE 4: MATCH-LEVEL OUTCOME AGGREGATOR
# =============================================================================

class MatchOutcomeAggregator:
    """
    Produces match-level summary features from session data.
    These are used for the OVERALL MATCH WIN PREDICTION (bonus objective).

    Aggregation Strategy:
    - For each match, take the WP and momentum index at the END of each innings.
    - The final innings WP (end of 4th innings or last completed innings) is
      our strongest predictor of match outcome.
    - We also compute session-level momentum volatility (std dev of WP across sessions)
      as a measure of how 'closely contested' the match was.
    """

    def aggregate_to_match_level(self, session_df: pd.DataFrame) -> pd.DataFrame:
        """Produces one row per match with match-level features."""

        match_groups = []

        for match_id, match_data in session_df.groupby("match_id"):

            # Final WP reading (last session of the match)
            final_row = match_data.sort_values("innings_num").iloc[-1]
            final_wp  = final_row["win_probability"]

            # Momentum volatility: high std = closely contested match
            wp_volatility = match_data["win_probability"].std()
            momentum_vol  = match_data["session_momentum_index"].std()

            # Number of momentum reversals (sign changes in wp_shift)
            wp_shifts       = match_data["wp_shift"].dropna()
            momentum_reversals = (np.diff(np.sign(wp_shifts)) != 0).sum()

            # Peak batting momentum (max WP batting team achieved)
            peak_batting_wp = match_data["win_probability"].max()

            # Average session run rate across all innings
            avg_run_rate = match_data["session_run_rate"].mean()

            # Total wickets across match
            total_wickets = match_data["session_wickets"].sum()

            # Match context
            winner   = final_row["winner"]
            result   = final_row["result"]
            home_team = final_row["home_team"]

            match_groups.append({
                "match_id"           : match_id,
                "winner"             : winner,
                "result"             : result,
                "home_team"          : home_team,
                "final_wp"           : final_wp,
                "wp_volatility"      : wp_volatility,
                "momentum_volatility": momentum_vol,
                "momentum_reversals" : momentum_reversals,
                "peak_batting_wp"    : peak_batting_wp,
                "avg_run_rate"       : avg_run_rate,
                "total_wickets"      : total_wickets,
                "num_sessions"       : len(match_data),
            })

        match_df = pd.DataFrame(match_groups)
        print(f"\n[Match Aggregator] {len(match_df)} match summaries generated.")
        return match_df


# =============================================================================
# MAIN PIPELINE — Run this to process your Cricsheet data
# =============================================================================

def run_feature_engineering_pipeline(
    data_dir: str,
    output_dir: str = "outputs/"
) -> tuple:
    """
    Full Phase 1 & 2 pipeline:
    1. Load all Test match JSON files
    2. Engineer session features
    3. Compute Win Probability
    4. Aggregate to match level
    5. Save all outputs for Phase 3 (XGBoost modeling)

    Parameters
    ----------
    data_dir   : str — path to folder containing Cricsheet JSON files
    output_dir : str — path to save processed DataFrames and models

    Returns
    -------
    tuple : (ball_df, session_df, match_df)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load raw data ──────────────────────────────────────────────
    loader   = CricsheetLoader(data_dir)
    ball_df  = loader.load_all_matches(match_type="test")

    # ── 2. Feature Engineering ────────────────────────────────────────
    engineer   = SessionFeatureEngineer()
    session_df = engineer.engineer_session_features(ball_df)

    # ── 3. Win Probability (Logistic Baseline) ────────────────────────
    wp_computer = WinProbabilityComputer()
    wp_computer.fit(session_df)
    session_df = wp_computer.predict_wp(session_df)
    wp_computer.save(output_dir)

    # ── 4. Match-Level Aggregation ────────────────────────────────────
    aggregator = MatchOutcomeAggregator()
    match_df   = aggregator.aggregate_to_match_level(session_df)

    # ── 5. Save outputs ───────────────────────────────────────────────
    ball_df.to_csv(   os.path.join(output_dir, "ball_by_ball_processed.csv"),  index=False)
    session_df.to_csv(os.path.join(output_dir, "session_features.csv"),        index=False)
    match_df.to_csv(  os.path.join(output_dir, "match_level_features.csv"),    index=False)

    print(f"\n{'='*60}")
    print(f"✅ Phase 1 & 2 Complete. Outputs saved to: {output_dir}")
    print(f"   ball_by_ball_processed.csv  → {len(ball_df):,} rows")
    print(f"   session_features.csv        → {len(session_df):,} rows")
    print(f"   match_level_features.csv    → {len(match_df):,} rows")
    print(f"{'='*60}")

    return ball_df, session_df, match_df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    # ─── UPDATE THIS PATH to your Cricsheet data folder ───────────────────
    DATA_DIR   = sys.argv[1] if len(sys.argv) > 1 else "data/cricsheet_test/"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "outputs/"

    ball_df, session_df, match_df = run_feature_engineering_pipeline(DATA_DIR, OUTPUT_DIR)

    # Quick sanity checks
    print("\n── SESSION FEATURES SAMPLE ──────────────────────────────────────")
    print(session_df[["match_id", "innings_num", "session", "session_run_rate",
                       "session_wickets", "session_momentum_index",
                       "win_probability", "wp_shift", "momentum_label"]].head(10).to_string())

    print("\n── MATCH SUMMARY SAMPLE ─────────────────────────────────────────")
    print(match_df.head(5).to_string())