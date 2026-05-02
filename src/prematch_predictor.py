"""
prematch_predictor.py
=====================
Pre-match momentum prediction engine for UPCOMING Test matches.

Since no ball-by-ball data exists before a match starts, this module
predicts momentum by:

  1. Fetching historical session data from session_features.csv
  2. Computing head-to-head team profiles (avg session run rate, wicket rate,
     dot ball pressure, home/away splits) from past meetings
  3. Constructing a synthetic "expected session" feature vector for each of
     the 6 canonical sessions (Inn1-Morning ... Inn2-Evening)
  4. Running the trained XGBoost model on each synthetic vector
  5. Returning a full pre-match momentum forecast report

This is entirely model-driven — no live API calls needed.

Usage:
    python prematch_predictor.py --team1 "India" --team2 "Australia" --venue "MCG" --toss bat --home Australia
    python prematch_predictor.py --list-teams          # show available teams in your data
    python prematch_predictor.py --match-id <CricAPI ID>  # auto-fill from CricAPI
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
# Walk up the directory tree from this file's location until we find a folder
# that contains outputs/models/.  This handles being placed in src/, the project
# root, or any sub-directory without manual path editing.
def _find_project_root() -> Path:
    candidate = Path(__file__).resolve().parent
    for _ in range(4):                          # search up to 4 levels up
        if (candidate / "outputs" / "models").exists():
            return candidate
        if (candidate / "outputs").exists():    # outputs exists but models not yet
            return candidate
        candidate = candidate.parent
    # Final fallback: directory of this file
    return Path(__file__).resolve().parent

BASE_DIR    = _find_project_root()
OUTPUT_DIR  = BASE_DIR / "outputs"
MODEL_DIR   = OUTPUT_DIR / "models"
SESSION_CSV = OUTPUT_DIR / "session_features.csv"

# ── Feature list — MUST match modeling.py ─────────────────────────────────────
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

# ── Session catalogue ─────────────────────────────────────────────────────────
# Each entry: (innings_num, session_label, is_morning, is_evening, ball_age_start)
SESSION_CATALOGUE = [
    (1, "Morning",   True,  False, 1),
    (1, "Afternoon", False, False, 181),
    (1, "Evening",   False, True,  361),
    (2, "Morning",   True,  False, 1),
    (2, "Afternoon", False, False, 181),
    (2, "Evening",   False, True,  361),
]

LABEL_MAP = {1: "▲ Batting Momentum", 0: "→ Neutral", -1: "▼ Bowling Momentum"}
LABEL_COLOR = {1: "🟢", 0: "🟡", -1: "🔴"}


# =============================================================================
# TEAM PROFILER
# =============================================================================

class TeamProfiler:
    """
    Computes batting and bowling profiles for each team from historical
    session_features.csv data.

    Batting profile  — features when the team is batting
    Bowling profile  — features when the team is fielding (opponent batting)
    """

    def __init__(self, session_df: pd.DataFrame):
        self.df = session_df.copy()
        self._normalise_team_names()

    def _normalise_team_names(self):
        """Lowercase + strip for fuzzy matching."""
        for col in ["batting_team", "fielding_team"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.strip()

    def available_teams(self) -> list[str]:
        teams = set()
        for col in ["batting_team", "fielding_team"]:
            if col in self.df.columns:
                teams.update(self.df[col].dropna().unique())
        return sorted(teams)

    def _fuzzy_match(self, name: str) -> str:
        """Case-insensitive partial match against available teams."""
        available = self.available_teams()
        name_lower = name.lower()
        # Exact match first
        for t in available:
            if t.lower() == name_lower:
                return t
        # Partial match
        for t in available:
            if name_lower in t.lower() or t.lower() in name_lower:
                return t
        return name  # fallback — may not match anything

    def batting_profile(self, team: str, home_only: bool = False) -> dict:
        """
        Returns mean session stats when `team` is batting.
        home_only filters to home ground sessions only.
        """
        team_matched = self._fuzzy_match(team)
        mask = self.df["batting_team"] == team_matched
        if home_only and "is_home_batting" in self.df.columns:
            mask &= self.df["is_home_batting"] == 1
        sub = self.df[mask]

        if len(sub) < 3:
            # Not enough data — return global mean
            sub = self.df[self.df["batting_team"].str.contains(
                team.split()[0], case=False, na=False
            )]

        return self._agg_profile(sub, role="batting")

    def bowling_profile(self, team: str, home_only: bool = False) -> dict:
        """
        Returns mean session stats when `team` is FIELDING (opponent batting).
        A strong bowling profile = low opponent run rate, high wickets.
        """
        team_matched = self._fuzzy_match(team)
        mask = self.df["fielding_team"] == team_matched
        if home_only and "is_home_batting" in self.df.columns:
            mask &= self.df["is_home_batting"] == 0
        sub = self.df[mask]

        if len(sub) < 3:
            sub = self.df[self.df["fielding_team"].str.contains(
                team.split()[0], case=False, na=False
            )]

        return self._agg_profile(sub, role="bowling")

    def head_to_head_profile(self, team1: str, team2: str) -> dict:
        """
        Returns stats from historical meetings between team1 and team2.
        """
        t1 = self._fuzzy_match(team1)
        t2 = self._fuzzy_match(team2)

        mask = (
            ((self.df["batting_team"] == t1) & (self.df["fielding_team"] == t2)) |
            ((self.df["batting_team"] == t2) & (self.df["fielding_team"] == t1))
        )
        sub = self.df[mask]
        return {
            "n_sessions"    : len(sub),
            "n_matches"     : sub["match_id"].nunique() if "match_id" in sub.columns else 0,
            "team1_batting" : self._agg_profile(sub[sub["batting_team"] == t1], "batting"),
            "team2_batting" : self._agg_profile(sub[sub["batting_team"] == t2], "batting"),
        }

    @staticmethod
    def _agg_profile(sub: pd.DataFrame, role: str) -> dict:
        """Aggregate key features from a session subset."""
        if sub.empty:
            return {
                "session_run_rate"   : 3.0,
                "session_wickets"    : 2.0,
                "dot_ball_pct"       : 0.45,
                "boundary_rate"      : 0.08,
                "max_dot_streak"     : 5.0,
                "wickets_per_over"   : 0.33,
                "total_pressure_balls": 8.0,
                "session_runs"       : 55.0,
                "session_extras"     : 3.0,
                "n_sessions"         : 0,
            }

        numeric = sub.select_dtypes(include=[np.number])
        cols = [c for c in [
            "session_run_rate", "session_wickets", "dot_ball_pct",
            "boundary_rate", "max_dot_streak", "wickets_per_over",
            "total_pressure_balls", "session_runs", "session_extras",
        ] if c in numeric.columns]

        profile = numeric[cols].mean().to_dict()
        profile["n_sessions"] = len(sub)
        return profile


# =============================================================================
# SYNTHETIC FEATURE BUILDER
# =============================================================================

class PreMatchFeatureBuilder:
    """
    Constructs a synthetic 24-feature session vector for a session
    that has NOT yet been played, using:
      - Batting team's historical batting profile
      - Bowling team's historical bowling profile
      - Match context (home/away, toss, innings)
    
    Strategy: for each feature, take the AVERAGE of what the batting
    team historically produces and what the bowling team historically
    concedes. This gives a balanced prior for the expected session state.
    """

    def __init__(
        self,
        profiler: TeamProfiler,
        team1: str,
        team2: str,
        home_team: str,
        toss_winner: str,
        toss_decision: str,   # "bat" or "field"
    ):
        self.profiler       = profiler
        self.team1          = team1
        self.team2          = team2
        self.home_team      = home_team
        self.toss_winner    = toss_winner
        self.toss_decision  = toss_decision

        # Pre-compute batting / bowling profiles for both teams
        self.profiles = {
            team1: {
                "bat" : profiler.batting_profile(team1),
                "bowl": profiler.bowling_profile(team1),
            },
            team2: {
                "bat" : profiler.batting_profile(team2),
                "bowl": profiler.bowling_profile(team2),
            },
        }

        # Determine batting order from toss
        if toss_decision.lower() == "bat":
            self.batting_first  = toss_winner
            self.fielding_first = team2 if toss_winner == team1 else team1
        else:
            self.fielding_first = toss_winner
            self.batting_first  = team2 if toss_winner == team1 else team1

    def _batting_team_for_innings(self, innings_num: int) -> str:
        """Who bats in innings 1/2/3/4."""
        if innings_num in (1, 3):
            return self.batting_first
        return self.fielding_first

    def build_session(
        self,
        innings_num: int,
        session_label: str,
        is_morning: bool,
        is_evening: bool,
        ball_age_start: int,
        prev_features: Optional[dict] = None,
    ) -> dict:
        """
        Build one synthetic feature vector for a session.
        """
        batting_team  = self._batting_team_for_innings(innings_num)
        fielding_team = self.team2 if batting_team == self.team1 else self.team1

        bat_profile  = self.profiles[batting_team]["bat"]
        bowl_profile = self.profiles[fielding_team]["bowl"]

        # Blend: 50% from batting team history, 50% from bowling team history
        def blend(bat_key: str, bowl_key: str, default: float) -> float:
            b = bat_profile.get(bat_key, default)
            w = bowl_profile.get(bowl_key, default)
            return (b + w) / 2.0

        run_rate    = blend("session_run_rate",    "session_run_rate",    3.0)
        wickets     = blend("session_wickets",     "session_wickets",     2.0)
        dot_pct     = blend("dot_ball_pct",        "dot_ball_pct",        0.45)
        boundary_r  = blend("boundary_rate",       "boundary_rate",       0.08)
        max_dot     = blend("max_dot_streak",      "max_dot_streak",      5.0)
        wpo         = blend("wickets_per_over",    "wickets_per_over",    0.33)
        pressure_b  = blend("total_pressure_balls","total_pressure_balls",8.0)
        sess_runs   = blend("session_runs",        "session_runs",        55.0)
        sess_extras = blend("session_extras",      "session_extras",      3.0)

        # Contextual flags
        is_home_batting     = 1 if batting_team == self.home_team else 0
        toss_bat_first      = 1 if self.toss_decision.lower() == "bat" else 0
        toss_winner_batting = 1 if self.toss_winner == batting_team else 0

        # Wickets at session end — cumulative; approximate using innings averages
        # Earlier sessions have fewer cumulative wickets
        session_order_in_innings = {"Morning": 1, "Afternoon": 2, "Evening": 3}
        s_order = session_order_in_innings.get(session_label, 2)
        wickets_at_end = min(wickets * s_order, 9.0)

        # Delta features — relative to previous session (or zeros if first)
        if prev_features:
            run_rate_delta     = run_rate - prev_features.get("session_run_rate", run_rate)
            wickets_delta      = wickets  - prev_features.get("session_wickets",  wickets)
            dot_pct_delta      = dot_pct  - prev_features.get("dot_ball_pct",     dot_pct)
        else:
            run_rate_delta = 0.0
            wickets_delta  = 0.0
            dot_pct_delta  = 0.0

        momentum_index = run_rate_delta - (wickets_delta * 2.5)

        features = {
            "session_run_rate"       : round(run_rate,    4),
            "session_runs"           : round(sess_runs,   1),
            "dot_ball_pct"           : round(dot_pct,     4),
            "boundary_rate"          : round(boundary_r,  4),
            "session_extras"         : round(sess_extras, 1),
            "session_wickets"        : round(wickets,     2),
            "wickets_per_over"       : round(wpo,         4),
            "wickets_at_session_end" : round(wickets_at_end, 1),
            "max_dot_streak"         : round(max_dot,     1),
            "total_pressure_balls"   : round(pressure_b,  1),
            "run_rate_delta"         : round(run_rate_delta, 4),
            "wickets_delta"          : round(wickets_delta,  4),
            "dot_ball_pct_delta"     : round(dot_pct_delta,  4),
            "session_momentum_index" : round(momentum_index, 4),
            "ball_age_start"         : ball_age_start,
            "innings_num"            : innings_num,
            "is_home_batting"        : is_home_batting,
            "toss_bat_first"         : toss_bat_first,
            "toss_winner_batting"    : toss_winner_batting,
            "is_fourth_innings"      : 1 if innings_num == 4 else 0,
            "is_first_innings"       : 1 if innings_num == 1 else 0,
            "is_morning_session"     : 1 if is_morning  else 0,
            "is_evening_session"     : 1 if is_evening  else 0,
            "top_order_exposed"      : 1 if wickets_at_end <= 4 else 0,
        }

        return {
            "features"     : features,
            "innings_num"  : innings_num,
            "session_label": session_label,
            "batting_team" : batting_team,
            "fielding_team": fielding_team,
        }


# =============================================================================
# PRE-MATCH PREDICTOR
# =============================================================================

class PreMatchPredictor:
    """
    Orchestrates the full pre-match prediction pipeline:
      1. Load historical data + trained model
      2. Build team profiles
      3. Synthesise session feature vectors
      4. Run model inference on all 6 sessions
      5. Return structured forecast report
    """

    def __init__(self):
        self.model      = None
        self.le         = None
        self.session_df = None
        self._load()

    def _load(self):
        """Load model, label encoder, and historical session data."""
        import joblib

        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Model directory not found: {MODEL_DIR}\n"
                "Run python main.py first to train the models."
            )

        xgb_path = MODEL_DIR / "session_momentum_xgb.pkl"
        le_path  = MODEL_DIR / "label_encoder.pkl"

        if not xgb_path.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at {xgb_path}\n"
                "Run python main.py to train the models."
            )

        self.model = joblib.load(xgb_path)
        self.le    = joblib.load(le_path)

        if SESSION_CSV.exists():
            self.session_df = pd.read_csv(SESSION_CSV)
            print(f"[PreMatch] Loaded {len(self.session_df):,} historical sessions "
                  f"from {self.session_df['match_id'].nunique()} matches.")
        else:
            raise FileNotFoundError(
                f"Session features CSV not found at {SESSION_CSV}\n"
                "Run python main.py first."
            )

    def predict(
        self,
        team1: str,
        team2: str,
        venue: str        = "Unknown",
        home_team: str    = "Unknown",
        toss_winner: str  = "",
        toss_decision: str = "bat",
    ) -> dict:
        """
        Generate a full pre-match momentum forecast.

        Parameters
        ----------
        team1, team2    : Team names (partial match against your historical data)
        venue           : Venue name (informational)
        home_team       : Which team is playing at home
        toss_winner     : Which team won the toss (defaults to team1)
        toss_decision   : "bat" or "field"

        Returns
        -------
        dict with session_forecasts, team_profiles, match_summary
        """
        if not toss_winner:
            toss_winner = team1

        profiler = TeamProfiler(self.session_df)

        # Warn if teams not found in data
        available = profiler.available_teams()
        for t in [team1, team2]:
            matched = profiler._fuzzy_match(t)
            if matched not in available:
                print(f"[Warning] '{t}' not found in historical data. "
                      f"Using global averages. Available: {available[:10]}")

        # H2H profile summary
        h2h = profiler.head_to_head_profile(team1, team2)

        # Feature builder
        builder = PreMatchFeatureBuilder(
            profiler, team1, team2, home_team, toss_winner, toss_decision
        )

        # ── Run inference for each of the 6 canonical sessions ────────────────
        session_forecasts = []
        prev_features     = None

        for (innings_num, sess_label, is_morning, is_evening, ball_age) in SESSION_CATALOGUE:
            build_result = builder.build_session(
                innings_num   = innings_num,
                session_label = sess_label,
                is_morning    = is_morning,
                is_evening    = is_evening,
                ball_age_start= ball_age,
                prev_features = prev_features,
            )

            features = build_result["features"]
            X        = pd.DataFrame([features])[SESSION_FEATURES].fillna(0)
            proba    = self.model.predict_proba(X)[0]
            classes  = list(self.le.classes_)

            p_bowl = float(proba[classes.index(-1)]) if -1 in classes else 0.0
            p_neut = float(proba[classes.index(0)])  if  0 in classes else 0.0
            p_bat  = float(proba[classes.index(1)])  if  1 in classes else 0.0

            pred_label = int(self.le.inverse_transform([int(np.argmax(proba))])[0])
            confidence = float(np.max(proba))

            session_forecasts.append({
                "innings"       : innings_num,
                "session"       : sess_label,
                "batting_team"  : build_result["batting_team"],
                "fielding_team" : build_result["fielding_team"],
                "predicted_label": pred_label,
                "confidence"    : confidence,
                "prob_batting"  : p_bat,
                "prob_neutral"  : p_neut,
                "prob_bowling"  : p_bowl,
                "exp_run_rate"  : round(features["session_run_rate"], 2),
                "exp_wickets"   : round(features["session_wickets"],  2),
                "exp_dot_pct"   : round(features["dot_ball_pct"] * 100, 1),
                "momentum_index": round(features["session_momentum_index"], 3),
                "features"      : features,
            })

            prev_features = features

        # ── Match-level summary ────────────────────────────────────────────────
        batting_wins  = sum(1 for s in session_forecasts if s["predicted_label"] == 1)
        bowling_wins  = sum(1 for s in session_forecasts if s["predicted_label"] == -1)
        neutral_count = sum(1 for s in session_forecasts if s["predicted_label"] == 0)
        avg_bat_prob  = np.mean([s["prob_batting"]  for s in session_forecasts])
        avg_bowl_prob = np.mean([s["prob_bowling"]  for s in session_forecasts])

        # Projected winner: team with more batting-momentum sessions
        t1_bat_sessions = sum(
            1 for s in session_forecasts
            if s["batting_team"] == profiler._fuzzy_match(team1)
            and s["predicted_label"] == 1
        )
        t2_bat_sessions = sum(
            1 for s in session_forecasts
            if s["batting_team"] == profiler._fuzzy_match(team2)
            and s["predicted_label"] == 1
        )

        if t1_bat_sessions > t2_bat_sessions:
            projected_winner = team1
            winner_confidence = t1_bat_sessions / max(t1_bat_sessions + t2_bat_sessions, 1)
        elif t2_bat_sessions > t1_bat_sessions:
            projected_winner = team2
            winner_confidence = t2_bat_sessions / max(t1_bat_sessions + t2_bat_sessions, 1)
        else:
            projected_winner = "Contested"
            winner_confidence = 0.5

        return {
            "match"  : {
                "team1"        : team1,
                "team2"        : team2,
                "venue"        : venue,
                "home_team"    : home_team,
                "toss_winner"  : toss_winner,
                "toss_decision": toss_decision,
            },
            "session_forecasts"  : session_forecasts,
            "h2h"                : h2h,
            "match_summary"      : {
                "batting_momentum_sessions" : batting_wins,
                "bowling_momentum_sessions" : bowling_wins,
                "neutral_sessions"          : neutral_count,
                "avg_prob_batting"          : round(avg_bat_prob,  3),
                "avg_prob_bowling"          : round(avg_bowl_prob, 3),
                "projected_winner"          : projected_winner,
                "winner_confidence"         : round(winner_confidence, 2),
                "team1_batting_sessions"    : t1_bat_sessions,
                "team2_batting_sessions"    : t2_bat_sessions,
            },
            "team_profiles" : {
                team1: {
                    "batting" : profiler.batting_profile(team1),
                    "bowling" : profiler.bowling_profile(team1),
                },
                team2: {
                    "batting" : profiler.batting_profile(team2),
                    "bowling" : profiler.bowling_profile(team2),
                },
            },
        }


# =============================================================================
# PRETTY PRINTER
# =============================================================================

def print_report(result: dict):
    """Print a formatted pre-match forecast to the terminal."""
    m   = result["match"]
    s   = result["match_summary"]
    sf  = result["session_forecasts"]
    h2h = result["h2h"]

    divider = "═" * 68

    print(f"\n{divider}")
    print(f"  🏏  PRE-MATCH MOMENTUM FORECAST")
    print(f"  {m['team1']}  vs  {m['team2']}")
    print(f"  Venue: {m['venue']}  |  Home: {m['home_team']}")
    print(f"  Toss: {m['toss_winner']} chose to {m['toss_decision'].upper()}")
    print(divider)

    # H2H
    print(f"\n  HEAD-TO-HEAD HISTORY")
    print(f"  ─────────────────────")
    print(f"  Historical meetings : {h2h.get('n_matches', 0)} matches, "
          f"{h2h.get('n_sessions', 0)} sessions")
    for team, role in [(m['team1'], 'team1_batting'), (m['team2'], 'team2_batting')]:
        p = h2h.get(role, {})
        if p.get('n_sessions', 0) > 0:
            print(f"  {team} batting: RR={p.get('session_run_rate',0):.2f}  "
                  f"Wkts/sess={p.get('session_wickets',0):.1f}  "
                  f"Dot%={p.get('dot_ball_pct',0)*100:.1f}%  "
                  f"(n={p['n_sessions']})")

    # Session table
    print(f"\n  SESSION-BY-SESSION FORECAST")
    print(f"  ─────────────────────────────────────────────────────────────")
    header = f"  {'Inn':<4} {'Session':<12} {'Batting':<22} {'Forecast':<22} {'RR':>5} {'Wkt':>5} {'Conf':>6}"
    print(header)
    print(f"  {'─'*64}")

    for s_data in sf:
        icon  = LABEL_COLOR[s_data["predicted_label"]]
        label = LABEL_MAP[s_data["predicted_label"]].replace("▲ ","").replace("▼ ","").replace("→ ","")
        print(
            f"  {s_data['innings']:<4} "
            f"{s_data['session']:<12} "
            f"{s_data['batting_team'][:20]:<22} "
            f"{icon} {label:<19} "
            f"{s_data['exp_run_rate']:>5.2f} "
            f"{s_data['exp_wickets']:>5.1f} "
            f"{s_data['confidence']*100:>5.1f}%"
        )

    # Match summary
    print(f"\n  MATCH PREDICTION SUMMARY")
    print(f"  ─────────────────────────")
    t1 = m['team1']; t2 = m['team2']
    ms = result['match_summary']
    print(f"  Batting momentum sessions : {ms['batting_momentum_sessions']}/6")
    print(f"  Bowling momentum sessions : {ms['bowling_momentum_sessions']}/6")
    print(f"  Neutral sessions          : {ms['neutral_sessions']}/6")
    print(f"  {t1} batting-dominant sessions : {ms['team1_batting_sessions']}")
    print(f"  {t2} batting-dominant sessions : {ms['team2_batting_sessions']}")
    print(f"\n  ┌─ PROJECTED OUTCOME ─────────────────────────────────┐")
    pw  = ms['projected_winner']
    wc  = ms['winner_confidence']
    icon = "🟢" if pw != "Contested" else "🟡"
    print(f"  │  {icon}  {pw:<30}  ({wc:.0%} confidence)  │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print(f"\n{divider}\n")


# =============================================================================
# FETCH FROM CRICAPI (optional — populates team/venue from match ID)
# =============================================================================

def fetch_match_metadata(match_id: str) -> dict:
    """Pull team names and venue from CricAPI for an upcoming match."""
    try:
        sys.path.insert(0, str(BASE_DIR))
        from live_feed import get_match_info
        info = get_match_info(match_id)
        teams = info.get("teams", [])
        toss  = info.get("toss", {})
        return {
            "team1"         : teams[0] if len(teams) > 0 else "Team A",
            "team2"         : teams[1] if len(teams) > 1 else "Team B",
            "venue"         : info.get("venue", "Unknown"),
            "toss_winner"   : toss.get("winner", ""),
            "toss_decision" : toss.get("decision", "bat"),
        }
    except Exception as e:
        print(f"[Warning] Could not fetch from CricAPI: {e}")
        return {}


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-match momentum forecast for upcoming Test matches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prematch_predictor.py --team1 "India" --team2 "Australia" \\
         --venue "MCG" --home "Australia" --toss "Australia" --decision bat

  python prematch_predictor.py --list-teams

  python prematch_predictor.py --match-id <CricAPI ID>
        """
    )
    parser.add_argument("--team1",      default="",        help="First team name")
    parser.add_argument("--team2",      default="",        help="Second team name")
    parser.add_argument("--venue",      default="Unknown", help="Venue name")
    parser.add_argument("--home",       default="",        help="Home team name")
    parser.add_argument("--toss",       default="",        help="Toss winner")
    parser.add_argument("--decision",   default="bat",     help="Toss decision: bat or field")
    parser.add_argument("--match-id",   default="",        help="CricAPI match ID (auto-fills team/venue)")
    parser.add_argument("--list-teams", action="store_true", help="List available teams in your data")
    parser.add_argument("--output-csv", default="",        help="Save forecast to CSV path")
    args = parser.parse_args()

    predictor = PreMatchPredictor()

    if args.list_teams:
        profiler = TeamProfiler(predictor.session_df)
        teams    = profiler.available_teams()
        print(f"\nAvailable teams in your dataset ({len(teams)}):")
        for i, t in enumerate(teams):
            print(f"  {i+1:>3}. {t}")
        sys.exit(0)

    # Auto-fill from CricAPI if match ID provided
    meta = {}
    if args.match_id:
        print(f"[PreMatch] Fetching metadata for match ID: {args.match_id}")
        meta = fetch_match_metadata(args.match_id)
        print(f"[PreMatch] Found: {meta.get('team1','?')} vs {meta.get('team2','?')} at {meta.get('venue','?')}")

    team1       = args.team1     or meta.get("team1", "")
    team2       = args.team2     or meta.get("team2", "")
    venue       = args.venue     or meta.get("venue",         "Unknown")
    home_team   = args.home      or meta.get("team1",         team1)
    toss_winner = args.toss      or meta.get("toss_winner",   team1)
    toss_dec    = args.decision  or meta.get("toss_decision", "bat")

    if not team1 or not team2:
        print("❌  Please provide --team1 and --team2  (or --match-id)")
        print("    Use --list-teams to see available team names.")
        sys.exit(1)

    print(f"\n[PreMatch] Generating forecast: {team1} vs {team2}")
    print(f"           Venue: {venue} | Home: {home_team} | Toss: {toss_winner} ({toss_dec})")

    result = predictor.predict(
        team1         = team1,
        team2         = team2,
        venue         = venue,
        home_team     = home_team,
        toss_winner   = toss_winner,
        toss_decision = toss_dec,
    )

    print_report(result)

    if args.output_csv:
        rows = []
        for sf in result["session_forecasts"]:
            rows.append({
                "innings"         : sf["innings"],
                "session"         : sf["session"],
                "batting_team"    : sf["batting_team"],
                "predicted_label" : sf["predicted_label"],
                "confidence"      : sf["confidence"],
                "prob_batting"    : sf["prob_batting"],
                "prob_neutral"    : sf["prob_neutral"],
                "prob_bowling"    : sf["prob_bowling"],
                "exp_run_rate"    : sf["exp_run_rate"],
                "exp_wickets"     : sf["exp_wickets"],
                "exp_dot_pct"     : sf["exp_dot_pct"],
                "momentum_index"  : sf["momentum_index"],
            })
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"[PreMatch] Forecast saved → {args.output_csv}")