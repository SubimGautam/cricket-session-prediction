# src/live_feature_builder.py
"""
LiveFeatureBuilder
==================
Converts a raw list of ball-by-ball delivery dicts (from live_feed.py) into
the exact 24-feature vector that SessionMomentumModel / the XGBoost model
was trained on.

All feature names are identical to SESSION_FEATURES in modeling.py and
Dashboard.py so the model receives correctly-shaped input at prediction time.

Typical call chain:
    balls   = live_feed.get_ball_by_ball(match_id)
    info    = live_feed.get_match_info(match_id)
    builder = LiveFeatureBuilder(info)
    result  = builder.build(balls)
    # result["features"] is a dict ready for model inference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ── Feature names — must match SESSION_FEATURES in modeling.py ──────────────
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

# Pressure crisis: a dot-ball streak of 6+ is a "pressure zone"
_PRESSURE_STREAK_THRESHOLD = 6


@dataclass
class _SessionAccumulator:
    """Mutable accumulator for the current live session."""
    balls_seen         : int   = 0
    runs               : int   = 0
    wickets            : int   = 0
    dot_balls          : int   = 0
    boundaries_4       : int   = 0
    boundaries_6       : int   = 0
    extras             : int   = 0
    current_dot_streak : int   = 0
    max_dot_streak     : int   = 0
    total_pressure_balls: int  = 0  # balls while streak >= threshold
    ball_age_start     : int   = 1  # absolute ball number of first ball in session

    # Previous session stats (for delta features)
    prev_run_rate     : float  = 0.0
    prev_wickets      : int    = 0
    prev_dot_ball_pct : float  = 0.0

    # Cumulative innings wickets (to determine top-order exposure)
    wickets_at_session_end: int = 0


def _infer_session_time(over: int) -> Tuple[bool, bool]:
    """
    Given the current over number, return (is_morning, is_evening).
    Approximate Test cricket session boundaries:
        Morning  : overs 1–30
        Afternoon: overs 31–60
        Evening  : overs 61+
    """
    if over < 30:
        return True, False
    if over >= 60:
        return False, True
    return False, False


class LiveFeatureBuilder:
    """
    Stateful feature builder for a live Test match.

    Parameters
    ----------
    match_info : dict
        Metadata dict returned by live_feed.get_match_info().
        Expected keys: teams, toss (winner + decision), id, score.
    prev_session_features : dict | None
        Feature dict from the immediately preceding session, used to compute
        delta features.  Pass None for the very first session of the match.
    """

    def __init__(
        self,
        match_info: dict,
        prev_session_features: Optional[dict] = None,
    ):
        self.match_info   = match_info or {}
        self.prev_feats   = prev_session_features or {}

        # Parse match metadata once
        teams          = self.match_info.get("teams", ["TeamA", "TeamB"])
        self.team1     = teams[0] if len(teams) > 0 else "TeamA"
        self.team2     = teams[1] if len(teams) > 1 else "TeamB"

        toss           = self.match_info.get("toss", {})
        self.toss_winner   = toss.get("winner",   "Unknown")
        self.toss_decision = toss.get("decision", "bat")

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def build(self, balls: List[dict]) -> dict:
        """
        Compute the 24-feature vector from a list of delivery dicts.

        Parameters
        ----------
        balls : list of delivery dicts (standardised format from live_feed.py)
            Each dict must contain at minimum:
                over, ball, runs {batter, extras, total}, wickets (list)

        Returns
        -------
        dict with keys:
            "features"       : dict mapping SESSION_FEATURES → float
            "innings_num"    : int
            "balls_in_session": int
            "session_name"   : str  ("morning" | "afternoon" | "evening")
            "batting_team"   : str
        """
        if not balls:
            return self._empty_result()

        acc = _SessionAccumulator()
        acc.ball_age_start = self._infer_ball_age_start(balls)

        # ── Set prev-session deltas ──────────────────────────────────
        acc.prev_run_rate      = float(self.prev_feats.get("session_run_rate", 0.0))
        acc.prev_wickets       = int(  self.prev_feats.get("session_wickets",  0))
        acc.prev_dot_ball_pct  = float(self.prev_feats.get("dot_ball_pct",     0.0))

        # ── Walk every delivery ──────────────────────────────────────
        for delivery in balls:
            runs    = delivery.get("runs", {})
            if isinstance(runs, int):
                b_runs = runs; e_runs = 0; t_runs = runs
            else:
                b_runs = int(runs.get("batter", 0))
                e_runs = int(runs.get("extras", 0))
                t_runs = int(runs.get("total",  b_runs + e_runs))

            wickets_this_ball = delivery.get("wickets", [])
            is_wicket  = len(wickets_this_ball) > 0
            extras_raw = delivery.get("extras", {})
            is_wide    = "wides"  in extras_raw
            is_noball  = "noballs" in extras_raw
            is_dot     = (b_runs == 0 and not is_wide and not is_noball)

            acc.balls_seen  += 1
            acc.runs        += t_runs
            acc.extras      += e_runs
            if is_wicket:
                acc.wickets += 1
            if b_runs == 4:
                acc.boundaries_4 += 1
            if b_runs == 6:
                acc.boundaries_6 += 1

            # Dot streak / pressure
            if is_dot:
                acc.current_dot_streak += 1
                acc.max_dot_streak      = max(acc.max_dot_streak, acc.current_dot_streak)
                if acc.current_dot_streak >= _PRESSURE_STREAK_THRESHOLD:
                    acc.total_pressure_balls += 1
                if is_dot:
                    acc.dot_balls += 1
            else:
                acc.current_dot_streak = 0

        # ── Cumulative innings wickets ───────────────────────────────
        acc.wickets_at_session_end = (
            int(self.prev_feats.get("wickets_at_session_end", 0)) + acc.wickets
        )

        # ── Infer context from last ball ─────────────────────────────
        last      = balls[-1]
        innings_num = int(last.get("innings", self._infer_innings_from_score()))
        last_over   = int(last.get("over", 0))
        is_morning, is_evening = _infer_session_time(last_over)
        session_name = "morning" if is_morning else ("evening" if is_evening else "afternoon")

        # Batting team from score summary or info
        batting_team = self._infer_batting_team(innings_num)

        toss_bat_first       = 1 if self.toss_decision.lower() == "bat" else 0
        toss_winner_batting  = 1 if self.toss_winner == batting_team   else 0
        # Home ground heuristic: treat team1 as home team
        is_home_batting      = 1 if batting_team == self.team1 else 0

        # ── Derived rates ────────────────────────────────────────────
        overs_bowled  = max(acc.balls_seen / 6.0, 0.1)
        run_rate      = acc.runs / overs_bowled
        dot_ball_pct  = acc.dot_balls / max(acc.balls_seen, 1)
        boundary_rate = (acc.boundaries_4 + acc.boundaries_6) / max(acc.balls_seen, 1)
        wkts_per_over = acc.wickets / overs_bowled

        run_rate_delta    = run_rate     - acc.prev_run_rate
        wickets_delta     = acc.wickets  - acc.prev_wickets
        dot_ball_pct_delta= dot_ball_pct - acc.prev_dot_ball_pct

        # Momentum index (mirrors feature_engineering.py formula)
        # Positive = batting momentum gaining; Negative = bowling
        momentum_index = run_rate_delta - (wickets_delta * 2.5)

        top_order_exposed = 1 if acc.wickets_at_session_end <= 4 else 0

        features = {
            "session_run_rate"        : round(run_rate,        4),
            "session_runs"            : acc.runs,
            "dot_ball_pct"            : round(dot_ball_pct,    4),
            "boundary_rate"           : round(boundary_rate,   4),
            "session_extras"          : acc.extras,
            "session_wickets"         : acc.wickets,
            "wickets_per_over"        : round(wkts_per_over,   4),
            "wickets_at_session_end"  : acc.wickets_at_session_end,
            "max_dot_streak"          : acc.max_dot_streak,
            "total_pressure_balls"    : acc.total_pressure_balls,
            "run_rate_delta"          : round(run_rate_delta,   4),
            "wickets_delta"           : wickets_delta,
            "dot_ball_pct_delta"      : round(dot_ball_pct_delta, 4),
            "session_momentum_index"  : round(momentum_index,  4),
            "ball_age_start"          : acc.ball_age_start,
            "innings_num"             : innings_num,
            "is_home_batting"         : is_home_batting,
            "toss_bat_first"          : toss_bat_first,
            "toss_winner_batting"     : toss_winner_batting,
            "is_fourth_innings"       : 1 if innings_num == 4 else 0,
            "is_first_innings"        : 1 if innings_num == 1 else 0,
            "is_morning_session"      : 1 if is_morning   else 0,
            "is_evening_session"      : 1 if is_evening   else 0,
            "top_order_exposed"       : top_order_exposed,
        }

        return {
            "features"         : features,
            "innings_num"      : innings_num,
            "balls_in_session" : acc.balls_seen,
            "session_name"     : session_name,
            "batting_team"     : batting_team,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _infer_ball_age_start(self, balls: List[dict]) -> int:
        """
        Estimate the absolute ball number at the start of the session.
        If deliveries carry no absolute counter, derive from over / ball fields.
        """
        if not balls:
            return 1
        first = balls[0]
        over  = int(first.get("over", 0))
        ball  = int(first.get("ball", 1))
        return max(1, over * 6 + ball)

    def _infer_innings_from_score(self) -> int:
        """
        Fallback: guess innings number from the score summary list length.
        CricAPI score list has one entry per innings played.
        """
        score = self.match_info.get("score", [])
        return max(1, len(score))

    def _infer_batting_team(self, innings_num: int) -> str:
        """
        Derive the batting team from innings number and toss decision.
        innings 1 & 3 → toss winner bats first (if toss = bat)
        innings 2 & 4 → other team

        This is approximate; for accuracy, use the 'batting' field
        from the score summary if available.
        """
        score = self.match_info.get("score", [])
        if score and isinstance(score, list) and innings_num <= len(score):
            inning_score = score[innings_num - 1]
            if isinstance(inning_score, dict):
                team = inning_score.get("inning", "").replace(" Inning 1", "").replace(" Inning 2", "")
                if team:
                    return team

        # Fallback: use toss decision
        if self.toss_decision.lower() == "bat":
            batting_first_team = self.toss_winner
        else:
            batting_first_team = self.team2 if self.toss_winner == self.team1 else self.team1

        if innings_num in (1, 3):
            return batting_first_team
        else:
            return self.team2 if batting_first_team == self.team1 else self.team1

    def _empty_result(self) -> dict:
        """Return a zeroed-out result when no ball data is available."""
        features = {k: 0 for k in SESSION_FEATURES}
        features["innings_num"]     = 1
        features["is_first_innings"]= 1
        return {
            "features"         : features,
            "innings_num"      : 1,
            "balls_in_session" : 0,
            "session_name"     : "morning",
            "batting_team"     : self.team1,
        }