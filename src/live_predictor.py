# src/live_predictor.py
"""
LivePredictor
=============
Ties together live_feed, live_feature_builder, and the trained XGBoost model
to produce a momentum prediction for a live Test match.

Usage (one-shot):
    from live_predictor import predict_current_session
    result = predict_current_session("match_id_here")
    print(result["predicted_label"], result["confidence"])

Used by Dashboard.py (Live Match tab) and by the polling thread in the
start_polling() helper.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Resolve paths whether running as src/live_predictor.py or directly ───────
_BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(_BASE))

MODEL_DIR = _BASE.parent / "outputs" / "models"

# Feature list — must match modeling.py & Dashboard.py SESSION_FEATURES
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


# ── Lazy model loader (cache after first load) ────────────────────────────────

_model_cache: dict = {}

def _load_models() -> dict:
    """Load XGBoost model + LabelEncoder, caching after first call."""
    if _model_cache:
        return _model_cache

    try:
        import joblib
    except ImportError:
        return {"error": "joblib not installed — run: pip install joblib"}

    xgb_path = MODEL_DIR / "session_momentum_xgb.pkl"
    le_path  = MODEL_DIR / "label_encoder.pkl"

    if not xgb_path.exists():
        return {"error": f"Model not found at {xgb_path}. Run the pipeline first."}
    if not le_path.exists():
        return {"error": f"LabelEncoder not found at {le_path}. Run the pipeline first."}

    _model_cache["xgb"] = joblib.load(xgb_path)
    _model_cache["le"]  = joblib.load(le_path)
    return _model_cache


# ── Main prediction function ──────────────────────────────────────────────────

def predict_current_session(
    match_id: str,
    prev_session_features: Optional[dict] = None,
) -> dict:
    """
    Fetch live ball-by-ball data, build features, and run model inference.

    Parameters
    ----------
    match_id : str
        The CricAPI match ID (from get_live_test_matches()).
    prev_session_features : dict | None
        Feature dict from the previous session for delta computation.
        If None, deltas are computed relative to zeros (first session of match).

    Returns
    -------
    dict with keys:
        predicted_label  : int   (-1 = bowling momentum, 0 = neutral, 1 = batting)
        confidence       : float (max class probability)
        prob_batting     : float
        prob_neutral     : float
        prob_bowling     : float
        session_name     : str
        innings_num      : int
        balls_in_session : int
        batting_team     : str
        match_id         : str
        features         : dict  (the 24-feature vector used for inference)

        On failure:
        error            : str
    """
    # ── 1. Load model ──────────────────────────────────────────────────
    models = _load_models()
    if "error" in models:
        return {"error": models["error"], "match_id": match_id}

    xgb_model = models["xgb"]
    le        = models["le"]

    # ── 2. Fetch live data ─────────────────────────────────────────────
    try:
        from live_feed import get_ball_by_ball, get_match_info
    except ImportError:
        return {"error": "live_feed.py not found in sys.path", "match_id": match_id}

    try:
        match_info = get_match_info(match_id)
        balls      = get_ball_by_ball(match_id)
    except Exception as e:
        return {"error": f"API fetch failed: {e}", "match_id": match_id}

    if not balls:
        return {
            "error"  : (
                "No ball-by-ball data returned. "
                "Ball-by-ball requires a paid CricAPI plan, or the match "
                "may not be live. Check https://cricapi.com/pricing"
            ),
            "match_id": match_id,
        }

    # ── 3. Build feature vector ───────────────────────────────────────
    try:
        from live_feature_builder import LiveFeatureBuilder
    except ImportError:
        return {"error": "live_feature_builder.py not found", "match_id": match_id}

    builder = LiveFeatureBuilder(match_info, prev_session_features)
    build_result = builder.build(balls)
    features     = build_result["features"]

    # ── 4. Inference ──────────────────────────────────────────────────
    try:
        X    = pd.DataFrame([features])[SESSION_FEATURES].fillna(0)
        proba = xgb_model.predict_proba(X)[0]
        classes = list(le.classes_)

        prob_bowling = float(proba[classes.index(-1)]) if -1 in classes else 0.0
        prob_neutral = float(proba[classes.index(0)])  if  0 in classes else 0.0
        prob_batting = float(proba[classes.index(1)])  if  1 in classes else 0.0

        predicted_idx   = int(np.argmax(proba))
        predicted_label = int(le.inverse_transform([predicted_idx])[0])
        confidence      = float(proba[predicted_idx])

    except Exception as e:
        return {"error": f"Model inference failed: {e}", "match_id": match_id}

    return {
        "match_id"        : match_id,
        "predicted_label" : predicted_label,
        "confidence"      : confidence,
        "prob_batting"    : prob_batting,
        "prob_neutral"    : prob_neutral,
        "prob_bowling"    : prob_bowling,
        "session_name"    : build_result["session_name"],
        "innings_num"     : build_result["innings_num"],
        "balls_in_session": build_result["balls_in_session"],
        "batting_team"    : build_result["batting_team"],
        "features"        : features,
    }


def predict_from_features(features: dict) -> dict:
    """
    Run inference directly on a pre-built feature dict.
    Used by the manual 'Live Prediction' tab in Dashboard.py.

    Parameters
    ----------
    features : dict  — must contain all SESSION_FEATURES keys

    Returns
    -------
    dict with keys: predicted_label, confidence, prob_batting,
                    prob_neutral, prob_bowling, features
    """
    models = _load_models()
    if "error" in models:
        return {"error": models["error"]}

    xgb_model = models["xgb"]
    le        = models["le"]

    try:
        X    = pd.DataFrame([features])[SESSION_FEATURES].fillna(0)
        proba = xgb_model.predict_proba(X)[0]
        classes = list(le.classes_)

        prob_bowling    = float(proba[classes.index(-1)]) if -1 in classes else 0.0
        prob_neutral    = float(proba[classes.index(0)])  if  0 in classes else 0.0
        prob_batting    = float(proba[classes.index(1)])  if  1 in classes else 0.0
        predicted_idx   = int(np.argmax(proba))
        predicted_label = int(le.inverse_transform([predicted_idx])[0])
        confidence      = float(proba[predicted_idx])

    except Exception as e:
        return {"error": f"Inference failed: {e}"}

    return {
        "predicted_label": predicted_label,
        "confidence"     : confidence,
        "prob_batting"   : prob_batting,
        "prob_neutral"   : prob_neutral,
        "prob_bowling"   : prob_bowling,
        "features"       : features,
    }