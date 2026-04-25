"""
=============================================================================
SESSION-BASED MOMENTUM PREDICTION IN TEST CRICKET
Phase 3: XGBoost / Random Forest Modeling + SHAP Interpretability
=============================================================================
Author      : [Your Name] — Master's Thesis
Inputs      : session_features.csv, match_level_features.csv (from Phase 1/2)
Outputs     : Trained models, SHAP plots, evaluation metrics CSV
Description : Two models are trained:
              (A) SESSION MOMENTUM MODEL — predicts WP shift label per session
              (B) MATCH OUTCOME MODEL    — predicts overall match winner
              Both are evaluated with Log Loss, AUC, RMSE, and SHAP values.
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, log_loss,
    mean_squared_error, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import shap
import joblib

warnings.filterwarnings("ignore")

# ── Plot Style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
THESIS_BLUE  = "#1B4F72"
THESIS_RED   = "#C0392B"
THESIS_GREEN = "#1E8449"


# =============================================================================
# FEATURE SETS
# =============================================================================

# Features used in the SESSION MOMENTUM model
SESSION_FEATURES = [
    # Scoring
    "session_run_rate",
    "session_runs",
    "dot_ball_pct",
    "boundary_rate",
    "session_extras",
    # Wickets
    "session_wickets",
    "wickets_per_over",
    "wickets_at_session_end",
    # Pressure
    "max_dot_streak",
    "total_pressure_balls",
    # Momentum deltas
    "run_rate_delta",
    "wickets_delta",
    "dot_ball_pct_delta",
    "session_momentum_index",
    # Ball age
    "ball_age_start",
    # Context
    "innings_num",
    "is_home_batting",
    "toss_bat_first",
    "toss_winner_batting",
    "is_fourth_innings",
    "is_first_innings",
    "is_morning_session",
    "is_evening_session",
    "top_order_exposed",
]

# Features used in the MATCH OUTCOME model
MATCH_FEATURES = [
    "final_wp",
    "wp_volatility",
    "momentum_volatility",
    "momentum_reversals",
    "peak_batting_wp",
    "avg_run_rate",
    "total_wickets",
    "num_sessions",
]


# =============================================================================
# MODULE 5: XGBOOST SESSION MOMENTUM MODEL
# =============================================================================

class SessionMomentumModel:
    """
    XGBoost multi-class classifier that predicts session momentum shift label:
      -1 → bowling team gaining momentum (WP shift < −0.05)
       0 → neutral session
      +1 → batting team gaining momentum (WP shift > +0.05)

    Why XGBoost:
    ------------
    XGBoost handles mixed feature types natively, is robust to missing values
    (which arise when sessions have sparse data), and provides native feature
    importance. Combined with SHAP, it offers both predictive power and the
    interpretability required for a thesis.

    Evaluation Metrics:
    -------------------
    - Log Loss     : Penalizes confident wrong predictions — key for probabilistic outputs
    - Weighted AUC : Appropriate for imbalanced 3-class momentum labels
    - Macro F1     : Treats all classes equally, important since neutral >> others
    - SHAP values  : Direction + magnitude of each feature's contribution per prediction
    """

    # XGBoost hyperparameters — tuned for moderate dataset sizes (500–5000 sessions)
    XGB_PARAMS = {
        "n_estimators"     : 400,
        "max_depth"        : 5,
        "learning_rate"    : 0.05,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "min_child_weight" : 3,
        "gamma"            : 0.1,
        "reg_alpha"        : 0.1,    # L1 regularization (feature sparsity)
        "reg_lambda"       : 1.0,    # L2 regularization
        "use_label_encoder": False,
        "eval_metric"      : "mlogloss",
        "random_state"     : 42,
        "n_jobs"           : -1,
    }

    def __init__(self, output_dir: str = "outputs/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_xgb  = None
        self.model_rf   = None
        self.le         = LabelEncoder()
        self.shap_values = None
        self.feature_names = SESSION_FEATURES

    def prepare_data(self, session_df: pd.DataFrame):
        """Clean and encode session data for modeling."""
        df = session_df.dropna(subset=["momentum_label"]).copy()

        # Fill NaN features with 0 (e.g., delta features for first session)
        X = df[self.feature_names].fillna(0)
        y_raw = df["momentum_label"].astype(int)

        # Encode labels: {-1, 0, 1} → {0, 1, 2} for XGBoost compatibility
        y = self.le.fit_transform(y_raw)  # -1→0, 0→1, 1→2

        print(f"\n[Session Model] Dataset: {len(X):,} samples × {len(self.feature_names)} features")
        print(f"  Label distribution: {dict(zip(self.le.classes_, np.bincount(y)))}")
        return X, y, df

    def train_and_evaluate(self, session_df: pd.DataFrame) -> dict:
        """
        Train XGBoost + Random Forest on session features.
        Perform 5-fold stratified cross-validation and report metrics.
        """
        X, y, df = self.prepare_data(session_df)

        # ── XGBoost ────────────────────────────────────────────────────
        print("\n[Session Model] Training XGBoost...")
        self.model_xgb = xgb.XGBClassifier(**self.XGB_PARAMS)

        cv_results_xgb = self._cross_validate_model(self.model_xgb, X, y, "XGBoost")

        # Refit on full data for SHAP analysis
        self.model_xgb.fit(X, y)

        # ── Random Forest (comparison baseline) ───────────────────────
        print("\n[Session Model] Training Random Forest...")
        self.model_rf = RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        cv_results_rf = self._cross_validate_model(self.model_rf, X, y, "Random Forest")
        self.model_rf.fit(X, y)

        # ── Save models ───────────────────────────────────────────────
        joblib.dump(self.model_xgb, self.output_dir / "session_momentum_xgb.pkl")
        joblib.dump(self.model_rf,  self.output_dir / "session_momentum_rf.pkl")
        joblib.dump(self.le,        self.output_dir / "label_encoder.pkl")

        print(f"\n[Session Model] Models saved to {self.output_dir}/")
        return {"xgb": cv_results_xgb, "rf": cv_results_rf}

    def _cross_validate_model(self, model, X, y, name: str) -> dict:
        """5-fold stratified CV — returns dict of metrics."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(
            model, X, y, cv=skf,
            scoring=["accuracy", "f1_macro", "neg_log_loss"],
            return_train_score=True
        )

        metrics = {
            "accuracy_mean" : cv_res["test_accuracy"].mean(),
            "accuracy_std"  : cv_res["test_accuracy"].std(),
            "f1_macro_mean" : cv_res["test_f1_macro"].mean(),
            "f1_macro_std"  : cv_res["test_f1_macro"].std(),
            "log_loss_mean" : -cv_res["test_neg_log_loss"].mean(),
            "log_loss_std"  : cv_res["test_neg_log_loss"].std(),
        }

        print(f"\n  ── {name} Cross-Validation Results (5-fold) ──")
        print(f"  Accuracy  : {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
        print(f"  F1 Macro  : {metrics['f1_macro_mean']:.4f} ± {metrics['f1_macro_std']:.4f}")
        print(f"  Log Loss  : {metrics['log_loss_mean']:.4f} ± {metrics['log_loss_std']:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # SHAP Interpretability
    # ------------------------------------------------------------------

    def compute_shap(self, session_df: pd.DataFrame):
        """
        Compute SHAP values for the XGBoost model.

        SHAP (SHapley Additive exPlanations) decomposes each prediction
        into additive feature contributions. For a thesis, this is critical:
        it answers 'WHICH features drove momentum shifts?' with statistical grounding.

        We compute TreeExplainer SHAP values — O(n_features × n_samples) but
        exact for tree-based models, unlike kernel SHAP approximations.
        """
        X, y, df = self.prepare_data(session_df)

        print("\n[SHAP] Computing TreeExplainer SHAP values...")
        explainer = shap.TreeExplainer(self.model_xgb)
        self.shap_values = explainer.shap_values(X)

        # shap_values shape: [n_classes, n_samples, n_features]
        # For momentum label +1 (class index 2): batting team gaining momentum
        self._plot_shap_summary(X, class_idx=2, class_label="Batting Momentum (+1)")
        self._plot_shap_summary(X, class_idx=0, class_label="Bowling Momentum (−1)")
        self._plot_shap_bar(X)

        return self.shap_values

    def _plot_shap_summary(self, X, class_idx: int, class_label: str):
        """Beeswarm SHAP summary plot for one class."""
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            self.shap_values[class_idx], X,
            feature_names=self.feature_names,
            plot_type="dot", show=False,
            max_display=15
        )
        plt.title(f"SHAP Feature Impact — {class_label}", fontsize=13,
                  color=THESIS_BLUE, fontweight="bold")
        plt.tight_layout()
        fname = f"shap_summary_class{class_idx}.png"
        plt.savefig(self.output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Saved: {fname}")

    def _plot_shap_bar(self, X):
        """Mean |SHAP| bar chart — global feature importance."""
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            # Average absolute SHAP across all classes
            np.abs(np.array(self.shap_values)).mean(axis=0),
            X,
            feature_names=self.feature_names,
            plot_type="bar", show=False,
            max_display=15
        )
        plt.title("Mean |SHAP| — Global Feature Importance (All Classes)",
                  fontsize=13, color=THESIS_BLUE, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_global_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[SHAP] Saved: shap_global_importance.png")


# =============================================================================
# MODULE 6: MATCH OUTCOME PREDICTION MODEL
# =============================================================================

class MatchOutcomeModel:
    """
    Predicts the overall Test match winner using match-level aggregated features.
    This answers the secondary thesis objective: 'Which team has the upper hand?'

    Model: XGBoost binary classifier
    Target: Did Team A (first-batting team) win the match? (1 = yes, 0 = no)

    Key Feature: final_wp is the strongest single predictor — it essentially
    carries forward all session-level information into match-level prediction.
    Supporting features (wp_volatility, momentum_reversals) add context about
    *how* the win probability arrived at its final value.
    """

    XGB_PARAMS = {
        "n_estimators"     : 300,
        "max_depth"        : 4,
        "learning_rate"    : 0.05,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "use_label_encoder": False,
        "eval_metric"      : "logloss",
        "random_state"     : 42,
        "n_jobs"           : -1,
    }

    def __init__(self, output_dir: str = "outputs/"):
        self.output_dir  = Path(output_dir)
        self.model       = None
        self.shap_values = None

    def train_and_evaluate(self, match_df: pd.DataFrame) -> dict:
        """Train match outcome model and evaluate."""

        # Filter: known outcomes only (exclude draws/no result for binary prediction)
        df = match_df[match_df["result"] == "normal"].copy()

        # For binary label: 1 = home team won (or first-listed team won)
        # This needs domain-specific encoding — adjust to your labeling convention
        df["target"] = (df["winner"] == df["home_team"]).astype(int)

        X = df[MATCH_FEATURES].fillna(0)
        y = df["target"]

        print(f"\n[Match Model] Dataset: {len(X):,} matches")
        print(f"  Home team win rate: {y.mean():.3f}")

        self.model = xgb.XGBClassifier(**self.XGB_PARAMS)

        # Cross-validate
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(
            self.model, X, y, cv=skf,
            scoring=["accuracy", "roc_auc", "neg_log_loss"]
        )

        metrics = {
            "accuracy_mean" : cv_res["test_accuracy"].mean(),
            "auc_mean"      : cv_res["test_roc_auc"].mean(),
            "log_loss_mean" : -cv_res["test_neg_log_loss"].mean(),
        }

        print(f"\n  ── Match Outcome XGBoost (5-fold CV) ──")
        print(f"  Accuracy  : {metrics['accuracy_mean']:.4f}")
        print(f"  AUC-ROC   : {metrics['auc_mean']:.4f}")
        print(f"  Log Loss  : {metrics['log_loss_mean']:.4f}")

        # Refit on full dataset
        self.model.fit(X, y)

        # SHAP for match outcome
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X)
        self._plot_match_shap(X)

        # Save predicted probabilities into df
        df["predicted_win_prob"] = self.model.predict_proba(X)[:, 1]
        df[["match_id", "winner", "home_team", "result",
            "predicted_win_prob", "target"]].to_csv(
            self.output_dir / "match_outcome_predictions.csv", index=False
        )

        joblib.dump(self.model, self.output_dir / "match_outcome_xgb.pkl")
        print(f"\n[Match Model] Saved model + predictions to {self.output_dir}/")

        return metrics

    def _plot_match_shap(self, X: pd.DataFrame):
        """SHAP bar plot for match outcome model."""
        fig, ax = plt.subplots(figsize=(9, 5))
        shap.summary_plot(
            self.shap_values, X,
            feature_names=MATCH_FEATURES,
            plot_type="bar", show=False
        )
        plt.title("SHAP Feature Importance — Match Outcome Prediction",
                  fontsize=13, color=THESIS_BLUE, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_match_outcome.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("[SHAP] Saved: shap_match_outcome.png")


# =============================================================================
# MODULE 7: EVALUATION REPORT GENERATOR
# =============================================================================

class EvaluationReporter:
    """
    Generates a consolidated CSV + text report of all model metrics.
    Suitable for direct inclusion in the thesis appendix.
    """

    @staticmethod
    def generate_report(session_metrics: dict, match_metrics: dict, output_dir: str):
        rows = []
        for model_name, metrics in session_metrics.items():
            rows.append({
                "Model"    : f"Session Momentum — {model_name.upper()}",
                "Accuracy" : f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}",
                "F1 Macro" : f"{metrics['f1_macro_mean']:.4f} ± {metrics['f1_macro_std']:.4f}",
                "Log Loss" : f"{metrics['log_loss_mean']:.4f} ± {metrics['log_loss_std']:.4f}",
                "AUC"      : "N/A (multiclass)",
            })

        rows.append({
            "Model"    : "Match Outcome — XGBoost",
            "Accuracy" : f"{match_metrics['accuracy_mean']:.4f}",
            "F1 Macro" : "N/A",
            "Log Loss" : f"{match_metrics['log_loss_mean']:.4f}",
            "AUC"      : f"{match_metrics['auc_mean']:.4f}",
        })

        report_df = pd.DataFrame(rows)
        report_df.to_csv(os.path.join(output_dir, "model_evaluation_report.csv"), index=False)

        print("\n" + "="*60)
        print("THESIS MODEL EVALUATION SUMMARY")
        print("="*60)
        print(report_df.to_string(index=False))
        print("="*60)
        print(f"\nReport saved to: {output_dir}/model_evaluation_report.csv")


# =============================================================================
# MAIN PIPELINE — Phase 3
# =============================================================================

def run_modeling_pipeline(
    session_csv: str = "outputs/session_features.csv",
    match_csv:   str = "outputs/match_level_features.csv",
    output_dir:  str = "outputs/"
) -> None:
    """
    Full Phase 3 pipeline:
    1. Load phase 1/2 outputs
    2. Train session momentum model (XGBoost + RF)
    3. Compute SHAP values
    4. Train match outcome model
    5. Generate evaluation report

    Parameters
    ----------
    session_csv : path to session_features.csv from Phase 1/2
    match_csv   : path to match_level_features.csv from Phase 1/2
    output_dir  : where to save models, plots, reports
    """
    print("\n" + "="*60)
    print("PHASE 3: MODELING + SHAP INTERPRETABILITY")
    print("="*60)

    session_df = pd.read_csv(session_csv)
    match_df   = pd.read_csv(match_csv)

    # ── A. Session Momentum Model ─────────────────────────────────────
    session_model  = SessionMomentumModel(output_dir=output_dir)
    session_metrics = session_model.train_and_evaluate(session_df)
    session_model.compute_shap(session_df)

    # ── B. Match Outcome Model ────────────────────────────────────────
    match_model   = MatchOutcomeModel(output_dir=output_dir)
    match_metrics = match_model.train_and_evaluate(match_df)

    # ── C. Consolidated Report ────────────────────────────────────────
    EvaluationReporter.generate_report(session_metrics, match_metrics, output_dir)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    SESSION_CSV = sys.argv[1] if len(sys.argv) > 1 else "outputs/session_features.csv"
    MATCH_CSV   = sys.argv[2] if len(sys.argv) > 2 else "outputs/match_level_features.csv"
    OUTPUT_DIR  = sys.argv[3] if len(sys.argv) > 3 else "outputs/"

    run_modeling_pipeline(SESSION_CSV, MATCH_CSV, OUTPUT_DIR)