"""
=============================================================================
SESSION-BASED MOMENTUM PREDICTION IN TEST CRICKET
Phase 3: XGBoost / Random Forest Modeling + SHAP Interpretability
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import shap
import joblib

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
THESIS_BLUE  = "#1B4F72"
THESIS_RED   = "#C0392B"
THESIS_GREEN = "#1E8449"

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

MATCH_FEATURES = [
    "final_wp", "wp_volatility", "momentum_volatility",
    "momentum_reversals", "peak_batting_wp", "avg_run_rate",
    "total_wickets", "num_sessions",
]


class SessionMomentumModel:
    """
    XGBoost + Random Forest predicting session momentum shift label.

    SHAP Strategy — One-vs-Rest binary models:
    XGBoost 2.x multiclass models store base_score as a per-class vector,
    which shap.TreeExplainer cannot parse (ValueError on float conversion).
    Fix: train one binary XGBoost per class. Each binary model has a scalar
    base_score, making SHAP fully compatible. Interpretation is identical
    to multiclass SHAP and is valid for thesis reporting.
    """

    XGB_PARAMS = {
        "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
        "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
    }

    def __init__(self, output_dir="outputs/"):
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_xgb     = None
        self.model_rf      = None
        self.ovr_models    = {}
        self.le            = LabelEncoder()
        self.shap_values   = {}
        self.feature_names = SESSION_FEATURES

    def prepare_data(self, session_df):
        df    = session_df.dropna(subset=["momentum_label"]).copy()
        X     = df[self.feature_names].fillna(0)
        y_raw = df["momentum_label"].astype(int)
        y     = self.le.fit_transform(y_raw)
        print(f"\n[Session Model] Dataset: {len(X):,} samples x {len(self.feature_names)} features")
        print(f"  Label distribution: {dict(zip(self.le.classes_, np.bincount(y)))}")
        return X, y, df

    def train_and_evaluate(self, session_df):
        X, y, df = self.prepare_data(session_df)

        print("\n[Session Model] Training XGBoost (multiclass for CV metrics)...")
        self.model_xgb = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, **self.XGB_PARAMS
        )
        cv_xgb = self._cross_validate(self.model_xgb, X, y, "XGBoost")
        self.model_xgb.fit(X, y)

        print("\n[Session Model] Training Random Forest...")
        self.model_rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        cv_rf = self._cross_validate(self.model_rf, X, y, "Random Forest")
        self.model_rf.fit(X, y)

        # One-vs-Rest binary XGBoost — used exclusively for SHAP
        print("\n[Session Model] Training One-vs-Rest binary models for SHAP...")
        for class_idx, class_name in {0: "Bowling", 1: "Neutral", 2: "Batting"}.items():
            y_bin = (y == class_idx).astype(int)
            m = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            )
            m.fit(X, y_bin)
            self.ovr_models[class_idx] = m
            print(f"  Trained OvR model for class {class_idx} ({class_name})")

        joblib.dump(self.model_xgb, self.output_dir / "session_momentum_xgb.pkl")
        joblib.dump(self.model_rf,  self.output_dir / "session_momentum_rf.pkl")
        joblib.dump(self.le,        self.output_dir / "label_encoder.pkl")
        print(f"\n[Session Model] Models saved to {self.output_dir}/")
        return {"xgb": cv_xgb, "rf": cv_rf}

    def _cross_validate(self, model, X, y, name):
        skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(
            model, X, y, cv=skf,
            scoring=["accuracy", "f1_macro", "neg_log_loss"],
            return_train_score=True
        )
        metrics = {
            "accuracy_mean": cv_res["test_accuracy"].mean(),
            "accuracy_std" : cv_res["test_accuracy"].std(),
            "f1_macro_mean": cv_res["test_f1_macro"].mean(),
            "f1_macro_std" : cv_res["test_f1_macro"].std(),
            "log_loss_mean": -cv_res["test_neg_log_loss"].mean(),
            "log_loss_std" : cv_res["test_neg_log_loss"].std(),
        }
        print(f"\n  -- {name} (5-fold CV) --")
        print(f"  Accuracy : {metrics['accuracy_mean']:.4f} +/- {metrics['accuracy_std']:.4f}")
        print(f"  F1 Macro : {metrics['f1_macro_mean']:.4f} +/- {metrics['f1_macro_std']:.4f}")
        print(f"  Log Loss : {metrics['log_loss_mean']:.4f} +/- {metrics['log_loss_std']:.4f}")
        return metrics

    def compute_shap(self, session_df):
        """SHAP via One-vs-Rest binary models — fully XGBoost 2.x compatible."""
        X, y, df = self.prepare_data(session_df)
        print("\n[SHAP] Computing SHAP values via One-vs-Rest binary models...")

        class_labels = {
            0: "Bowling Momentum (-1)",
            1: "Neutral (0)",
            2: "Batting Momentum (+1)"
        }

        for class_idx, class_label in class_labels.items():
            print(f"  Computing SHAP for: {class_label}...")
            explainer = shap.TreeExplainer(self.ovr_models[class_idx])
            sv = explainer.shap_values(X)
            # Binary XGBoost may return list [neg_class, pos_class] — take pos class
            if isinstance(sv, list):
                sv = sv[1]
            self.shap_values[class_idx] = sv
            self._plot_shap_summary(X, class_idx, class_label)

        self._plot_shap_bar(X)
        print(f"\n[SHAP] All plots saved to {self.output_dir}/")
        return self.shap_values

    def _plot_shap_summary(self, X, class_idx, class_label):
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            self.shap_values[class_idx], X,
            feature_names=self.feature_names,
            plot_type="dot", show=False, max_display=15,
        )
        plt.title(f"SHAP Feature Impact -- {class_label}",
                  fontsize=13, color=THESIS_BLUE, fontweight="bold")
        plt.tight_layout()
        fname = f"shap_summary_class{class_idx}.png"
        plt.savefig(self.output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SHAP] Saved: {fname}")

    def _plot_shap_bar(self, X):
        all_shap  = np.array([self.shap_values[i] for i in sorted(self.shap_values)])
        mean_abs  = np.abs(all_shap).mean(axis=0)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            mean_abs, X,
            feature_names=self.feature_names,
            plot_type="bar", show=False, max_display=15,
        )
        plt.title("Mean |SHAP| -- Global Feature Importance (All Classes)",
                  fontsize=13, color=THESIS_BLUE, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_global_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [SHAP] Saved: shap_global_importance.png")


class MatchOutcomeModel:
    """Binary XGBoost predicting overall Test match winner."""

    XGB_PARAMS = {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "eval_metric": "logloss", "random_state": 42, "n_jobs": -1,
    }

    def __init__(self, output_dir="outputs/"):
        self.output_dir  = Path(output_dir)
        self.model       = None
        self.shap_values = None

    def train_and_evaluate(self, match_df):
        df = match_df[match_df["result"] == "normal"].copy()
        df["target"] = (df["winner"] == df["home_team"]).astype(int)

        X = df[MATCH_FEATURES].fillna(0)
        y = df["target"]

        print(f"\n[Match Model] Dataset: {len(X):,} matches")
        print(f"  Home team win rate: {y.mean():.3f}")

        self.model = xgb.XGBClassifier(**self.XGB_PARAMS)
        skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(
            self.model, X, y, cv=skf,
            scoring=["accuracy", "roc_auc", "neg_log_loss"]
        )
        metrics = {
            "accuracy_mean": cv_res["test_accuracy"].mean(),
            "auc_mean"     : cv_res["test_roc_auc"].mean(),
            "log_loss_mean": -cv_res["test_neg_log_loss"].mean(),
        }
        print(f"\n  -- Match Outcome XGBoost (5-fold CV) --")
        print(f"  Accuracy : {metrics['accuracy_mean']:.4f}")
        print(f"  AUC-ROC  : {metrics['auc_mean']:.4f}")
        print(f"  Log Loss : {metrics['log_loss_mean']:.4f}")

        self.model.fit(X, y)

        # Binary model — SHAP works without any workaround
        explainer        = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        self._plot_match_shap(X)

        df["predicted_win_prob"] = self.model.predict_proba(X)[:, 1]
        df[["match_id", "winner", "home_team", "result",
            "predicted_win_prob", "target"]].to_csv(
            self.output_dir / "match_outcome_predictions.csv", index=False
        )
        joblib.dump(self.model, self.output_dir / "match_outcome_xgb.pkl")
        print(f"\n[Match Model] Saved to {self.output_dir}/")
        return metrics

    def _plot_match_shap(self, X):
        plt.figure(figsize=(9, 5))
        shap.summary_plot(
            self.shap_values, X,
            feature_names=MATCH_FEATURES,
            plot_type="bar", show=False
        )
        plt.title("SHAP Feature Importance -- Match Outcome Prediction",
                  fontsize=13, color=THESIS_BLUE, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_match_outcome.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [SHAP] Saved: shap_match_outcome.png")


class EvaluationReporter:
    @staticmethod
    def generate_report(session_metrics, match_metrics, output_dir):
        rows = []
        for model_name, metrics in session_metrics.items():
            rows.append({
                "Model"   : f"Session Momentum -- {model_name.upper()}",
                "Accuracy": f"{metrics['accuracy_mean']:.4f} +/- {metrics['accuracy_std']:.4f}",
                "F1 Macro": f"{metrics['f1_macro_mean']:.4f} +/- {metrics['f1_macro_std']:.4f}",
                "Log Loss": f"{metrics['log_loss_mean']:.4f} +/- {metrics['log_loss_std']:.4f}",
                "AUC"     : "N/A (multiclass)",
            })
        if match_metrics:
            rows.append({
                "Model"   : "Match Outcome -- XGBoost",
                "Accuracy": f"{match_metrics['accuracy_mean']:.4f}",
                "F1 Macro": "N/A",
                "Log Loss": f"{match_metrics['log_loss_mean']:.4f}",
                "AUC"     : f"{match_metrics['auc_mean']:.4f}",
            })
        report_df = pd.DataFrame(rows)
        report_df.to_csv(os.path.join(output_dir, "model_evaluation_report.csv"), index=False)
        print("\n" + "="*60)
        print("THESIS MODEL EVALUATION SUMMARY")
        print("="*60)
        print(report_df.to_string(index=False))
        print("="*60)
        print(f"\nSaved -> {output_dir}/model_evaluation_report.csv")


if __name__ == "__main__":
    import sys
    SESSION_CSV = sys.argv[1] if len(sys.argv) > 1 else "outputs/session_features.csv"
    MATCH_CSV   = sys.argv[2] if len(sys.argv) > 2 else "outputs/match_level_features.csv"
    OUTPUT_DIR  = sys.argv[3] if len(sys.argv) > 3 else "outputs/"

    session_df = pd.read_csv(SESSION_CSV)
    match_df   = pd.read_csv(MATCH_CSV)

    sm = SessionMomentumModel(output_dir=OUTPUT_DIR)
    session_metrics = sm.train_and_evaluate(session_df)
    sm.compute_shap(session_df)

    mm = MatchOutcomeModel(output_dir=OUTPUT_DIR)
    match_metrics = mm.train_and_evaluate(match_df)

    EvaluationReporter.generate_report(session_metrics, match_metrics, OUTPUT_DIR)