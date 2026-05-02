# src/scheduler.py
"""
Scheduler
=========
Lightweight APScheduler-based background service that runs two periodic jobs:

    1. CricsheetPoller  — checks for new Cricsheet ZIP once per day
    2. Live prediction  — polls CricAPI for the current session every N minutes
       (only while a live match is active)

This module is designed to run as a standalone background process alongside
the Streamlit dashboard, keeping data fresh without user intervention.

Usage:
    # Start the background scheduler (blocking — run in a separate terminal)
    python -m src.scheduler

    # Or import and start programmatically
    from src.scheduler import start_scheduler, stop_scheduler
    start_scheduler()
    ...
    stop_scheduler()

Dependencies:
    pip install apscheduler
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Scheduler] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scheduler")

# ── APScheduler import (graceful fallback if not installed) ───────────────────
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval    import IntervalTrigger
    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _APSCHEDULER_AVAILABLE = False
    log.warning(
        "apscheduler not installed — scheduler disabled.\n"
        "Install with:  pip install apscheduler"
    )

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE      = Path(__file__).resolve().parent
_OUTPUT    = _BASE.parent / "outputs"
_MODEL_DIR = _OUTPUT / "models"

sys.path.insert(0, str(_BASE))

# ── Configuration ─────────────────────────────────────────────────────────────
# Intervals are in seconds
CRICSHEET_POLL_INTERVAL_HOURS = 24     # Check for new Cricsheet files daily
LIVE_PREDICTION_INTERVAL_MINS = 5      # Predict every 5 minutes during live play
MAX_CONSECUTIVE_ERRORS        = 5      # Stop live polling after N failures

# ── State ─────────────────────────────────────────────────────────────────────
_scheduler: Optional[object]  = None
_active_match_id: Optional[str] = None
_consecutive_errors: int      = 0


# ─────────────────────────────────────────────────────────────────────────────
# Job 1 — Cricsheet daily refresh
# ─────────────────────────────────────────────────────────────────────────────

def job_cricsheet_poll():
    """
    Background job: download latest Cricsheet ZIP and extract new YAML files.
    If new files are found, writes a flag file so the dashboard can prompt
    the user to re-run the feature engineering pipeline.
    """
    log.info("Running Cricsheet poll job…")
    try:
        from cricsheet_poller import CricsheetPoller
        try:
            from config import DATA_RAW
            data_dir = DATA_RAW
        except ImportError:
            data_dir = os.path.expanduser("~/Documents/cricket data/tests")

        poller    = CricsheetPoller(data_dir=data_dir)
        new_files = poller.poll()

        if new_files:
            log.info(f"Cricsheet: {len(new_files)} new file(s) downloaded.")
            # Write a flag so the dashboard can surface a "refresh available" notice
            flag = _OUTPUT / "new_cricsheet_files.flag"
            flag.write_text(f"{len(new_files)} new files downloaded")
        else:
            log.info("Cricsheet: dataset up-to-date.")

    except Exception as e:
        log.error(f"Cricsheet poll job failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Job 2 — Live prediction polling
# ─────────────────────────────────────────────────────────────────────────────

def job_live_predict():
    """
    Background job: fetch current session data, run inference, and persist
    the prediction via state_manager.
    """
    global _active_match_id, _consecutive_errors

    if not _active_match_id:
        # Auto-discover live Test match if none set
        try:
            from live_feed import get_live_test_matches
            matches = get_live_test_matches()
            if matches:
                _active_match_id = matches[0]["id"]
                log.info(f"Auto-detected live match: {_active_match_id}")
            else:
                log.info("No live Test matches — skipping prediction job.")
                return
        except Exception as e:
            log.warning(f"Could not discover live matches: {e}")
            return

    log.info(f"Running live prediction for match: {_active_match_id}")
    try:
        from live_predictor  import predict_current_session
        from state_manager   import save_prediction, save_wp_point

        result = predict_current_session(_active_match_id)

        if "error" in result:
            log.warning(f"Prediction error: {result['error']}")
            _consecutive_errors += 1
            if _consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.error(
                    f"Reached {MAX_CONSECUTIVE_ERRORS} consecutive errors. "
                    "Clearing active match ID."
                )
                _active_match_id   = None
                _consecutive_errors = 0
            return

        # Persist
        save_prediction(result)
        save_wp_point(
            _active_match_id,
            result["prob_batting"],
            result["prob_bowling"],
            result["session_name"],
        )

        log.info(
            f"Prediction saved: label={result['predicted_label']}, "
            f"confidence={result['confidence']:.2%}, "
            f"session={result['session_name']}"
        )
        _consecutive_errors = 0  # reset on success

    except Exception as e:
        log.error(f"Live prediction job failed: {e}")
        _consecutive_errors += 1


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler lifecycle
# ─────────────────────────────────────────────────────────────────────────────

def start_scheduler(live_interval_mins: int = LIVE_PREDICTION_INTERVAL_MINS):
    """
    Start the background scheduler with both jobs.

    Parameters
    ----------
    live_interval_mins : int
        How often (in minutes) to run the live prediction job.
    """
    global _scheduler

    if not _APSCHEDULER_AVAILABLE:
        log.error("Cannot start scheduler — apscheduler not installed.")
        return

    if _scheduler and _scheduler.running:
        log.info("Scheduler already running.")
        return

    _scheduler = BackgroundScheduler(timezone="UTC")

    # Job 1: Cricsheet daily refresh
    _scheduler.add_job(
        job_cricsheet_poll,
        trigger=IntervalTrigger(hours=CRICSHEET_POLL_INTERVAL_HOURS),
        id="cricsheet_poll",
        name="Cricsheet daily refresh",
        replace_existing=True,
        next_run_time=None,   # Don't run immediately on startup
    )

    # Job 2: Live prediction
    _scheduler.add_job(
        job_live_predict,
        trigger=IntervalTrigger(minutes=live_interval_mins),
        id="live_predict",
        name=f"Live prediction (every {live_interval_mins} min)",
        replace_existing=True,
    )

    _scheduler.start()
    log.info(
        f"Scheduler started.\n"
        f"  • Cricsheet refresh: every {CRICSHEET_POLL_INTERVAL_HOURS}h\n"
        f"  • Live prediction  : every {live_interval_mins} min"
    )


def stop_scheduler():
    """Gracefully shut down the background scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("Scheduler stopped.")
    _scheduler = None


def set_active_match(match_id: str):
    """Tell the live prediction job which match to monitor."""
    global _active_match_id, _consecutive_errors
    _active_match_id   = match_id
    _consecutive_errors = 0
    log.info(f"Active match set to: {match_id}")


def clear_active_match():
    """Stop monitoring any specific match (revert to auto-discover)."""
    global _active_match_id
    _active_match_id = None
    log.info("Active match cleared — will auto-discover on next poll.")


def run_cricsheet_now():
    """Trigger an immediate Cricsheet poll (bypasses the schedule)."""
    job_cricsheet_poll()


def run_prediction_now(match_id: Optional[str] = None):
    """Trigger an immediate live prediction (bypasses the schedule)."""
    global _active_match_id
    if match_id:
        _active_match_id = match_id
    job_live_predict()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cricket Momentum Background Scheduler")
    parser.add_argument("--interval", type=int, default=LIVE_PREDICTION_INTERVAL_MINS,
                        help="Live prediction interval in minutes (default: 5)")
    parser.add_argument("--match-id", default=None,
                        help="Pin the scheduler to a specific match ID")
    parser.add_argument("--cricsheet-only", action="store_true",
                        help="Run a single Cricsheet poll and exit")
    args = parser.parse_args()

    if args.cricsheet_only:
        log.info("Running one-shot Cricsheet poll…")
        job_cricsheet_poll()
        sys.exit(0)

    if args.match_id:
        _active_match_id = args.match_id

    start_scheduler(live_interval_mins=args.interval)

    log.info("Scheduler running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        stop_scheduler()
        log.info("Scheduler terminated.")