# src/state_manager.py
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime

DB_PATH = Path("outputs/live_state.db")

def init_db():
    """Create tables if they don't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id    TEXT,
                innings_num INTEGER,
                session_name TEXT,
                predicted_label INTEGER,
                confidence  REAL,
                features_json TEXT,
                created_at  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS wp_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id    TEXT,
                timestamp   TEXT,
                wp_batting  REAL,
                wp_bowling  REAL,
                session_name TEXT
            )
        """)

def save_prediction(result: dict):
    """Persist a prediction result to SQLite."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions
            (match_id, innings_num, session_name, predicted_label,
             confidence, features_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result["match_id"],
            result["innings_num"],
            result["session_name"],
            result["predicted_label"],
            result["confidence"],
            json.dumps(result["features"]),
            datetime.utcnow().isoformat(),
        ))

def save_wp_point(match_id: str, wp_batting: float,
                  wp_bowling: float, session_name: str):
    """Append one WP data point — called after each prediction."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO wp_history
            (match_id, timestamp, wp_batting, wp_bowling, session_name)
            VALUES (?, ?, ?, ?, ?)
        """, (match_id, datetime.utcnow().isoformat(),
              wp_batting, wp_bowling, session_name))

def get_wp_history(match_id: str) -> list[dict]:
    """Retrieve full WP trajectory for a match — for the chart."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT timestamp, wp_batting, wp_bowling, session_name
            FROM wp_history
            WHERE match_id = ?
            ORDER BY id ASC
        """, (match_id,)).fetchall()
    return [
        {"timestamp": r[0], "wp_batting": r[1],
         "wp_bowling": r[2], "session": r[3]}
        for r in rows
    ]

def get_prediction_log(match_id: str) -> list[dict]:
    """All predictions made for a match — for the audit trail."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT innings_num, session_name, predicted_label,
                   confidence, created_at
            FROM predictions
            WHERE match_id = ?
            ORDER BY id ASC
        """, (match_id,)).fetchall()
    return [
        {"innings": r[0], "session": r[1], "label": r[2],
         "confidence": r[3], "at": r[4]}
        for r in rows
    ]