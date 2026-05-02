# src/state_manager.py
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

DB_PATH = Path("outputs/live_predictions.db")

def get_db_connection():
    """Get SQLite connection with row_factory for dict results."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Predictions log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            innings_num INTEGER,
            session_name TEXT,
            label INTEGER,
            confidence REAL,
            prob_batting REAL,
            prob_neutral REAL,
            prob_bowling REAL,
            features_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Win probability history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wp_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            wp_batting REAL,
            wp_bowling REAL,
            session_name TEXT,
            poll_number INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def save_prediction(result: Dict[str, Any]):
    """Save a prediction result to the database."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO predictions (
            match_id, timestamp, innings_num, session_name,
            label, confidence, prob_batting, prob_neutral,
            prob_bowling, features_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result["match_id"],
        datetime.utcnow().isoformat(),
        result.get("innings_num"),
        result.get("session_name"),
        result["predicted_label"],
        result["confidence"],
        result["prob_batting"],
        result["prob_neutral"],
        result["prob_bowling"],
        json.dumps(result.get("features", {}))
    ))
    
    conn.commit()
    conn.close()

def save_wp_point(match_id: str, wp_batting: float, wp_bowling: float, session_name: str):
    """Save a win probability point for trajectory tracking."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get next poll number
    cursor.execute("SELECT COUNT(*) as count FROM wp_history WHERE match_id = ?", (match_id,))
    poll_number = cursor.fetchone()["count"] + 1
    
    cursor.execute("""
        INSERT INTO wp_history (
            match_id, timestamp, wp_batting, wp_bowling,
            session_name, poll_number
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        match_id,
        datetime.utcnow().isoformat(),
        wp_batting,
        wp_bowling,
        session_name,
        poll_number
    ))
    
    conn.commit()
    conn.close()

def get_prediction_log(match_id: str, limit: int = 50) -> List[Dict]:
    """Retrieve prediction history for a match."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT timestamp, innings_num, session_name, label, confidence,
               prob_batting, prob_neutral, prob_bowling
        FROM predictions
        WHERE match_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (match_id, limit))
    
    return [dict(row) for row in cursor.fetchall()]

def get_wp_history(match_id: str) -> List[Dict]:
    """Retrieve win probability history for trajectory chart."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT timestamp, wp_batting, wp_bowling, session_name, poll_number
        FROM wp_history
        WHERE match_id = ?
        ORDER BY poll_number ASC
    """, (match_id,))
    
    return [dict(row) for row in cursor.fetchall()]