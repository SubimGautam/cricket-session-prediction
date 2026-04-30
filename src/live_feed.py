# src/live_feed.py
import requests
import pandas as pd
from datetime import datetime

CRICDATA_API_KEY = "your_api_key_here"  # from cricketdata.org
BASE_URL = "https://api.cricapi.com/v1"

def get_live_test_matches() -> list[dict]:
    """Fetch all currently live Test matches."""
    resp = requests.get(
        f"{BASE_URL}/currentMatches",
        params={"apikey": CRICDATA_API_KEY, "offset": 0},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    # Filter to Test matches only
    return [m for m in data if m.get("matchType", "").lower() == "test"]

def get_scorecard(match_id: str) -> dict:
    """Full scorecard for a match — includes innings, overs, players."""
    resp = requests.get(
        f"{BASE_URL}/match_scorecard",
        params={"apikey": CRICDATA_API_KEY, "id": match_id},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("data", {})

def get_ball_by_ball(match_id: str) -> list[dict]:
    """Ball-by-ball feed — available on paid tier."""
    resp = requests.get(
        f"{BASE_URL}/match_bbb",
        params={"apikey": CRICDATA_API_KEY, "id": match_id},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])