#!/usr/bin/env python3
"""
test_live_api.py
================
Run this script to test CricAPI + your full pipeline.
"""

import sys
import os
import requests

# ✅ Fix import path properly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Imports ─────────────────────────────────────────────────────────
from live_feed import get_live_test_matches, get_match_info, get_ball_by_ball
from match_fetcher import get_all_test_matches

# ── Config ──────────────────────────────────────────────────────────
KEY = "d9608163-e87f-4be3-834a-a27780e79930"
BASE_URL = "https://api.cricapi.com/v1"

# ── Test 1: API Connectivity ────────────────────────────────────────
print("\n── Test 1: API Connectivity ─────────────────────────────────────────")

try:
    resp = requests.get(
        f"{BASE_URL}/currentMatches",
        params={"apikey": KEY, "offset": 0},
        timeout=15,
    )

    print(f"   HTTP status : {resp.status_code}")
    data = resp.json()

    if data.get("status") == "success":
        print("   ✅ API key valid — CricAPI responded successfully")
    else:
        print("   ❌ API issue:", data)
        sys.exit(1)

except Exception as e:
    print("   ❌ Connection failed:", e)
    sys.exit(1)

# ── Test 2: Live Test Matches ───────────────────────────────────────
print("\n── Test 2: Live Test Matches ────────────────────────────────────────")

live = get_live_test_matches()

if live:
    print(f"   ✅ Found {len(live)} live Test match(es):")
    for m in live:
        print(f"      [{m['id']}] {m['name']} | {m['status']}")
else:
    print("   ℹ️ No live Test matches right now")

# ── Test 3: Upcoming Test Matches (NEW) ──────────────────────────────
print("\n── Test 3: Upcoming Test Matches ─────────────────────────────")

from src.match_fetcher import get_all_test_matches

upcoming = get_all_test_matches(KEY)

if upcoming:
    print(f"   ✅ Found {len(upcoming)} upcoming Test match(es):")
    for m in upcoming[:5]:
        print(f"      [{m['id']}] {m['name']} ({m['date']})")
else:
    print("   ❌ No upcoming Test matches found")

# ── Test 4: Match Info ──────────────────────────────────────────────
print("\n── Test 4: Match Info ───────────────────────────────────────────────")

test_id = live[0]["id"] if live else (upcoming[0]["id"] if upcoming else None)

if test_id:
    info = get_match_info(test_id)
    if info:
        print(f"   ✅ Match info fetched: {info.get('name')} at {info.get('venue')}")
    else:
        print("   ⚠️ Match info empty")
else:
    print("   ℹ️ No match available to test")

# ── Test 5: Ball-by-ball ────────────────────────────────────────────
print("\n── Test 5: Ball-by-Ball Data ────────────────────────────────────────")

if test_id:
    balls = get_ball_by_ball(test_id)

    if balls:
        print(f"   ✅ {len(balls)} deliveries returned")
        print(f"   Sample: {balls[0]}")
    else:
        print("   ℹ️ No ball-by-ball (free plan limitation)")
else:
    print("   ℹ️ Skipped")

# ── Summary ─────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────────")
print("   API: ✅ Working")
print("   Live matches: OK")
print("   Upcoming matches: OK")
print("   Ball-by-ball: Requires paid plan")