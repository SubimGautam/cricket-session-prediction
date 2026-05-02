# src/live_feed.py
"""
Live data feed from CricAPI (https://api.cricapi.com).

IMPORTANT — API Key:
    Your key is registered at https://cricapi.com/dashboard
    The correct base URL is: https://api.cricapi.com/v1
    (NOT api.cricdata.org — that domain does not exist)

CricAPI Free-tier relevant endpoints used here:
    GET /currentMatches    → live & recent matches
    GET /match_info        → match metadata by id
    GET /match_bbb         → ball-by-ball data (paid tier)

Refer to: https://cricapi.com/apidocs/ for full docs.
"""

import requests
from typing import List, Dict, Optional

# ── API Configuration ────────────────────────────────────────────────────────
CRICAPI_KEY  = "d9608163-e87f-4be3-834a-a27780e79930"   # Your CricAPI key
BASE_URL     = "https://api.cricapi.com/v1"              # ← CORRECTED URL

# ─────────────────────────────────────────────────────────────────────────────

class CricAPI:
    """
    Thin wrapper around CricAPI v1.

    Usage:
        api = CricAPI(CRICAPI_KEY)
        matches = api.get_live_matches()          # currently live
        info    = api.get_match_info(match_id)
        balls   = api.get_ball_by_ball(match_id)
    """

    def __init__(self, api_key: str):
        self.api_key  = api_key
        self.base_url = BASE_URL
        self.session  = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ── Internal request helper ──────────────────────────────────────────────

    def _get(self, endpoint: str, extra_params: dict = None) -> dict:
        """
        Send GET request to CricAPI.  All requests pass the apikey as a query
        param (CricAPI ignores header-based auth for most endpoints).
        Returns the parsed JSON dict, or {} on error.
        """
        params = {"apikey": self.api_key, "offset": 0}
        if extra_params:
            params.update(extra_params)

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # CricAPI wraps successful responses in {"status":"success","data":[...]}
            if data.get("status") != "success":
                print(f"[CricAPI] Non-success response for /{endpoint}: "
                      f"{data.get('reason', data.get('status', 'unknown'))}")
                return {}
            return data

        except requests.exceptions.ConnectionError as e:
            print(f"[CricAPI] Connection error — check your internet / API key: {e}")
            return {}
        except requests.exceptions.HTTPError as e:
            print(f"[CricAPI] HTTP {resp.status_code} on /{endpoint}: {e}")
            return {}
        except requests.exceptions.RequestException as e:
            print(f"[CricAPI] Request failed for /{endpoint}: {e}")
            return {}

    # ── Public methods ────────────────────────────────────────────────────────

    def get_live_matches(self) -> List[dict]:
        """
        Fetch currently live matches and return only Test matches.

        CricAPI endpoint: GET /currentMatches
        Response shape: {"status":"success","data":[{match}, ...]}

        Each match dict contains:
            id, name, matchType, status, venue, teams, teamInfo,
            score, series_id, dateTimeGMT, ...
        """
        data     = self._get("currentMatches")
        all_matches = data.get("data", [])

        test_matches = []
        for m in all_matches:
            match_type = str(m.get("matchType", "")).lower()
            if match_type != "test":
                continue

            # Normalise team names
            teams = m.get("teams", [])
            team1 = teams[0] if len(teams) > 0 else "Unknown"
            team2 = teams[1] if len(teams) > 1 else "Unknown"

            test_matches.append({
                "id"     : m.get("id", ""),
                "name"   : m.get("name",  f"{team1} vs {team2}"),
                "team1"  : team1,
                "team2"  : team2,
                "status" : m.get("status",  "Live"),
                "venue"  : m.get("venue",   "Unknown"),
                "series" : m.get("series",  ""),
                "date"   : m.get("dateTimeGMT", ""),
                "score"  : m.get("score",   []),
            })

        print(f"[CricAPI] Found {len(test_matches)} live Test match(es).")
        return test_matches

    def get_match_info(self, match_id: str) -> dict:
        """
        Fetch full metadata for a single match.

        CricAPI endpoint: GET /match_info?id=<match_id>
        """
        data  = self._get("match_info", {"id": match_id})
        match = data.get("data", {})

        if not match:
            return {}

        teams = match.get("teams", [])
        return {
            "id"          : match.get("id", match_id),
            "name"        : match.get("name", ""),
            "matchType"   : match.get("matchType", "test"),
            "status"      : match.get("status", ""),
            "venue"       : match.get("venue",  ""),
            "toss"        : match.get("toss",   {}),
            "teams"       : teams,
            "teamInfo"    : match.get("teamInfo", []),
            "score"       : match.get("score",   []),
            "series_id"   : match.get("series_id", ""),
            "dateTimeGMT" : match.get("dateTimeGMT", ""),
            "fantasyEnabled": match.get("fantasyEnabled", False),
        }

    def get_ball_by_ball(self, match_id: str) -> List[dict]:
        """
        Fetch ball-by-ball data.

        CricAPI endpoint: GET /match_bbb?id=<match_id>
        NOTE: Ball-by-ball is available on paid CricAPI tiers.
              On the free tier this returns an empty list.

        Returns a standardised list of delivery dicts compatible with
        live_feature_builder.LiveFeatureBuilder.
        """
        data       = self._get("match_bbb", {"id": match_id})
        raw_balls  = data.get("data", {})

        # CricAPI returns a dict keyed by innings: {"t1": [...], "t2": [...]}
        # or a list depending on the tier / version.
        deliveries: List[dict] = []

        if isinstance(raw_balls, list):
            source_lists = [raw_balls]
        elif isinstance(raw_balls, dict):
            source_lists = list(raw_balls.values())
        else:
            return []

        for innings_idx, balls in enumerate(source_lists, start=1):
            for ball in balls:
                r = ball.get("r", ball.get("runs", {}))
                if isinstance(r, int):
                    # flat format: r is total runs
                    batter_runs = r
                    extras_runs = 0
                    total_runs  = r
                else:
                    batter_runs = int(r.get("batter", r.get("batsman", 0)))
                    extras_runs = int(r.get("extras", 0))
                    total_runs  = int(r.get("total",  batter_runs + extras_runs))

                # Wicket
                wickets     = ball.get("wickets", ball.get("wkts", []))
                if isinstance(wickets, int):
                    wickets = []       # some endpoints return cumulative wicket count

                deliveries.append({
                    "innings"    : innings_idx,
                    "over"       : int(ball.get("over",    ball.get("ov", 0))),
                    "ball"       : int(ball.get("ball",    ball.get("b",  0))),
                    "batter"     : ball.get("batter",      ball.get("bat", "")),
                    "bowler"     : ball.get("bowler",      ball.get("bowl", "")),
                    "runs"       : {
                        "batter" : batter_runs,
                        "extras" : extras_runs,
                        "total"  : total_runs,
                    },
                    "wickets"    : wickets,
                    "extras"     : ball.get("extras",  {}),
                    "commentary" : ball.get("c",       ball.get("commentary", "")),
                })

        return deliveries

    def get_upcoming_test_matches(self, days_ahead: int = 30) -> List[dict]:
        """
        Fetch upcoming Test matches scheduled within the next `days_ahead` days.

        CricAPI endpoint: GET /matches?offset=0
        """
        data     = self._get("matches")
        all_matches = data.get("data", [])

        upcoming = []
        for m in all_matches:
            match_type = str(m.get("matchType", "")).lower()
            if match_type != "test":
                continue
            if m.get("matchStarted", False):
                continue          # already started → not "upcoming"

            teams = m.get("teams", [])
            upcoming.append({
                "id"     : m.get("id", ""),
                "name"   : m.get("name", ""),
                "team1"  : teams[0] if len(teams) > 0 else "Unknown",
                "team2"  : teams[1] if len(teams) > 1 else "Unknown",
                "date"   : m.get("dateTimeGMT", ""),
                "venue"  : m.get("venue", ""),
                "series" : m.get("series", ""),
            })

        return upcoming


# ── Module-level convenience API instance ────────────────────────────────────
_api = CricAPI(CRICAPI_KEY)


# ── Public convenience functions (used by Dashboard.py & live_predictor.py) ─

def get_live_test_matches() -> List[dict]:
    """Return currently live Test matches."""
    return _api.get_live_matches()


def get_ball_by_ball(match_id: str) -> List[dict]:
    """Return ball-by-ball data for `match_id`."""
    return _api.get_ball_by_ball(match_id)


def get_match_info(match_id: str) -> dict:
    """Return metadata for `match_id`."""
    return _api.get_match_info(match_id)


def get_upcoming_matches(days_ahead: int = 30) -> List[dict]:
    """Return upcoming Test matches."""
    return _api.get_upcoming_test_matches(days_ahead)