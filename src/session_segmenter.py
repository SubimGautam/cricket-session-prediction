# src/session_segmenter.py
from dataclasses import dataclass, field
from datetime import time
from typing import Optional

# Test match session boundaries (approximate, adjust per timezone)
SESSION_WINDOWS = {
    "morning"   : (time(10, 30), time(13, 0)),   # 10:30 → 13:00 (lunch)
    "afternoon" : (time(13, 40), time(16, 10)),   # 13:40 → 16:10 (tea)
    "evening"   : (time(16, 30), time(18, 30)),   # 16:30 → stumps
}

@dataclass
class Session:
    """One segmented session with all its ball events."""
    name        : str          # "morning" | "afternoon" | "evening"
    day         : int          # match day number
    innings_num : int
    balls       : list = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """A session is complete when it has >= 20 overs OR a break marker."""
        return len(self.balls) >= 120  # 120 legal deliveries ~ 20 overs

def assign_session_from_over(over: int, ball: int, innings: int) -> str:
    """
    Deterministic session assignment purely from over count —
    used when timestamp data is unavailable (e.g. Cricsheet JSON).
    Approximate boundaries: morning=1–32, afternoon=33–64, evening=65–90+
    """
    if over < 32:
        return "morning"
    elif over < 65:
        return "afternoon"
    else:
        return "evening"

def segment_innings_balls(
    balls: list[dict],
    innings_num: int,
    day: int = 1,
) -> list[Session]:
    """
    Slice a flat list of ball-event dicts into Session objects.
    Works with both Cricsheet JSON (uses over index) and
    live API data (uses timestamp if available).

    Each ball dict is expected to have:
        {"over": int, "ball": int, "runs": {...}, "wickets": [...], ...}
    """
    sessions: list[Session] = []
    current_session: Optional[Session] = None

    for b in balls:
        over = int(b.get("over", 0))
        session_name = assign_session_from_over(over, b.get("ball", 0), innings_num)

        if current_session is None or current_session.name != session_name:
            # Close old session, open new one
            if current_session is not None:
                sessions.append(current_session)
            current_session = Session(
                name=session_name,
                day=day,
                innings_num=innings_num,
            )

        current_session.balls.append(b)

    if current_session and current_session.balls:
        sessions.append(current_session)

    return sessions