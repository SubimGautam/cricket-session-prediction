"""
src/load_data.py
Reads all .yaml cricket files from your data folder and converts them
into a single ball-by-ball pandas DataFrame.

Usage:
    from src.load_data import load_all_matches
    df = load_all_matches()
"""

import os
import yaml
import pandas as pd
from tqdm import tqdm
from src.config import DATA_RAW


def parse_match(filepath: str) -> pd.DataFrame:
    """Parse a single Cricsheet YAML file into ball-by-ball rows.
    Supports both old format (batsman, 0.1 keys) and new format (batter, over blocks).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        match = yaml.safe_load(f)

    info = match.get("info", {})
    meta = {
        "match_id"     : os.path.splitext(os.path.basename(filepath))[0],
        "teams"        : " vs ".join(info.get("teams", [])),
        "venue"        : info.get("venue", "Unknown"),
        "city"         : info.get("city", "Unknown"),
        "dates"        : str(info.get("dates", [""])[0]),
        "toss_winner"  : info.get("toss", {}).get("winner", ""),
        "toss_decision": info.get("toss", {}).get("decision", ""),
        "winner"       : info.get("outcome", {}).get("winner", "draw/no result"),
    }

    rows = []
    for innings_num, innings_block in enumerate(match.get("innings", []), start=1):
        # Old format: {"1st innings": {"team": "...", "deliveries": [...]}}
        # New format: {"team": "...", "overs": [...]}
        for innings_key, innings_data in innings_block.items():
            if not isinstance(innings_data, dict):
                continue

            batting_team = innings_data.get("team", innings_key)

            # ── Old format: flat deliveries list with "0.1" style keys ──
            if "deliveries" in innings_data:
                for ball_block in innings_data["deliveries"]:
                    for ball_key, delivery in ball_block.items():
                        # ball_key like "0.1" → over=1, ball=1
                        try:
                            over_f = float(ball_key)
                            over_num = int(over_f) + 1          # 0-indexed → 1-indexed
                            ball_num = round((over_f % 1) * 10) # 0.1 → ball 1
                        except (ValueError, TypeError):
                            over_num, ball_num = 0, 0

                        row = _parse_delivery_old(
                            delivery, innings_num, batting_team, over_num, ball_num
                        )
                        row.update(meta)
                        rows.append(row)

            # ── New format: overs list with delivery dicts ──
            elif "overs" in innings_data:
                for over_block in innings_data["overs"]:
                    over_num = over_block.get("over", 0) + 1
                    for delivery in over_block.get("deliveries", []):
                        row = _parse_delivery_new(
                            delivery, innings_num, batting_team, over_num
                        )
                        row.update(meta)
                        rows.append(row)

    return pd.DataFrame(rows)


def _parse_delivery_old(delivery: dict, innings: int, batting_team: str,
                        over: int, ball_num) -> dict:
    """Parse old Cricsheet format (uses 'batsman', flat runs dict)."""
    runs        = delivery.get("runs", {})
    batter_runs = runs.get("batsman", 0)   # old format uses "batsman"
    extras_val  = runs.get("extras", 0)
    total_runs  = runs.get("total", batter_runs + extras_val)

    wicket      = delivery.get("wicket", {})  # old format: single dict not list
    is_wicket   = 1 if wicket else 0
    wicket_kind = wicket.get("kind", "") if wicket else ""
    dismissed   = wicket.get("player_out", "") if wicket else ""

    extras_dict = delivery.get("extras", {})

    return {
        "innings"       : innings,
        "batting_team"  : batting_team,
        "over"          : over,
        "ball"          : ball_num,
        "batter"        : delivery.get("batsman", ""),
        "bowler"        : delivery.get("bowler", ""),
        "batter_runs"   : batter_runs,
        "extras"        : extras_val,
        "total_runs"    : total_runs,
        "is_wicket"     : is_wicket,
        "wicket_kind"   : wicket_kind,
        "dismissed"     : dismissed,
        "wides"         : extras_dict.get("wides", 0),
        "noballs"       : extras_dict.get("noballs", 0),
        "byes"          : extras_dict.get("byes", 0),
        "is_boundary_4" : 1 if batter_runs == 4 else 0,
        "is_boundary_6" : 1 if batter_runs == 6 else 0,
        "is_dot"        : 1 if total_runs == 0 else 0,
    }


def _parse_delivery_new(delivery: dict, innings: int, batting_team: str,
                        over: int) -> dict:
    """Parse new Cricsheet format (uses 'batter', wickets as list)."""
    runs        = delivery.get("runs", {})
    batter_runs = runs.get("batter", 0)
    extras_val  = runs.get("extras", 0)
    total_runs  = runs.get("total", batter_runs + extras_val)

    wicket_info = delivery.get("wickets", [])
    is_wicket   = 1 if wicket_info else 0
    wicket_kind = wicket_info[0].get("kind", "") if wicket_info else ""
    dismissed   = wicket_info[0].get("player_out", "") if wicket_info else ""

    extras_dict = delivery.get("extras", {})

    return {
        "innings"       : innings,
        "batting_team"  : batting_team,
        "over"          : over,
        "ball"          : delivery.get("ball", 0),
        "batter"        : delivery.get("batter", ""),
        "bowler"        : delivery.get("bowler", ""),
        "batter_runs"   : batter_runs,
        "extras"        : extras_val,
        "total_runs"    : total_runs,
        "is_wicket"     : is_wicket,
        "wicket_kind"   : wicket_kind,
        "dismissed"     : dismissed,
        "wides"         : extras_dict.get("wides", 0),
        "noballs"       : extras_dict.get("noballs", 0),
        "byes"          : extras_dict.get("byes", 0),
        "is_boundary_4" : 1 if batter_runs == 4 else 0,
        "is_boundary_6" : 1 if batter_runs == 6 else 0,
        "is_dot"        : 1 if total_runs == 0 else 0,
    }


def load_all_matches(max_files: int = None, verbose: bool = True) -> pd.DataFrame:
    """
    Load all YAML files from DATA_RAW.

    Args:
        max_files: limit number of files (useful for testing, e.g. max_files=50)
        verbose:   show progress bar

    Returns:
        Combined ball-by-ball DataFrame
    """
    if not os.path.exists(DATA_RAW):
        raise FileNotFoundError(
            f"Data folder not found: {DATA_RAW}\n"
            "Please update DATA_RAW in src/config.py"
        )

    yaml_files = sorted([
        os.path.join(DATA_RAW, f)
        for f in os.listdir(DATA_RAW)
        if f.endswith(".yaml")
    ])

    if max_files:
        yaml_files = yaml_files[:max_files]

    print(f"📂 Loading {len(yaml_files)} YAML files from:\n   {DATA_RAW}\n")

    dfs = []
    errors = []
    for fp in tqdm(yaml_files, disable=not verbose):
        try:
            dfs.append(parse_match(fp))
        except Exception as e:
            errors.append((fp, str(e)))

    if errors:
        print(f"\n⚠️  {len(errors)} files failed to parse (skipped).")

    df = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Loaded {len(df):,} deliveries from {len(dfs)} matches.")
    return df


def save_processed(df: pd.DataFrame, filename: str = "ball_by_ball.csv"):
    """Save cleaned DataFrame to data/processed/"""
    from src.config import DATA_PROC
    os.makedirs(DATA_PROC, exist_ok=True)
    path = os.path.join(DATA_PROC, filename)
    df.to_csv(path, index=False)
    print(f"💾 Saved to {path}")


if __name__ == "__main__":
    df = load_all_matches()
    save_processed(df)
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
