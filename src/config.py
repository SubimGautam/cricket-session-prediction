"""
config.py — Central configuration for the Cricket Momentum project.
Update DATA_PATH if your data is in a different location.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW   = os.path.expanduser("~/Documents/cricket data/tests")  # ← your YAML folder
DATA_PROC  = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS    = os.path.join(BASE_DIR, "outputs", "charts")

# ── Session boundaries (overs) ─────────────────────────────────────────────
SESSION_MAP = {1: (1, 30), 2: (31, 60), 3: (61, 90)}

# ── Momentum thresholds ────────────────────────────────────────────────────
MOMENTUM_GAIN_THRESHOLD  =  1.5
MOMENTUM_LOSS_THRESHOLD  = -1.5

# ── Model ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
