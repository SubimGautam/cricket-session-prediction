# 🏏 Cricket Momentum Prediction System

Predict momentum shifts across Test cricket sessions using ball-by-ball data.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Check your data path
Edit `src/config.py` and make sure `DATA_RAW` points to your YAML folder:
```python
DATA_RAW = "~/Documents/cricket-data/tests"   # default — update if needed
```

### 3. Test with a small sample first
```bash
python main.py --sample 20
```
This loads just 20 matches so you can verify everything works quickly.

### 4. Run the full pipeline
```bash
python main.py
```

### 5. View your charts
Open the `outputs/charts/` folder — you'll find 6 PNG files:

| File | What it shows |
|------|--------------|
| `01_overview.png` | Runs distribution, wicket types, run rate by over |
| `02_session_analysis.png` | S1 vs S2 vs S3 comparison |
| `03_momentum_heatmap.png` | Momentum index across all overs & innings |
| `04_momentum_wave.png` | Single match momentum wave with wicket markers |
| `05_top_players.png` | Top batters & bowlers |
| `06_wicket_clusters.png` | How wickets kill momentum |

## Project Structure
```
cricket_momentum/
├── main.py              ← Run this
├── setup.py             ← Run first to install packages
├── requirements.txt
├── src/
│   ├── config.py        ← Set your data path here
│   ├── load_data.py     ← YAML → DataFrame
│   └── visualize.py     ← All charts
├── data/
│   ├── raw/             ← (your YAMLs stay in Documents/)
│   └── processed/       ← ball_by_ball.csv saved here
└── outputs/
    └── charts/          ← All PNGs saved here
```

## Useful Commands
```bash
# Quick test (20 files)
python main.py --sample 20

# Full run
python main.py

# Re-run charts without re-loading data
python main.py --charts-only
```
