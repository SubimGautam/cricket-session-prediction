"""
main.py — Run the full Cricket Momentum pipeline.

Steps:
    1. Load all YAML files
    2. Save processed CSV
    3. Generate all visualizations

Usage:
    python main.py                  # full run (all 983 files)
    python main.py --sample 50      # quick test with 50 files
    python main.py --charts-only    # skip loading, use existing CSV
"""

import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Cricket Momentum System")
    parser.add_argument("--sample", type=int, default=None,
                        help="Load only N files (for quick testing)")
    parser.add_argument("--charts-only", action="store_true",
                        help="Skip loading, use existing processed CSV")
    args = parser.parse_args()

    from src.config import DATA_PROC

    if args.charts_only:
        csv_path = os.path.join(DATA_PROC, "ball_by_ball.csv")
        if not os.path.exists(csv_path):
            print("❌ No processed CSV found. Run without --charts-only first.")
            return
        print(f"📂 Loading from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df):,} rows.")
    else:
        from src.load_data import load_all_matches, save_processed
        df = load_all_matches(max_files=args.sample)
        save_processed(df)

    from src.visualize import run_all_charts
    run_all_charts(df)

    print("\n🏁 Done! Open outputs/charts/ to see your visualizations.")

if __name__ == "__main__":
    main()
