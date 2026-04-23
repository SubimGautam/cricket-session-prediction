"""
RUN THIS FIRST: python setup.py
Installs all required packages and verifies your data folder.
"""
import subprocess
import sys
import os

packages = [
    "pandas", "numpy", "matplotlib", "seaborn",
    "plotly", "pyyaml", "scikit-learn", "xgboost", "tqdm"
]

print("📦 Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

print("\n✅ All packages installed!\n")

# Check data path
DATA_PATH = os.path.expanduser("~/Documents/cricket-data/tests")
if os.path.exists(DATA_PATH):
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".yaml")]
    print(f"✅ Found {len(files)} YAML files in {DATA_PATH}")
else:
    print(f"⚠️  Could not find data at: {DATA_PATH}")
    print("   Please update DATA_PATH in src/config.py if your folder is elsewhere.")
