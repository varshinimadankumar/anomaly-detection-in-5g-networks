import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import pandas as pd
from src.preprocessing.windowing import generate_windows
from src.preprocessing.feature_engineering import build_feature_matrix

# 1. Load normalized parquet
df = pd.read_parquet("data/processed/normalized.parquet")

# 2. Create windows
windows = generate_windows(df, window=5, step=2)
windows.to_parquet("data/processed/windows.parquet")
print("✓ Windows saved.")

# 3. Build ML-ready features
features = build_feature_matrix(windows)
features.to_parquet("data/processed/features.parquet")
print("✓ Features saved.")
