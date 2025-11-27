import sys, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd
from src.preprocessing.feature_engineering import build_feature_matrix
from src.models.ocsvm import OCSVMModel
import joblib

df = pd.read_parquet("data/processed/features_clean.parquet")
X = build_feature_matrix(df).values

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
scaler = joblib.load("models/scaler.joblib")
Xs = scaler.transform(X)

model = OCSVMModel()
model.train(Xs)
model.save("models/ocsvm.joblib")
