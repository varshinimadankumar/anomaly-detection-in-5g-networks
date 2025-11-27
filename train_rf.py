import sys, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd
from src.preprocessing.feature_engineering import build_feature_matrix
from src.models.rf import RFModel
import joblib

df = pd.read_parquet("data/processed/labeled_features.parquet")

X = build_feature_matrix(df).values
y = df["label"].values

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

scaler = joblib.load("models/scaler.joblib")
Xs = scaler.transform(X)

model = RFModel()
model.train(Xs, y)
model.save("models/rf.joblib")
