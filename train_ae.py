import sys, os
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.ae import Autoencoder
from src.preprocessing.feature_engineering import build_feature_matrix
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_parquet("data/processed/features_clean.parquet")
X = build_feature_matrix(df).values

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.joblib")

tensor = torch.tensor(Xs, dtype=torch.float32)
loader = DataLoader(TensorDataset(tensor), batch_size=128, shuffle=True)

model = Autoencoder(input_dim=tensor.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epoch in range(20):
    for (batch,) in loader:
        opt.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch)
        loss.backward()
        opt.step()
    print(f"Epoch {epoch} Loss {loss.item():.4f}")

torch.save(model.state_dict(), "models/ae.pt")
