import sys, os
import numpy as np

# Add project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import torch
import joblib

from src.models.ae import Autoencoder
from src.models.ensemble import fuse_scores
from src.preprocessing.feature_engineering import build_feature_matrix


def main():
    print("🔹 Loading cleaned features...")
    df = pd.read_parquet("data/processed/features_clean.parquet")
    X = build_feature_matrix(df).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler_path = "models/scaler.joblib"
    ae_path = "models/ae.pt"
    ocsvm_path = "models/ocsvm.joblib"
    rf_path = "models/rf.joblib"

    if not (os.path.exists(scaler_path) and os.path.exists(ae_path)):
        raise FileNotFoundError("AE and scaler must be trained before the ensemble.")

    print("🔹 Loading scaler...")
    scaler = joblib.load(scaler_path)
    Xs = scaler.transform(X)

    # ---- AE SCORE (reconstruction error) ----
    print("🔹 Loading AE model and computing scores...")
    device = "cpu"
    input_dim = Xs.shape[1]
    ae = Autoencoder(input_dim)
    ae.load_state_dict(torch.load(ae_path, map_location=device))
    ae.to(device)
    ae.eval()

    with torch.no_grad():
        tX = torch.tensor(Xs, dtype=torch.float32, device=device)
        recon = ae(tX).cpu().numpy()
    ae_score = ((recon - Xs) ** 2).mean(axis=1)

    # ---- OCSVM SCORE ----
    if os.path.exists(ocsvm_path):
        print("🔹 Loading OCSVM and computing scores...")
        ocsvm = joblib.load(ocsvm_path)
        # Decision_function gives distances from boundary (higher = more normal)
        # We invert so that higher = more anomalous
        oc_dec = ocsvm.decision_function(Xs)
        ocsvm_score = -oc_dec
    else:
        print("⚠️ OCSVM model not found, using zeros.")
        ocsvm_score = np.zeros_like(ae_score)

    # ---- RF SCORE ----
    if os.path.exists(rf_path):
        print("🔹 Loading RF and computing scores...")
        rf = joblib.load(rf_path)
        if hasattr(rf, "predict_proba"):
            rf_score = rf.predict_proba(Xs)[:, 1]   # prob of anomaly class
        else:
            rf_score = rf.predict(Xs)
    else:
        print("⚠️ RF model not found, using zeros.")
        rf_score = np.zeros_like(ae_score)

    # ---- FUSE SCORES ----
    print("🔹 Fusing scores (AE + OCSVM + RF)...")
    scores_dict = {
        "ae": ae_score,
        "ocsvm": ocsvm_score,
        "rf": rf_score,
    }
    fused = fuse_scores(scores_dict)

    out_df = df.copy()
    out_df["ae_score"] = ae_score
    out_df["ocsvm_score"] = ocsvm_score
    out_df["rf_score"] = rf_score
    out_df["fused_score"] = fused

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/ensemble_scores.parquet"
    out_df.to_parquet(out_path)
    print(f"✅ Ensemble scores saved to {out_path}")


if __name__ == "__main__":
    main()
