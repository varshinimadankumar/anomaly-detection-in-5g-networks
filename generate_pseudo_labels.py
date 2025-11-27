import sys, os
import numpy as np

# Add project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd
import torch
import joblib

from src.models.ae import Autoencoder
from src.preprocessing.feature_engineering import build_feature_matrix


def generate_pseudo_labels():

    print("🔹 Loading features ...")
    df = pd.read_parquet("data/processed/features_clean.parquet")
    X = build_feature_matrix(df).values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------------------------------------------------
    # Load models
    # ---------------------------------------------------------
    print("🔹 Loading AE, scaler, OCSVM...")

    scaler = joblib.load("models/scaler.joblib")

    ae = Autoencoder(X.shape[1])
    ae.load_state_dict(torch.load("models/ae.pt", map_location="cpu"))
    ae.eval()

    try:
        ocsvm = joblib.load("models/ocsvm.joblib")
        ocsvm_available = True
    except:
        print("⚠️ OCSVM not found. Will only use AE & statistical methods.")
        ocsvm_available = False

    Xs = scaler.transform(X)

    # ---------------------------------------------------------
    # AE reconstruction score
    # ---------------------------------------------------------
    print("🔹 Computing AE reconstruction error...")
    with torch.no_grad():
        tX = torch.tensor(Xs, dtype=torch.float32)
        recon = ae(tX).numpy()

    ae_score = ((Xs - recon) ** 2).mean(axis=1)

    # ---------------------------------------------------------
    # OCSVM score
    # ---------------------------------------------------------
    if ocsvm_available:
        oc_dec = ocsvm.decision_function(Xs)
        ocsvm_score = -oc_dec
    else:
        ocsvm_score = np.zeros_like(ae_score)

    # ---------------------------------------------------------
    # Statistical anomaly score
    # ---------------------------------------------------------
    print("🔹 Computing statistical anomaly scores...")
    z_scores = np.abs((Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-8))
    stat_score = z_scores.mean(axis=1)

    # ---------------------------------------------------------
    # Traffic burst anomaly indicators
    # ---------------------------------------------------------
    packet_count = df["packet_count"].values
    byte_count = df["byte_count"].values

    packet_z = (packet_count - packet_count.mean()) / (packet_count.std() + 1e-8)
    byte_z   = (byte_count - byte_count.mean())   / (byte_count.std() + 1e-8)

    burst_score = (np.abs(packet_z) + np.abs(byte_z)) / 2

    # ---------------------------------------------------------
    # Combine all scores (weighted fusion)
    # ---------------------------------------------------------
    print("🔹 Fusing scores...")
    fused = (
        0.40 * ae_score +
        0.25 * ocsvm_score +
        0.20 * stat_score +
        0.15 * burst_score
    )

    # normalize fused score 0–1
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

    # ---------------------------------------------------------
    # Threshold for anomaly labeling
    # ---------------------------------------------------------
    threshold = np.quantile(fused, 0.90)  # Top 10% considered anomalies
    labels = (fused >= threshold).astype(int)

    df_out = df.copy()
    df_out["ae_score"] = ae_score
    df_out["ocsvm_score"] = ocsvm_score
    df_out["stat_score"] = stat_score
    df_out["burst_score"] = burst_score
    df_out["fused_score"] = fused
    df_out["label"] = labels

    # Save labeled dataset
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/labeled_features.parquet"
    df_out.to_parquet(out_path)

    print(f"✅ Labeled dataset saved to {out_path}")
    print("   (label=1 → anomaly, label=0 → normal)")


if __name__ == "__main__":
    generate_pseudo_labels()
