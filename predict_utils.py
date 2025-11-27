print("💡 USING PREDICT_UTILS FROM:", __file__)

import os, sys, numpy as np, pandas as pd
import torch, joblib
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.models.ae import Autoencoder
from src.models.ensemble import fuse_scores
from src.preprocessing.feature_engineering import build_feature_matrix
from src.models.gnn import GraphSAGE


class Predictor:
    def __init__(
        self,
        scaler_path="models/scaler.joblib",
        ae_path="models/ae.pt",
        ocsvm_path="models/ocsvm.joblib",
        rf_path="models/rf.joblib",
        gnn_path="models/gnn.pt",
        device="cpu"
    ):
        self.device = device

        self.scaler = joblib.load(scaler_path)
        INPUT_DIM = len(self.scaler.mean_)

        # AE
        self.ae = Autoencoder(INPUT_DIM)
        self.ae.load_state_dict(torch.load(ae_path, map_location=device))
        self.ae.eval()

        # OCSVM
        self.ocsvm = joblib.load(ocsvm_path) if os.path.exists(ocsvm_path) else None

        # RF
        self.rf = joblib.load(rf_path) if os.path.exists(rf_path) else None

        # GNN
        self.gnn = GraphSAGE(in_dim=INPUT_DIM, hidden_dim=64, out_dim=2)
        self.gnn.load_state_dict(torch.load(gnn_path, map_location=device))
        self.gnn.eval()

    def _compute_gnn_score(self, df):
        print("🔍 GNN SCORE FUNCTION EXECUTED — rows:", len(df))

        if self.gnn is None:
            print("⚠️ No GNN model loaded — returning zeros")
            return np.zeros(len(df))

        # Build feature matrix
        X = build_feature_matrix(df).values
        print("🔧 FEATURE MATRIX SHAPE:", X.shape)

        X = np.nan_to_num(X)
        Xs = self.scaler.transform(X)
        feats = torch.tensor(Xs, dtype=torch.float32)

        num_nodes = len(df)
        print("🧩 NUM NODES:", num_nodes)

        # Build chain edges
        if num_nodes > 1:
            src = torch.arange(0, num_nodes - 1)
            dst = torch.arange(1, num_nodes)
            edge_index = torch.stack([src, dst], dim=0)
            print("🔗 EDGE INDEX SHAPE:", edge_index.shape)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            print("🟦 SINGLE NODE — NO EDGES")

        # Run GNN
        out = self.gnn(feats, edge_index)
        prob = torch.softmax(out, dim=1)[:, 1].detach().numpy()

        print("📌 GNN OUTPUT SHAPE:", out.shape)

        return prob

    def predict(self, df):
        X = build_feature_matrix(df).values
        X = self.scaler.transform(np.nan_to_num(X))

        # AE
        with torch.no_grad():
            tX = torch.tensor(X, dtype=torch.float32)
            recon = self.ae(tX).numpy()
        ae_score = ((X - recon) ** 2).mean(axis=1)

        # OCSVM
        ocsvm_score = -self.ocsvm.decision_function(X) if self.ocsvm else np.zeros(len(df))

        # RF
        if self.rf:
            if hasattr(self.rf, "predict_proba"):
                rf_score = self.rf.predict_proba(X)[:, 1]
            else:
                rf_score = self.rf.predict(X)
        else:
            rf_score = np.zeros(len(df))

        # GNN
        gnn_score = self._compute_gnn_score(df)

        fused = fuse_scores({
            "ae": ae_score,
            "ocsvm": ocsvm_score,
            "rf": rf_score,
            "gnn": gnn_score
        })

        out = df.copy()
        out["ae_score"] = ae_score
        out["ocsvm_score"] = ocsvm_score
        out["rf_score"] = rf_score
        out["gnn_score"] = gnn_score
        out["fused_score"] = fused

        return out
