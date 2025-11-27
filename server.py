import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from src.inference.predict_utils import Predictor

app = FastAPI(title="5G Anomaly Detection API")

predictor = Predictor()


class PredictRequest(BaseModel):
    data: list  # list of dict rows


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame(req.data)
    scored = predictor.predict(df)

    return {
        "count": len(scored),
        "scores": scored[
            ["ae_score", "ocsvm_score", "rf_score", "gnn_score", "fused_score"]
        ].to_dict(orient="records")
    }
