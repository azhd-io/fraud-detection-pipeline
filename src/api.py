# src/api.py
import os, json
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

MODEL_DIR = "models"

# ---- Load model + config at startup
cfg_path = os.path.join(MODEL_DIR, "inference_config.json")
with open(cfg_path, "r") as f:
    cfg = json.load(f)

model_name = cfg.get("model", "RandomForest")

# Choose which threshold to use: env var beats defaults
choice = os.getenv("THRESHOLD_CHOICE", "f1").lower()  # "f1" or "recall95"
if choice == "recall95":
    threshold = float(cfg.get("threshold_recall95", cfg.get("threshold_f1", 0.5)))
else:
    threshold = float(cfg.get("threshold_f1", cfg.get("threshold_recall95", 0.5)))

model_path = os.path.join(MODEL_DIR, f"model_{model_name}.joblib")
model = joblib.load(model_path)

app = FastAPI(title="Fraud Detection API")

# ---- Define request schema
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running", "model": model_name, "threshold": threshold}

@app.post("/predict_one")
def predict_one(tx: Transaction):
    df = pd.DataFrame([tx.dict()])
    prob = model.predict_proba(df)[:, 1][0]
    pred = int(prob >= threshold)
    return {"fraud_prob": prob, "is_fraud_pred": pred}

@app.post("/predict_batch")
def predict_batch(transactions: List[Transaction]):
    df = pd.DataFrame([t.dict() for t in transactions])
    prob = model.predict_proba(df)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "results": [
            {"fraud_prob": float(p), "is_fraud_pred": int(cls)}
            for p, cls in zip(prob, pred)
        ]
    }
