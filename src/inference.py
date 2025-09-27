# src/inference.py
import os, json, sys
import pandas as pd
import joblib

MODEL_DIR = "models"
DATA_IN   = "data/creditcard.csv"        # change if you want to score another file
DATA_OUT  = "data/predictions.csv"

def load_model_and_config(model_dir=MODEL_DIR):
    # read threshold + which model was chosen
    cfg_path = os.path.join(model_dir, "inference_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing {cfg_path}. Run train_baseline.py first.")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    model_name = cfg.get("model")
    threshold  = float(cfg.get("threshold", 0.5))
    model_path = os.path.join(model_dir, f"model_{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Run train_baseline.py first.")
    model = joblib.load(model_path)
    return model, threshold, cfg

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # If the CSV includes the label, drop it for inference
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    return df

def predict_file(input_csv=DATA_IN, output_csv=DATA_OUT):
    model, threshold, cfg = load_model_and_config()

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    X  = prepare_features(df)

    # proba for the positive class
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)

    out = df.copy()
    out["fraud_prob"]    = proba
    out["is_fraud_pred"] = pred

    # sort by highest risk first (optional but handy)
    out_sorted = out.sort_values("fraud_prob", ascending=False).reset_index(drop=True)
    out_sorted.to_csv(output_csv, index=False)

    print("✅ Inference complete")
    print(f"Model      : {cfg.get('model')}")
    print(f"Threshold  : {threshold:.3f}")
    print(f"Input CSV  : {os.path.abspath(input_csv)}")
    print(f"Output CSV : {os.path.abspath(output_csv)}")

if __name__ == "__main__":
    # Allow optional custom input/output paths: python src/inference.py data/my.csv data/my_preds.csv
    in_path  = sys.argv[1] if len(sys.argv) > 1 else DATA_IN
    out_path = sys.argv[2] if len(sys.argv) > 2 else DATA_OUT
    predict_file(in_path, out_path)
