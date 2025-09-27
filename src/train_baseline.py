# src/train_baseline.py
import os, json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    precision_recall_curve, roc_curve, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42
DATA_PATH = "data/creditcard.csv"
REPORT_DIR = "reports"
MODEL_DIR = "models"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y

def build_pipelines():
    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None,
            random_state=RANDOM_STATE
        ))
    ])

    rf_base = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    # Wrap in calibration
    rf = Pipeline([
        ("clf", CalibratedClassifierCV(rf_base, method="isotonic", cv=3))
    ])
    
    return {"LogReg": lr, "RandomForest": rf}

def evaluate(model, X_test, y_test, name):
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, proba)
    pr  = average_precision_score(y_test, proba)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n{name} — ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", cm)

    # Curves
    prec, rec, thr = precision_recall_curve(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"pr_{name}.png")); plt.clf()

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC Curve — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"roc_{name}.png")); plt.clf()

    # Auto-pick a threshold to hit >= 0.85 recall (tweak for your goal)
    best_thr = 0.5
    for p, r, t in zip(prec, rec, np.r_[thr, 1.0]):
        if r >= 0.85:
            best_thr = float(t); break

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "confusion_matrix": cm.tolist(),
        "threshold_recall>=0.85": best_thr
    }

def main():
    print("🔎 Loading data…")
    X, y = load_data()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = build_pipelines()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    metrics = {}
    best_name, best_score = None, -1.0

    # Cross-validate & fit
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_tr, y_tr, scoring="roc_auc", cv=cv, n_jobs=-1)
        print(f"{name} CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

        pipe.fit(X_tr, y_tr)
        res = evaluate(pipe, X_te, y_te, name)
        res["cv_roc_auc_mean"] = float(scores.mean())
        res["cv_roc_auc_std"]  = float(scores.std())
        metrics[name] = res

        if res["pr_auc"] > best_score:   # choose by PR-AUC due to imbalance
            best_score = res["pr_auc"]
            best_name  = name

    # Save metrics
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n📝 Saved metrics to {os.path.join(REPORT_DIR, 'metrics.json')}")

    # Persist best model
    best_model = models[best_name]
    model_path = os.path.join(MODEL_DIR, f"model_{best_name}.joblib")
    joblib.dump(best_model, model_path)
    print(f"✅ Saved best model: {best_name} → {model_path}")

    # Save chosen threshold for that model
    thr = metrics[best_name]["threshold_recall>=0.85"]
    with open(os.path.join(MODEL_DIR, "inference_config.json"), "w") as f:
        json.dump({"model": best_name, "threshold": thr}, f, indent=2)
    print(f"🎯 Stored threshold (recall>=0.85): {thr:.3f}")

if __name__ == "__main__":
    main()
