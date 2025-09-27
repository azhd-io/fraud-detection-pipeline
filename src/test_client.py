import pandas as pd, requests, json, random, os
import numpy as np


import pandas as pd, requests, matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

API = "http://127.0.0.1:8000/predict_one"

df = pd.read_csv("data/creditcard.csv")

def send_row(row):
    payload = row.drop(labels=["Class"]).to_dict()
    payload = {k: float(v) for k, v in payload.items()}
    r = requests.post(API, json=payload, timeout=10)
    return r.status_code, r.json()

# 1) A random non-fraud row
nf = df[df["Class"]==0].sample(1, random_state=1).iloc[0]
code, res = send_row(nf)
print("\nNON-FRAUD sample →", "HTTP", code, res)

# 2) A random fraud row
fr = df[df["Class"]==1].sample(1, random_state=2).iloc[0]
code, res = send_row(fr)
print("\nFRAUD sample     →", "HTTP", code, res)

# Show the ground truth classes for reference
print("\nGround truth of the two rows:")
print("non-fraud row Class:", int(nf["Class"]))
print("fraud row Class    :", int(fr["Class"]))




API = "http://127.0.0.1:8000/predict_batch"

df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Class"])
y_true = df["Class"].values

all_preds, all_probs = [], []

batch_size = 10000
for i in range(0, len(X), batch_size):
    batch = X.iloc[i:i+batch_size]
    payload = [{k: float(v) for k, v in row.to_dict().items()} for _, row in batch.iterrows()]

    resp = requests.post(API, json=payload, timeout=120)
    resp.raise_for_status()
    results = resp.json()["results"]

    all_probs.extend([r["fraud_prob"] for r in results])
    all_preds.extend([r["is_fraud_pred"] for r in results])

y_pred, y_prob = all_preds, all_probs


# --- Metrics ---
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
print("F1-score :", f1_score(y_true, y_pred, zero_division=0))

roc_auc = roc_auc_score(y_true, y_prob)
pr_auc  = average_precision_score(y_true, y_prob)
print("\nROC-AUC :", roc_auc)
print("PR-AUC  :", pr_auc)

# --- Curves ---
prec, rec, thr = precision_recall_curve(y_true, y_prob)
fpr, tpr, _ = roc_curve(y_true, y_prob)

plt.figure(figsize=(6,4))
plt.plot(rec, prec, label=f"PR AUC={pr_auc:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve (API predictions)")
plt.legend(); plt.tight_layout()
plt.savefig("reports/api_pr_curve.png")
plt.close()

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curve (API predictions)")
plt.legend(); plt.tight_layout()
plt.savefig("reports/api_roc_curve.png")
plt.close()

print("\n📊 Saved curves to reports/api_pr_curve.png and reports/api_roc_curve.png")

out = df.copy()
out["fraud_prob"] = y_prob
out["is_fraud_pred"] = y_pred
out = out.sort_values("fraud_prob", ascending=False)
os.makedirs("reports", exist_ok=True)
out.to_csv("reports/full_api_scored.csv", index=False)
frauds_only = out[out["is_fraud_pred"]==1]
frauds_only.to_csv("reports/frauds_only.csv", index=False)
print("Saved:", "reports/full_api_scored.csv", "and", "reports/frauds_only.csv")


# --- Threshold tuning ---
prec, rec, thr = precision_recall_curve(y_true, y_prob)

# Example policy 1: maximize F1
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
best_idx = np.argmax(f1_scores)
best_thr = float(thr[best_idx])
print(f"\nBest F1 threshold: {best_thr:.4f} (F1={f1_scores[best_idx]:.3f})")

# Example policy 2: enforce recall >= 0.95, pick lowest threshold that achieves it
target_recall = 0.95
thr_recall = 0.5
for p, r, t in zip(prec, rec, np.r_[thr, 1.0]):
    if r >= target_recall:
        thr_recall = float(t)
        break
print(f"Threshold for recall ≥{target_recall*100:.0f}%: {thr_recall:.4f}")

# --- Save to inference_config.json ---
config = {
    "model": "RandomForest",  # or whichever you used
    "threshold_f1": best_thr,
    "threshold_recall95": thr_recall
}
os.makedirs("models", exist_ok=True)
with open("models/inference_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("🎯 Updated models/inference_config.json with tuned thresholds.")
