
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

df = pd.read_csv("data/predictions.csv")
print("Total predicted frauds:", (df["is_fraud_pred"]==1).sum())
print("Total transactions:", len(df))

frauds = df[df["is_fraud_pred"]==1]
print(frauds.head(10))   # first 10 fraud predictions
print("\nTop 5 fraud probs:\n", frauds["fraud_prob"].nlargest(5))


print(df.sort_values("fraud_prob", ascending=False).head(10))


y_true = df["Class"]
y_pred = df["is_fraud_pred"]
y_prob = df["fraud_prob"]

print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

print("ROC-AUC:", roc_auc_score(y_true, y_prob))
print("PR-AUC :", average_precision_score(y_true, y_prob))

