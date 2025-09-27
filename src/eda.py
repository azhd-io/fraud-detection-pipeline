# src/eda.py
import os, sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render to files (no GUI pop-ups)
import matplotlib.pyplot as plt

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Show which interpreter is running (helps debug venv issues)
print("🔎 Python exe:", sys.executable)

# ---- Load data
CSV_PATH = "data/creditcard.csv"
df = pd.read_csv(CSV_PATH)
print("✅ Dataset loaded:", CSV_PATH)
print("Shape:", df.shape)

# ---- Basic checks
print("\nClass distribution (counts):")
print(df["Class"].value_counts())
print("\nClass distribution (%):")
print((df["Class"].value_counts(normalize=True) * 100).round(4))

print("\nMissing values (total):", int(df.isna().sum().sum()))
print("\n'Amount' stats:")
print(df["Amount"].describe())

# ---- Plots (saved to reports/)
# 1) Fraud vs Non-fraud count
ax = df["Class"].value_counts().sort_index().plot(kind="bar")
ax.set_title("Fraud (1) vs Non-Fraud (0)")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "class_balance.png"))
plt.clf()

# 2) Amount histogram (clip extreme tail for readability)
df["Amount"].clip(upper=df["Amount"].quantile(0.99)).plot(kind="hist", bins=50)
plt.title("Amount Distribution (clipped at 99th percentile)")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "amount_hist.png"))
plt.clf()

# 3) Correlation heatmap (small subset to keep image readable)
corr = df.sample(min(len(df), 20000), random_state=42).corr(numeric_only=True)
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.title("Correlation (sampled)")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "corr_heatmap.png"))
plt.clf()

print("\n📁 Reports saved to:", os.path.abspath(REPORT_DIR))
print(" - class_balance.png")
print(" - amount_hist.png")
print(" - corr_heatmap.png")
print("\n✅ EDA complete.")
