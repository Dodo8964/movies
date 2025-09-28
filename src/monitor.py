# src/monitor.py
import pandas as pd
from scipy import stats
import joblib
import json
import os

INFERENCE_LOG = "models/inference_log.csv"
TRAIN_SNAP = "models/train_snapshot.csv"  # create during training (optional)

def compute_basic_metrics():
    if not os.path.exists(INFERENCE_LOG):
        print("No inference log yet.")
        return
    df = pd.read_csv(INFERENCE_LOG)
    print("Total inferences:", len(df))
    print("Predictions distribution:\n", df["prediction"].value_counts(normalize=True))

def check_drift(feature="budget"):
    if not os.path.exists(INFERENCE_LOG) or not os.path.exists(TRAIN_SNAP):
        print("Need both inference log and train snapshot for drift.")
        return
    train = pd.read_csv(TRAIN_SNAP)
    inf = pd.read_csv(INFERENCE_LOG)
    # two-sample KS test (continuous)
    stat, p = stats.ks_2samp(train[feature].dropna(), inf[feature].dropna())
    print(f"KS stat={stat:.3f}, p-value={p:.3f} -> {'no drift' if p>0.01 else 'possible drift'}")

if __name__ == "__main__":
    compute_basic_metrics()
    # optionally check drift
    # check_drift("budget")
