#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pickle
import warnings
import numpy as np
import pandas as pd

# ---- Matplotlib (headless-safe) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb
import shap

# ==============================
# CONFIG
# ==============================
DATA_PATH = "Prediction models/Data/prepared_weekly.pkl"
TSCV_PATH = "Prediction models/Data/tscv_weekly.pkl"

OUTPUT_DIR = "Prediction models/XGBoost"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "Explainability")
SHAP_DIR = os.path.join(EXPLAIN_DIR, "shap_outputs")
FEATURE_IMPORTANCE_DIR = os.path.join(EXPLAIN_DIR, "feature_importance")

for d in [
    OUTPUT_DIR, RESULTS_DIR, PREDICTIONS_DIR,
    EXPLAIN_DIR, SHAP_DIR, FEATURE_IMPORTANCE_DIR
]:
    os.makedirs(d, exist_ok=True)

# ==============================
# WHAT-IF CONFIG (TOP FEATURES)
# ==============================
TOP_FEATURES = [
    "WITHDRWLS_LOG_sum_LAG12",
    "WITHDRWLS_LOG_sum_LAG8",
    "WITHDRWLS_LOG_sum_LAG4",
]

SCENARIOS = {
    "baseline": 1.0,
    "x0p9": 0.9,
    "x1p1": 1.1,
}

# ==============================
# UTILS
# ==============================
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0

def inv_target(y_scaler, arr, h):
    return arr * y_scaler.scale_[h] + y_scaler.mean_[h]

# ==============================
# LOAD DATA
# ==============================
print("Loading prepared weekly data...")
with open(DATA_PATH, "rb") as f:
    prepared = pickle.load(f)

with open(TSCV_PATH, "rb") as f:
    tscv_splits = pickle.load(f)

data     = prepared["data"]
features = prepared["features"]
targets  = prepared["targets"]
ATM_IDs  = prepared["ATM_IDs"]
scalers  = prepared["scalers"]

TOP_IDX = [features.index(f) for f in TOP_FEATURES]

print(f"Loaded {len(data)} weekly samples for {len(ATM_IDs)} ATMs")

# ==============================
# STORAGE
# ==============================
cv_pred_rows = []

# ==============================
# MODELING
# ==============================
for atm_id in ATM_IDs:
    print(f"\nProcessing ATM {atm_id}...")

    atm_df = data[data["ATM_ID"] == atm_id].copy().sort_values("WEEK_START")
    if atm_df.empty:
        continue

    X_all = atm_df[features].values
    y_all = atm_df[targets].values
    weeks = atm_df["WEEK_START"].values
    y_scaler = scalers[atm_id]["y_scaler"]

    # ------------------------------
    # Train one model per horizon
    # ------------------------------
    models = []

    for h in range(7):
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_all, y_all[:, h])
        models.append(model)

    # ------------------------------
    # SCENARIO PREDICTIONS
    # ------------------------------
    for scenario, factor in SCENARIOS.items():

        X_scn = X_all.copy()
        if factor != 1.0:
            X_scn[:, TOP_IDX] *= factor

        for h in range(7):
            y_pred_sc = models[h].predict(X_scn)
            y_pred = inv_target(y_scaler, y_pred_sc, h)
            y_true = inv_target(y_scaler, y_all[:, h], h)

            for w, yt, yp in zip(weeks, y_true, y_pred):
                cv_pred_rows.append({
                    "ATM_ID": atm_id,
                    "Scenario": scenario,
                    "Horizon": h + 1,
                    "Week_Start": pd.to_datetime(w),
                    "Actual": float(yt),
                    "Predicted": float(yp),
                    "Residual": float(yp - yt),
                })

# ==============================
# SAVE OUTPUT
# ==============================
cv_pred_df = pd.DataFrame(cv_pred_rows)
cv_pred_df.sort_values(
    ["Scenario", "ATM_ID", "Week_Start", "Horizon"],
    inplace=True
)

out_path = os.path.join(
    PREDICTIONS_DIR,
    "XGBoost_weekly_CV_predictions_whatif_top3.csv"
)
cv_pred_df.to_csv(out_path, index=False)

print("\nXGBoost what-if prediction completed successfully!")
print(f"Saved → {out_path}")
print("Scenarios:", list(SCENARIOS.keys()))
