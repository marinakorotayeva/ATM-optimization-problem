#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Linear Regression model using ElasticNetCV (Ridge + Lasso hybrid)
Predicts next 7 days (Mon–Sun) for each ATM using prepared weekly data.
Still linear, but regularized and much more accurate.
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================================================
# CONFIG
# ===============================================================
DATA_PATH = "Prediction models/Data/prepared_weekly.pkl"
TSCV_PATH = "Prediction models/Data/tscv_weekly.pkl"

OUTPUT_DIR = "Prediction models/Linear Regression"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ===============================================================
# UTILS
# ===============================================================
def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100

def inverse_one_horizon(y_scaler, y_scaled_1d, h_index, n_h=7):
    y_scaled_1d = np.asarray(y_scaled_1d).reshape(-1)
    temp = np.zeros((y_scaled_1d.shape[0], n_h))
    temp[:, h_index] = y_scaled_1d
    inv = y_scaler.inverse_transform(temp)
    return inv[:, h_index]

def format_weeks(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()

def plot_cv_mean_graphs_for_atm(atm_id, mean_df, save_dir):
    if mean_df.empty:
        return
     # --- Actual vs Predicted (mean across horizons)
    dates = pd.to_datetime(mean_df["Week_Start"])
    plt.figure(figsize=(12, 5))
    plt.plot(dates, mean_df["Actual_Mean"], label="Actual (mean)", linewidth=1.2)
    plt.plot(dates, mean_df["Predicted_Mean"], label="Predicted (mean)", linestyle="--", linewidth=1.2)
    plt.title(f"ATM {atm_id} – CV Actual vs Predicted (mean across horizons)")
    plt.xlabel("Week Start"); plt.ylabel("log1p(Withdrawals)")
    plt.grid(True); plt.legend()
    format_weeks(plt.gca()); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{atm_id}_CV_mean_allhorizons_actual_vs_pred.png"))
    plt.close()

    # --- Residuals (Predicted - Actual)
    plt.figure(figsize=(12, 4))
    plt.plot(dates, mean_df["Residual_Mean"], linewidth=1.2, color="green")
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    plt.title(f"ATM {atm_id} – CV Residuals (Pred - Actual, mean across horizons)")
    plt.xlabel("Week Start")
    plt.ylabel("Residual")
    plt.ylim(-2, 2)                  # Fixed residual scale
    plt.grid(True)
    format_weeks(plt.gca())
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{atm_id}_CV_mean_allhorizons_residuals.png"))
    plt.close()

# ===============================================================
# LOAD
# ===============================================================
print("Loading prepared weekly data...")
with open(DATA_PATH, "rb") as f:
    prepared = pickle.load(f)

data     = prepared["data"]
features = prepared["features"]
targets  = prepared["targets"]
ATM_IDs  = prepared["ATM_IDs"]
scalers  = prepared["scalers"]

with open(TSCV_PATH, "rb") as f:
    tscv_splits = pickle.load(f)

print(f"Loaded {len(data)} weekly samples for {len(ATM_IDs)} ATMs")
assert len(targets) == 7, "Expected 7 target columns (Mon–Sun)."

# ===============================================================
# CV TRAINING
# ===============================================================
cv_rows_by_h, cv_rows_mean = [], []
cv_preds_rows, cv_preds_mean_rows = [], []

print("\nRunning ElasticNetCV (regularized Linear Regression)...")

for atm_id in ATM_IDs:
    if atm_id not in tscv_splits:
        print(f"No CV splits for ATM {atm_id}, skipping.")
        continue

    atm_df = data[data["ATM_ID"] == atm_id].copy().sort_values("WEEK_START").reset_index(drop=True)
    n_samples = len(atm_df)
    if n_samples < 25:
        continue

    X_all = atm_df[features].values
    y_all = atm_df[targets].values
    weeks = atm_df["WEEK_START"].values
    y_scaler = scalers[atm_id]["y_scaler"]
    folds = len(tscv_splits[atm_id])

    per_h_metrics = {h: {"rmse": [], "mae": [], "smape": []} for h in range(7)}
    preds_all_horizons = []

    for h_idx in range(7):
        for split in tscv_splits[atm_id]:
            tr, te = np.array(split["train_idx"]), np.array(split["test_idx"])
            if len(tr) < 10 or len(te) < 5:
                continue
            X_tr, X_te = X_all[tr], X_all[te]
            y_tr, y_te = y_all[tr, h_idx], y_all[te, h_idx]
            wk_te = weeks[te]

            # Scale features (within fold to avoid leakage)
            scaler_X = StandardScaler()
            X_tr_scaled = scaler_X.fit_transform(X_tr)
            X_te_scaled = scaler_X.transform(X_te)

            model = ElasticNetCV(
            l1_ratio=[.1, .3, .5, .7, .9],
            alphas=np.logspace(-3, 2, 30),
            cv=5,
            max_iter=50000,                # increase iterations 5x
            tol=1e-4,                      # smaller tolerance = better convergence
            random_state=42,
            n_jobs=-1                      # parallelize to make it faster
            )
            model.fit(X_tr_scaled, y_tr)
            y_pred_scaled = model.predict(X_te_scaled)

            y_true = inverse_one_horizon(y_scaler, y_te, h_idx)
            y_pred = inverse_one_horizon(y_scaler, y_pred_scaled, h_idx)

            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            s = smape(y_true, y_pred)

            per_h_metrics[h_idx]["rmse"].append(rmse)
            per_h_metrics[h_idx]["mae"].append(mae)
            per_h_metrics[h_idx]["smape"].append(s)

            for w, yt, yp in zip(wk_te, y_true, y_pred):
                preds_all_horizons.append({
                    "ATM_ID": atm_id,
                    "Horizon": h_idx + 1,
                    "Week_Start": w,
                    "Actual": yt,
                    "Predicted": yp,
                    "Residual": yp - yt
                })

    # === Per horizon results
    for h_idx in range(7):
        r, m, s = per_h_metrics[h_idx]["rmse"], per_h_metrics[h_idx]["mae"], per_h_metrics[h_idx]["smape"]
        if not r:
            continue
        cv_rows_by_h.append({
            "ATM_ID": atm_id,
            "Horizon": h_idx + 1,
            "CV_RMSE_Mean": np.mean(r),
            "CV_RMSE_Std": np.std(r, ddof=1) if len(r) > 1 else np.nan,
            "CV_MAE_Mean": np.mean(m),
            "CV_MAE_Std": np.std(m, ddof=1) if len(m) > 1 else np.nan,
            "CV_SMAPE_Mean": np.mean(s),
            "CV_SMAPE_Std": np.std(s, ddof=1) if len(s) > 1 else np.nan,
            "Folds": folds,
            "Samples": n_samples
        })

    # === Mean across horizons
    rmse_means = [np.mean(per_h_metrics[h]["rmse"]) for h in range(7) if per_h_metrics[h]["rmse"]]
    mae_means = [np.mean(per_h_metrics[h]["mae"]) for h in range(7) if per_h_metrics[h]["mae"]]
    s_means = [np.mean(per_h_metrics[h]["smape"]) for h in range(7) if per_h_metrics[h]["smape"]]
    if rmse_means:
        cv_rows_mean.append({
            "ATM_ID": atm_id,
            "CV_RMSE_Hmean": np.mean(rmse_means),
            "CV_RMSE_Hstd": np.std(rmse_means, ddof=1) if len(rmse_means) > 1 else np.nan,
            "CV_MAE_Hmean": np.mean(mae_means),
            "CV_MAE_Hstd": np.std(mae_means, ddof=1) if len(mae_means) > 1 else np.nan,
            "CV_SMAPE_Hmean": np.mean(s_means),
            "CV_SMAPE_Hstd": np.std(s_means, ddof=1) if len(s_means) > 1 else np.nan,
            "Horizons": 7,
            "Samples": n_samples
        })

    # === Per-week mean predictions + graphs
    atm_cv_df = pd.DataFrame(preds_all_horizons)
    cv_preds_rows.extend(preds_all_horizons)
    if not atm_cv_df.empty:
        g = atm_cv_df.groupby("Week_Start", as_index=False).agg(
            Actual_Mean=("Actual", "mean"),
            Predicted_Mean=("Predicted", "mean")
        )
        g["Residual_Mean"] = g["Predicted_Mean"] - g["Actual_Mean"]
        g.insert(0, "ATM_ID", atm_id)
        cv_preds_mean_rows.extend(g.to_dict("records"))
        plot_cv_mean_graphs_for_atm(atm_id, g, GRAPHS_DIR)

# ===============================================================
# SAVE
# ===============================================================
cv_by_h_df = pd.DataFrame(cv_rows_by_h)
cv_mean_df = pd.DataFrame(cv_rows_mean)
cv_preds_df = pd.DataFrame(cv_preds_rows)
cv_preds_mean_df = pd.DataFrame(cv_preds_mean_rows)

cv_by_h_df.to_csv(os.path.join(RESULTS_DIR, "LR_weekly_CV_results.csv"), index=False)
cv_mean_df.to_csv(os.path.join(RESULTS_DIR, "LR_weekly_CV_results_mean.csv"), index=False)
cv_preds_df.to_csv(os.path.join(PREDICTIONS_DIR, "LR_weekly_CV_predictions.csv"), index=False)
cv_preds_mean_df.to_csv(os.path.join(PREDICTIONS_DIR, "LR_weekly_CV_predictions_mean.csv"), index=False)

print("\nElasticNet (regularized Linear Regression) completed successfully!")
print(f"Results → {RESULTS_DIR}")
print(f"Predictions → {PREDICTIONS_DIR}")
print(f"Graphs → {GRAPHS_DIR}")
