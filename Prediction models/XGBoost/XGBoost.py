#!/usr/bin/env python3
import os
import pickle
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ===============================================================
# CONFIG
# ===============================================================
DATA_PATH = "Prediction models/Data/prepared_data.pkl"
TSCV_PATH = "Prediction models/Data/tscv.pkl"

OUTPUT_DIR = "Prediction models/XGBoost"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# ===============================================================
# UTILS
# ===============================================================
def smape_numpy(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100

def format_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()

# ===============================================================
# LOAD DATA
# ===============================================================
print("Loading prepared data...")
with open(DATA_PATH, "rb") as f:
    prepared = pickle.load(f)
data = prepared["data"]
features = prepared["features"]
target = prepared["target"]
ATM_IDs = prepared["ATM_IDs"]
scalers = prepared["scalers"]

print(f"Loaded {len(data)} samples across {len(ATM_IDs)} ATMs")

print("Loading TimeSeries CV splits...")
with open(TSCV_PATH, "rb") as f:
    tscv_splits = pickle.load(f)
print(f"Loaded CV splits for {len(tscv_splits)} ATMs")

# ===============================================================
# PER-ATM TIME SERIES CV
# ===============================================================
results = []
all_preds = []

print("\n🔁 Starting per-ATM XGBoost time series CV...\n")

for atm_id in ATM_IDs:
    if atm_id not in tscv_splits:
        print(f"ATM {atm_id}: no CV splits found, skipping.")
        continue

    atm_df = data[data["ATM_ID"] == atm_id].copy().sort_values("DATE").reset_index(drop=True)
    if len(atm_df) < 200:
        print(f"ATM {atm_id}: too few samples, skipping.")
        continue

    print(f"🏧 Processing ATM {atm_id} ({len(atm_df)} samples)")

    X_all = atm_df[features].values
    y_all = atm_df[target].values
    dates_all = atm_df["DATE"].values
    y_scaler = scalers[atm_id]["y_scaler"]

    fold_metrics = []
    fold_preds = []

    for split in tscv_splits[atm_id]:
        fold = int(split["fold"])
        train_idx = np.array(split["train_idx"])
        test_idx = np.array(split["test_idx"])

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]
        test_dates = dates_all[test_idx]

        if len(X_train) < 30 or len(X_test) < 10:
            continue

        # Train model
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, verbose=False)

        # Predict (still in scaled log space)
        y_pred_scaled = model.predict(X_test)

        # Convert to log units (inverse of scaler)
        y_true_log = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Metrics
        rmse = math.sqrt(mean_squared_error(y_true_log, y_pred_log))
        mae = mean_absolute_error(y_true_log, y_pred_log)
        smape = smape_numpy(y_true_log, y_pred_log)
        fold_metrics.append((rmse, mae, smape))
        print(f"   🔹 Fold {fold} → RMSE={rmse:.3f}, MAE={mae:.3f}, SMAPE={smape:.2f}%")

        # Save predictions
        for d, yt, yp in zip(test_dates, y_true_log, y_pred_log):
            fold_preds.append({
                "ATM_ID": atm_id,
                "Fold": fold,
                "DATE": d,
                "Actual_LOG": yt,
                "Predicted_LOG": yp,
                "Residual_LOG": yp - yt
            })

    # Aggregate CV metrics
    if fold_metrics:
        rmses, maes, smapes = zip(*fold_metrics)
        results.append({
            "ATM_ID": atm_id,
            "CV_RMSE_Mean": np.mean(rmses),
            "CV_MAE_Mean": np.mean(maes),
            "CV_SMAPE_Mean": np.mean(smapes),
            "CV_RMSE_Std": np.std(rmses, ddof=1) if len(rmses) > 1 else np.nan,
            "CV_MAE_Std": np.std(maes, ddof=1) if len(maes) > 1 else np.nan,
            "CV_SMAPE_Std": np.std(smapes, ddof=1) if len(smapes) > 1 else np.nan,
            "Samples": len(atm_df)
        })
        all_preds.extend(fold_preds)

# ===============================================================
# SAVE CV RESULTS
# ===============================================================
cv_results_df = pd.DataFrame(results)
cv_results_path = os.path.join(RESULTS_DIR, "XGBoost_per_ATM_CV_results.csv")
cv_results_df.to_csv(cv_results_path, index=False)
print(f"\nSaved CV results → {cv_results_path}")

preds_df = pd.DataFrame(all_preds)
preds_path = os.path.join(PREDICTIONS_DIR, "XGBoost_CV_predictions.csv")
preds_df.to_csv(preds_path, index=False)
print(f"Saved CV predictions → {preds_path}")

# ===============================================================
# FINAL FULL-DATA TRAIN + GRAPHS
# ===============================================================
print("\n🔁 Training final models on full data (for each ATM)...")
full_results = []
full_preds = []

for atm_id in ATM_IDs:
    atm_df = data[data["ATM_ID"] == atm_id].copy().sort_values("DATE").reset_index(drop=True)
    if len(atm_df) < 100:
        continue

    y_scaler = scalers[atm_id]["y_scaler"]
    X_all = atm_df[features].values
    y_all = atm_df[target].values
    dates_all = atm_df["DATE"].values

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_all, y_all, verbose=False)

    y_pred_scaled = model.predict(X_all)
    y_true_log = y_scaler.inverse_transform(y_all.reshape(-1, 1)).ravel()
    y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    full_preds.extend([
        {"ATM_ID": atm_id, "DATE": d, "Actual_LOG": yt, "Predicted_LOG": yp, "Residual_LOG": yp - yt}
        for d, yt, yp in zip(dates_all, y_true_log, y_pred_log)
    ])

    rmse = math.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae = mean_absolute_error(y_true_log, y_pred_log)
    smape = smape_numpy(y_true_log, y_pred_log)

    full_results.append({
        "ATM_ID": atm_id,
        "RMSE_Full_LOG": rmse,
        "MAE_Full_LOG": mae,
        "SMAPE_Full_LOG": smape,
        "Samples": len(y_true_log)
    })

    # === GRAPHS ===
    plt.figure(figsize=(12, 5))
    plt.plot(dates_all, y_true_log, label="Actual", linewidth=1.1)
    plt.plot(dates_all, y_pred_log, label="Predicted", linestyle="--", linewidth=1.1)
    plt.title(f"ATM {atm_id} – Actual vs Predicted (Scaled)")
    plt.xlabel("Date"); plt.ylabel("log1p(Withdrawals)")
    plt.legend(); plt.grid(True); format_dates(plt.gca()); plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f"{atm_id}_actual_vs_predicted.png"))
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(dates_all, y_pred_log - y_true_log)
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"ATM {atm_id} – Residuals Over Time (Scaled)")
    plt.xlabel("Date"); plt.ylabel("Residual (Pred - Actual)")
    plt.grid(True); format_dates(plt.gca()); plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f"{atm_id}_residuals.png"))
    plt.close()

# ===============================================================
# SAVE FULL RESULTS
# ===============================================================
pd.DataFrame(full_results).to_csv(os.path.join(RESULTS_DIR, "XGBoost_full_results.csv"), index=False)
pd.DataFrame(full_preds).to_csv(os.path.join(PREDICTIONS_DIR, "XGBoost_full_predictions.csv"), index=False)

print("\nXGBoost modeling completed successfully")
print(f"Results saved to {OUTPUT_DIR}")
