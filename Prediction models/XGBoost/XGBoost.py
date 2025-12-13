#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pickle
import warnings
import numpy as np
import pandas as pd

# ---- Matplotlib/plot setup (headless-safe) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb

# ==============================
# CONFIG / FOLDERS
# ==============================
DATA_PATH = "Prediction models/Data/prepared_weekly.pkl"
TSCV_PATH = "Prediction models/Data/tscv_weekly.pkl"

OUTPUT_DIR = "Prediction models/XGBoost"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")
SHAP_DIR = os.path.join(OUTPUT_DIR, "shap")
FEATURE_IMPORTANCE_DIR = os.path.join(OUTPUT_DIR, "feature_importance")

for d in [OUTPUT_DIR, RESULTS_DIR, PREDICTIONS_DIR, GRAPHS_DIR, SHAP_DIR, FEATURE_IMPORTANCE_DIR]:
    os.makedirs(d, exist_ok=True)

# ==============================
# UTILS
# ==============================
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not np.any(mask):
        return np.nan
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0

def inv_target_col(y_scaler, arr_1d, h_index):
    arr_1d = np.asarray(arr_1d, dtype=float).ravel()
    return arr_1d * y_scaler.scale_[h_index] + y_scaler.mean_[h_index]

def safe_colorbar_warning_filter():
    warnings.filterwarnings(
        "ignore",
        message="No data for colormapping provided via 'c'.*",
        category=UserWarning
    )

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

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

print(f"Loaded {len(data)} weekly samples for {len(ATM_IDs)} ATMs\n")

# ==============================
# STORAGE FOR RESULTS
# ==============================
cv_rows = []
cv_mean_rows = []
cv_pred_rows = []
full_pred_mean_rows = []

# ==============================
# MODELING
# ==============================
for atm_id in ATM_IDs:
    print(f" Processing ATM {atm_id}...")

    atm_df = data[data["ATM_ID"] == atm_id].copy().sort_values("WEEK_START").reset_index(drop=True)
    if atm_df.empty:
        print(f"   No data, skipping.")
        continue

    X_all = atm_df[features].values
    y_all = atm_df[targets].values
    weeks = atm_df["WEEK_START"].values
    y_scaler = scalers[atm_id]["y_scaler"]

    atm_h_rmse, atm_h_mae, atm_h_smape = [], [], []

    for h, tgt in enumerate(targets, start=1):
        folds = tscv_splits.get(atm_id, [])
        fold_metrics = []

        for split in folds:
            tr_idx = np.array(split["train_idx"])
            te_idx = np.array(split["test_idx"])

            X_tr, X_te = X_all[tr_idx], X_all[te_idx]
            y_tr = y_all[tr_idx, h-1]
            y_te = y_all[te_idx, h-1]
            weeks_te = weeks[te_idx]

            if len(X_tr) < 10 or len(X_te) < 5:
                continue

            model = XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1
            )
            model.fit(X_tr, y_tr)

            y_pred_sc = model.predict(X_te)
            y_true = inv_target_col(y_scaler, y_te, h-1)
            y_pred = inv_target_col(y_scaler, y_pred_sc, h-1)

            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            mae  = mean_absolute_error(y_true, y_pred)
            sp   = smape(y_true, y_pred)
            fold_metrics.append((rmse, mae, sp))

            for w, yt, yp in zip(weeks_te, y_true, y_pred):
                cv_pred_rows.append({
                    "ATM_ID": atm_id,
                    "Horizon": h,
                    "Week_Start": pd.to_datetime(w),
                    "Actual": float(yt),
                    "Predicted": float(yp),
                    "Residual": float(yp - yt)
                })

        if fold_metrics:
            rmses, maes, smapes = map(np.array, zip(*fold_metrics))
            cv_rows.append({
                "ATM_ID": atm_id,
                "Horizon": h,
                "CV_RMSE_Mean": float(rmses.mean()),
                "CV_RMSE_Std": float(rmses.std(ddof=1)) if len(rmses) > 1 else np.nan,
                "CV_MAE_Mean": float(maes.mean()),
                "CV_MAE_Std": float(maes.std(ddof=1)) if len(maes) > 1 else np.nan,
                "CV_SMAPE_Mean": float(smapes.mean()),
                "CV_SMAPE_Std": float(smapes.std(ddof=1)) if len(smapes) > 1 else np.nan,
                "Folds": int(len(fold_metrics)),
                "Samples": int(len(atm_df))
            })
            atm_h_rmse.append(rmses.mean())
            atm_h_mae.append(maes.mean())
            atm_h_smape.append(smapes.mean())

    if len(atm_h_rmse) > 0:
        atm_h_rmse = np.array(atm_h_rmse)
        atm_h_mae = np.array(atm_h_mae)
        atm_h_smape = np.array(atm_h_smape)
        cv_mean_rows.append({
            "ATM_ID": atm_id,
            "CV_RMSE_Hmean": float(atm_h_rmse.mean()),
            "CV_RMSE_Hstd": float(atm_h_rmse.std(ddof=1)) if len(atm_h_rmse) > 1 else np.nan,
            "CV_MAE_Hmean": float(atm_h_mae.mean()),
            "CV_MAE_Hstd": float(atm_h_mae.std(ddof=1)) if len(atm_h_mae) > 1 else np.nan,
            "CV_SMAPE_Hmean": float(atm_h_smape.mean()),
            "CV_SMAPE_Hstd": float(atm_h_smape.std(ddof=1)) if len(atm_h_smape) > 1 else np.nan,
            "Horizons": int(len(atm_h_rmse)),
            "Samples": int(len(atm_df))
        })

    # ==============================
    # FULL-DATA TRAINING (for plots/SHAP/FI)
    # ==============================
    full_true_matrix = np.zeros_like(y_all, dtype=float)
    full_pred_matrix = np.zeros_like(y_all, dtype=float)
    shap_abs_sum = np.zeros(X_all.shape[1], dtype=float)
    shap_vals_all_h = []
    fi_accum = {}

    for h, tgt in enumerate(targets, start=1):
        model = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        )
        model.fit(X_all, y_all[:, h-1])

        y_pred_sc = model.predict(X_all)
        full_true_matrix[:, h-1] = inv_target_col(y_scaler, y_all[:, h-1], h-1)
        full_pred_matrix[:, h-1] = inv_target_col(y_scaler, y_pred_sc, h-1)

        try:
            dmat = xgb.DMatrix(X_all, feature_names=[f"f{i}" for i in range(X_all.shape[1])])
            shap_contrib = model.get_booster().predict(dmat, pred_contribs=True)[:, :-1]
            shap_abs_sum += np.abs(shap_contrib).mean(axis=0)
            shap_vals_all_h.append(shap_contrib)
        except Exception as e:
            print(f"   SHAP failed for ATM {atm_id}, Horizon {h}: {e}")

        try:
            booster = model.get_booster()
            fmap = {f"f{i}": i for i in range(X_all.shape[1])}
            gain_dict = booster.get_score(importance_type="gain")
            for k, v in gain_dict.items():
                idx = fmap.get(k, None)
                if idx is not None:
                    fi_accum[idx] = fi_accum.get(idx, 0.0) + float(v)
        except Exception as e:
            print(f"   Feature importance failed for ATM {atm_id}, Horizon {h}: {e}")

    # ==============================
    # PLOTS (Actual/Pred + Residuals)
    # ==============================
    mean_true = full_true_matrix.mean(axis=1)
    mean_pred = full_pred_matrix.mean(axis=1)
    std_pred  = full_pred_matrix.std(axis=1)
    residuals = mean_pred - mean_true

    # --- Actual vs Predicted (mean) ---
    plt.figure(figsize=(11, 5))
    plt.plot(weeks, mean_true, label="Actual (mean H1–H7)", linewidth=1.2)
    plt.plot(weeks, mean_pred, "--", label="Predicted (mean H1–H7)", linewidth=1.2)
    #plt.fill_between(weeks, mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, label="Pred ±1σ")
    plt.title(f"{atm_id} — Actual vs Predicted (mean across horizons)")
    plt.xlabel("Week Start")
    plt.ylabel("Withdrawals (real scale)")
    plt.grid(True); plt.legend()
    save_fig(os.path.join(GRAPHS_DIR, f"{atm_id}_allhorizons_predictions.png"))

    # --- Residuals (Predicted - Actual) ---
    plt.figure(figsize=(11, 4))
    plt.plot(weeks, residuals, color="green", linewidth=1.2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    plt.title(f"{atm_id} — Residuals (Pred - Actual, mean across horizons)")
    plt.xlabel("Week Start")
    plt.ylabel("Residual (Pred - Actual)")
    plt.ylim(-0.005, 0.005)
    plt.grid(True)
    save_fig(os.path.join(GRAPHS_DIR, f"{atm_id}_allhorizons_residuals.png"))

    for w, yt, yp in zip(weeks, mean_true, mean_pred):
        full_pred_mean_rows.append({
            "ATM_ID": atm_id,
            "Week_Start": pd.to_datetime(w),
            "Actual_Mean": float(yt),
            "Predicted_Mean": float(yp),
            "Residual_Mean": float(yp - yt)
        })

    # ==============================
    # SHAP + FEATURE IMPORTANCE
    # ==============================
    if len(shap_vals_all_h) > 0:
        shap_mean_abs = shap_abs_sum / len(shap_vals_all_h)
        order = np.argsort(shap_mean_abs)[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(np.array(features)[order][::-1], shap_mean_abs[order][::-1])
        plt.title(f"{atm_id} — Mean(|SHAP|) across horizons")
        plt.xlabel("Mean(|SHAP|)")
        save_fig(os.path.join(SHAP_DIR, f"{atm_id}_shap_bar.png"))

        safe_colorbar_warning_filter()
        shap_stack = np.vstack(shap_vals_all_h)
        max_points = min(5000, shap_stack.shape[0])
        if shap_stack.shape[0] > max_points:
            idx = np.random.RandomState(42).choice(shap_stack.shape[0], size=max_points, replace=False)
            shap_stack = shap_stack[idx]
        reps = int(np.ceil(shap_stack.shape[0] / X_all.shape[0]))
        X_rep = np.tile(X_all, (reps, 1))[:shap_stack.shape[0], :]

        import shap as _shap
        try:
            _plt_fig = plt.figure(figsize=(10, 6))
            _shap.summary_plot(
                shap_stack, X_rep,
                feature_names=features,
                show=False, plot_type="dot",
                color_bar=False
            )
            save_fig(os.path.join(SHAP_DIR, f"{atm_id}_shap_beeswarm.png"))
        except Exception as e:
            print(f"   SHAP beeswarm failed for ATM {atm_id}: {e}")

    if len(fi_accum) > 0:
        fi_idx = np.array(sorted(fi_accum.keys()))
        fi_vals = np.array([fi_accum[i] for i in fi_idx])
        order = np.argsort(fi_vals)[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(np.array(features)[fi_idx][order][::-1], fi_vals[order][::-1])
        plt.title(f"{atm_id} — XGBoost Gain (aggregated H1–H7)")
        plt.xlabel("Total Gain")
        save_fig(os.path.join(FEATURE_IMPORTANCE_DIR, f"{atm_id}_feature_importance.png"))

# ==============================
# SAVE RESULTS
# ==============================
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(os.path.join(RESULTS_DIR, "XGBoost_weekly_CV_results.csv"), index=False)
cv_mean_df = pd.DataFrame(cv_mean_rows)
cv_mean_df.to_csv(os.path.join(RESULTS_DIR, "XGBoost_weekly_CV_results_mean.csv"), index=False)

cv_pred_df = pd.DataFrame(cv_pred_rows)
cv_pred_df.sort_values(["ATM_ID", "Horizon", "Week_Start"]).to_csv(
    os.path.join(PREDICTIONS_DIR, "XGBoost_weekly_CV_predictions.csv"), index=False
)
full_pred_mean_df = pd.DataFrame(full_pred_mean_rows)
full_pred_mean_df.sort_values(["ATM_ID", "Week_Start"]).to_csv(
    os.path.join(PREDICTIONS_DIR, "XGBoost_full_mean_predictions.csv"), index=False
)

print("\nXGBoost weekly modeling completed successfully!")
print(f"Results → {RESULTS_DIR}")
print(f"Predictions → {PREDICTIONS_DIR}")
print(f"SHAP (bar + beeswarm) → {SHAP_DIR}")
print(f"Feature importance → {FEATURE_IMPORTANCE_DIR}")
print(f"Graphs → {GRAPHS_DIR}")
