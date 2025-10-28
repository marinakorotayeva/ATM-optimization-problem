#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===============================================================
# CONFIG
# ===============================================================
DATA_PATH = "Prediction models/Data/prepared_weekly.pkl"
TSCV_PATH = "Prediction models/Data/tscv_weekly.pkl"

OUTPUT_DIR = "Prediction models/LSTM"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

SEQ_LEN = 4       # use last 4 weeks to predict next 7 days
EPOCHS = 80
BATCH_SIZE = 16
VAL_FRAC = 0.1
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ===============================================================
# UTILS
# ===============================================================
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if not np.any(mask):
        return np.nan
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0

def create_sequences(X, y, seq_len=4):
    """Build rolling windows of SEQ_LEN weeks to predict next 7 days."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def build_lstm(input_shape, output_dim=7):
    """Multi-output LSTM → predict 7 next-day values."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.LayerNormalization(),
        layers.Dropout(0.3),

        layers.LSTM(64),
        layers.LayerNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dense(output_dim)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model

def inv_target(y_scaled, y_scaler, h):
    return y_scaled * y_scaler.scale_[h] + y_scaler.mean_[h]

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# ===============================================================
# LOAD WEEKLY DATA
# ===============================================================
print("📂 Loading prepared weekly data...")
with open(DATA_PATH, "rb") as f:
    prepared = pickle.load(f)
with open(TSCV_PATH, "rb") as f:
    tscv_splits = pickle.load(f)

data     = prepared["data"]
features = prepared["features"]
targets  = prepared["targets"]
ATM_IDs  = prepared["ATM_IDs"]
scalers  = prepared["scalers"]

print(f"✅ Loaded {len(data)} weekly samples for {len(ATM_IDs)} ATMs")

# ===============================================================
# STORAGE
# ===============================================================
cv_rows = []
cv_mean_rows = []
cv_pred_rows = []
full_pred_mean_rows = []

# ===============================================================
# MODELING LOOP
# ===============================================================
for atm_id in ATM_IDs:
    print(f"\n🏧 ATM {atm_id}")
    df = data[data["ATM_ID"] == atm_id].sort_values("WEEK_START").reset_index(drop=True)
    if len(df) < SEQ_LEN + 8:
        print("   ⚠️ Not enough samples, skipping.")
        continue

    X_all = df[features].values
    y_all = df[targets].values
    weeks = df["WEEK_START"].values
    y_scaler = scalers[atm_id]["y_scaler"]

    atm_h_rmse, atm_h_mae, atm_h_smape = [], [], []

    for split in tscv_splits.get(atm_id, []):
        tr, te = np.array(split["train_idx"]), np.array(split["test_idx"])
        if len(tr) <= SEQ_LEN or len(te) <= SEQ_LEN:
            continue

        X_tr, y_tr = X_all[tr], y_all[tr]
        X_te, y_te = X_all[te], y_all[te]
        weeks_te = weeks[te][SEQ_LEN:]

        Xtr_seq, ytr_seq = create_sequences(X_tr, y_tr, SEQ_LEN)
        Xte_seq, yte_seq = create_sequences(X_te, y_te, SEQ_LEN)
        if len(Xte_seq) == 0:
            continue

        model = build_lstm((SEQ_LEN, Xtr_seq.shape[-1]), output_dim=7)
        es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
        model.fit(
            Xtr_seq, ytr_seq,
            validation_split=VAL_FRAC,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=False,
            verbose=0,
            callbacks=[es, rlrop]
        )

        y_pred_sc = model.predict(Xte_seq, verbose=0)
        for h in range(7):
            y_true = inv_target(yte_seq[:, h], y_scaler, h)
            y_pred = inv_target(y_pred_sc[:, h], y_scaler, h)
            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            mae  = mean_absolute_error(y_true, y_pred)
            sm   = smape(y_true, y_pred)
            atm_h_rmse.append(rmse); atm_h_mae.append(mae); atm_h_smape.append(sm)

            for w, yt, yp in zip(weeks_te, y_true, y_pred):
                cv_pred_rows.append({
                    "ATM_ID": atm_id,
                    "Horizon": h+1,
                    "Week_Start": pd.to_datetime(w),
                    "Actual": float(yt),
                    "Predicted": float(yp),
                    "Residual": float(yp - yt)
                })

    if len(atm_h_rmse) == 0:
        continue

    # Summaries
    cv_mean_rows.append({
        "ATM_ID": atm_id,
        "CV_RMSE_Hmean": float(np.mean(atm_h_rmse)),
        "CV_RMSE_Hstd":  float(np.std(atm_h_rmse, ddof=1)),
        "CV_MAE_Hmean":  float(np.mean(atm_h_mae)),
        "CV_MAE_Hstd":   float(np.std(atm_h_mae, ddof=1)),
        "CV_SMAPE_Hmean":float(np.mean(atm_h_smape)),
        "CV_SMAPE_Hstd": float(np.std(atm_h_smape, ddof=1)),
        "Horizons": 7,
        "Samples": len(df)
    })

    # ===============================================================
    # FULL-DATA TRAIN for Graphs
    # ===============================================================
    X_seq, y_seq = create_sequences(X_all, y_all, SEQ_LEN)
    model_full = build_lstm((SEQ_LEN, X_seq.shape[-1]), output_dim=7)
    model_full.fit(X_seq, y_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=0)

    y_pred_full_sc = model_full.predict(X_seq, verbose=0)
    y_true_full = np.zeros_like(y_pred_full_sc)
    y_pred_full = np.zeros_like(y_pred_full_sc)

    for h in range(7):
        y_true_full[:, h] = inv_target(y_seq[:, h], y_scaler, h)
        y_pred_full[:, h] = inv_target(y_pred_full_sc[:, h], y_scaler, h)

    mean_true = y_true_full.mean(axis=1)
    mean_pred = y_pred_full.mean(axis=1)
    residuals = mean_pred - mean_true
    std_pred  = y_pred_full.std(axis=1)
    weeks_seq = weeks[SEQ_LEN:]

    # Actual vs Predicted
    plt.figure(figsize=(11, 5))
    plt.plot(weeks_seq, mean_true, label="Actual (mean H1–H7)", linewidth=1.2)
    plt.plot(weeks_seq, mean_pred, "--", label="Predicted (mean H1–H7)", linewidth=1.2)
    plt.fill_between(weeks_seq, mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, label="Pred ±1σ")
    plt.title(f"{atm_id} — Actual vs Predicted (mean across horizons)")
    plt.xlabel("Week Start"); plt.ylabel("Withdrawals (real scale)")
    plt.grid(True); plt.legend()
    save_fig(os.path.join(GRAPHS_DIR, f"{atm_id}_allhorizons_predictions.png"))

    # Residuals
    plt.figure(figsize=(11, 4))
    plt.plot(weeks_seq, residuals, color="green", linewidth=1.2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    plt.title(f"{atm_id} — Residuals (Pred - Actual, mean across horizons)")
    plt.xlabel("Week Start"); plt.ylabel("Residual (Pred - Actual)")
    plt.grid(True)
    save_fig(os.path.join(GRAPHS_DIR, f"{atm_id}_allhorizons_residuals.png"))

    for w, yt, yp in zip(weeks_seq, mean_true, mean_pred):
        full_pred_mean_rows.append({
            "ATM_ID": atm_id,
            "Week_Start": pd.to_datetime(w),
            "Actual_Mean": float(yt),
            "Predicted_Mean": float(yp),
            "Residual_Mean": float(yp - yt)
        })

# ===============================================================
# SAVE RESULTS
# ===============================================================
cv_pred_df = pd.DataFrame(cv_pred_rows)
cv_mean_df = pd.DataFrame(cv_mean_rows)
full_pred_mean_df = pd.DataFrame(full_pred_mean_rows)

cv_pred_df.to_csv(os.path.join(RESULTS_DIR, "LSTM_weekly_CV_results.csv"), index=False)
cv_mean_df.to_csv(os.path.join(RESULTS_DIR, "LSTM_weekly_CV_results_mean.csv"), index=False)
cv_pred_df.to_csv(os.path.join(PREDICTIONS_DIR, "LSTM_weekly_CV_predictions.csv"), index=False)
full_pred_mean_df.to_csv(os.path.join(PREDICTIONS_DIR, "LSTM_full_mean_predictions.csv"), index=False)

print("\n✅ LSTM weekly modeling completed successfully!")
print(f"Results → {RESULTS_DIR}")
print(f"Predictions → {PREDICTIONS_DIR}")
print(f"Graphs → {GRAPHS_DIR}")


