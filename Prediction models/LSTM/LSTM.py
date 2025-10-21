#!/usr/bin/env python3
import os
import math
import pickle
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===============================================================
# CONFIG
# ===============================================================
DATA_PATH = "Prediction models/Data/prepared_data.pkl"
TSCV_PATH = "Prediction models/Data/tscv.pkl"

OUTPUT_DIR = "Prediction models/LSTM"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")


os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


# Model / training
SEQ_LEN = 90          # context to capture weekly/monthly patterns
EPOCHS = 100
BATCH_SIZE = 32
VAL_FRAC = 0.1        # last 10% of training sequences for validation (time-ordered)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ===============================================================
# UTILS
# ===============================================================
def smape_numpy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100

def create_sequences(X, y, seq_length=60):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

def time_ordered_val_split(X, y, val_frac=0.1):
    """Use the LAST fraction as validation to avoid leakage."""
    n = len(X)
    if n == 0:
        return X, y, X, y
    cut = max(1, int(n * (1 - val_frac)))
    return X[:cut], y[:cut], X[cut:], y[cut:]

def build_bi_lstm_model(input_shape):
    """BiLSTM with normalization + dropout for stability."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.LayerNormalization(),
        layers.Dropout(0.2),

        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.LayerNormalization(),
        layers.Dropout(0.2),

        layers.Bidirectional(layers.LSTM(32)),
        layers.LayerNormalization(),
        layers.Dropout(0.2),

        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model

def train_one_model(Xtr_seq, ytr_seq):
    """Train with time-ordered validation; no shuffling."""
    X_tr, y_tr, X_val, y_val = time_ordered_val_split(Xtr_seq, ytr_seq, VAL_FRAC)
    model = build_bi_lstm_model((SEQ_LEN, Xtr_seq.shape[-1]))

    es = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0)
    rlrop = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=0)

    model.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        shuffle=False,             # critical for time series
        verbose=0,
        callbacks=[es, rlrop]
    )
    return model

def format_dates(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()

# ===============================================================
# LOAD DATA & SPLITS
# ===============================================================
print("📂 Loading prepared dataset...")
with open(DATA_PATH, "rb") as f:
    prepared = pickle.load(f)

data = prepared["data"].sort_values(["ATM_ID", "DATE"]).reset_index(drop=True)
features = prepared["features"]
target = prepared["target"]           # "TARGET_SCALED" → standardized LOG target
scalers = prepared["scalers"]         # per-ATM scalers; y_scaler is on LOG target
ATM_IDs = prepared["ATM_IDs"]

print(f"Loaded {len(data)} samples from {len(ATM_IDs)} ATMs.")

print("Loading pre-generated CV splits...")
with open(TSCV_PATH, "rb") as f:
    tscv_splits = pickle.load(f)
print(f"Loaded CV splits for {len(tscv_splits)} ATMs.")

# ===============================================================
# PER-ATM CROSS-VALIDATION (LOG units)
# ===============================================================
overall_rows = []
all_pred_rows = []

print("\n🔁 Starting per-ATM TimeSeries Cross-Validation (BiLSTM, log-scaled units)...\n")

for atm_id in ATM_IDs:
    if atm_id not in tscv_splits:
        print(f"No CV splits found for ATM {atm_id}, skipping.")
        continue

    atm_df = data[data["ATM_ID"] == atm_id].copy().reset_index(drop=True)

    if len(atm_df) < SEQ_LEN + 50:
        print(f"🏧 ATM: {atm_id} — Skipping (too few samples)")
        continue

    print(f"🏧 ATM: {atm_id}")
    y_scaler = scalers[atm_id]["y_scaler"]  # scaler on LOG target

    X_all = atm_df[features].values
    y_all = atm_df[target].values
    dates_all = atm_df["DATE"].values

    cv_rmse, cv_mae, cv_smape = [], [], []

    # We also write per-ATM/per-fold prediction files
    fold_pred_accum = []

    for split in tscv_splits[atm_id]:
        fold = int(split["fold"])
        train_idx = np.array(split["train_idx"])
        test_idx = np.array(split["test_idx"])

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

        if len(X_test) <= SEQ_LEN or len(X_train) <= SEQ_LEN:
            print(f"   Fold {fold}: segment too short for SEQ_LEN={SEQ_LEN}, skipping.")
            continue

        # Build sequences
        Xtr_seq, ytr_seq = create_sequences(X_train, y_train, SEQ_LEN)
        Xte_seq, yte_seq = create_sequences(X_test,  y_test,  SEQ_LEN)
        test_dates = dates_all[test_idx][SEQ_LEN:]  # align to yte_seq

        # Train
        model = train_one_model(Xtr_seq, ytr_seq)

        # Predict in standardized space → invert to LOG (still log units)
        y_pred_scaled = model.predict(Xte_seq, verbose=0).ravel()
        y_true_log = y_scaler.inverse_transform(yte_seq.reshape(-1, 1)).ravel()
        y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Metrics in LOG units
        rmse = math.sqrt(mean_squared_error(y_true_log, y_pred_log))
        mae = mean_absolute_error(y_true_log, y_pred_log)
        sm  = smape_numpy(y_true_log, y_pred_log)

        cv_rmse.append(rmse); cv_mae.append(mae); cv_smape.append(sm)
        print(f"   🔹 Fold {fold} → RMSE={rmse:.2f}, MAE={mae:.2f}, SMAPE={sm:.2f}%")

        # Store per-fold predictions (for joined CV file)
        fold_rows = [
            {
                "ATM_ID": atm_id,
                "Fold": fold,
                "DATE": dt,
                "Actual_LOG": yt,
                "Predicted_LOG": yp,
                "Residual_LOG": yp - yt,
                "AbsError_LOG": abs(yp - yt)
            }
            for dt, yt, yp in zip(test_dates, y_true_log, y_pred_log)
        ]
        all_pred_rows.extend(fold_rows)
        fold_pred_accum.extend(fold_rows)

        # Save per-ATM per-fold CSV immediately
        per_fold_df = pd.DataFrame(fold_rows)
        per_fold_path = os.path.join(PREDICTIONS_DIR, f"LSTM_predictions_{atm_id}_fold{fold}.csv")
        per_fold_df.to_csv(per_fold_path, index=False)

     

    # Per-ATM summary
    overall_rows.append({
        "ATM_ID": atm_id,
        "CV_RMSE_Mean": float(np.mean(cv_rmse)) if cv_rmse else np.nan,
        "CV_RMSE_Std":  float(np.std(cv_rmse, ddof=1)) if len(cv_rmse) > 1 else np.nan,
        "CV_MAE_Mean":  float(np.mean(cv_mae)) if cv_mae else np.nan,
        "CV_MAE_Std":   float(np.std(cv_mae, ddof=1)) if len(cv_mae) > 1 else np.nan,
        "CV_SMAPE_Mean":float(np.mean(cv_smape)) if cv_smape else np.nan,
        "CV_SMAPE_Std": float(np.std(cv_smape, ddof=1)) if len(cv_smape) > 1 else np.nan,
        "Total_Samples": int(len(atm_df))
    })

# ===============================================================
# SAVE JOINED CV PREDICTIONS & PER-ATM SUMMARY
# ===============================================================
if all_pred_rows:
    pred_df = pd.DataFrame(all_pred_rows).sort_values(["ATM_ID", "DATE"])
else:
    pred_df = pd.DataFrame(columns=["ATM_ID","Fold","DATE","Actual_LOG","Predicted_LOG","Residual_LOG","AbsError_LOG"])

pred_path = os.path.join(PREDICTIONS_DIR, "LSTM_predictions_CV_joined.csv")
pred_df.to_csv(pred_path, index=False)

cv_results_df = pd.DataFrame(overall_rows)
cv_results_path = os.path.join(RESULTS_DIR, "LSTM_per_ATM_results.csv")
cv_results_df.to_csv(cv_results_path, index=False)

print("\nSaved CV predictions  →", pred_path)
print("Saved CV per-ATM stats →", cv_results_path)














# ===============================================================
# FINAL FULL-DATA TRAIN + CONTINUOUS PREDICTIONS (LOG)
# ===============================================================
print("\n🔁 Training final model on full data (per ATM) and generating continuous predictions (LOG)...")
full_rows = []
full_pred_rows = []

for atm_id in ATM_IDs:
    atm_df = data[data["ATM_ID"] == atm_id].copy().reset_index(drop=True)
    if len(atm_df) < SEQ_LEN + 1:
        continue

    y_scaler = scalers[atm_id]["y_scaler"]
    X_all = atm_df[features].values
    y_all = atm_df[target].values
    dates_all = atm_df["DATE"].values

    X_seq, y_seq = create_sequences(X_all, y_all, SEQ_LEN)
    model = train_one_model(X_seq, y_seq)

    y_pred_scaled = model.predict(X_seq, verbose=0).ravel()
    y_true_log = y_scaler.inverse_transform(y_seq.reshape(-1, 1)).ravel()
    y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    pred_dates = dates_all[SEQ_LEN:]  # align with sequences

    # Store full predictions
    full_pred_rows.extend([
        {
            "ATM_ID": atm_id,
            "DATE": dt,
            "Actual_LOG": yt,
            "Predicted_LOG": yp,
            "Residual_LOG": yp - yt,
            "AbsError_LOG": abs(yp - yt)
        }
        for dt, yt, yp in zip(pred_dates, y_true_log, y_pred_log)
    ])

    # Full-data metrics (LOG units)
    full_rows.append({
        "ATM_ID": atm_id,
        "MAE_Full_LOG": mean_absolute_error(y_true_log, y_pred_log),
        "RMSE_Full_LOG": math.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        "SMAPE_Full_LOG": smape_numpy(y_true_log, y_pred_log),
        "Total_Sequences": int(len(y_true_log))
    })

# Save full predictions and metrics
full_pred_df = pd.DataFrame(full_pred_rows).sort_values(["ATM_ID","DATE"])
full_pred_path = os.path.join(PREDICTIONS_DIR, "LSTM_predictions_full_continuous.csv")
full_pred_df.to_csv(full_pred_path, index=False)

full_df = pd.DataFrame(full_rows)
full_path = os.path.join(RESULTS_DIR, "LSTM_full_evaluation.csv")
full_df.to_csv(full_path, index=False)

print("Saved full continuous predictions →", full_pred_path)
print("Saved full-data evaluation →", full_path)

# ===============================================================
# 📈 GRAPHS: FULL continuous (LOG)
# ===============================================================
if not full_pred_df.empty:
    print("\n📊 Generating FULL continuous graphs per ATM (LOG)...")
    for atm_id in full_pred_df["ATM_ID"].unique():
        ap = full_pred_df[full_pred_df["ATM_ID"] == atm_id].sort_values("DATE")
        if ap.empty:
            continue

        # Actual vs Predicted (continuous)
        plt.figure(figsize=(12, 5))
        plt.plot(ap["DATE"], ap["Actual_LOG"], label="Actual (log1p)", linewidth=1.1)
        plt.plot(ap["DATE"], ap["Predicted_LOG"], label="Predicted (log1p)", linestyle="--", linewidth=1.1)
        plt.title(f"ATM {atm_id} – FULL Actual vs Predicted (LOG, continuous)")
        plt.xlabel("Date"); plt.ylabel("log1p(Withdrawals)")
        plt.legend(); plt.grid(True); format_dates(plt.gca()); plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, f"{atm_id}_FULL_actual_vs_predicted_log.png"))
        plt.close()

        # Residuals over time (continuous)
        plt.figure(figsize=(12, 4))
        plt.plot(ap["DATE"], ap["Residual_LOG"], linewidth=1.0)
        plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
        plt.title(f"ATM {atm_id} – FULL Residuals Over Time (LOG)")
        plt.xlabel("Date"); plt.ylabel("Residual (Pred - Actual) in log1p")
        plt.grid(True); format_dates(plt.gca()); plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, f"{atm_id}_FULL_residuals_over_time_log.png"))
        plt.close()

        # Residual histogram (continuous)
        plt.figure(figsize=(8, 5))
        plt.hist(ap["Residual_LOG"], bins=40, alpha=0.8)
        plt.title(f"ATM {atm_id} – FULL Residual Histogram (LOG)")
        plt.xlabel("Residual (Pred - Actual) in log1p"); plt.ylabel("Frequency")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, f"{atm_id}_FULL_residuals_hist_log.png"))
        plt.close()

        # Residual scatter vs Predicted (continuous)
        plt.figure(figsize=(6, 6))
        plt.scatter(ap["Predicted_LOG"], ap["Residual_LOG"], s=12, alpha=0.7)
        plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
        plt.title(f"ATM {atm_id} – FULL Residual Scatter (LOG)")
        plt.xlabel("Predicted (log1p)"); plt.ylabel("Residual (Pred - Actual)")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(GRAPHS_DIR, f"{atm_id}_FULL_residual_scatter_log.png"))
        plt.close()
else:
    print("\n No full predictions produced → FULL graphs skipped.")

# ===============================================================
# OVERALL SUMMARY ACROSS ATMs (CV means)
# ===============================================================
if not cv_results_df.empty:
    summary = {
        "ATM_NUMBER": int(len(cv_results_df)),
        "CV_RMSE_Mean_Mean": float(cv_results_df["CV_RMSE_Mean"].mean()),
        "CV_RMSE_Mean_Std":  float(cv_results_df["CV_RMSE_Mean"].std()),
        "CV_MAE_Mean_Mean":  float(cv_results_df["CV_MAE_Mean"].mean()),
        "CV_MAE_Mean_Std":   float(cv_results_df["CV_MAE_Mean"].std()),
        "CV_SMAPE_Mean_Mean":float(cv_results_df["CV_SMAPE_Mean"].mean()),
        "CV_SMAPE_Mean_Std": float(cv_results_df["CV_SMAPE_Mean"].std()),
        "Created_At": datetime.datetime.now(datetime.UTC).isoformat()
    }
    summary_path = os.path.join(RESULTS_DIR, "LSTM_CV_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n Final Cross-Validation Summary (saved):")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}")
else:
    print("\n CV results were empty, summary not written.")

