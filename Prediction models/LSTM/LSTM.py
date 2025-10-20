import os
import math
import pickle
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# ===============================================================
# CONFIG
# ===============================================================
DATA_PATH = "Prediction models/Data/prepared_global_data.pkl"
OUTPUT_DIR = "Prediction models/LSTM"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

SEQ_LEN = 30
EPOCHS = 60
BATCH_SIZE = 32
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ===============================================================
# UTILS
# ===============================================================
def smape_numpy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100

def create_sequences(X, y, seq_length=30):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# ===============================================================
# LOAD DATA
# ===============================================================
print("📂 Loading prepared dataset...")
with open(DATA_PATH, "rb") as f:
    prepared = pickle.load(f)

data = prepared["data"].sort_values(["ATM_ID", "DATE"]).reset_index(drop=True)
features = prepared["features"]
target = prepared["target"]
scalers = prepared["scalers"]
ATM_IDs = prepared["ATM_IDs"]

print(f"✅ Loaded {len(data)} samples from {len(ATM_IDs)} ATMs.")

# ===============================================================
# PER-ATM CROSS-VALIDATION
# ===============================================================
overall_results = []
predictions_records = []

print("\n🔁 Starting per-ATM TimeSeries Cross-Validation...\n")

for atm_id in ATM_IDs:
    print(f"🏧 ATM: {atm_id}")
    atm_df = data[data["ATM_ID"] == atm_id].copy().reset_index(drop=True)

    if len(atm_df) < SEQ_LEN + 50:
        print("⚠️ Skipping (too few samples)")
        continue

    scaler = scalers[atm_id]["y_scaler"]

    X_all = atm_df[features].values
    y_all = atm_df[target].values

    tscv = TimeSeriesSplit(n_splits=5)
    atm_rmse, atm_mae, atm_smape = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), 1):
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        Xtr_seq, ytr_seq = create_sequences(X_train, y_train, SEQ_LEN)
        Xte_seq, yte_seq = create_sequences(X_test, y_test, SEQ_LEN)

        model = build_lstm_model((SEQ_LEN, len(features)))
        es = callbacks.EarlyStopping(patience=12, restore_best_weights=True, verbose=0)

        model.fit(
            Xtr_seq, ytr_seq,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=0,
            callbacks=[es]
        )

        yp_scaled = model.predict(Xte_seq, verbose=0).ravel()

        # Inverse-transform
        yte_inv = scaler.inverse_transform(yte_seq.reshape(-1, 1)).ravel()
        yp_inv = scaler.inverse_transform(yp_scaled.reshape(-1, 1)).ravel()

        rmse = math.sqrt(mean_squared_error(yte_inv, yp_inv))
        mae = mean_absolute_error(yte_inv, yp_inv)
        sm = smape_numpy(yte_inv, yp_inv)

        atm_rmse.append(rmse)
        atm_mae.append(mae)
        atm_smape.append(sm)

        # Store predictions
        test_dates = atm_df["DATE"].iloc[test_idx].values[SEQ_LEN:]
        for i in range(len(yte_inv)):
            predictions_records.append({
                "ATM_ID": atm_id,
                "Fold": fold,
                "DATE": test_dates[i],
                "Actual": yte_inv[i],
                "Predicted": yp_inv[i],
                "Residual": yp_inv[i] - yte_inv[i],
                "Absolute_Error": abs(yp_inv[i] - yte_inv[i])
            })

        print(f"   🔹 Fold {fold} → RMSE={rmse:.2f}, MAE={mae:.2f}, SMAPE={sm:.2f}%")

    # Per-ATM summary
    overall_results.append({
        "ATM_ID": atm_id,
        "CV_RMSE_Mean": np.mean(atm_rmse),
        "CV_MAE_Mean": np.mean(atm_mae),
        "CV_SMAPE_Mean": np.mean(atm_smape)
    })

# ===============================================================
# SAVE RESULTS
# ===============================================================
results_df = pd.DataFrame(overall_results)
results_df.to_csv(os.path.join(RESULTS_DIR, "LSTM_per_ATM_results.csv"), index=False)

pred_df = pd.DataFrame(predictions_records)
pred_df.to_csv(os.path.join(PREDICTIONS_DIR, "predictions_per_atm.csv"), index=False)

print("\n📊 === Overall Results (Mean Across ATMs) ===")
print(results_df.describe().loc[["mean", "std"]])

print("\n📁 Saved per-ATM results →", os.path.join(RESULTS_DIR, "LSTM_per_ATM_results.csv"))
print("📁 Saved per-ATM predictions →", os.path.join(PREDICTIONS_DIR, "predictions_per_atm.csv"))

# ===============================================================
# 📈 GRAPHS: ACTUAL VS PREDICTED + RESIDUALS
# ===============================================================
print("\n📊 Generating graphs for each ATM...")

for atm_id in pred_df["ATM_ID"].unique():
    atm_preds = pred_df[pred_df["ATM_ID"] == atm_id].sort_values("DATE")

    # ---- Plot 1: Actual vs Predicted ----
    plt.figure(figsize=(12, 5))
    plt.plot(atm_preds["DATE"], atm_preds["Actual"], label="Actual", color="blue")
    plt.plot(atm_preds["DATE"], atm_preds["Predicted"], label="Predicted", color="red", linestyle="--")
    plt.title(f"ATM {atm_id} – Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Withdrawals (log-transformed inverse)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f"{atm_id}_actual_vs_predicted.png"))
    plt.close()

    # ---- Plot 2: Residuals Histogram ----
    plt.figure(figsize=(8, 5))
    plt.hist(atm_preds["Residual"], bins=40, color="purple", alpha=0.7)
    plt.title(f"ATM {atm_id} – Residuals Distribution")
    plt.xlabel("Prediction Error (Residual)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, f"{atm_id}_residuals_hist.png"))
    plt.close()

print("✅ Graphs saved to:", GRAPHS_DIR)

# ===============================================================
# SUMMARY (Detailed Results File)
# ===============================================================
cv_rmse_mean = results_df["CV_RMSE_Mean"].mean()
cv_rmse_std = results_df["CV_RMSE_Mean"].std()

cv_mae_mean = results_df["CV_MAE_Mean"].mean()
cv_mae_std = results_df["CV_MAE_Mean"].std()

cv_smape_mean = results_df["CV_SMAPE_Mean"].mean()
cv_smape_std = results_df["CV_SMAPE_Mean"].std()

# ---- Full dataset evaluation (train + test on all) ----
# Build one final model per ATM and compute metrics on full data
full_mae_list, full_rmse_list, full_smape_list = [], [], []
total_samples = 0

for atm_id in ATM_IDs:
    atm_df = data[data["ATM_ID"] == atm_id].copy().reset_index(drop=True)
    if len(atm_df) < SEQ_LEN + 1:
        continue

    scaler = scalers[atm_id]["y_scaler"]
    X_all = atm_df[features].values
    y_all = atm_df[target].values
    X_seq, y_seq = create_sequences(X_all, y_all, SEQ_LEN)

    model = build_lstm_model((SEQ_LEN, len(features)))
    es = callbacks.EarlyStopping(patience=12, restore_best_weights=True, verbose=0)
    model.fit(X_seq, y_seq, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=0, callbacks=[es])

    y_pred_scaled = model.predict(X_seq, verbose=0).ravel()
    y_true_inv = scaler.inverse_transform(y_seq.reshape(-1, 1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    full_mae_list.append(mean_absolute_error(y_true_inv, y_pred_inv))
    full_rmse_list.append(math.sqrt(mean_squared_error(y_true_inv, y_pred_inv)))
    full_smape_list.append(smape_numpy(y_true_inv, y_pred_inv))

    total_samples += len(y_true_inv)

# Compute final full-data averages
mae_full = np.mean(full_mae_list) if full_mae_list else np.nan
rmse_full = np.mean(full_rmse_list) if full_rmse_list else np.nan
smape_full = np.mean(full_smape_list) if full_smape_list else np.nan

# ===============================================================
# SAVE SUMMARY CSV
# ===============================================================
summary = {
    "ATM_NUMBER": len(ATM_IDs),
    "CV_RMSE_Mean": cv_rmse_mean,
    "CV_RMSE_Std": cv_rmse_std,
    "CV_MAE_Mean": cv_mae_mean,
    "CV_MAE_Std": cv_mae_std,
    "CV_SMAPE_Mean": cv_smape_mean,
    "CV_SMAPE_Std": cv_smape_std,
    "MAE_Full": mae_full,
    "RMSE_Full": rmse_full,
    "SMAPE_Full": smape_full,
    "Total_Samples": total_samples
}

summary_path = os.path.join(RESULTS_DIR, "LSTM_CV_summary.csv")
pd.DataFrame([summary]).to_csv(summary_path, index=False)

print("\n✅ Final Cross-Validation Summary (saved):")
for k, v in summary.items():
    print(f"{k}: {v:.4f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}")
