#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pickle
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NEW: SHAP (for beeswarm explainability)
import shap

# ===============================================================
# CONFIG
# ===============================================================
DATA_PATH = "Prediction models/Data/prepared_weekly.pkl"
TSCV_PATH = "Prediction models/Data/tscv_weekly.pkl"

OUTPUT_DIR = "Prediction models/LSTM"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")

# NEW: Explainability folder (same structure as you requested for other models)
EXPLAIN_DIR = os.path.join(OUTPUT_DIR, "Explainability")
SHAP_DIR = os.path.join(EXPLAIN_DIR, "shap_outputs")
FEATURE_IMPORTANCE_DIR = os.path.join(EXPLAIN_DIR, "feature_importance")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(EXPLAIN_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)

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

# NEW: SHAP helper (avoid harmless matplotlib warnings)
def safe_colorbar_warning_filter():
    warnings.filterwarnings(
        "ignore",
        message="No data for colormapping provided via 'c'.*",
        category=UserWarning
    )

# NEW: Gradient-based feature importance for NN (1 plot per ATM)
def compute_gradient_importance(model, X_seq_explain):
    """
    Simple NN feature importance: mean absolute gradient of mean output wrt inputs.
    Returns importance per feature (shape: [n_features]).
    """
    X_tensor = tf.convert_to_tensor(X_seq_explain, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        y_out = model(X_tensor, training=False)          # (n, 7)
        y_scalar = tf.reduce_mean(y_out, axis=1)         # (n,)
        y_mean = tf.reduce_mean(y_scalar)                # scalar
    grads = tape.gradient(y_mean, X_tensor)              # (n, seq_len, n_features)
    imp = tf.reduce_mean(tf.abs(grads), axis=[0, 1])     # (n_features,)
    return imp.numpy()

# ===============================================================
# LOAD WEEKLY DATA
# ===============================================================
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

print(f"Loaded {len(data)} weekly samples for {len(ATM_IDs)} ATMs")

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
    print(f"\nATM {atm_id}")
    df = data[data["ATM_ID"] == atm_id].sort_values("WEEK_START").reset_index(drop=True)
    if len(df) < SEQ_LEN + 8:
        print(" Not enough samples, skipping.")
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
    # FULL-DATA TRAIN for Graphs + Explainability
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
    #plt.fill_between(weeks_seq, mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, label="Pred ±1σ")
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
    # EXPLAINABILITY (SHAP beeswarm + feature importance) — 1 each per ATM
    # ===============================================================

        # --- SHAP beeswarm (aggregate across outputs + time) ---
    try:
        safe_colorbar_warning_filter()

        # background and explain sets (keep small for speed/stability)
        n_bg = min(50, len(X_seq))
        n_explain = min(200, len(X_seq))

        # deterministic selection for explain points
        if n_explain < len(X_seq):
            idx_explain = np.linspace(0, len(X_seq) - 1, n_explain).astype(int)
            X_explain = X_seq[idx_explain]
        else:
            X_explain = X_seq

        X_bg = X_seq[:n_bg]

        # GradientExplainer is usually more robust for TF/Keras than DeepExplainer for RNNs
        explainer = shap.GradientExplainer(model_full, X_bg)
        shap_vals = explainer.shap_values(X_explain)

        # --- Normalize shap_vals into a single numpy array ---
        # Possible formats:
        # 1) list of outputs: [ (n, seq, f), ... ] OR [ (n, seq, f, outdim) ... ]
        # 2) array: (n, seq, f) or (n, seq, f, outdim) etc.
        if isinstance(shap_vals, list):
            # stack outputs first
            shap_arr = np.stack([np.asarray(sv) for sv in shap_vals], axis=0)
            # shap_arr now: (out, n, seq, f, ...) maybe
            # average over output dimension
            shap_arr = shap_arr.mean(axis=0)
        else:
            shap_arr = np.asarray(shap_vals)

        # Now shap_arr might still have extra trailing dims (e.g., outputs)
        # We want (n, seq, f). If there are extra dims, average them out.
        # Example: (n, seq, f, 7) -> mean over last axis
        while shap_arr.ndim > 3:
            shap_arr = shap_arr.mean(axis=-1)

        # If SHAP returned (seq, f) for a single sample, expand to (1, seq, f)
        if shap_arr.ndim == 2:
            shap_arr = shap_arr[None, :, :]

        # Reduce time dimension to get standard beeswarm input: (n, f)
        shap_2d = shap_arr.mean(axis=1)
        X_2d = X_explain.mean(axis=1)

        # Final safety check
        if shap_2d.shape != X_2d.shape:
            raise ValueError(f"After aggregation, SHAP shape {shap_2d.shape} != X shape {X_2d.shape}")

        # Limit points for plotting
        max_points = min(5000, shap_2d.shape[0])
        if shap_2d.shape[0] > max_points:
            rng = np.random.RandomState(42)
            pick = rng.choice(shap_2d.shape[0], size=max_points, replace=False)
            shap_2d = shap_2d[pick]
            X_2d = X_2d[pick]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_2d,
            X_2d,
            feature_names=features,
            show=False,
            plot_type="dot",
            color_bar=False
        )
        plt.title(f"{atm_id} — SHAP Beeswarm (aggregated over time & H1–H7)", pad=12)
        save_fig(os.path.join(SHAP_DIR, f"{atm_id}_shap_beeswarm.png"))

    except Exception as e:
        print(f"   SHAP beeswarm failed for ATM {atm_id}: {e}")


    # --- Feature importance (gradient-based) ---
    try:
        # Use same explain set as above (or fallback)
        if 'X_explain' not in locals():
            n_explain = min(200, len(X_seq))
            if n_explain < len(X_seq):
                idx_explain = np.linspace(0, len(X_seq) - 1, n_explain).astype(int)
                X_explain = X_seq[idx_explain]
            else:
                X_explain = X_seq

        fi_vals = compute_gradient_importance(model_full, X_explain)  # (n_features,)
        order = np.argsort(fi_vals)[::-1]

        plt.figure(figsize=(8, 6))
        plt.barh(np.array(features)[order][::-1], fi_vals[order][::-1])
        plt.title(f"{atm_id} — Feature importance (mean |gradient|)")
        plt.xlabel("Mean |∂output/∂feature|")
        save_fig(os.path.join(FEATURE_IMPORTANCE_DIR, f"{atm_id}_feature_importance.png"))

    except Exception as e:
        print(f"   Feature importance failed for ATM {atm_id}: {e}")

    # cleanup locals that may carry across loop (safe)
    if "X_explain" in locals():
        del X_explain

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

print("\nLSTM weekly modeling completed successfully!")
print(f"Results → {RESULTS_DIR}")
print(f"Predictions → {PREDICTIONS_DIR}")
print(f"Graphs → {GRAPHS_DIR}")
print(f"Explainability (SHAP beeswarm) → {SHAP_DIR}")
print(f"Explainability (feature importance) → {FEATURE_IMPORTANCE_DIR}")



