#!/usr/bin/env python3
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# ===============================================================
# CONFIG
# ===============================================================
os.makedirs("Prediction models/Data", exist_ok=True)

INPUT_CSV = "Prediction models/data_5_atms.csv"   # <-- change if needed
OUTPUT_PKL = "Prediction models/Data/prepared_data.pkl"
TSCV_PKL   = "Prediction models/Data/tscv.pkl"

# Lags / windows you want (rich but not insane)
LAG_LIST = [1, 2, 3, 7, 14, 30, 60]
ROLL_WINDOWS = [7, 14, 30, 60]

# ===============================================================
# LOAD
# ===============================================================
print("📂 Loading dataset...")
df = pd.read_csv(INPUT_CSV, sep=None, engine="python")

# Parse dates (keeps your dayfirst behavior but accepts ISO too)
df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
df = df.sort_values(["ATM_ID", "DATE"]).reset_index(drop=True)

# Basic hygiene
if "HOLIDAY" not in df.columns:
    df["HOLIDAY"] = 0
df["HOLIDAY"] = df["HOLIDAY"].fillna(0).astype(int)

# ===============================================================
# TARGET (Option A): LOG1P transform
# ===============================================================
df["WITHDRWLS_LOG"] = np.log1p(df["WITHDRWLS"].clip(lower=0))

# ===============================================================
# FEATURE ENGINEERING (per ATM)
# ===============================================================
def engineer_features(atm_df: pd.DataFrame) -> pd.DataFrame:
    g = atm_df.copy().sort_values("DATE")

    # ---------- Lags (log space) ----------
    for lag in LAG_LIST:
        g[f"LAG{lag}"] = g["WITHDRWLS_LOG"].shift(lag)

    # ---------- Rolling stats (log space) ----------
    for w in ROLL_WINDOWS:
        # shift(1) so we never leak the current target
        g[f"ROLL_MEAN_{w}"] = g["WITHDRWLS_LOG"].rolling(w, min_periods=1).mean().shift(1)
        g[f"ROLL_STD_{w}"]  = g["WITHDRWLS_LOG"].rolling(w, min_periods=1).std().shift(1)
        g[f"ROLL_MIN_{w}"]  = g["WITHDRWLS_LOG"].rolling(w, min_periods=1).min().shift(1)
        g[f"ROLL_MAX_{w}"]  = g["WITHDRWLS_LOG"].rolling(w, min_periods=1).max().shift(1)

    # ---------- Diffs / rates (log space) ----------
    g["DIFF_1"] = g["WITHDRWLS_LOG"].diff(1).shift(1)
    g["DIFF_7"] = g["WITHDRWLS_LOG"].diff(7).shift(1)

    # Relative to 30-day mean (log space)
    roll30 = g["WITHDRWLS_LOG"].rolling(30, min_periods=5).mean().shift(1)
    rel = g["WITHDRWLS_LOG"] / roll30
    rel = rel.replace([np.inf, -np.inf], 1.0).fillna(1.0)
    g["REL_TO_MEAN_30"] = rel

    # ---------- Calendar ----------
    g["DAY_OF_WEEK"]  = g["DATE"].dt.dayofweek
    g["DAY_IN_MONTH"] = g["DATE"].dt.day
    g["WEEK_OF_YEAR"] = g["DATE"].dt.isocalendar().week.astype(int)
    g["MONTH"]        = g["DATE"].dt.month
    g["IsWeekend"]    = g["DAY_OF_WEEK"].isin([5, 6]).astype(int)
    g["IsMonthStart"] = g["DATE"].dt.is_month_start.astype(int)
    g["IsMonthEnd"]   = g["DATE"].dt.is_month_end.astype(int)
    # Days to month end (helps month-end spikes)
    month_end = g["DATE"] + pd.offsets.MonthEnd(0)
    g["DaysToMonthEnd"] = (month_end - g["DATE"]).dt.days

    # ---------- Cyclical encodings ----------
    g["DOW_SIN"]   = np.sin(2 * np.pi * g["DAY_OF_WEEK"] / 7)
    g["DOW_COS"]   = np.cos(2 * np.pi * g["DAY_OF_WEEK"] / 7)
    g["MONTH_SIN"] = np.sin(2 * np.pi * g["MONTH"] / 12)
    g["MONTH_COS"] = np.cos(2 * np.pi * g["MONTH"] / 12)

    # ---------- Holiday features ----------
    g["HOLIDAY_LAG1"] = g["HOLIDAY"].shift(1)
    g["HOLIDAY_LAG2"] = g["HOLIDAY"].shift(2)
    g["HOLIDAY_FWD1"] = g["HOLIDAY"].shift(-1)
    g["HOLIDAY_ROLL7"] = g["HOLIDAY"].rolling(7, min_periods=1).sum().shift(1)

    # Holiday proximity: any holiday +/- 3 days
    # (dilates holiday effect to nearby days)
    h = g["HOLIDAY"].rolling(7, min_periods=1, center=True).max()
    g["HOLIDAY_NEAR_3D"] = h.fillna(0).astype(int)

    return g

print("⚙️ Engineering features...")
parts = []
for atm_id, sub in df.groupby("ATM_ID"):
    parts.append(engineer_features(sub))
df = pd.concat(parts, axis=0).sort_values(["ATM_ID", "DATE"]).reset_index(drop=True)

# Drop rows created by lags/rolls
df = df.dropna().reset_index(drop=True)

# ===============================================================
# FEATURE SET
# ===============================================================
FEATURES = (
    # Lags
    [f"LAG{lag}" for lag in LAG_LIST]
    +
    # Rolling stats
    sum([[f"ROLL_MEAN_{w}", f"ROLL_STD_{w}", f"ROLL_MIN_{w}", f"ROLL_MAX_{w}"] for w in ROLL_WINDOWS], [])
    +
    # Diffs / relative
    ["DIFF_1", "DIFF_7", "REL_TO_MEAN_30"]
    +
    # Calendar
    ["DAY_OF_WEEK", "DAY_IN_MONTH", "WEEK_OF_YEAR", "MONTH",
     "IsWeekend", "IsMonthStart", "IsMonthEnd", "DaysToMonthEnd"]
    +
    # Cyclical
    ["DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS"]
    +
    # Holiday
    ["HOLIDAY", "HOLIDAY_LAG1", "HOLIDAY_LAG2", "HOLIDAY_FWD1",
     "HOLIDAY_ROLL7", "HOLIDAY_NEAR_3D"]
)

TARGET_RAW  = "WITHDRWLS"       # original
TARGET_LOG  = "WITHDRWLS_LOG"   # model base
TARGET_SCALED_COL = "TARGET_SCALED"

# ===============================================================
# SCALE PER ATM
# ===============================================================
print("📊 Scaling per ATM...")
scaled_frames = []
scalers = {}

for atm_id, sub in df.groupby("ATM_ID"):
    g = sub.copy().sort_values("DATE")

    # Target scaler in log space
    y_scaler = StandardScaler()
    g[TARGET_SCALED_COL] = y_scaler.fit_transform(g[[TARGET_LOG]])

    # Feature scaler
    X_scaler = StandardScaler()
    g[FEATURES] = X_scaler.fit_transform(g[FEATURES])

    scalers[atm_id] = {"y_scaler": y_scaler, "X_scaler": X_scaler}
    scaled_frames.append(g)

df_scaled = pd.concat(scaled_frames, axis=0).sort_values(["ATM_ID", "DATE"]).reset_index(drop=True)

# ===============================================================
# GENERATE PER-ATM TIME SERIES SPLITS
# ===============================================================
print("🔁 Generating per-ATM TimeSeriesSplit folds...")
tscv_splits = {}
tscv = TimeSeriesSplit(n_splits=5)

for atm_id, sub in df_scaled.groupby("ATM_ID"):
    X = sub[FEATURES].values
    splits = []
    # Note: TimeSeriesSplit uses index positions (0..len-1) of "sub"
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        splits.append({
            "fold": fold,
            "train_idx": tr.tolist(),
            "test_idx": te.tolist()
        })
    tscv_splits[atm_id] = splits

# ===============================================================
# SAVE
# ===============================================================
prepared = {
    "data": df_scaled,              # includes ATM_ID, DATE, original target columns, scaled features, and TARGET_SCALED
    "features": FEATURES,
    "target": TARGET_SCALED_COL,    # what models should learn
    "ATM_IDs": df_scaled["ATM_ID"].unique().tolist(),
    "scalers": scalers
}

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(prepared, f)

with open(TSCV_PKL, "wb") as f:
    pickle.dump(tscv_splits, f)

# ===============================================================
# REPORT
# ===============================================================
print("\n✅ Data preparation completed successfully!")
print(f"Total ATMs: {len(prepared['ATM_IDs'])}")
print(f"Total samples: {len(df_scaled)}")
print(f"Features: {len(FEATURES)}")
print("📁 Saved prepared data →", OUTPUT_PKL)
print("📁 Saved CV splits   →", TSCV_PKL)

