#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# ===============================================================
# CONFIG
# ===============================================================
os.makedirs("Prediction models/Data", exist_ok=True)
INPUT_CSV = "Prediction models/data_5_atms.csv"
OUTPUT_PKL = "Prediction models/Data/prepared_global_data.pkl"

# ===============================================================
# LOAD DATA
# ===============================================================
print("📂 Loading dataset...")
df = pd.read_csv(INPUT_CSV, sep=None, engine="python")

# Parse and sort
df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
df = df.sort_values(by=["ATM_ID", "DATE"]).reset_index(drop=True)
df["HOLIDAY"] = df["HOLIDAY"].astype(int)

# ===============================================================
# LOG TRANSFORM TARGET
# ===============================================================
# Add log-transformed withdrawals
df["WITHDRWLS_LOG"] = np.log1p(df["WITHDRWLS"])

# ===============================================================
# FEATURE ENGINEERING
# ===============================================================
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    results = []

    for atm_id, atm_df in data.groupby("ATM_ID"):
        atm_df = atm_df.copy().sort_values("DATE")

        # Lag features (log space)
        for lag in [1, 2, 3, 7, 14, 30]:
            atm_df[f"LAG{lag}"] = atm_df["WITHDRWLS_LOG"].shift(lag)

        # Rolling statistics (log space)
        for w in [3, 7, 14, 30]:
            atm_df[f"ROLL_MEAN_{w}"] = atm_df["WITHDRWLS_LOG"].rolling(w, min_periods=1).mean().shift(1)
            atm_df[f"ROLL_STD_{w}"] = atm_df["WITHDRWLS_LOG"].rolling(w, min_periods=1).std().shift(1)

        # Relative change features
        atm_df["DIFF_1"] = atm_df["WITHDRWLS_LOG"].diff().shift(1)
        atm_df["DIFF_7"] = atm_df["WITHDRWLS_LOG"].diff(7).shift(1)

        # Relative to 30-day mean
        rolling_mean_30 = atm_df["WITHDRWLS_LOG"].rolling(30, min_periods=5).mean().shift(1)
        atm_df["REL_TO_MEAN_30"] = atm_df["WITHDRWLS_LOG"] / rolling_mean_30
        atm_df["REL_TO_MEAN_30"].replace([np.inf, -np.inf], 1, inplace=True)
        atm_df["REL_TO_MEAN_30"].fillna(1, inplace=True)

        # Calendar features
        atm_df["DAY_OF_WEEK"] = atm_df["DATE"].dt.dayofweek
        atm_df["DAY_IN_MONTH"] = atm_df["DATE"].dt.day
        atm_df["MONTH"] = atm_df["DATE"].dt.month
        atm_df["IsWeekend"] = atm_df["DAY_OF_WEEK"].isin([5, 6]).astype(int)

        # Cyclical encoding
        atm_df["DOW_SIN"] = np.sin(2 * np.pi * atm_df["DAY_OF_WEEK"] / 7)
        atm_df["DOW_COS"] = np.cos(2 * np.pi * atm_df["DAY_OF_WEEK"] / 7)
        atm_df["MONTH_SIN"] = np.sin(2 * np.pi * atm_df["MONTH"] / 12)
        atm_df["MONTH_COS"] = np.cos(2 * np.pi * atm_df["MONTH"] / 12)

        # Holiday lags
        atm_df["HOLIDAY_LAG1"] = atm_df["HOLIDAY"].shift(1)
        atm_df["HOLIDAY_LAG2"] = atm_df["HOLIDAY"].shift(2)
        atm_df["HOLIDAY_ROLL7"] = atm_df["HOLIDAY"].rolling(7, min_periods=1).sum().shift(1)

        results.append(atm_df)

    return pd.concat(results, axis=0)

print("⚙️ Engineering features...")
df = engineer_features(df)

# ===============================================================
# CLEAN DATA
# ===============================================================
# Drop rows with NaNs caused by lag/rolling
df = df.dropna().reset_index(drop=True)

# ===============================================================
# FEATURE SELECTION
# ===============================================================
FEATURES = [
    # Lag & rolling
    "LAG1", "LAG2", "LAG3", "LAG7", "LAG14", "LAG30",
    "ROLL_MEAN_3", "ROLL_STD_3",
    "ROLL_MEAN_7", "ROLL_STD_7",
    "ROLL_MEAN_14", "ROLL_STD_14",
    "ROLL_MEAN_30", "ROLL_STD_30",
    # Relative
    "DIFF_1", "DIFF_7", "REL_TO_MEAN_30",
    # Calendar
    "DAY_OF_WEEK", "DAY_IN_MONTH", "MONTH", "IsWeekend",
    # Cyclical
    "DOW_SIN", "DOW_COS", "MONTH_SIN", "MONTH_COS",
    # Holiday
    "HOLIDAY", "HOLIDAY_LAG1", "HOLIDAY_LAG2", "HOLIDAY_ROLL7"
]

TARGET = "WITHDRWLS_LOG"

# ===============================================================
# SCALE PER ATM
# ===============================================================
scaled_frames = []
scalers = {}

print("📊 Scaling per ATM...")

for atm_id, atm_df in df.groupby("ATM_ID"):
    atm_df = atm_df.copy().sort_values("DATE")

    # Target scaler
    y_scaler = StandardScaler()
    atm_df["TARGET_SCALED"] = y_scaler.fit_transform(atm_df[[TARGET]])

    # Feature scaler
    X_scaler = StandardScaler()
    atm_df[FEATURES] = X_scaler.fit_transform(atm_df[FEATURES])

    scalers[atm_id] = {"y_scaler": y_scaler, "X_scaler": X_scaler}
    scaled_frames.append(atm_df)

df_scaled = pd.concat(scaled_frames).reset_index(drop=True)

# ===============================================================
# SAVE OUTPUT
# ===============================================================
prepared_global_data = {
    "data": df_scaled,
    "features": FEATURES,
    "target": "TARGET_SCALED",
    "ATM_IDs": df_scaled["ATM_ID"].unique().tolist(),
    "scalers": scalers
}

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(prepared_global_data, f)

print("\n✅ Data preparation completed successfully!")
print(f"Total ATMs: {len(prepared_global_data['ATM_IDs'])}")
print(f"Total samples: {len(df_scaled)}")
print(f"Features: {len(FEATURES)}")
print("📁 Saved →", OUTPUT_PKL)



