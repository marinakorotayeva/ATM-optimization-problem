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

INPUT_CSV = "Prediction models/data_5_atms.csv"
OUTPUT_PKL = "Prediction models/Data/prepared_weekly.pkl"
TSCV_PKL   = "Prediction models/Data/tscv_weekly.pkl"

# ===============================================================
# LOAD & CLEAN
# ===============================================================
print("📂 Loading dataset...")
df = pd.read_csv(INPUT_CSV, sep=None, engine="python")
df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
df = df.sort_values(["ATM_ID", "DATE"]).reset_index(drop=True)

if "HOLIDAY" not in df.columns:
    df["HOLIDAY"] = 0
df["HOLIDAY"] = df["HOLIDAY"].fillna(0).astype(int)

df["WITHDRWLS_LOG"] = np.log1p(df["WITHDRWLS"].clip(lower=0))
df["Year"] = df["DATE"].dt.isocalendar().year.astype(int)
df["Week"] = df["DATE"].dt.isocalendar().week.astype(int)

# ===============================================================
# AGGREGATE WEEKLY FEATURES
# ===============================================================
print("🧮 Aggregating weekly data...")
weekly_parts = []
for atm_id, sub in df.groupby("ATM_ID"):
    sub = sub.copy()
    weekly = sub.groupby(["Year", "Week"]).agg({
        "WITHDRWLS_LOG": ["mean", "std", "min", "max", "sum"],
        "HOLIDAY": ["sum", "max"],
    })
    weekly.columns = ["_".join(col) for col in weekly.columns]
    weekly = weekly.reset_index()
    weekly["ATM_ID"] = atm_id
    weekly["WEEK_START"] = pd.to_datetime(
        weekly["Year"].astype(str) + weekly["Week"].astype(str) + "1",
        format="%G%V%u", errors="coerce"
    )
    weekly = weekly.sort_values("WEEK_START").reset_index(drop=True)

    # Cyclical week-of-year
    weekly["WEEK_SIN"] = np.sin(2 * np.pi * weekly["Week"] / 52.0)
    weekly["WEEK_COS"] = np.cos(2 * np.pi * weekly["Week"] / 52.0)

    weekly_parts.append(weekly)

weekly_df = pd.concat(weekly_parts, axis=0).sort_values(["ATM_ID", "WEEK_START"])

# ===============================================================
# GENERATE FUTURE TARGETS
# ===============================================================
print("🎯 Generating next-week (7-day) targets...")
df["Week_Start"] = df["DATE"] - pd.to_timedelta(df["DATE"].dt.dayofweek, unit="D")

targets = []
for atm_id, sub in df.groupby("ATM_ID"):
    sub = sub.sort_values("DATE").reset_index(drop=True)
    weeks = sorted(sub["Week_Start"].unique())

    for w in range(len(weeks) - 1):
        current_week = weeks[w]
        next_week = weeks[w + 1]
        this_week_data = sub[sub["Week_Start"] == current_week]
        next_week_data = sub[sub["Week_Start"] == next_week]
        if len(next_week_data) < 7:
            continue

        row = {"ATM_ID": atm_id, "WEEK_START": current_week}
        for i, day in enumerate(sorted(next_week_data["DATE"].unique())):
            row[f"TARGET_h{i+1}"] = np.log1p(
                next_week_data.loc[next_week_data["DATE"] == day, "WITHDRWLS"].values[0]
            )
        targets.append(row)

targets_df = pd.DataFrame(targets)
df_final = pd.merge(weekly_df, targets_df, on=["ATM_ID", "WEEK_START"], how="inner")

# ===============================================================
# ADD TIME-SERIES FEATURES
# ===============================================================
print("🧩 Adding lag & rolling features...")

def add_time_features(df):
    df = df.sort_values("WEEK_START").copy()
    for lag in [1, 2, 4, 8, 12]:
        df[f"WITHDRWLS_LOG_sum_LAG{lag}"] = df["WITHDRWLS_LOG_sum"].shift(lag)
        df[f"WITHDRWLS_LOG_mean_LAG{lag}"] = df["WITHDRWLS_LOG_mean"].shift(lag)

    for w in [4, 8, 13, 26, 52]:
        df[f"ROLL_MEAN_{w}"] = df["WITHDRWLS_LOG_sum"].shift(1).rolling(w, min_periods=2).mean()
        df[f"ROLL_STD_{w}"] = df["WITHDRWLS_LOG_sum"].shift(1).rolling(w, min_periods=2).std()

    df["SUM_DIFF_1"] = df["WITHDRWLS_LOG_sum"].diff(1)
    df["SUM_PCT_1"] = df["WITHDRWLS_LOG_sum"].pct_change(1)
    df["IS_MONTH_END_WEEK"] = (df["WEEK_START"] + pd.Timedelta(days=6)).dt.is_month_end.astype(int)
    df["IS_MONTH_START_WEEK"] = (df["WEEK_START"].dt.is_month_start).astype(int)
    return df

enhanced_parts = []
for atm_id, sub in df_final.groupby("ATM_ID"):
    enhanced_parts.append(add_time_features(sub))

df_final = pd.concat(enhanced_parts, axis=0).dropna().reset_index(drop=True)

# ===============================================================
# SCALE PER ATM
# ===============================================================
print("📊 Scaling per ATM...")
scalers = {}
scaled_frames = []

target_cols = [f"TARGET_h{i}" for i in range(1, 8)]
feature_cols = [c for c in df_final.columns if c not in
                ["ATM_ID", "Year", "Week", "WEEK_START"] + target_cols]

for atm_id, sub in df_final.groupby("ATM_ID"):
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    g = sub.copy().sort_values("WEEK_START")

    g[feature_cols] = X_scaler.fit_transform(g[feature_cols])
    y_scaled = y_scaler.fit_transform(g[target_cols])
    y_scaled_df = pd.DataFrame(y_scaled, columns=[f"{t}_SCALED" for t in target_cols])
    g = pd.concat([g.reset_index(drop=True), y_scaled_df], axis=1)

    scalers[atm_id] = {"X_scaler": X_scaler, "y_scaler": y_scaler}
    scaled_frames.append(g)

df_scaled = pd.concat(scaled_frames, axis=0).sort_values(["ATM_ID", "WEEK_START"]).reset_index(drop=True)

# ===============================================================
# TIME SERIES SPLITS
# ===============================================================
print("🔁 Generating weekly TimeSeriesSplit folds...")
tscv_splits = {}
tscv = TimeSeriesSplit(n_splits=5)

for atm_id, sub in df_scaled.groupby("ATM_ID"):
    X = sub[feature_cols].values
    splits = []
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        splits.append({"fold": fold, "train_idx": tr.tolist(), "test_idx": te.tolist()})
    tscv_splits[atm_id] = splits

# ===============================================================
# SAVE
# ===============================================================
prepared = {
    "data": df_scaled,
    "features": feature_cols,
    "targets": [f"{t}_SCALED" for t in target_cols],
    "ATM_IDs": df_scaled["ATM_ID"].unique().tolist(),
    "scalers": scalers,
}

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(prepared, f)
with open(TSCV_PKL, "wb") as f:
    pickle.dump(tscv_splits, f)

print("\n✅ Enhanced weekly data preparation completed successfully!")
print(f"Features: {len(feature_cols)} → {feature_cols[:10]} ...")
print(f"Saved → {OUTPUT_PKL}")
print(f"CV Splits → {TSCV_PKL}")

