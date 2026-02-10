#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export 12 scenario files from:
  Prediction models/XGBoost/predictions/XGBoost_weekly_CV_predictions_whatif_top3.csv

We use 4 fixed weeks:
  - low_demand      : 2007-01-01
  - high_volatility : 2007-06-18
  - holiday         : 2008-01-28
  - normal          : 2008-01-21

For each week, export 3 scenario files:
  - baseline
  - x0p9
  - x1p1

Output directory:
  Prediction models/XGBoost/predictions/
"""

import os
import pandas as pd

# =========================
# CONFIG
# =========================
IN_CSV = "Prediction models/XGBoost/predictions/XGBoost_weekly_CV_predictions_whatif_top3.csv"
OUT_DIR = "Prediction models/XGBoost/predictions"

FIXED_WEEKS = {
    "low_demand": "2007-01-01",
    "high_volatility": "2007-06-18",
    "holiday": "2008-01-28",
    "normal": "2008-01-21",
}

SCENARIOS = ["baseline", "x0p9", "x1p1"]

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD
# =========================
df = pd.read_csv(IN_CSV)
df["Week_Start"] = pd.to_datetime(df["Week_Start"])

if "Scenario" not in df.columns:
    raise ValueError("Input CSV must contain a 'Scenario' column (baseline/x0p9/x1p1).")

# =========================
# EXPORT 12 FILES
# =========================
exported = 0

for week_type, week_str in FIXED_WEEKS.items():
    week_dt = pd.to_datetime(week_str)

    for scn in SCENARIOS:
        out_name = f"XGBoost_week_{week_dt.strftime('%Y-%m-%d')}_{week_type}_{scn}.csv"
        out_path = os.path.join(OUT_DIR, out_name)

        wk = df[(df["Week_Start"] == week_dt) & (df["Scenario"] == scn)].copy()

        if wk.empty:
            print(f"WARNING: No rows found for week={week_str} | type={week_type} | scenario={scn}")
            continue

        # Optional: keep consistent ordering
        wk.sort_values(["ATM_ID", "Horizon"], inplace=True)

        wk.to_csv(out_path, index=False)
        exported += 1
        print(f"Saved → {out_path}  ({len(wk)} rows)")

print(f"\nDone. Exported {exported} files to: {OUT_DIR}")

