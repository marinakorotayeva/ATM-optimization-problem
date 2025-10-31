#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline ATM Refill Simulation
==============================
A simple non-optimized baseline model for comparison.

Logic:
- Every Monday, refill each ATM to full capacity.
- On other days, do nothing.
- Tracks stock, refills, cash-outs, and costs.

Outputs:
- Daily simulation results (CSV + readable TXT)
- Summary metrics (CSV)
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
# 🔧 Choose your prediction source file here
PRED_FILE = "Prediction models/XGBoost/predictions/XGBoost_weekly_CV_predictions.csv"
# PRED_FILE = "Prediction models/Linear Regression/predictions/LR_weekly_CV_predictions.csv"
# PRED_FILE = "Prediction models/LSTM/predictions/LSTM_weekly_CV_predictions.csv"

DEFAULT_INITIAL_STOCK = 500_000
DEFAULT_CAPACITY = 20_000_000
REFILL_COST_PER_UNIT = 0.01
PENALTY_OUT_OF_CASH = 10_000

# Output folder
RESULTS_DIR = "Baseline simulation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# LOAD FORECAST DATA
# ============================================================
if not os.path.exists(PRED_FILE):
    raise FileNotFoundError(f"❌ Prediction file not found: {PRED_FILE}")

print(f"📂 Loading predictions from: {PRED_FILE}")
df = pd.read_csv(PRED_FILE)
df["Week_Start"] = pd.to_datetime(df["Week_Start"], errors="coerce")
df = df.dropna(subset=["Week_Start"])
df = df.sort_values(["ATM_ID", "Week_Start", "Horizon"]).reset_index(drop=True)

# Convert horizons (1–7) to actual calendar dates
df["Date"] = df["Week_Start"] + pd.to_timedelta(df["Horizon"] - 1, unit="D")

# If predictions are in log1p scale → convert back to real scale
if df["Predicted"].mean() < 100:  # heuristic threshold
    print("🔁 Detected log-scale predictions — converting to real cash values...")
    df["Predicted"] = np.expm1(df["Predicted"])
else:
    print("✅ Predictions already in real cash scale.")

# ============================================================
# SIMULATION FUNCTION
# ============================================================
def simulate_monday_refill(pred_df, capacity, init_stock, refill_cost, penalty):
    """Simulate simple policy: refill to full every Monday."""
    atms = pred_df["ATM_ID"].unique()
    stocks = {atm: init_stock for atm in atms}
    results = []

    for date in sorted(pred_df["Date"].unique()):
        day_df = pred_df[pred_df["Date"] == date]

        for _, row in day_df.iterrows():
            atm = row["ATM_ID"]
            withdrawal = row["Predicted"]
            stock_prev = stocks[atm]

            # --- Simple rule ---
            refill = 0.0
            if date.weekday() == 0:  # Monday
                refill = max(0, capacity - stock_prev)

            # --- Stock update ---
            stock_today = stock_prev + refill - withdrawal

            out_cash = 1 if stock_today < 0 else 0
            stock_today = max(stock_today, 0)

            # --- Daily cost ---
            day_cost = refill * refill_cost + out_cash * penalty

            # Save results
            results.append({
                "ATM": atm,
                "Date": date.strftime("%Y-%m-%d"),
                "Predicted_Withdrawal": withdrawal,
                "Refill_Amount": refill,
                "Stock": stock_today,
                "Out_of_Cash": out_cash,
                "Day_Cost": day_cost
            })

            # Update stock
            stocks[atm] = stock_today

    return pd.DataFrame(results)

# ============================================================
# RUN SIMULATION
# ============================================================
print("\n🏧 Running baseline simulation (refill every Monday to full capacity)...")

sim_df = simulate_monday_refill(
    pred_df=df,
    capacity=DEFAULT_CAPACITY,
    init_stock=DEFAULT_INITIAL_STOCK,
    refill_cost=REFILL_COST_PER_UNIT,
    penalty=PENALTY_OUT_OF_CASH
)

# ============================================================
# CALCULATE METRICS
# ============================================================
metrics = {
    "Model": "Baseline_MondayFull",
    "Total_Cost": sim_df["Day_Cost"].sum(),
    "CashOut_Days": sim_df["Out_of_Cash"].sum(),
    "Refill_Count": (sim_df["Refill_Amount"] > 0).sum(),
    "Total_Refilled": sim_df["Refill_Amount"].sum()
}

metrics_df = pd.DataFrame([metrics])

# ============================================================
# SAVE RESULTS
# ============================================================
daily_csv = os.path.join(RESULTS_DIR, "baseline_daily_results.csv")
readable_txt = os.path.join(RESULTS_DIR, "baseline_daily_results_readable.txt")
metrics_csv = os.path.join(RESULTS_DIR, "baseline_metrics.csv")

# Save CSV
sim_df.to_csv(daily_csv, index=False)
metrics_df.to_csv(metrics_csv, index=False)

# Save readable text report
with open(readable_txt, "w", encoding="utf-8") as f:
    f.write(sim_df.to_string(index=False, float_format="{:,.0f}".format))

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n✅ Baseline simulation complete!")
print("📁 Daily results (CSV):", daily_csv)
print("📁 Readable report (TXT):", readable_txt)
print("📁 Summary metrics (CSV):", metrics_csv)
print("\n📊 Summary Metrics:")
# Properly format numbers with thousand separators
for col in ["Total_Cost", "CashOut_Days", "Refill_Count", "Total_Refilled"]:
    metrics_df[col] = metrics_df[col].map("{:,.0f}".format)
print(metrics_df.to_string(index=False))
