#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ATM Cash Refill Optimization
============================

This script connects your forecasting models (Linear Regression, XGBoost, and LSTM)
with the weekly optimization model. For each model, it minimizes refill costs and
out-of-cash penalties under budget and operational constraints.

Results are saved separately per model in:
    Optimization model/results_<ModelName>/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from pulp import *

# ============================================================
# CONFIGURATION
# ============================================================
refill_cost_per_unit = 0.01         # Cost per unit refilled
penalty_out_of_cash = 10000         # Large penalty for cash-out days
daily_budget = 25_000_000           # Maximum total refill budget per day
default_atm_capacity = 20_000_000   # Maximum ATM capacity
default_initial_stock = 500_000     # Starting cash if no prior data
max_refills_per_week = 3            # Operational limit per ATM per week

# Forecast model result files
MODEL_PATHS = {
    "LinearRegression": "Prediction models/Linear Regression/predictions/LR_weekly_CV_predictions.csv",
    "XGBoost": "Prediction models/XGBoost/predictions/XGBoost_weekly_CV_predictions.csv",
    "LSTM": "Prediction models/LSTM/predictions/LSTM_weekly_CV_predictions.csv"
}

# Output directory
BASE_OUT_DIR = "Optimization model"
os.makedirs(BASE_OUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTION — Optimization per model
# ============================================================
def run_optimization(model_name: str, pred_file: str):
    print(f"\n🚀 Starting optimization using {model_name} predictions")

    # --- Load forecast data ---
    if not os.path.exists(pred_file):
        print(f"❌ File not found: {pred_file}")
        return

    df = pd.read_csv(pred_file)
    df["Week_Start"] = pd.to_datetime(df["Week_Start"], errors="coerce")
    df = df.dropna(subset=["Week_Start"]).sort_values(["ATM_ID", "Week_Start", "Horizon"]).reset_index(drop=True)

    # ✅ Convert predicted withdrawals from log-scale to real-scale if necessary
    if df["Predicted"].mean() < 100:  # heuristic check for log-space
        print("🔁 Detected log-scale predictions — converting with exp(x) - 1 ...")
        df["Predicted"] = np.exp(df["Predicted"]) - 1
    else:
        print("✅ Predictions already in real cash scale.")

    # Convert horizons (1–7) to actual dates
    df["Date"] = df["Week_Start"] + pd.to_timedelta(df["Horizon"] - 1, unit="D")

    # --- Create output folders ---
    model_dir = os.path.join(BASE_OUT_DIR, f"results_{model_name}")
    os.makedirs(model_dir, exist_ok=True)

    stock_file = os.path.join(model_dir, "last_stock_levels.csv")
    if os.path.exists(stock_file):
        s0_df = pd.read_csv(stock_file)
        initial_stocks = s0_df.set_index("ATM")["Stock"].to_dict()
    else:
        initial_stocks = {atm: default_initial_stock for atm in df["ATM_ID"].unique()}

    all_results = []
    unique_weeks = sorted(df["Week_Start"].unique())

    # ============================================================
    # WEEKLY OPTIMIZATION LOOP
    # ============================================================
    for week_start in unique_weeks:
        week_end = week_start + timedelta(days=6)
        print(f"\n📅 Week: {week_start.date()} → {week_end.date()}")

        week_atms = df[df["Week_Start"] == week_start]["ATM_ID"].unique()
        M = len(week_atms)
        T = 7

        # Predicted withdrawals per day (7xM matrix)
        w = [[0] * T for _ in range(M)]
        for m, atm in enumerate(week_atms):
            atm_data = df[(df["ATM_ID"] == atm) & (df["Week_Start"] == week_start)]
            for _, row in atm_data.iterrows():
                t = int(row["Horizon"]) - 1
                w[m][t] = row["Predicted"]

        # Capacities & initial stocks
        C = [default_atm_capacity] * M
        s0 = [initial_stocks.get(atm, default_initial_stock) for atm in week_atms]
        B = [daily_budget] * T

        # ========================================================
        # DEFINE OPTIMIZATION MODEL
        # ========================================================
        model = LpProblem(f"ATM_Optimization_{model_name}_{week_start.date()}", LpMinimize)

        N = max(C) * 2
        epsilon = 1e-3

        x = [[LpVariable(f"x_{m}_{t}", lowBound=0, cat="Continuous") for t in range(T)] for m in range(M)]
        y = [[LpVariable(f"y_{m}_{t}", cat="Binary") for t in range(T)] for m in range(M)]
        s = [[LpVariable(f"s_{m}_{t}", lowBound=0, upBound=C[m], cat="Continuous") for t in range(T)] for m in range(M)]
        z = [[LpVariable(f"z_{m}_{t}", cat="Binary") for t in range(T)] for m in range(M)]

        # Objective: minimize refill + penalty costs
        model += lpSum(refill_cost_per_unit * x[m][t] + penalty_out_of_cash * z[m][t]
                       for m in range(M) for t in range(T))

        # Constraints
        for m in range(M):
            model += lpSum(y[m][t] for t in range(T)) <= max_refills_per_week
            for t in range(T):
                withdrawal = w[m][t]
                # Stock balance
                if t == 0:
                    model += s[m][t] == s0[m] + x[m][t] - withdrawal
                else:
                    model += s[m][t] == s[m][t-1] + x[m][t] - withdrawal

                # Refill link constraints
                model += x[m][t] <= N * y[m][t]
                model += x[m][t] >= epsilon * y[m][t]

                # Capacity
                model += s[m][t] <= C[m]

                # Out-of-cash indicator
                model += s[m][t] >= -N * z[m][t]
                model += s[m][t] <= N * (1 - z[m][t])

        # Daily budget
        for t in range(T):
            model += lpSum(x[m][t] for m in range(M)) <= B[t]

        # ========================================================
        # SOLVE
        # ========================================================
        status = model.solve(PULP_CBC_CMD(msg=0))
        print(f"🧮 Solver status: {LpStatus[status]}")

        # ========================================================
        # COLLECT RESULTS
        # ========================================================
        week_cost = 0
        for m in range(M):
            atm = week_atms[m]
            atm_cost = 0

            for t in range(T):
                date = week_start + timedelta(days=t)
                refill = value(x[m][t])
                stock = value(s[m][t])
                refilled = int(value(y[m][t]))
                out_cash = int(value(z[m][t]))
                withdrawal = w[m][t]
                day_cost = refill_cost_per_unit * refill + penalty_out_of_cash * out_cash
                atm_cost += day_cost

                all_results.append({
                    "Model": model_name,
                    "ATM": atm,
                    "Date": date.strftime("%Y-%m-%d"),
                    "Predicted_Withdrawal": withdrawal,
                    "Refill_Amount": refill,
                    "Stock": stock,
                    "Refilled": refilled,
                    "Out_of_Cash": out_cash,
                    "Day_Cost": day_cost
                })

            week_cost += atm_cost
            initial_stocks[atm] = value(s[m][T - 1])

        print(f"💰 Total weekly cost ({model_name}): {week_cost:,.2f}")

    # ============================================================
    # SAVE RESULTS + METRICS
    # ============================================================
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(model_dir, f"atm_refill_results_{model_name}.csv")
    results_df.to_csv(csv_path, index=False)

    # Summary metrics
    metrics = {
        "Model": model_name,
        "Total_Cost": results_df["Day_Cost"].sum(),
        "CashOut_Days": results_df["Out_of_Cash"].sum(),
        "Refill_Count": (results_df["Refill_Amount"] > 0).sum(),
        "Total_Refilled": results_df["Refill_Amount"].sum(),
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(model_dir, f"metrics_{model_name}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\n📊 Summary Metrics:")
    print(metrics_df.to_string(index=False, float_format="%.2f"))

    # Save readable report
    with open(os.path.join(model_dir, f"atm_refill_results_{model_name}_readable.txt"), "w", encoding="utf-8") as f:
        f.write(results_df.to_string(index=False, float_format="{:,.0f}".format))

    # Save final stock levels
    last_stock = pd.DataFrame.from_dict(initial_stocks, orient="index", columns=["Stock"]).reset_index()
    last_stock.columns = ["ATM", "Stock"]
    last_stock.to_csv(stock_file, index=False)

    print(f"✅ Results saved for {model_name} → {csv_path}")

    # ============================================================
    # PLOTS PER ATM (IMPROVED)
    # ============================================================
    print(f"📊 Creating improved plots for {model_name}...")
    for atm in results_df["ATM"].unique():
        atm_df = results_df[results_df["ATM"] == atm].copy()
        if atm_df.empty:
            continue

        dates = pd.to_datetime(atm_df["Date"])
        refill = atm_df["Refill_Amount"]
        stock = atm_df["Stock"]

        fig, ax1 = plt.subplots(figsize=(16, 5))  # ⬅️ Wider figure for readability
        ax1.bar(dates, refill, color="tab:blue", alpha=0.6, width=2)
        ax1.set_ylabel("Refill Amount", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Format X-axis dates
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(dates, stock, color="tab:orange", marker="o", markersize=3, linewidth=1)
        ax2.set_ylabel("Stock Level", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title(f"{model_name} - ATM {atm} Refill & Stock")
        fig.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(model_dir, f"ATM_{atm}_{model_name}_refill_plot.png"), bbox_inches="tight", dpi=150)
        plt.close()

    print(f"✅ Improved plots saved for {model_name}")


# ============================================================
# RUN OPTIMIZATION FOR EACH MODEL
# ============================================================
for model_name, path in MODEL_PATHS.items():
    run_optimization(model_name, path)

print("\n🎯 Optimization completed for all forecasting models!")
