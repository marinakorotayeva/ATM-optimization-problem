#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stability analysis for ATM refill optimization
==============================================

What it does:
- For each forecasting model (LR, XGBoost, LSTM)
- For each forecast multiplier (0.900..1.100 in steps of 0.001)
  -> scales predicted withdrawals
  -> runs the weekly optimization
  -> saves results + metrics
- Builds a stability summary table (deltas vs multiplier=1.000 baseline)
- Produces ONE combined stability figure (3 panels):
    1) ΔTotal cost (%)
    2) ΔRefill count
    3) Refill-days Jaccard vs baseline (schedule stability)

Change requested:
- Keep lines, REMOVE marker dots from the plot.
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    PULP_CBC_CMD,
    LpStatus,
    value,
)

# ============================================================
# CONFIG (match the Optimization model settings)
# ============================================================
REFILL_COST_PER_UNIT = 0.01
PENALTY_OUT_OF_CASH = 10000
DAILY_BUDGET = 25_000_000
DEFAULT_ATM_CAPACITY = 20_000_000
DEFAULT_INITIAL_STOCK = 500_000
MAX_REFILLS_PER_WEEK = 3

# Where the forecast CSVs are
MODEL_PATHS = {
    "LinearRegression": "Prediction models/Linear Regression/predictions/LR_weekly_CV_predictions.csv",
    "XGBoost": "Prediction models/XGBoost/predictions/XGBoost_weekly_CV_predictions.csv",
    "LSTM": "Prediction models/LSTM/predictions/LSTM_weekly_CV_predictions.csv",
}

# Multipliers to test — now 0.001 steps: 0.900, 0.901, ..., 1.100
# Integer-based construction avoids floating point artifacts.
MULTIPLIERS = [i / 1000 for i in range(900, 1100 + 1)]

# Output root for stability runs
OUT_ROOT = os.path.join("Optimization model", "Stability")
os.makedirs(OUT_ROOT, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def _ensure_real_scale_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    The predictions may be log-space. Use the same heuristic:
    if mean is small (<100), treat as log and convert with exp(x)-1.
    """
    df = df.copy()
    if df["Predicted"].mean() < 100:
        df["Predicted_Real"] = np.exp(df["Predicted"]) - 1
    else:
        df["Predicted_Real"] = df["Predicted"]
    return df


def load_predictions(pred_file: str) -> pd.DataFrame:
    df = pd.read_csv(pred_file)
    df["Week_Start"] = pd.to_datetime(df["Week_Start"], errors="coerce")
    df = (
        df.dropna(subset=["Week_Start"])
        .sort_values(["ATM_ID", "Week_Start", "Horizon"])
        .reset_index(drop=True)
    )
    df = _ensure_real_scale_predictions(df)
    return df


def run_weekly_optimization(
    week_df: pd.DataFrame,
    initial_stocks: dict,
    model_tag: str,
    week_start: pd.Timestamp,
) -> tuple[list[dict], dict]:
    """
    Solve a single week for all ATMs in week_df and return:
      - list of daily result rows
      - updated initial_stocks (ending stock becomes next week's initial)
    """
    week_atms = week_df["ATM_ID"].unique()
    M = len(week_atms)
    T = 7

    # Build predicted withdrawals w[m][t]
    w = [[0.0] * T for _ in range(M)]
    for m, atm in enumerate(week_atms):
        atm_data = week_df[week_df["ATM_ID"] == atm]
        for _, row in atm_data.iterrows():
            t = int(row["Horizon"]) - 1
            if 0 <= t < T:
                w[m][t] = float(row["Predicted_Real"])

    # Capacities, starting stocks, daily budgets
    C = [DEFAULT_ATM_CAPACITY] * M
    s0 = [float(initial_stocks.get(atm, DEFAULT_INITIAL_STOCK)) for atm in week_atms]
    B = [DAILY_BUDGET] * T

    # Optimization model
    model = LpProblem(f"ATM_Optimization_{model_tag}_{week_start.date()}", LpMinimize)

    N = max(C) * 2
    epsilon = 1e-3

    x = [[LpVariable(f"x_{m}_{t}", lowBound=0, cat="Continuous") for t in range(T)] for m in range(M)]
    y = [[LpVariable(f"y_{m}_{t}", cat="Binary") for t in range(T)] for m in range(M)]
    s = [[LpVariable(f"s_{m}_{t}", lowBound=0, upBound=C[m], cat="Continuous") for t in range(T)] for m in range(M)]
    z = [[LpVariable(f"z_{m}_{t}", cat="Binary") for t in range(T)] for m in range(M)]

    # Objective
    model += lpSum(
        REFILL_COST_PER_UNIT * x[m][t] + PENALTY_OUT_OF_CASH * z[m][t]
        for m in range(M)
        for t in range(T)
    )

    # Constraints
    for m in range(M):
        model += lpSum(y[m][t] for t in range(T)) <= MAX_REFILLS_PER_WEEK

        for t in range(T):
            withdrawal = w[m][t]
            if t == 0:
                model += s[m][t] == s0[m] + x[m][t] - withdrawal
            else:
                model += s[m][t] == s[m][t - 1] + x[m][t] - withdrawal

            model += x[m][t] <= N * y[m][t]
            model += x[m][t] >= epsilon * y[m][t]
            model += s[m][t] <= C[m]

            # out-of-cash indicator
            model += s[m][t] >= -N * z[m][t]
            model += s[m][t] <= N * (1 - z[m][t])

    for t in range(T):
        model += lpSum(x[m][t] for m in range(M)) <= B[t]

    # Solve
    status = model.solve(PULP_CBC_CMD(msg=0))
    _ = LpStatus[status]

    # Collect results
    rows = []
    for m, atm in enumerate(week_atms):
        for t in range(T):
            date = (week_start + timedelta(days=t)).strftime("%Y-%m-%d")
            refill_amt = float(value(x[m][t]))
            stock = float(value(s[m][t]))
            refilled = int(round(float(value(y[m][t]))))
            out_cash = int(round(float(value(z[m][t]))))
            withdrawal = float(w[m][t])
            day_cost = REFILL_COST_PER_UNIT * refill_amt + PENALTY_OUT_OF_CASH * out_cash

            rows.append(
                {
                    "Model": model_tag,
                    "ATM": atm,
                    "Date": date,
                    "Predicted_Withdrawal": withdrawal,
                    "Refill_Amount": refill_amt,
                    "Stock": stock,
                    "Refilled": refilled,
                    "Out_of_Cash": out_cash,
                    "Day_Cost": day_cost,
                }
            )

        # Update ending stock for next week
        initial_stocks[atm] = float(value(s[m][T - 1]))

    return rows, initial_stocks


def run_optimization_for_multiplier(
    model_name: str,
    pred_df: pd.DataFrame,
    multiplier: float,
    out_dir: str,
) -> tuple[str, dict]:
    """
    Run the whole rolling weekly optimization for a given model + multiplier.
    Returns:
      - path to results CSV
      - metrics dict
    """
    os.makedirs(out_dir, exist_ok=True)

    # Scale predictions (in REAL scale)
    df = pred_df.copy()
    df["Predicted_Real"] = df["Predicted_Real"] * float(multiplier)

    # Week-by-week loop
    all_rows = []
    initial_stocks = {atm: DEFAULT_INITIAL_STOCK for atm in df["ATM_ID"].unique()}

    unique_weeks = sorted(df["Week_Start"].unique())
    for week_start in unique_weeks:
        week_df = df[df["Week_Start"] == week_start]
        model_tag = f"{model_name}_x{multiplier:.3f}"
        week_rows, initial_stocks = run_weekly_optimization(
            week_df=week_df,
            initial_stocks=initial_stocks,
            model_tag=model_tag,
            week_start=pd.Timestamp(week_start),
        )
        all_rows.extend(week_rows)

    results_df = pd.DataFrame(all_rows)

    results_path = os.path.join(out_dir, f"atm_refill_results_{model_name}_x{multiplier:.3f}.csv")
    results_df.to_csv(results_path, index=False)

    metrics = {
        "Model": model_name,
        "Forecast_Multiplier": float(multiplier),
        "Total_Cost": float(results_df["Day_Cost"].sum()),
        "CashOut_Days": int(results_df["Out_of_Cash"].sum()),
        "Refill_Count": int((results_df["Refill_Amount"] > 0).sum()),
        "Total_Refilled": float(results_df["Refill_Amount"].sum()),
        "Results_File": results_path,
    }

    metrics_path = os.path.join(out_dir, f"metrics_{model_name}_x{multiplier:.3f}.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    return results_path, metrics


def refill_days_set(results_csv: str) -> set[tuple[str, str]]:
    """
    Set of refill 'events' as (ATM, Date) where Refill_Amount > 0.
    Used for Jaccard similarity vs baseline.
    """
    df = pd.read_csv(results_csv)
    df = df[df["Refill_Amount"] > 0]
    return set(zip(df["ATM"].astype(str), df["Date"].astype(str)))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    uni = len(a.union(b))
    return inter / uni if uni > 0 else 0.0


# ============================================================
# MAIN: run stability + plot
# ============================================================
def main():
    all_metrics = []

    # 1) Run optimization for all models & multipliers
    for model_name, pred_path in MODEL_PATHS.items():
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")

        pred_df = load_predictions(pred_path)
        model_out_dir = os.path.join(OUT_ROOT, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        for mult in MULTIPLIERS:
            out_dir = model_out_dir

            # Cache: if results already exist, reuse
            results_csv = os.path.join(out_dir, f"atm_refill_results_{model_name}_x{mult:.3f}.csv")
            metrics_csv = os.path.join(out_dir, f"metrics_{model_name}_x{mult:.3f}.csv")

            if os.path.exists(results_csv) and os.path.exists(metrics_csv):
                metrics = pd.read_csv(metrics_csv).iloc[0].to_dict()
                metrics["Results_File"] = results_csv  # ensure correct
            else:
                _, metrics = run_optimization_for_multiplier(
                    model_name=model_name,
                    pred_df=pred_df,
                    multiplier=mult,
                    out_dir=out_dir,
                )

            all_metrics.append(metrics)

    summary = pd.DataFrame(all_metrics)

    # 2) Compute deltas + Jaccard vs baseline (multiplier = 1.000)
    rows = []
    for model_name in summary["Model"].unique():
        sub = summary[summary["Model"] == model_name].copy()
        base = sub[np.isclose(sub["Forecast_Multiplier"], 1.0, atol=1e-12)]
        if base.empty:
            raise ValueError(f"No baseline (multiplier=1.000) found for model {model_name}")

        base_cost = float(base["Total_Cost"].iloc[0])
        base_refills = int(base["Refill_Count"].iloc[0])
        base_refilled_amt = float(base["Total_Refilled"].iloc[0])
        base_set = refill_days_set(str(base["Results_File"].iloc[0]))

        for _, r in sub.iterrows():
            m = float(r["Forecast_Multiplier"])
            cost = float(r["Total_Cost"])
            refill_cnt = int(r["Refill_Count"])
            total_refilled = float(r["Total_Refilled"])
            res_file = str(r["Results_File"])

            this_set = refill_days_set(res_file)
            jac = jaccard(base_set, this_set)

            rows.append(
                {
                    "Model": model_name,
                    "Forecast_Multiplier": m,
                    "Total_Cost": cost,
                    "CashOut_Days": int(r["CashOut_Days"]),
                    "Refill_Count": refill_cnt,
                    "Total_Refilled": total_refilled,
                    "Results_File": res_file,
                    "Delta_Cost_%": (cost / base_cost - 1.0) * 100.0 if base_cost != 0 else np.nan,
                    "Delta_RefillCount": refill_cnt - base_refills,
                    "Delta_TotalRefilled": total_refilled - base_refilled_amt,
                    "RefillDays_Jaccard_vs_Base": jac,
                }
            )

    stability = (
        pd.DataFrame(rows)
        .sort_values(["Model", "Forecast_Multiplier"])
        .reset_index(drop=True)
    )

    stability_csv = os.path.join(OUT_ROOT, "stability_summary_all_models.csv")
    stability.to_csv(stability_csv, index=False)
    print(f"Saved stability summary -> {stability_csv}")

    # 3) Plot: one combined figure (LINES ONLY — no markers)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for model_name in stability["Model"].unique():
        sub = stability[stability["Model"] == model_name].sort_values("Forecast_Multiplier")
        x = sub["Forecast_Multiplier"].to_numpy()

        # NO marker="o" here -> removes dots
        axes[0].plot(x, sub["Delta_Cost_%"].to_numpy(), label=model_name)
        axes[1].plot(x, sub["Delta_RefillCount"].to_numpy(), label=model_name)
        axes[2].plot(x, sub["RefillDays_Jaccard_vs_Base"].to_numpy(), label=model_name)

    for ax in axes:
        ax.axvline(1.0, linestyle="--", linewidth=1)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_title("Stability analysis — sensitivity to forecast multiplier (all models)")
    axes[0].set_ylabel("ΔTotal cost (%) vs baseline")
    axes[1].set_ylabel("ΔRefill count vs baseline")
    axes[2].set_ylabel("Refill-days Jaccard vs baseline")
    axes[2].set_xlabel("Forecast multiplier")

    axes[0].legend(loc="best")

    plot_path = os.path.join(OUT_ROOT, "stability_all_models.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved stability plot -> {plot_path}")


if __name__ == "__main__":
    main()
