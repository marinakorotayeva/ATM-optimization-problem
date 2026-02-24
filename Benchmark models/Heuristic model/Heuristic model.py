import os
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

PRED_FILE = "Prediction models/XGBoost/predictions/XGBoost_weekly_CV_predictions.csv"

# Explicit scale (avoid heuristics)
DATA_SCALE = "log1p"  # "log1p" or "original"

DEFAULT_INITIAL_STOCK = 500_000
DEFAULT_CAPACITY = 20_000_000

REFILL_COST_PER_UNIT = 0.01
PENALTY_OUT_OF_CASH = 10_000

# Service days when refills are possible (0=Mon ... 6=Sun)
SERVICE_DAYS = {0, 3}  # Monday & Thursday

# --- Heuristic parameters (tuneable) ---
# Reorder point: cover demand until next service day with a multiplier (risk buffer)
REORDER_MULT = 1.35     # >1 increases safety against forecast error (try 1.2–1.6)
ABS_BUFFER = 250_000    # absolute extra cash buffer, protects low-demand weeks too

# Target level: refill to a higher level than s (this prevents frequent cash-outs)
TARGET_MULT = 2.50      # S roughly = TARGET_MULT * s (try 1.8–3.5)
EXTRA_TARGET = 0        # additional absolute target cash on top of TARGET_MULT*s

# Optional: do NOT refill tiny amounts (reduces operational noise)
MIN_REFILL_AMOUNT = 0   # e.g. 50_000 to ignore very small refills

# Output folder
RESULTS_DIR = "Benchmark models/Heuristic model/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(PRED_FILE)

required_cols = {"ATM_ID", "Week_Start", "Horizon", "Actual", "Predicted"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns {missing} in {PRED_FILE}. Found: {list(df.columns)}")

df["Week_Start"] = pd.to_datetime(df["Week_Start"], errors="coerce")
df = df.dropna(subset=["Week_Start"]).copy()

df["Horizon"] = pd.to_numeric(df["Horizon"], errors="coerce")
df = df.dropna(subset=["Horizon"]).copy()
df["Horizon"] = df["Horizon"].astype(int)

# Convert horizons (1–7) to actual dates
df["Date"] = df["Week_Start"] + pd.to_timedelta(df["Horizon"] - 1, unit="D")
df = df.sort_values(["ATM_ID", "Date"]).reset_index(drop=True)

# Convert scale
if DATA_SCALE.lower() == "log1p":
    df["Actual_real"] = np.expm1(df["Actual"].astype(float))
    df["Forecast_real"] = np.expm1(df["Predicted"].astype(float))
else:
    df["Actual_real"] = df["Actual"].astype(float)
    df["Forecast_real"] = df["Predicted"].astype(float)

# Safety: no negative values
df["Actual_real"] = df["Actual_real"].clip(lower=0)
df["Forecast_real"] = df["Forecast_real"].clip(lower=0)

# ============================================================
# HELPERS
# ============================================================
def next_service_day_date(current_date: pd.Timestamp) -> pd.Timestamp:
    """Next date >= current_date that is a service day."""
    d = current_date
    for _ in range(8):  # within a week must exist
        if d.weekday() in SERVICE_DAYS:
            return d
        d += pd.Timedelta(days=1)
    return current_date


def forecast_sum_until_next_service(atm_df: pd.DataFrame, current_date: pd.Timestamp) -> float:
    """Sum of forecasted demand from current_date to next service day inclusive."""
    nsd = next_service_day_date(current_date)
    window = atm_df[(atm_df["Date"] >= current_date) & (atm_df["Date"] <= nsd)]
    return float(window["Forecast_real"].sum())


# ============================================================
# SIMULATION
# ============================================================
def simulate_heuristic_sS_xgb(pred_df: pd.DataFrame) -> pd.DataFrame:
    atms = pred_df["ATM_ID"].unique()
    stocks = {atm: float(DEFAULT_INITIAL_STOCK) for atm in atms}

    # Pre-split by ATM for fast slicing
    atm_groups = {atm: pred_df[pred_df["ATM_ID"] == atm].copy() for atm in atms}

    results = []
    all_dates = sorted(pred_df["Date"].unique())

    for date in all_dates:
        day_rows = pred_df[pred_df["Date"] == date]

        for _, row in day_rows.iterrows():
            atm = row["ATM_ID"]
            stock_prev = stocks[atm]
            withdrawal_real = float(row["Actual_real"])

            atm_df = atm_groups[atm]
            expected = forecast_sum_until_next_service(atm_df, date)

            # Reorder point and target
            s = expected * REORDER_MULT + ABS_BUFFER
            S = min(DEFAULT_CAPACITY, s * TARGET_MULT + EXTRA_TARGET)

            refill = 0.0
            if date.weekday() in SERVICE_DAYS and stock_prev < s:
                refill = max(0.0, S - stock_prev)
                if refill < MIN_REFILL_AMOUNT:
                    refill = 0.0

            stock_today = stock_prev + refill - withdrawal_real

            out_cash = int(stock_today < 0)
            stock_today = max(stock_today, 0.0)

            day_cost = refill * REFILL_COST_PER_UNIT + out_cash * PENALTY_OUT_OF_CASH

            results.append({
                "ATM": atm,
                "Date": date.strftime("%Y-%m-%d"),
                "Actual_Withdrawal": withdrawal_real,
                "Forecast_XGB": float(row["Forecast_real"]),
                "Expected_Until_NextService": expected,
                "s_ReorderPoint": s,
                "S_Target": S,
                "Refill_Amount": refill,
                "Stock": stock_today,
                "Out_of_Cash": out_cash,
                "Day_Cost": day_cost
            })

            stocks[atm] = stock_today

    return pd.DataFrame(results)


# ============================================================
# RUN
# ============================================================
print("Running improved heuristic (s,S) policy using XGBoost forecasts...")
sim_df = simulate_heuristic_sS_xgb(df)

metrics = {
    "Model": f"Heuristic_sS_XGBoost_REORDER{REORDER_MULT}_TARGET{TARGET_MULT}_Svc{sorted(list(SERVICE_DAYS))}",
    "Total_Cost": float(sim_df["Day_Cost"].sum()),
    "CashOut_Days": int(sim_df["Out_of_Cash"].sum()),
    "Refill_Count": int((sim_df["Refill_Amount"] > 0).sum()),
    "Total_Refilled": float(sim_df["Refill_Amount"].sum()),
}

metrics_df = pd.DataFrame([metrics])

# ============================================================
# SAVE
# ============================================================
daily_csv = os.path.join(RESULTS_DIR, "heuristic_xgb_daily_results.csv")
metrics_csv = os.path.join(RESULTS_DIR, "heuristic_xgb_metrics.csv")
readable_txt = os.path.join(RESULTS_DIR, "heuristic_xgb_daily_results_readable.txt")

sim_df.to_csv(daily_csv, index=False)
metrics_df.to_csv(metrics_csv, index=False)

with open(readable_txt, "w", encoding="utf-8") as f:
    f.write(sim_df.to_string(index=False, float_format="{:,.0f}".format))

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\nHeuristic simulation complete!")
print("Daily results (CSV):", daily_csv)
print("Metrics (CSV):", metrics_csv)
print("Readable report (TXT):", readable_txt)

print("\nSummary Metrics:")
pretty = metrics_df.copy()
for col in ["Total_Cost", "CashOut_Days", "Refill_Count", "Total_Refilled"]:
    pretty[col] = pretty[col].map(lambda x: f"{x:,.0f}")
print(pretty.to_string(index=False))

print("\nTuning tip (if cash-outs still > 0):")
print("- Increase REORDER_MULT (e.g., 1.35 -> 1.5) and/or ABS_BUFFER (e.g., 250k -> 500k)")
print("- Increase TARGET_MULT (e.g., 2.5 -> 3.0)")
print("- Add one more service day (e.g., SERVICE_DAYS = {0,2,4})")
