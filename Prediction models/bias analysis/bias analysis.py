from __future__ import annotations

import os
import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ====== CONFIG ======
# If your Actual/Predicted are in log1p scale (as in your thesis), keep "log1p".
# If they are already in original scale, set "original".
TARGET_SCALE = "log1p"  # "log1p" or "original"

EPS = 1e-9

BASE_DIR = os.path.join("Prediction models")
OUT_DIR = os.path.join(BASE_DIR, "bias analysis")
OUT_FILE = os.path.join(OUT_DIR, "bias_report_models.csv")

MODEL_INPUTS = {
    "LR": os.path.join(BASE_DIR, "Linear Regression", "predictions", "LR_weekly_CV_predictions*"),
    "LSTM": os.path.join(BASE_DIR, "LSTM", "predictions", "LSTM_weekly_CV_predictions*"),
    "XGBoost": os.path.join(BASE_DIR, "XGBoost", "predictions", "XGBoost_weekly_CV_predictions*"),
}


# ====== HELPERS ======
def ensure_outdir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def find_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    return [f for f in files if os.path.isfile(f)]


def _find_col(lower_map: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures these columns exist:
      ATM_ID, Week_Start, Actual, Predicted
    """
    df = df.rename(columns={c: c.strip() for c in df.columns})
    lower = {c.lower(): c for c in df.columns}

    atm_col = _find_col(lower, ["atm_id", "atm", "cashp_id", "cashpoint_id", "cashpoint"])
    week_col = _find_col(lower, ["week_start", "weekstart", "week", "date", "week_start_date"])

    actual_col = _find_col(lower, ["actual", "y", "y_true", "true", "target", "actual_mean"])
    pred_col = _find_col(lower, ["predicted", "y_pred", "yhat", "prediction", "forecast", "predicted_mean"])

    if atm_col is None:
        raise ValueError(f"Missing ATM column. Found: {list(df.columns)}")
    if week_col is None:
        raise ValueError(f"Missing Week_Start/Date column. Found: {list(df.columns)}")
    if actual_col is None:
        raise ValueError(f"Missing Actual (or Actual_Mean) column. Found: {list(df.columns)}")
    if pred_col is None:
        raise ValueError(f"Missing Predicted (or Predicted_Mean) column. Found: {list(df.columns)}")

    df = df.rename(columns={
        atm_col: "ATM_ID",
        week_col: "Week_Start",
        actual_col: "Actual",
        pred_col: "Predicted",
    })

    df["ATM_ID"] = df["ATM_ID"].astype(str)
    df["Week_Start"] = pd.to_datetime(df["Week_Start"], errors="coerce")
    df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")
    df["Predicted"] = pd.to_numeric(df["Predicted"], errors="coerce")

    df = df.dropna(subset=["ATM_ID", "Week_Start", "Actual", "Predicted"]).copy()
    return df


def load_model(model_name: str, pattern: str) -> pd.DataFrame:
    files = find_files(pattern)
    if not files:
        raise FileNotFoundError(f"[{model_name}] No files found for pattern: {pattern}")

    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        d = standardize_columns(d)
        d["Model"] = model_name
        dfs.append(d)

    if not dfs:
        raise RuntimeError(f"[{model_name}] Found files but none readable as CSV: {files}")

    return pd.concat(dfs, ignore_index=True)


def compute_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns ONE row per model with the key bias metrics (on ORIGINAL scale if TARGET_SCALE=log1p).
    """
    df = df.copy()

    if TARGET_SCALE.lower() == "log1p":
        # Convert to original demand units
        df["Actual_use"] = np.expm1(df["Actual"].values)
        df["Pred_use"] = np.expm1(df["Predicted"].values)
    else:
        df["Actual_use"] = df["Actual"].values
        df["Pred_use"] = df["Predicted"].values

    df["Error"] = df["Pred_use"] - df["Actual_use"]
    df["Under"] = (df["Error"] < 0).astype(int)
    df["Over"] = (df["Error"] > 0).astype(int)

    denom = np.where(np.abs(df["Actual_use"].values) < EPS, EPS, df["Actual_use"].values)
    df["RelError"] = df["Error"] / denom

    g = df.groupby("Model", dropna=False)

    report = g.agg(
        N=("Error", "size"),
        Mean_Actual=("Actual_use", "mean"),
        Mean_Predicted=("Pred_use", "mean"),
        Bias_ME=("Error", "mean"),          # MAIN: mean signed error
        Bias_MedE=("Error", "median"),
        Under_Share=("Under", "mean"),
        Over_Share=("Over", "mean"),
        Bias_MRE=("RelError", "mean"),      # optional: mean relative signed error
    ).reset_index()

    report["Bias_Direction"] = np.where(
        report["Bias_ME"] < 0, "UNDER (negative bias)",
        np.where(report["Bias_ME"] > 0, "OVER (positive bias)", "NEUTRAL (zero mean)")
    )

    # Nice additional “magnitude” view (how large bias is relative to mean actual)
    report["Bias_ME_as_%_of_MeanActual"] = 100.0 * report["Bias_ME"] / np.where(
        np.abs(report["Mean_Actual"].values) < EPS, EPS, report["Mean_Actual"].values
    )

    # Sort models in a stable order
    order = {"LR": 0, "LSTM": 1, "XGBoost": 2}
    report["__order"] = report["Model"].map(order).fillna(999).astype(int)
    report = report.sort_values("__order").drop(columns="__order")

    return report


# ====== MAIN ======
def main() -> None:
    ensure_outdir()

    frames = []
    for model, pattern in MODEL_INPUTS.items():
        frames.append(load_model(model, pattern))

    df_all = pd.concat(frames, ignore_index=True)

    report = compute_report(df_all)
    report.to_csv(OUT_FILE, index=False)

    print(f"\nSaved ONE report file:\n  {OUT_FILE}\n")
    print("How to read it:")
    print("- Bias_ME < 0  => UNDERprediction (systematically too low)")
    print("- Bias_ME > 0  => OVERprediction  (systematically too high)")
    print("\nPreview:")
    print(report[["Model", "N", "Bias_ME", "Under_Share", "Over_Share", "Bias_Direction"]].to_string(index=False))


if __name__ == "__main__":
    main()
