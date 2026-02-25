import pandas as pd
import numpy as np
import os
from tabulate import tabulate

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "Prediction models/Comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD MODEL RESULTS
# ==========================================
print("Loading model results...")

def load_results(path: str, model_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{model_name} results not found: {path}")
    df = pd.read_csv(path)
    df["Model"] = model_name
    df.rename(columns={
        "CV_RMSE_Hmean": "RMSE",
        "CV_MAE_Hmean": "MAE",
        "CV_SMAPE_Hmean": "SMAPE"
    }, inplace=True)
    # Keep only relevant columns
    return df[["ATM_ID", "RMSE", "MAE", "SMAPE", "Model"]]

lr_path   = "Prediction models/Linear Regression/results/LR_weekly_CV_results_mean.csv"
xgb_path  = "Prediction models/XGBoost/results/XGBoost_weekly_CV_results_mean.csv"
lstm_path = "Prediction models/LSTM/results/LSTM_weekly_CV_results_mean.csv"

lr   = load_results(lr_path, "Linear Regression")
xgb  = load_results(xgb_path, "XGBoost")
lstm = load_results(lstm_path, "LSTM")

print("Loaded Linear Regression, XGBoost, LSTM results")

# ==========================================
# COMPARISON PER ATM
# ==========================================
comparison_rows = []

atm_list = sorted(set(lr["ATM_ID"]).intersection(xgb["ATM_ID"]).intersection(lstm["ATM_ID"]))

for atm in atm_list:
    lr_r   = lr[lr["ATM_ID"] == atm].iloc[0]
    xgb_r  = xgb[xgb["ATM_ID"] == atm].iloc[0]
    lstm_r = lstm[lstm["ATM_ID"] == atm].iloc[0]

    # Improvements vs LR (%): (LR - Model) / LR * 100
    def improv(lr_val, other_val):
        return (lr_val - other_val) / lr_val * 100 if lr_val != 0 else np.nan

    xgb_rmse_improv  = improv(lr_r["RMSE"],  xgb_r["RMSE"])
    lstm_rmse_improv = improv(lr_r["RMSE"],  lstm_r["RMSE"])

    xgb_mae_improv   = improv(lr_r["MAE"],   xgb_r["MAE"])
    lstm_mae_improv  = improv(lr_r["MAE"],   lstm_r["MAE"])

    xgb_smape_improv  = improv(lr_r["SMAPE"], xgb_r["SMAPE"])
    lstm_smape_improv = improv(lr_r["SMAPE"], lstm_r["SMAPE"])

    # Determine best models (lowest metric)
    rmse_dict = {"Linear Regression": lr_r["RMSE"], "XGBoost": xgb_r["RMSE"], "LSTM": lstm_r["RMSE"]}
    mae_dict  = {"Linear Regression": lr_r["MAE"],  "XGBoost": xgb_r["MAE"],  "LSTM": lstm_r["MAE"]}
    smape_dict= {"Linear Regression": lr_r["SMAPE"],"XGBoost": xgb_r["SMAPE"],"LSTM": lstm_r["SMAPE"]}

    best_rmse  = min(rmse_dict, key=rmse_dict.get)
    best_mae   = min(mae_dict,  key=mae_dict.get)
    best_smape = min(smape_dict,key=smape_dict.get)

    comparison_rows.append({
        "ATM_ID": atm,

        "LR_RMSE": lr_r["RMSE"],
        "XGB_RMSE": xgb_r["RMSE"],
        "LSTM_RMSE": lstm_r["RMSE"],
        "XGB_RMSE_Improv_%": xgb_rmse_improv,
        "LSTM_RMSE_Improv_%": lstm_rmse_improv,
        "Best_RMSE_Model": best_rmse,

        "LR_MAE": lr_r["MAE"],
        "XGB_MAE": xgb_r["MAE"],
        "LSTM_MAE": lstm_r["MAE"],
        "XGB_MAE_Improv_%": xgb_mae_improv,
        "LSTM_MAE_Improv_%": lstm_mae_improv,
        "Best_MAE_Model": best_mae,

        "LR_SMAPE": lr_r["SMAPE"],
        "XGB_SMAPE": xgb_r["SMAPE"],
        "LSTM_SMAPE": lstm_r["SMAPE"],
        "XGB_SMAPE_Improv_%": xgb_smape_improv,
        "LSTM_SMAPE_Improv_%": lstm_smape_improv,
        "Best_SMAPE_Model": best_smape
    })

comparison_df = pd.DataFrame(comparison_rows)

# ==========================================
# OVERALL STATISTICS (mean across ATMs)
# ==========================================
def overall_row(metric_name: str):
    lr_mean  = lr[metric_name].mean()
    xgb_mean = xgb[metric_name].mean()
    lstm_mean= lstm[metric_name].mean()

    return {
        "Metric": metric_name,
        "Linear Regression": f"{lr_mean:.3f}",
        "XGBoost": f"{xgb_mean:.3f}",
        "LSTM": f"{lstm_mean:.3f}",
        "XGB_Improvement_vs_LR": f"{((lr_mean - xgb_mean) / lr_mean * 100):+.1f}%" if lr_mean != 0 else "n/a",
        "LSTM_Improvement_vs_LR": f"{((lr_mean - lstm_mean) / lr_mean * 100):+.1f}%" if lr_mean != 0 else "n/a",
    }

overall_df = pd.DataFrame([
    overall_row("RMSE"),
    overall_row("MAE"),
    overall_row("SMAPE"),
])

# ==========================================
# BEST MODEL COUNT
# ==========================================
best_rmse_count  = comparison_df["Best_RMSE_Model"].value_counts()
best_mae_count   = comparison_df["Best_MAE_Model"].value_counts()
best_smape_count = comparison_df["Best_SMAPE_Model"].value_counts()

best_model_summary = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "SMAPE"],
    "Linear Regression": [
        best_rmse_count.get("Linear Regression", 0),
        best_mae_count.get("Linear Regression", 0),
        best_smape_count.get("Linear Regression", 0),
    ],
    "XGBoost": [
        best_rmse_count.get("XGBoost", 0),
        best_mae_count.get("XGBoost", 0),
        best_smape_count.get("XGBoost", 0),
    ],
    "LSTM": [
        best_rmse_count.get("LSTM", 0),
        best_mae_count.get("LSTM", 0),
        best_smape_count.get("LSTM", 0),
    ],
    "Total ATMs": [len(comparison_df)] * 3
})

# ==========================================
# SAVE OUTPUTS
# ==========================================
comparison_csv = os.path.join(OUTPUT_DIR, "model_comparison_per_ATM.csv")
overall_csv    = os.path.join(OUTPUT_DIR, "model_comparison_overall.csv")
best_csv       = os.path.join(OUTPUT_DIR, "best_model_counts.csv")

comparison_df.to_csv(comparison_csv, index=False)
overall_df.to_csv(overall_csv, index=False)
best_model_summary.to_csv(best_csv, index=False)

# ==========================================
# TEXT REPORT
# ==========================================
report_path = os.path.join(OUTPUT_DIR, "comparison_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 100 + "\n")
    f.write("MODEL COMPARISON REPORT (Linear Regression vs XGBoost vs LSTM)\n")
    f.write("=" * 100 + "\n\n")

    f.write("PER-ATM PERFORMANCE:\n")
    f.write(tabulate(comparison_df, headers="keys", tablefmt="grid", showindex=False))
    f.write("\n\n")

    f.write("OVERALL AVERAGE PERFORMANCE (mean across ATMs):\n")
    f.write(tabulate(overall_df, headers="keys", tablefmt="grid", showindex=False))
    f.write("\n\n")

    f.write("BEST MODEL COUNTS (number of ATMs where the model is best):\n")
    f.write(tabulate(best_model_summary, headers="keys", tablefmt="grid", showindex=False))
    f.write("\n")

print("Done.")
print(f"Saved:\n- {comparison_csv}\n- {overall_csv}\n- {best_csv}\n- {report_path}")


