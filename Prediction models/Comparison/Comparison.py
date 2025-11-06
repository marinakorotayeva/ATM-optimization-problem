#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ---- Linear Regression ----
lr_path = "Prediction models/Linear Regression/results/LR_weekly_CV_results_mean.csv"
if not os.path.exists(lr_path):
    raise FileNotFoundError(f"Linear Regression results not found: {lr_path}")
lr = pd.read_csv(lr_path)
lr["Model"] = "Linear Regression"
lr.rename(columns={
    "CV_RMSE_Hmean": "RMSE",
    "CV_MAE_Hmean": "MAE",
    "CV_SMAPE_Hmean": "SMAPE"
}, inplace=True)
print("Loaded Linear Regression results")

# ---- XGBoost ----
xgb_path = "Prediction models/XGBoost/results/XGBoost_weekly_CV_results_mean.csv"
if not os.path.exists(xgb_path):
    raise FileNotFoundError(f"XGBoost results not found: {xgb_path}")
xgb = pd.read_csv(xgb_path)
xgb["Model"] = "XGBoost"
xgb.rename(columns={
    "CV_RMSE_Hmean": "RMSE",
    "CV_MAE_Hmean": "MAE",
    "CV_SMAPE_Hmean": "SMAPE"
}, inplace=True)
print("Loaded XGBoost results")

# ---- LSTM ----
lstm_path = "Prediction models/LSTM/results/LSTM_weekly_CV_results_mean.csv"
if not os.path.exists(lstm_path):
    raise FileNotFoundError(f"LSTM results not found: {lstm_path}")
lstm = pd.read_csv(lstm_path)
lstm["Model"] = "LSTM"
lstm.rename(columns={
    "CV_RMSE_Hmean": "RMSE",
    "CV_MAE_Hmean": "MAE",
    "CV_SMAPE_Hmean": "SMAPE"
}, inplace=True)
print("Loaded LSTM results")

# ==========================================
# ALIGN STRUCTURES
# ==========================================
# Keep only relevant columns
lr = lr[["ATM_ID", "RMSE", "MAE", "SMAPE", "Model"]]
xgb = xgb[["ATM_ID", "RMSE", "MAE", "SMAPE", "Model"]]
lstm = lstm[["ATM_ID", "RMSE", "MAE", "SMAPE", "Model"]]

# Merge into one DataFrame
combined = pd.concat([lr, xgb, lstm], ignore_index=True)

# ==========================================
# COMPARISON PER ATM
# ==========================================
comparison_rows = []

atm_list = sorted(set(lr["ATM_ID"]).intersection(xgb["ATM_ID"]).intersection(lstm["ATM_ID"]))

for atm in atm_list:
    lr_r = lr[lr["ATM_ID"] == atm].iloc[0]
    xgb_r = xgb[xgb["ATM_ID"] == atm].iloc[0]
    lstm_r = lstm[lstm["ATM_ID"] == atm].iloc[0]

    # RMSE & SMAPE improvements over LR
    xgb_rmse_improv = (lr_r["RMSE"] - xgb_r["RMSE"]) / lr_r["RMSE"] * 100
    lstm_rmse_improv = (lr_r["RMSE"] - lstm_r["RMSE"]) / lr_r["RMSE"] * 100

    xgb_smape_improv = (lr_r["SMAPE"] - xgb_r["SMAPE"]) / lr_r["SMAPE"] * 100
    lstm_smape_improv = (lr_r["SMAPE"] - lstm_r["SMAPE"]) / lr_r["SMAPE"] * 100

    # Determine best model
    rmse_dict = {
        "Linear Regression": lr_r["RMSE"],
        "XGBoost": xgb_r["RMSE"],
        "LSTM": lstm_r["RMSE"]
    }
    smape_dict = {
        "Linear Regression": lr_r["SMAPE"],
        "XGBoost": xgb_r["SMAPE"],
        "LSTM": lstm_r["SMAPE"]
    }

    best_rmse = min(rmse_dict, key=rmse_dict.get)
    best_smape = min(smape_dict, key=smape_dict.get)

    comparison_rows.append({
        "ATM_ID": atm,
        "LR_RMSE": lr_r["RMSE"],
        "XGB_RMSE": xgb_r["RMSE"],
        "LSTM_RMSE": lstm_r["RMSE"],
        "XGB_RMSE_Improv_%": xgb_rmse_improv,
        "LSTM_RMSE_Improv_%": lstm_rmse_improv,
        "Best_RMSE_Model": best_rmse,
        "LR_SMAPE": lr_r["SMAPE"],
        "XGB_SMAPE": xgb_r["SMAPE"],
        "LSTM_SMAPE": lstm_r["SMAPE"],
        "XGB_SMAPE_Improv_%": xgb_smape_improv,
        "LSTM_SMAPE_Improv_%": lstm_smape_improv,
        "Best_SMAPE_Model": best_smape
    })

comparison_df = pd.DataFrame(comparison_rows)

# ==========================================
# OVERALL STATISTICS
# ==========================================
overall_stats = {
    "Metric": ["RMSE", "SMAPE"],
    "Linear Regression": [
        f"{lr['RMSE'].mean():.2f}",
        f"{lr['SMAPE'].mean():.2f}"
    ],
    "XGBoost": [
        f"{xgb['RMSE'].mean():.2f}",
        f"{xgb['SMAPE'].mean():.2f}"
    ],
    "LSTM": [
        f"{lstm['RMSE'].mean():.2f}",
        f"{lstm['SMAPE'].mean():.2f}"
    ],
    "XGB_Improvement_vs_LR": [
        f"{(lr['RMSE'].mean() - xgb['RMSE'].mean()) / lr['RMSE'].mean() * 100:+.1f}%",
        f"{(lr['SMAPE'].mean() - xgb['SMAPE'].mean()) / lr['SMAPE'].mean() * 100:+.1f}%"
    ],
    "LSTM_Improvement_vs_LR": [
        f"{(lr['RMSE'].mean() - lstm['RMSE'].mean()) / lr['RMSE'].mean() * 100:+.1f}%",
        f"{(lr['SMAPE'].mean() - lstm['SMAPE'].mean()) / lr['SMAPE'].mean() * 100:+.1f}%"
    ]
}

overall_df = pd.DataFrame(overall_stats)

# ==========================================
# BEST MODEL COUNT
# ==========================================
best_rmse_count = comparison_df["Best_RMSE_Model"].value_counts()
best_smape_count = comparison_df["Best_SMAPE_Model"].value_counts()

best_model_summary = pd.DataFrame({
    "Metric": ["RMSE", "SMAPE"],
    "Linear Regression": [best_rmse_count.get("Linear Regression", 0), best_smape_count.get("Linear Regression", 0)],
    "XGBoost": [best_rmse_count.get("XGBoost", 0), best_smape_count.get("XGBoost", 0)],
    "LSTM": [best_rmse_count.get("LSTM", 0), best_smape_count.get("LSTM", 0)],
    "Total ATMs": [len(comparison_df), len(comparison_df)]
})

# ==========================================
# SAVE OUTPUTS
# ==========================================
comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_per_ATM.csv"), index=False)
overall_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_overall.csv"), index=False)

# ==========================================
# TEXT REPORT
# ==========================================
with open(os.path.join(OUTPUT_DIR, "comparison_report.txt"), "w") as f:
    f.write("=" * 100 + "\n")
    f.write("MODEL COMPARISON REPORT (Linear Regression vs XGBoost vs LSTM)\n")
    f.write("=" * 100 + "\n\n")
    f.write("PER-ATM PERFORMANCE:\n")
    f.write(tabulate(comparison_df, headers="keys", tablefmt="grid", showindex=False))
    f.write("\n\n")
    f.write("OVERALL AVERAGE PERFORMANCE:\n")
    f.write(tabulate(overall_df, headers="keys", tablefmt="grid", showindex=False))
    f.write("\n\n")
    f.write("BEST MODEL COUNTS:\n")
    f.write(tabulate(best_model_summary, headers="keys", tablefmt="grid", showindex=False))
    f.write("\n")

print("Comparison report saved")

# ==========================================
# VISUALIZATIONS
# ==========================================
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 13
})

x_pos = np.arange(len(comparison_df))
bar_width = 0.25

# --- RMSE Bar Chart ---
plt.figure(figsize=(14, 6))
plt.bar(x_pos - bar_width, comparison_df["LR_RMSE"], bar_width, label="Linear Regression", color="skyblue")
plt.bar(x_pos, comparison_df["XGB_RMSE"], bar_width, label="XGBoost", color="lightcoral")
plt.bar(x_pos + bar_width, comparison_df["LSTM_RMSE"], bar_width, label="LSTM", color="lightgreen")
plt.xticks(x_pos, comparison_df["ATM_ID"], rotation=45)
plt.ylabel("RMSE")
plt.title("RMSE Comparison per ATM")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "RMSE_comparison.png"), dpi=300)
plt.close()

# --- SMAPE Bar Chart ---
plt.figure(figsize=(14, 6))
plt.bar(x_pos - bar_width, comparison_df["LR_SMAPE"], bar_width, label="Linear Regression", color="skyblue")
plt.bar(x_pos, comparison_df["XGB_SMAPE"], bar_width, label="XGBoost", color="lightcoral")
plt.bar(x_pos + bar_width, comparison_df["LSTM_SMAPE"], bar_width, label="LSTM", color="lightgreen")
plt.xticks(x_pos, comparison_df["ATM_ID"], rotation=45)
plt.ylabel("SMAPE (%)")
plt.title("SMAPE Comparison per ATM")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "SMAPE_comparison.png"), dpi=300)
plt.close()

print("Visualizations saved")
print(f"All comparison files saved to: {OUTPUT_DIR}")

