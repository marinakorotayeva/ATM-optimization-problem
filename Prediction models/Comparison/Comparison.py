import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

# Create directory for comparison results
os.makedirs("Prediction models/Comparison", exist_ok=True)

# Load results from all models
print("Loading results from all models...")

try:
    # Load Linear Regression results
    lr_results = pd.read_csv("Prediction models/Linear Regression/results/linear_regression_cv_results.csv")
    lr_results['Model'] = 'Linear Regression'
    print("✓ Loaded Linear Regression results")
except FileNotFoundError:
    print("✗ Linear Regression results not found. Please run the Linear Regression code first.")
    exit(1)

try:
    # Load XGBoost results
    xgb_results = pd.read_csv("Prediction models/XGBoost/results/xgboost_cv_results.csv")
    xgb_results['Model'] = 'XGBoost'
    print("✓ Loaded XGBoost results")
except FileNotFoundError:
    print("✗ XGBoost results not found. Please run the XGBoost code first.")
    exit(1)

try:
    # Load LSTM results
    lstm_results = pd.read_csv("Prediction models/LSTM/results/lstm_cv_results.csv")
    lstm_results['Model'] = 'LSTM'
    print("✓ Loaded LSTM results")
except FileNotFoundError:
    print("✗ LSTM results not found. Please run the LSTM code first.")
    exit(1)

# Combine results
combined_results = pd.concat([lr_results, xgb_results, lstm_results], ignore_index=True)

# Create detailed comparison table
comparison_data = []

for atm_num in sorted(combined_results['ATM_NUMBER'].unique()):
    lr_atm = lr_results[lr_results['ATM_NUMBER'] == atm_num]
    xgb_atm = xgb_results[xgb_results['ATM_NUMBER'] == atm_num]
    lstm_atm = lstm_results[lstm_results['ATM_NUMBER'] == atm_num]
    
    if not lr_atm.empty and not xgb_atm.empty and not lstm_atm.empty:
        # Extract metrics
        lr_rmse = lr_atm['CV_RMSE_Mean'].values[0]
        xgb_rmse = xgb_atm['CV_RMSE_Mean'].values[0]
        lstm_rmse = lstm_atm['CV_RMSE_Mean'].values[0]
        
        lr_smape = lr_atm['CV_SMAPE_Mean'].values[0]
        xgb_smape = xgb_atm['CV_SMAPE_Mean'].values[0]
        lstm_smape = lstm_atm['CV_SMAPE_Mean'].values[0]
        
        # Calculate improvements vs Linear Regression (baseline)
        xgb_rmse_improv = ((lr_rmse - xgb_rmse) / lr_rmse * 100) if lr_rmse > 0 else 0
        lstm_rmse_improv = ((lr_rmse - lstm_rmse) / lr_rmse * 100) if lr_rmse > 0 else 0
        
        xgb_smape_improv = ((lr_smape - xgb_smape) / lr_smape * 100) if lr_smape > 0 else 0
        lstm_smape_improv = ((lr_smape - lstm_smape) / lr_smape * 100) if lr_smape > 0 else 0
        
        # Determine best model for each metric
        rmse_values = {'Linear Regression': lr_rmse, 'XGBoost': xgb_rmse, 'LSTM': lstm_rmse}
        smape_values = {'Linear Regression': lr_smape, 'XGBoost': xgb_smape, 'LSTM': lstm_smape}
        
        best_rmse = min(rmse_values, key=rmse_values.get)
        best_smape = min(smape_values, key=smape_values.get)
        
        comparison_data.append({
            'ATM': atm_num,
            'LR_RMSE': f"{lr_rmse:.0f} ± {lr_atm['CV_RMSE_Std'].values[0]:.0f}",
            'XGB_RMSE': f"{xgb_rmse:.0f} ± {xgb_atm['CV_RMSE_Std'].values[0]:.0f}",
            'LSTM_RMSE': f"{lstm_rmse:.0f} ± {lstm_atm['CV_RMSE_Std'].values[0]:.0f}",
            'XGB_RMSE_Improv': f"{xgb_rmse_improv:+.1f}%",
            'LSTM_RMSE_Improv': f"{lstm_rmse_improv:+.1f}%",
            'Best_RMSE': best_rmse,
            
            'LR_SMAPE': f"{lr_smape:.1f}% ± {lr_atm['CV_SMAPE_Std'].values[0]:.1f}",
            'XGB_SMAPE': f"{xgb_smape:.1f}% ± {xgb_atm['CV_SMAPE_Std'].values[0]:.1f}",
            'LSTM_SMAPE': f"{lstm_smape:.1f}% ± {lstm_atm['CV_SMAPE_Std'].values[0]:.1f}",
            'XGB_SMAPE_Improv': f"{xgb_smape_improv:+.1f}%",
            'LSTM_SMAPE_Improv': f"{lstm_smape_improv:+.1f}%",
            'Best_SMAPE': best_smape
        })

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_data)

# Calculate overall statistics
overall_stats = {
    'Metric': ['RMSE (Mean ± Std)', 'SMAPE (Mean ± Std)'],
    'Linear Regression': [
        f"{lr_results['CV_RMSE_Mean'].mean():.0f} ± {lr_results['CV_RMSE_Std'].mean():.0f}",
        f"{lr_results['CV_SMAPE_Mean'].mean():.1f}% ± {lr_results['CV_SMAPE_Std'].mean():.1f}"
    ],
    'XGBoost': [
        f"{xgb_results['CV_RMSE_Mean'].mean():.0f} ± {xgb_results['CV_RMSE_Std'].mean():.0f}",
        f"{xgb_results['CV_SMAPE_Mean'].mean():.1f}% ± {xgb_results['CV_SMAPE_Std'].mean():.1f}"
    ],
    'LSTM': [
        f"{lstm_results['CV_RMSE_Mean'].mean():.0f} ± {lstm_results['CV_RMSE_Std'].mean():.0f}",
        f"{lstm_results['CV_SMAPE_Mean'].mean():.1f}% ± {lstm_results['CV_SMAPE_Std'].mean():.1f}"
    ],
    'XGB_Improvement': [
        f"{(lr_results['CV_RMSE_Mean'].mean() - xgb_results['CV_RMSE_Mean'].mean()) / lr_results['CV_RMSE_Mean'].mean() * 100:+.1f}%",
        f"{(lr_results['CV_SMAPE_Mean'].mean() - xgb_results['CV_SMAPE_Mean'].mean()) / lr_results['CV_SMAPE_Mean'].mean() * 100:+.1f}%"
    ],
    'LSTM_Improvement': [
        f"{(lr_results['CV_RMSE_Mean'].mean() - lstm_results['CV_RMSE_Mean'].mean()) / lr_results['CV_RMSE_Mean'].mean() * 100:+.1f}%",
        f"{(lr_results['CV_SMAPE_Mean'].mean() - lstm_results['CV_SMAPE_Mean'].mean()) / lr_results['CV_SMAPE_Mean'].mean() * 100:+.1f}%"
    ]
}

overall_df = pd.DataFrame(overall_stats)

# Count best models
best_rmse_count = comparison_df['Best_RMSE'].value_counts()
best_smape_count = comparison_df['Best_SMAPE'].value_counts()

best_model_summary = pd.DataFrame({
    'Metric': ['RMSE', 'SMAPE'],
    'Linear Regression': [best_rmse_count.get('Linear Regression', 0), best_smape_count.get('Linear Regression', 0)],
    'XGBoost': [best_rmse_count.get('XGBoost', 0), best_smape_count.get('XGBoost', 0)],
    'LSTM': [best_rmse_count.get('LSTM', 0), best_smape_count.get('LSTM', 0)],
    'Total ATMs': [len(comparison_df), len(comparison_df)]
})

# Save only the required overall CSV file
overall_df.to_csv("Prediction models/Comparison/model_comparison_overall.csv", index=False)

print("✓ Comparison table saved")

# Create formatted text output for better readability
with open("Prediction models/Comparison/comparison_report.txt", "w") as f:
    f.write("=" * 100 + "\n")
    f.write("MODEL COMPARISON REPORT (Linear Regression vs XGBoost vs LSTM)\n")
    f.write("=" * 100 + "\n\n")
    
    f.write("DETAILED ATM PERFORMANCE COMPARISON:\n")
    f.write("-" * 100 + "\n")
    f.write(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))
    f.write("\n\n")
    
    f.write("OVERALL PERFORMANCE SUMMARY:\n")
    f.write("-" * 100 + "\n")
    f.write(tabulate(overall_df, headers='keys', tablefmt='grid', showindex=False))
    f.write("\n\n")
    
    f.write("BEST MODEL COUNT:\n")
    f.write("-" * 100 + "\n")
    f.write(tabulate(best_model_summary, headers='keys', tablefmt='grid', showindex=False))
    f.write("\n\n")
    
    f.write("KEY INSIGHTS:\n")
    f.write("-" * 100 + "\n")
    f.write(f"1. XGBoost shows better RMSE performance for {best_rmse_count.get('XGBoost', 0)} out of {len(comparison_df)} ATMs\n")
    f.write(f"2. LSTM shows better RMSE performance for {best_rmse_count.get('LSTM', 0)} out of {len(comparison_df)} ATMs\n")
    f.write(f"3. XGBoost shows better SMAPE performance for {best_smape_count.get('XGBoost', 0)} out of {len(comparison_df)} ATMs\n")
    f.write(f"4. LSTM shows better SMAPE performance for {best_smape_count.get('LSTM', 0)} out of {len(comparison_df)} ATMs\n")
    f.write(f"5. Overall XGBoost RMSE improvement: {overall_stats['XGB_Improvement'][0]}\n")
    f.write(f"6. Overall LSTM RMSE improvement: {overall_stats['LSTM_Improvement'][0]}\n")
    f.write(f"7. Overall XGBoost SMAPE improvement: {overall_stats['XGB_Improvement'][1]}\n")
    f.write(f"8. Overall LSTM SMAPE improvement: {overall_stats['LSTM_Improvement'][1]}\n")

print("✓ Text report saved")

# Create only the required visualizations
print("Creating comparison visualizations...")

# Set larger font sizes for all plots
plt.rcParams.update({
    'font.size': 14,           # Default font size
    'axes.titlesize': 18,      # Title font size
    'axes.labelsize': 16,      # Axis label font size
    'xtick.labelsize': 12,     # X-axis tick label size
    'ytick.labelsize': 12,     # Y-axis tick label size
    'legend.fontsize': 14,     # Legend font size
})

# 1. RMSE Comparison Bar Chart
plt.figure(figsize=(18, 10))  # Slightly larger figure
x_pos = np.arange(len(comparison_df))
bar_width = 0.25

# Extract numeric values for plotting
lr_rmse_values = [float(x.split(' ± ')[0]) for x in comparison_df['LR_RMSE']]
xgb_rmse_values = [float(x.split(' ± ')[0]) for x in comparison_df['XGB_RMSE']]
lstm_rmse_values = [float(x.split(' ± ')[0]) for x in comparison_df['LSTM_RMSE']]

plt.bar(x_pos - bar_width, lr_rmse_values, bar_width, 
        label='Linear Regression', alpha=0.8, color='skyblue')
plt.bar(x_pos, xgb_rmse_values, bar_width, 
        label='XGBoost', alpha=0.8, color='lightcoral')
plt.bar(x_pos + bar_width, lstm_rmse_values, bar_width, 
        label='LSTM', alpha=0.8, color='lightgreen')

plt.xlabel('ATM Number', fontsize=16, fontweight='bold')
plt.ylabel('RMSE', fontsize=16, fontweight='bold')
plt.title('RMSE Comparison: Linear Regression vs XGBoost vs LSTM', fontsize=18, fontweight='bold')
plt.xticks(x_pos, comparison_df['ATM'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("Prediction models/Comparison/rmse_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. SMAPE Comparison Bar Chart
plt.figure(figsize=(18, 10))  # Slightly larger figure
lr_smape_values = [float(x.split(' ± ')[0].replace('%', '')) for x in comparison_df['LR_SMAPE']]
xgb_smape_values = [float(x.split(' ± ')[0].replace('%', '')) for x in comparison_df['XGB_SMAPE']]
lstm_smape_values = [float(x.split(' ± ')[0].replace('%', '')) for x in comparison_df['LSTM_SMAPE']]

plt.bar(x_pos - bar_width, lr_smape_values, bar_width, 
        label='Linear Regression', alpha=0.8, color='skyblue')
plt.bar(x_pos, xgb_smape_values, bar_width, 
        label='XGBoost', alpha=0.8, color='lightcoral')
plt.bar(x_pos + bar_width, lstm_smape_values, bar_width, 
        label='LSTM', alpha=0.8, color='lightgreen')

plt.xlabel('ATM Number', fontsize=16, fontweight='bold')
plt.ylabel('SMAPE (%)', fontsize=16, fontweight='bold')
plt.title('SMAPE Comparison: Linear Regression vs XGBoost vs LSTM', fontsize=18, fontweight='bold')
plt.xticks(x_pos, comparison_df['ATM'], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("Prediction models/Comparison/smape_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Required visualizations saved")

# Print summary to console
print("\n" + "="*100)
print("MODEL COMPARISON SUMMARY (Linear Regression vs XGBoost vs LSTM)")
print("="*100)

print(f"\nOverall Performance:")
print(f"Linear Regression - RMSE: {lr_results['CV_RMSE_Mean'].mean():.0f} ± {lr_results['CV_RMSE_Std'].mean():.0f}")
print(f"Linear Regression - SMAPE: {lr_results['CV_SMAPE_Mean'].mean():.1f}% ± {lr_results['CV_SMAPE_Std'].mean():.1f}")
print(f"XGBoost - RMSE: {xgb_results['CV_RMSE_Mean'].mean():.0f} ± {xgb_results['CV_RMSE_Std'].mean():.0f}")
print(f"XGBoost - SMAPE: {xgb_results['CV_SMAPE_Mean'].mean():.1f}% ± {xgb_results['CV_SMAPE_Std'].mean():.1f}")
print(f"LSTM - RMSE: {lstm_results['CV_RMSE_Mean'].mean():.0f} ± {lstm_results['CV_RMSE_Std'].mean():.0f}")
print(f"LSTM - SMAPE: {lstm_results['CV_SMAPE_Mean'].mean():.1f}% ± {lstm_results['CV_SMAPE_Std'].mean():.1f}")

print(f"\nImprovement vs Linear Regression:")
xgb_rmse_improv = (lr_results['CV_RMSE_Mean'].mean() - xgb_results['CV_RMSE_Mean'].mean()) / lr_results['CV_RMSE_Mean'].mean() * 100
xgb_smape_improv = (lr_results['CV_SMAPE_Mean'].mean() - xgb_results['CV_SMAPE_Mean'].mean()) / lr_results['CV_SMAPE_Mean'].mean() * 100
lstm_rmse_improv = (lr_results['CV_RMSE_Mean'].mean() - lstm_results['CV_RMSE_Mean'].mean()) / lr_results['CV_RMSE_Mean'].mean() * 100
lstm_smape_improv = (lr_results['CV_SMAPE_Mean'].mean() - lstm_results['CV_SMAPE_Mean'].mean()) / lr_results['CV_SMAPE_Mean'].mean() * 100

print(f"XGBoost RMSE: {xgb_rmse_improv:+.1f}%")
print(f"XGBoost SMAPE: {xgb_smape_improv:+.1f}%")
print(f"LSTM RMSE: {lstm_rmse_improv:+.1f}%")
print(f"LSTM SMAPE: {lstm_smape_improv:+.1f}%")

print(f"\nBest Model Count:")
print(f"By RMSE: Linear Regression: {best_rmse_count.get('Linear Regression', 0)}, XGBoost: {best_rmse_count.get('XGBoost', 0)}, LSTM: {best_rmse_count.get('LSTM', 0)}")
print(f"By SMAPE: Linear Regression: {best_smape_count.get('Linear Regression', 0)}, XGBoost: {best_smape_count.get('XGBoost', 0)}, LSTM: {best_smape_count.get('LSTM', 0)}")

print(f"\nFiles saved in: Prediction models/Comparison/")
print("="*100)
