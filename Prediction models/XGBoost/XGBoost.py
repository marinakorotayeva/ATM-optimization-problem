import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shap
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline

# Create directories if they don't exist
os.makedirs("Prediction models/XGBoost/results", exist_ok=True)
os.makedirs("Prediction models/XGBoost/predictions", exist_ok=True)
os.makedirs("Prediction models/XGBoost/graphs/actual_vs_predicted", exist_ok=True)
os.makedirs("Prediction models/XGBoost/graphs/residuals", exist_ok=True)
os.makedirs("Prediction models/XGBoost/graphs/feature_importance", exist_ok=True)
os.makedirs("Prediction models/XGBoost/graphs/shap", exist_ok=True)

# Load prepared data
try:
    with open("Prediction models/Data/prepared_data.pkl", "rb") as f:
        prepared_data = pickle.load(f)
except FileNotFoundError:
    print("Error: Prepared data file not found. Please run data_preparation.py first.")
    print("Expected file: Prediction models/data/prepared_data.pkl")
    exit(1)

# Load cross-validator
try:
    with open("Prediction models/Data/tscv.pkl", "rb") as f:
        tscv = pickle.load(f)
except FileNotFoundError:
    print("Error: Cross-validator file not found. Please run data_preparation.py first.")
    print("Expected file: Prediction models/data/tscv.pkl")
    exit(1)

# Function to calculate SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero when both true and pred are zero
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    return 100 * np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])

# Store results and predictions
combined_results = []
prediction_rows = []
feature_importance_data = []
shap_values_data = []

# Process each ATM
for atm_num, data_dict in prepared_data.items():
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    train_dates = data_dict['train_dates']
    test_dates = data_dict['test_dates']
    features = data_dict['features']
    
    # XGBoost Model with optimized parameters
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Cross-validation with 10 folds
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_smape = symmetric_mean_absolute_percentage_error(y_train, train_pred)
    test_smape = symmetric_mean_absolute_percentage_error(y_test, test_pred)
    
    # Store results
    combined_results.append({
        'ATM_NUMBER': atm_num,
        'CV_RMSE_Mean': np.mean(cv_rmse),
        'CV_RMSE_Std': np.std(cv_rmse),
        'MAE_Train': train_mae,
        'MAE_Test': test_mae,
        'RMSE_Train': train_rmse,
        'RMSE_Test': test_rmse,
        'SMAPE_Train': train_smape,
        'SMAPE_Test': test_smape
    })
    
    # Store predictions
    train_df = pd.DataFrame({
        'ATM_NUMBER': atm_num,
        'DATE': train_dates,
        'Set_Type': 'Train',
        'Actual': y_train,
        'Predicted': train_pred
    })
    
    test_df = pd.DataFrame({
        'ATM_NUMBER': atm_num,
        'DATE': test_dates,
        'Set_Type': 'Test',
        'Actual': y_test,
        'Predicted': test_pred
    })
    
    prediction_rows.extend([train_df, test_df])
    
    # Feature Importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance,
        'ATM_NUMBER': atm_num
    })
    feature_importance_data.append(feature_importance_df)
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    indices = np.argsort(importance)[::-1]
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'ATM {atm_num} - XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(f"Prediction models/XGBoost/graphs/feature_importance/ATM_{atm_num}.png")
    plt.close()
    
    # SHAP Analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Store SHAP values for summary
    shap_summary = pd.DataFrame({
        'Feature': features,
        'Mean_ABS_SHAP': np.mean(np.abs(shap_values), axis=0),
        'ATM_NUMBER': atm_num
    })
    shap_values_data.append(shap_summary)
    
    # Create SHAP summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_train, feature_names=features, show=False)
    plt.title(f'ATM {atm_num} - SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(f"Prediction models/XGBoost/graphs/shap/ATM_{atm_num}_summary.png")
    plt.close()
    
    # Create SHAP dependence plots for top 3 features
    top_features = feature_importance_df.nlargest(3, 'Importance')['Feature'].values
    for i, feature in enumerate(top_features):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_train, feature_names=features, show=False)
        plt.title(f'ATM {atm_num} - SHAP Dependence for {feature}')
        plt.tight_layout()
        plt.savefig(f"Prediction models/XGBoost/graphs/shap/ATM_{atm_num}_dependence_{feature}.png")
        plt.close()
    
    # Create actual vs predicted plot
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, label='Actual', color='blue', alpha=0.7)
    plt.plot(test_dates, test_pred, label='Predicted', color='red', linestyle='--', alpha=0.7)
    plt.title(f'ATM {atm_num} - Actual vs Predicted (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Withdrawals')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis for better date display
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"Prediction models/XGBoost/graphs/actual_vs_predicted/ATM_{atm_num}.png")
    plt.close()
    
    # Create improved residual plot with connecting lines
    residuals = y_test - test_pred
    plt.figure(figsize=(12, 6))
    
    # Plot connecting lines first (behind dots)
    plt.plot(test_dates, residuals, color='green', linestyle='-', alpha=0.3, label='Residual Trend')
    
    # Plot dots on top
    plt.scatter(test_dates, residuals, color='green', alpha=0.7, label='Daily Residuals')
    
    # Zero line
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
    
    # Add rolling mean of residuals (7-day window)
    rolling_mean = pd.Series(residuals, index=test_dates).rolling(7).mean()
    plt.plot(test_dates, rolling_mean, color='purple', linestyle='-', linewidth=2, label='7-Day Rolling Mean')
    
    plt.title(f'ATM {atm_num} - Residuals Over Time (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"Prediction models/XGBoost/graphs/residuals/ATM_{atm_num}.png")
    plt.close()

# Save results
results_df = pd.DataFrame(combined_results)
results_df.to_csv("Prediction models/XGBoost/results/xgboost_results.csv", index=False)

# Save predictions
predictions_df = pd.concat(prediction_rows, ignore_index=True)
predictions_df.to_csv("Prediction models/XGBoost/predictions/predictions.csv", index=False)

# Save feature importance data
feature_importance_df = pd.concat(feature_importance_data, ignore_index=True)
feature_importance_df.to_csv("Prediction models/XGBoost/results/feature_importance.csv", index=False)

# Save SHAP summary data
shap_values_df = pd.concat(shap_values_data, ignore_index=True)
shap_values_df.to_csv("Prediction models/XGBoost/results/shap_values_summary.csv", index=False)

# Create summary statistics for features across all ATMs
feature_importance_summary = (feature_importance_df.groupby('Feature')['Importance']
                             .agg(['mean', 'std', 'count'])
                             .sort_values('mean', ascending=False)
                             .reset_index())

shap_summary = (shap_values_df.groupby('Feature')['Mean_ABS_SHAP']
               .agg(['mean', 'std', 'count'])
               .sort_values('mean', ascending=False)
               .reset_index())

feature_importance_summary.to_csv("Prediction models/XGBoost/results/feature_importance_summary.csv", index=False)
shap_summary.to_csv("Prediction models/XGBoost/results/shap_summary.csv", index=False)

print("XGBoost analysis completed!")
print(f"Processed {len(combined_results)} ATMs")
print(f"Cross-validation performed with {tscv.n_splits} folds")
print("Results saved in: Prediction models/XGBoost/")
print("\nAdditional outputs:")
print("- Feature importance plots and CSV")
print("- SHAP summary plots and dependence plots")
print("- SHAP values summary CSV")
print("- Global feature importance statistics across all ATMs")
