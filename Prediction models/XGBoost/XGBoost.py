import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shap
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

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
best_params_list = []

# Hyperparameter grid for tuning
param_distributions = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [300, 500, 700],
    'max_depth': [3, 4, 5],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha': [1, 5, 10],
    'reg_lambda': [5, 10, 15],
    'gamma': [1, 3, 5]
}

# Process each ATM
for atm_num, data_dict in prepared_data.items():
    print(f"\n--- Processing ATM {atm_num} ---")
    
    X_train = data_dict['X_train']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    train_dates = data_dict['train_dates']
    test_dates = data_dict['test_dates']
    features = data_dict['features']
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Hyperparameter tuning with RandomizedSearchCV
    print("Performing hyperparameter tuning...")
    base_model = XGBRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=20,  # Number of parameter combinations to try
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    # Get the best model and parameters
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_params_list.append({'ATM_NUMBER': atm_num, **best_params})
    
    print(f"Best parameters for ATM {atm_num}:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Cross-validation with the best model
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    # Predictions
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    
    # Calculate residuals
    test_residuals = y_test - test_pred
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_smape = symmetric_mean_absolute_percentage_error(y_train, train_pred)
    test_smape = symmetric_mean_absolute_percentage_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
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
        'SMAPE_Test': test_smape,
        'R2_Train': train_r2,
        'R2_Test': test_r2,
        'Train_Test_Ratio': test_rmse / train_rmse
    })
    
    print(f"Results - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}, Ratio: {test_rmse/train_rmse:.2f}")
    
    # Store predictions (without residuals)
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
    
    # Create Time Series Residuals Plot (without confidence bands)
    print("Creating time series residuals plot...")
    plt.figure(figsize=(14, 6))
    
    # Plot connecting lines
    plt.plot(test_dates, test_residuals, color='green', linestyle='-', alpha=0.5, label='Residual Trend')
    
    # Plot dots
    plt.scatter(test_dates, test_residuals, color='green', alpha=0.7, s=20, label='Daily Residuals')
    
    # Zero line
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Error', linewidth=2)
    
    plt.title(f'ATM {atm_num} - Residuals Over Time (Test Set)\nRMSE: {test_rmse:.0f}, MAE: {test_mae:.0f}')
    plt.xlabel('Date')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"Prediction models/XGBoost/graphs/residuals/ATM_{atm_num}_residuals_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature Importance
    importance = best_model.feature_importances_
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
    plt.title(f'ATM {atm_num} - XGBoost Feature Importance (Tuned)')
    plt.tight_layout()
    plt.savefig(f"Prediction models/XGBoost/graphs/feature_importance/ATM_{atm_num}_tuned.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP Analysis (only if dataset is not too large)
    if len(X_train) < 10000:  # Limit to avoid memory issues
        try:
            explainer = shap.TreeExplainer(best_model)
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
            plt.title(f'ATM {atm_num} - SHAP Feature Importance (Tuned)')
            plt.tight_layout()
            plt.savefig(f"Prediction models/XGBoost/graphs/shap/ATM_{atm_num}_summary_tuned.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"SHAP analysis failed for ATM {atm_num}: {e}")
    
    # Create actual vs predicted plot
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_test, label='Actual', color='blue', alpha=0.8, linewidth=1.5)
    plt.plot(test_dates, test_pred, label='Predicted', color='red', linestyle='--', alpha=0.8, linewidth=1.5)
    
    plt.title(f'ATM {atm_num} - Actual vs Predicted (Test Set, Tuned)\nRMSE: {test_rmse:.0f}, MAE: {test_mae:.0f}, SMAPE: {test_smape:.1f}%')
    plt.xlabel('Date')
    plt.ylabel('Withdrawals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis for better date display
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"Prediction models/XGBoost/graphs/actual_vs_predicted/ATM_{atm_num}_tuned.png", dpi=300, bbox_inches='tight')
    plt.close()

# Save results
results_df = pd.DataFrame(combined_results)
results_df.to_csv("Prediction models/XGBoost/results/xgboost_tuned_results.csv", index=False)

# Save best parameters
best_params_df = pd.DataFrame(best_params_list)
best_params_df.to_csv("Prediction models/XGBoost/results/best_parameters.csv", index=False)

# Save predictions (without residuals)
predictions_df = pd.concat(prediction_rows, ignore_index=True)
predictions_df.to_csv("Prediction models/XGBoost/predictions/predictions.csv", index=False)

# Save feature importance data
if feature_importance_data:
    feature_importance_df = pd.concat(feature_importance_data, ignore_index=True)
    feature_importance_df.to_csv("Prediction models/XGBoost/results/feature_importance_tuned.csv", index=False)

# Save SHAP summary data
if shap_values_data:
    shap_values_df = pd.concat(shap_values_data, ignore_index=True)
    shap_values_df.to_csv("Prediction models/XGBoost/results/shap_values_summary_tuned.csv", index=False)

# Create summary statistics for features across all ATMs
if feature_importance_data:
    feature_importance_summary = (feature_importance_df.groupby('Feature')['Importance']
                                 .agg(['mean', 'std', 'count'])
                                 .sort_values('mean', ascending=False)
                                 .reset_index())
    feature_importance_summary.to_csv("Prediction models/XGBoost/results/feature_importance_summary_tuned.csv", index=False)

if shap_values_data:
    shap_summary = (shap_values_df.groupby('Feature')['Mean_ABS_SHAP']
                   .agg(['mean', 'std', 'count'])
                   .sort_values('mean', ascending=False)
                   .reset_index())
    shap_summary.to_csv("Prediction models/XGBoost/results/shap_summary_tuned.csv", index=False)

print("\n" + "="*60)
print("XGBoost tuning and analysis completed!")
print(f"Processed {len(combined_results)} ATMs")
print(f"Cross-validation performed with {tscv.n_splits} folds")
print("\nResults saved in: Prediction models/XGBoost/")
print("\nOutputs include:")
print("- Time series residual plots")
print("- Model performance metrics")
print("- Feature importance analysis")
print("- SHAP summary analysis")
print("="*60)

# Show summary of results
print("\nSummary of Results:")
summary_cols = ['ATM_NUMBER', 'RMSE_Train', 'RMSE_Test', 'Train_Test_Ratio', 'R2_Test', 'SMAPE_Test']
print(results_df[summary_cols].round(2).to_string(index=False))
