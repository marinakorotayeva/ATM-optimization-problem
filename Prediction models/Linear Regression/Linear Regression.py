import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Create directories if they don't exist
os.makedirs("Prediction models/Linear Regression/results", exist_ok=True)
os.makedirs("Prediction models/Linear Regression/predictions", exist_ok=True)
os.makedirs("Prediction models/Linear Regression/graphs/actual_vs_predicted", exist_ok=True)
os.makedirs("Prediction models/Linear Regression/graphs/residuals", exist_ok=True)

# Load dataset
file_path = "Prediction models/data.csv"
data = pd.read_csv(file_path)
data['DATE'] = pd.to_datetime(data['DATE'], dayfirst=True, errors='coerce')
data.sort_values(by=['ATM_NUMBER', 'DATE'], inplace=True)

# Feature Engineering - Simplified lag features and 7-day rolling stats only
def add_features(df):
    df = df.copy()
    for atm in df['ATM_NUMBER'].unique():
        atm_mask = df['ATM_NUMBER'] == atm
        
        # Lag features (1-day and 1-week)
        df.loc[atm_mask, 'ATM_WITHDRWLS_1DAYAGO'] = df.loc[atm_mask, 'ATM_WITHDRWLS'].shift(1)
        df.loc[atm_mask, 'ATM_WITHDRWLS_7DAYAGO'] = df.loc[atm_mask, 'ATM_WITHDRWLS'].shift(7)
        
        # 7-day rolling features only
        df.loc[atm_mask, 'ROLLING_MEAN_7D'] = (
            df.loc[atm_mask, 'ATM_WITHDRWLS'].rolling(window=7).mean()
        )
        df.loc[atm_mask, 'ROLLING_STD_7D'] = (
            df.loc[atm_mask, 'ATM_WITHDRWLS'].rolling(window=7).std()
        )
    return df

data = add_features(data)

# Define features (only 7-day rolling + key lags)
features = [
    'ATM_WITHDRWLS_1DAYAGO',
    'ATM_WITHDRWLS_7DAYAGO',
    'ROLLING_MEAN_7D',
    'ROLLING_STD_7D',
    'HOLIDAY',
    'DAY_OF_WEEK',
    'IsSaturday',
    'IsSunday'
]

# ATMs to analyze
atms_to_analyze = data["ATM_NUMBER"].unique()

# Store results and predictions
combined_results = []
prediction_rows = []

# Define time series cross-validator
tscv = TimeSeriesSplit(n_splits=5)

# Function to calculate SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero when both true and pred are zero
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    return 100 * np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])

# Process each ATM
for atm_num in atms_to_analyze:
    atm_data = data[data['ATM_NUMBER'] == atm_num].copy()
    atm_data.set_index('DATE', inplace=True)
    
    # Filter by fixed date range and drop rows with missing values
    start_date = pd.to_datetime("2006-02-25")
    end_date = pd.to_datetime("2008-02-25")
    atm_data_filtered = atm_data[["ATM_WITHDRWLS"] + features].loc[start_date:end_date].dropna()
    
    # Apply fixed train/test date split
    train = atm_data_filtered.loc[:pd.to_datetime("2007-10-02")]
    test = atm_data_filtered.loc[pd.to_datetime("2007-10-03"):]
    
    if train.empty or test.empty:
        continue
    
    X_train = train[features]
    X_test = test[features]
    y_train = train["ATM_WITHDRWLS"].values
    y_test = test["ATM_WITHDRWLS"].values
    
    # Linear Regression Model only
    model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                              scoring='neg_mean_squared_error')
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
        'DATE': train.index,
        'Set_Type': 'Train',
        'Actual': y_train,
        'Predicted': train_pred
    })
    
    test_df = pd.DataFrame({
        'ATM_NUMBER': atm_num,
        'DATE': test.index,
        'Set_Type': 'Test',
        'Actual': y_test,
        'Predicted': test_pred
    })
    
    prediction_rows.extend([train_df, test_df])
    
    # Create actual vs predicted plot
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, y_test, label='Actual', color='blue', alpha=0.7)
    plt.plot(test.index, test_pred, label='Predicted', color='red', linestyle='--', alpha=0.7)
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
    plt.savefig(f"Prediction models/Linear Regression/graphs/actual_vs_predicted/ATM_{atm_num}.png")
    plt.close()
    
    # Create improved residual plot with connecting lines
    residuals = y_test - test_pred
    plt.figure(figsize=(12, 6))
    
    # Plot connecting lines first (behind dots)
    plt.plot(test.index, residuals, 
             color='green', 
             linestyle='-', 
             alpha=0.3, 
             label='Residual Trend')
    
    # Plot dots on top
    plt.scatter(test.index, residuals, 
                color='green', 
                alpha=0.7, 
                label='Daily Residuals')
    
    # Zero line
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
    
    # Add rolling mean of residuals (7-day window)
    rolling_mean = pd.Series(residuals, index=test.index).rolling(7).mean()
    plt.plot(test.index, rolling_mean, 
             color='purple', 
             linestyle='-', 
             linewidth=2, 
             label='7-Day Rolling Mean')
    
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
    plt.savefig(f"Prediction models/Linear Regression/graphs/residuals/ATM_{atm_num}.png")
    plt.close()

# Save results
results_df = pd.DataFrame(combined_results)
results_df.to_csv("Prediction models/Linear Regression/results/linear_regression_results.csv", index=False)

# Save predictions
predictions_df = pd.concat(prediction_rows)
predictions_df.to_csv("Prediction models/Linear Regression/predictions/predictions.csv", index=False)

print("Analysis completed. Results, predictions, and graphs saved.")
