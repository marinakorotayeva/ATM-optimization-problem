import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit
import pickle

# Create directories if they don't exist
os.makedirs("Prediction models/Data", exist_ok=True)

# Load dataset
file_path = "Prediction models/data.csv"
data = pd.read_csv(file_path)
data['DATE'] = pd.to_datetime(data['DATE'], dayfirst=True, errors='coerce')
data.sort_values(by=['ATM_NUMBER', 'DATE'], inplace=True)

# Feature Engineering - CORRECTED to avoid data leakage
def add_features(df):
    df = df.copy()
    for atm in df['ATM_NUMBER'].unique():
        atm_mask = df['ATM_NUMBER'] == atm
        
        # Lag features (1-day and 1-week)
        df.loc[atm_mask, 'ATM_WITHDRWLS_1DAYAGO'] = df.loc[atm_mask, 'ATM_WITHDRWLS'].shift(1)
        df.loc[atm_mask, 'ATM_WITHDRWLS_7DAYAGO'] = df.loc[atm_mask, 'ATM_WITHDRWLS'].shift(7)
        
        # CORRECTED: Calculate rolling stats and shift by 1 to avoid data leakage
        # The value for day 't' will be the rolling mean of days 't-7' to 't-1'
        rolling_mean = df.loc[atm_mask, 'ATM_WITHDRWLS'].rolling(window=7, min_periods=1).mean()
        rolling_std = df.loc[atm_mask, 'ATM_WITHDRWLS'].rolling(window=7, min_periods=1).std()
        
        # Shift the result by 1 to avoid data leakage
        df.loc[atm_mask, 'ROLLING_MEAN_7D'] = rolling_mean.shift(1)
        df.loc[atm_mask, 'ROLLING_STD_7D'] = rolling_std.shift(1)
    
    return df

print("Adding features...")
data = add_features(data)

# Define features
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
print(f"Found {len(atms_to_analyze)} ATMs: {atms_to_analyze}")

# Define time series cross-validator with 10 folds
tscv = TimeSeriesSplit(n_splits=10)

# Function to prepare and split data for each ATM
def prepare_atm_data(atm_data, features):
    atm_data = atm_data.copy()
    atm_data.set_index('DATE', inplace=True)
    
    # Filter by fixed date range
    start_date = pd.to_datetime("2006-02-25")
    end_date = pd.to_datetime("2008-02-25")
    atm_data_filtered = atm_data.loc[start_date:end_date]
    
    # Apply fixed train/test date split
    train_cutoff = pd.to_datetime("2007-10-02")
    test_start = pd.to_datetime("2007-10-03")
    
    train = atm_data_filtered.loc[:train_cutoff]
    test = atm_data_filtered.loc[test_start:]
    
    if train.empty or test.empty:
        print(f"Warning: Empty train or test set for ATM")
        return None, None, None, None, None, None
    
    # Drop rows with missing values from each set independently
    train_clean = train.dropna(subset=features + ['ATM_WITHDRWLS'])
    test_clean = test.dropna(subset=features + ['ATM_WITHDRWLS'])
    
    if train_clean.empty or test_clean.empty:
        print(f"Warning: Empty train or test set after dropping NA for ATM")
        return None, None, None, None, None, None
    
    X_train = train_clean[features]
    X_test = test_clean[features]
    y_train = train_clean['ATM_WITHDRWLS'].values
    y_test = test_clean['ATM_WITHDRWLS'].values
    
    return X_train, X_test, y_train, y_test, train_clean.index, test_clean.index

# Store prepared data for all ATMs
prepared_data = {}
successful_atms = 0

print("Preparing data for each ATM...")
for atm_num in atms_to_analyze:
    atm_data = data[data['ATM_NUMBER'] == atm_num].copy()
    X_train, X_test, y_train, y_test, train_dates, test_dates = prepare_atm_data(atm_data, features)
    
    if X_train is not None:
        prepared_data[atm_num] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'features': features
        }
        successful_atms += 1
        print(f"  ATM {atm_num}: {len(X_train)} train samples, {len(X_test)} test samples")
    else:
        print(f"  ATM {atm_num}: Failed to prepare data")

# Save the prepared data
with open("Prediction models/Data/prepared_data.pkl", "wb") as f:
    pickle.dump(prepared_data, f)

# Save the cross-validator
with open("Prediction models/Data/tscv.pkl", "wb") as f:
    pickle.dump(tscv, f)

print("\nData preparation completed!")
print(f"Successfully prepared data for {successful_atms}/{len(atms_to_analyze)} ATMs")
print(f"Data saved in: Prediction models/Data/")

# Show sample statistics for the first ATM
if prepared_data:
    first_atm = list(prepared_data.keys())[0]
    print(f"\nSample statistics for ATM {first_atm}:")
    print(f"  Training period: {prepared_data[first_atm]['train_dates'].min()} to {prepared_data[first_atm]['train_dates'].max()}")
    print(f"  Testing period:  {prepared_data[first_atm]['test_dates'].min()} to {prepared_data[first_atm]['test_dates'].max()}")
    print(f"  Training samples: {len(prepared_data[first_atm]['X_train'])}")
    print(f"  Testing samples:  {len(prepared_data[first_atm]['X_test'])}")
