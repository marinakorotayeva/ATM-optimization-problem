import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit

# Create directories if they don't exist
os.makedirs("Prediction models/Data", exist_ok=True)

# Load dataset
file_path = "Prediction models/data.csv"
data = pd.read_csv(file_path)
data['DATE'] = pd.to_datetime(data['DATE'], dayfirst=True, errors='coerce')
data.sort_values(by=['ATM_NUMBER', 'DATE'], inplace=True)

# Feature Engineering
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

# Define time series cross-validator with 10 folds
tscv = TimeSeriesSplit(n_splits=10)

# Function to prepare and split data for each ATM
def prepare_atm_data(atm_data, features):
    atm_data = atm_data.copy()
    atm_data.set_index('DATE', inplace=True)
    
    # Filter by fixed date range and drop rows with missing values
    start_date = pd.to_datetime("2006-02-25")
    end_date = pd.to_datetime("2008-02-25")
    atm_data_filtered = atm_data[["ATM_WITHDRWLS"] + features].loc[start_date:end_date].dropna()
    
    # Apply fixed train/test date split
    train = atm_data_filtered.loc[:pd.to_datetime("2007-10-02")]
    test = atm_data_filtered.loc[pd.to_datetime("2007-10-03"):]
    
    if train.empty or test.empty:
        return None, None, None, None, None, None
    
    X_train = train[features]
    X_test = test[features]
    y_train = train["ATM_WITHDRWLS"].values
    y_test = test["ATM_WITHDRWLS"].values
    
    return X_train, X_test, y_train, y_test, train.index, test.index

# Store prepared data for all ATMs
prepared_data = {}

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

# Save the prepared data
import pickle

with open("Prediction models/Data/prepared_data.pkl", "wb") as f:
    pickle.dump(prepared_data, f)

# Save the cross-validator
with open("Prediction models/Data/tscv.pkl", "wb") as f:
    pickle.dump(tscv, f)

print("Data preparation completed. Prepared data saved for modeling.")
print(f"Processed {len(prepared_data)} ATMs")
print(f"Data saved in: Prediction models/data/")
