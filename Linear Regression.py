import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import os

# ----------------- Load Data -----------------
file_path = "Prediction models/Linear Regression/data for prediction.csv"
df = pd.read_csv(file_path)

# Ensure DATE is datetime
df["DATE"] = pd.to_datetime(df["DATE"])

# Prepare result storage
future_predictions = []

# Process each ATM separately
for atm in df["ATM"].unique():
    atm_df = df[df["ATM"] == atm].copy()
    atm_df = atm_df.sort_values("DATE")

    # Features and target
    features = ["Day of the week", "Day in a month", "Month", "Year", "Holiday (Yes -1, No -0)", "Avg daily withdrawal per ATM"]
    target = "WITHDRWLS"

    # Train model
    X = atm_df[features]
    y = atm_df[target]
    model = LinearRegression()
    model.fit(X, y)

    # Predict next 5 days
    last_date = atm_df["DATE"].max()
    for i in range(1, 6):
        future_date = last_date + timedelta(days=i)
        dow = future_date.weekday() + 1  # Monday = 1
        dom = future_date.day
        month = future_date.month
        year = future_date.year
        holiday = 0  # You can customize this if you have a holiday calendar

        avg_withdrawal = atm_df["WITHDRWLS"].mean()
        features_input = np.array([[dow, dom, month, year, holiday, avg_withdrawal]])
        predicted_withdrawal = model.predict(features_input)[0]

        # Recreate the full row structure
        prediction_row = {
            "CASHP_ID": f"Pred_{atm}_{i}",
            "ATM": atm,
            "DATE": future_date,
            "WITHDRWLS": predicted_withdrawal,
            "DEPOSITS": 0,  # Placeholder
            "WITHDRWLS Scaled": predicted_withdrawal / atm_df["WITHDRWLS"].max(),
            "DEPOSITS Scaled": 0,
            "Day of the week": dow,
            "Day in a month": dom,
            "Month": month,
            "Year": year,
            "Date": f"{dom}-{month}-{year}",
            "Holiday (Yes -1, No -0)": holiday,
            "Avg daily withdrawal per ATM": avg_withdrawal
        }

        future_predictions.append(prediction_row)

# ----------------- Save Results -----------------
pred_df = pd.DataFrame(future_predictions)
pred_df.to_csv("Prediction models/Linear Regression/atm_withdrawal_predictions_5_days.csv", index=False)
print("Predictions saved to atm_withdrawal_predictions_5_days.csv")
