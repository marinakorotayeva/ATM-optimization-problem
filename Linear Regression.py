import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

# --------- Paths and Constants ---------
data_file = "Prediction models/Linear Regression/data for prediction.csv"
status_file = "Prediction models/Linear Regression/last_prediction_date.txt"
output_folder = "Prediction models/Linear Regression"
plot_folder = os.path.join(output_folder, "prediction_plots")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

# --------- Load and Prepare Data ---------
df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values(["ATM", "DATE"])

# --------- Read Last Prediction Point ---------
if os.path.exists(status_file):
    with open(status_file, "r") as f:
        last_date = pd.to_datetime(f.read().strip())
else:
    last_date = None

# --------- Parameters ---------
prediction_horizon = 5
features = ["DEPOSITS", "Day of the week", "Day in a month", "Month", "Year", "Holiday (Yes -1, No -0)"]
default_atm_capacity = 20000000

# --------- Run Per ATM ---------
prediction_results = []

for atm_id, atm_df in df.groupby("ATM"):
    atm_df = atm_df.copy().sort_values("DATE")
    atm_df = atm_df.dropna(subset=features + ["WITHDRWLS"])

    if last_date:
        train_df = atm_df[atm_df["DATE"] <= last_date]
        pred_df = atm_df[(atm_df["DATE"] > last_date)].head(prediction_horizon)
    else:
        train_df = atm_df.iloc[:-prediction_horizon]
        pred_df = atm_df.iloc[-prediction_horizon:]

    if len(pred_df) < prediction_horizon or len(train_df) < 5:
        print(f"⚠️ Not enough data to predict for ATM {atm_id}. Skipping.")
        continue

    # Train model
    model = LinearRegression()
    model.fit(train_df[features], train_df["WITHDRWLS"])

    # Predict next 5 days
    pred_df["Predicted_WITHDRWLS"] = model.predict(pred_df[features])
    prediction_results.append(pred_df)

    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(pred_df["DATE"], pred_df["WITHDRWLS"], marker='o', label="Actual")
    plt.plot(pred_df["DATE"], pred_df["Predicted_WITHDRWLS"], marker='x', label="Predicted")
    plt.title(f"ATM {atm_id} - Forecast for Next 5 Days")
    plt.xlabel("Date")
    plt.ylabel("WITHDRWLS")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"ATM_{atm_id}_forecast_until_{pred_df['DATE'].max().date()}.png"))
    plt.close()

# --------- Save Prediction Outputs ---------
if prediction_results:
    all_predictions = pd.concat(prediction_results)

    # Add DAY number (1–5) per ATM
    all_predictions["DAY"] = all_predictions.groupby("ATM").cumcount() + 1

    # Rename predicted column to match optimizer input
    all_predictions["WITHDRWLS"] = all_predictions["Predicted_WITHDRWLS"]

    # Add ATM capacity column for optimizer (static for now)
    all_predictions["ATM capacity"] = default_atm_capacity

    # Reorder columns (optional)
    ordered_cols = ["ATM", "DATE", "DAY", "WITHDRWLS", "ATM capacity"]
    other_cols = [col for col in all_predictions.columns if col not in ordered_cols]
    all_predictions = all_predictions[ordered_cols + other_cols]

    # Save output file
    prediction_output = os.path.join(output_folder, "latest_prediction_output.csv")
    all_predictions.to_csv(prediction_output, index=False)

    # Update last prediction date
    latest_date = all_predictions["DATE"].max()
    with open(status_file, "w") as f:
        f.write(str(latest_date.date()))

    print(f"✅ Prediction completed up to {latest_date.date()}")
    print(f"📁 Prediction file saved to: {prediction_output}")
    print(f"📈 Forecast plots saved in: {plot_folder}")
else:
    print("🚫 No new prediction could be made (possibly at end of data).")
