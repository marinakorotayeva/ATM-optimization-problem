import pandas as pd
import matplotlib.pyplot as plt
import os
from pulp import *
from datetime import timedelta

# ----------------- Parameters -----------------
refill_cost_per_unit = 0.01
penalty_out_of_cash = 10000
daily_budget = 25_000_000
default_atm_capacity = 20_000_000  
default_initial_stock = 500_000   
max_refills_per_week = 3

# ----------------- Load Data -----------------
weekly_file = "Prediction models/Linear Regression/predictions/predictions.csv"
df = pd.read_csv(weekly_file)

# Convert DATE to datetime and sort
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values(by=["ATM_NUMBER", "DATE"])

# Get all unique weeks in the data
df['Week_Start'] = df['DATE'] - pd.to_timedelta(df['DATE'].dt.dayofweek, unit='d')
unique_weeks = sorted(df['Week_Start'].unique())

# Initialize stock levels
stock_file = "Optimization model/last_stock_levels.csv"
if os.path.exists(stock_file):
    s0_df = pd.read_csv(stock_file)
    initial_stocks = s0_df.set_index('ATM')['Stock'].to_dict()
else:
    initial_stocks = {atm: default_initial_stock for atm in df['ATM_NUMBER'].unique()}

# Prepare to collect all results
all_results = []

# ----------------- Process Each Week -----------------
for week_start in unique_weeks:
    week_end = week_start + timedelta(days=6)
    print(f"\nProcessing week from {week_start.date()} to {week_end.date()}")
    
    # Get all ATMs that have at least one record in this week
    week_atms = df[(df['DATE'] >= week_start) & (df['DATE'] <= week_end)]['ATM_NUMBER'].unique()
    M = len(week_atms)
    T = 7  # Days in a week
    
    # Create a complete date range for the week
    full_date_range = pd.date_range(start=week_start, end=week_end)
    
    # Prepare withdrawals matrix with zeros as default
    w = [[0]*T for _ in range(M)]
    
    # Fill in actual predicted values where available
    for m, atm in enumerate(week_atms):
        atm_data = df[(df['ATM_NUMBER'] == atm) & 
                     (df['DATE'] >= week_start) & 
                     (df['DATE'] <= week_end)]
        
        for _, row in atm_data.iterrows():
            day_index = (row['DATE'] - week_start).days
            w[m][day_index] = row['Predicted']
    
    # ATM capacities - use default 
    C = [default_atm_capacity] * M  
    
    # Initial stocks for this week
    s0 = [initial_stocks.get(atm, default_initial_stock) for atm in week_atms]
    
    # Daily budget
    B = [daily_budget] * T
    
    # ----------------- Optimization Model -----------------
    model = LpProblem(f"ATM_Refill_Optimization_{week_start.date()}", LpMinimize)
    
    # Decision variables
    N = max(C) * 2  # Large number for big-M constraints
    epsilon = 1e-3   # Small number for minimum refill amount
    
    x = [[LpVariable(f"x_{m}_{t}", lowBound=0, cat='Continuous') for t in range(T)] for m in range(M)]
    y = [[LpVariable(f"y_{m}_{t}", cat='Binary') for t in range(T)] for m in range(M)]
    s = [[LpVariable(f"s_{m}_{t}", lowBound=0, upBound=C[m], cat='Continuous') for t in range(T)] for m in range(M)]
    z = [[LpVariable(f"z_{m}_{t}", cat='Binary') for t in range(T)] for m in range(M)]
    
    # Objective function
    model += lpSum(refill_cost_per_unit * x[m][t] + penalty_out_of_cash * z[m][t]
                   for m in range(M) for t in range(T))
    
    # Constraints
    for m in range(M):
        # Limit refills per week
        model += lpSum(y[m][t] for t in range(T)) <= max_refills_per_week
        
        for t in range(T):
            withdrawal = w[m][t]
            
            # Stock balance constraint
            if t == 0:
                model += s[m][t] == s0[m] + x[m][t] - withdrawal
            else:
                model += s[m][t] == s[m][t-1] + x[m][t] - withdrawal
            
            # Refill amount constraints
            model += x[m][t] <= N * y[m][t]
            model += x[m][t] >= epsilon * y[m][t]
            
            # Stock capacity constraints
            model += s[m][t] <= C[m]
            
            # Out-of-cash indicator constraints
            model += s[m][t] >= -N * z[m][t]
            model += s[m][t] <= N * (1 - z[m][t])
    
    # Daily budget constraints
    for t in range(T):
        model += lpSum(x[m][t] for m in range(M)) <= B[t]
    
    # ----------------- Solve -----------------
    status = model.solve(PULP_CBC_CMD(msg=1))
    print(f"Solver status: {LpStatus[status]}")
    if LpStatus[status] != "Optimal":
        print(f"Warning: Problem for week {week_start.date()} not optimal or feasible!")
        # Still proceed but use zero values for this week
        for m in range(M):
            for t in range(T):
                x[m][t] = 0
                y[m][t] = 0
                s[m][t] = s0[m] if t == 0 else (s[m][t-1] if t > 0 else s0[m])
                z[m][t] = 0
    
    # ----------------- Collect Results -----------------
    week_cost = 0
    
    for m in range(M):
        atm_number = week_atms[m]
        atm_cost = 0
        
        for t in range(T):
            current_date = week_start + timedelta(days=t)
            refill_amount = value(x[m][t])
            stock_amount = value(s[m][t])
            refilled_binary = int(value(y[m][t]))
            out_of_cash_binary = int(value(z[m][t]))
            day_cost = refill_cost_per_unit * refill_amount + penalty_out_of_cash * out_of_cash_binary
            atm_cost += day_cost
            
            all_results.append({
                "ATM": atm_number,
                "Date": current_date.strftime('%Y-%m-%d'),
                "Refill_Amount": refill_amount,
                "Stock": stock_amount,
                "Refilled": refilled_binary,
                "Out_of_Cash": out_of_cash_binary,
                "Day_Cost": day_cost,
                "Predicted_Withdrawal": w[m][t]
            })
        
        week_cost += atm_cost
        # Update initial stock for next week
        initial_stocks[atm_number] = value(s[m][T-1])
    
    print(f"Total cost for week {week_start.date()}: {week_cost:,.2f}")

# ----------------- Save Results -----------------
df_refill = pd.DataFrame(all_results)

# Save complete results
output_file = "Optimization model/atm_refill_results_total.csv"
df_refill.to_csv(output_file, index=False)
print(f"\nAll results saved to {output_file}")

# Save readable version
with open("Optimization model/atm_refill_results_readable.txt", "w", encoding="utf-8") as f:
    f.write(df_refill.to_string(index=False, float_format='{:,.0f}'.format))
print("Readable results saved to atm_refill_results_readable.txt")

# Save final stock levels for next run
last_stock = pd.DataFrame.from_dict(initial_stocks, orient='index', columns=['Stock'])
last_stock.reset_index(inplace=True)
last_stock.columns = ['ATM', 'Stock']
last_stock.to_csv(stock_file, index=False)
print(f"Saved final stock levels to {stock_file}")

# ----------------- Plotting -----------------
for atm_number in df['ATM_NUMBER'].unique():
    atm_data = df_refill[df_refill["ATM"] == atm_number]
    if len(atm_data) == 0:
        continue
        
    dates = pd.to_datetime(atm_data["Date"])
    refill_amt = atm_data["Refill_Amount"]
    stock_amt = atm_data["Stock"]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_title(f"ATM {atm_number} - Refill Amount and Stock Over Time")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Refill Amount", color='tab:blue')
    ax1.bar(dates, refill_amt, color='tab:blue', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Stock Level", color='tab:orange')
    ax2.plot(dates, stock_amt, color='tab:orange', marker='o')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f"Optimization model/atm_{atm_number}_refill_plot.png", bbox_inches='tight')
    plt.close()

print("\nOptimization completed for all weeks!")
