import pandas as pd
import matplotlib.pyplot as plt
import os
from pulp import *

# ----------------- Load Weekly Data -----------------
weekly_file = "Example prediction 5 days, 5 ATMs, one week.csv"  # Change this to your uploaded file
df = pd.read_csv(weekly_file)

# Sort data
df = df.sort_values(by=["ATM", "DATE"])

# Determine ATM count and number of days (should be 5 per week)
M = df["ATM"].nunique()
T = int(len(df) / M)

# Withdrawals per ATM x Day
w = df.pivot(index="ATM", columns="DAY", values="WITHDRWLS").values.tolist()

# ATM capacity
C = df.groupby("ATM")["ATM capacity"].first().tolist()

# Initial stock (If previous file exists, take last week's ending stock)
stock_file = "last_stock_levels.csv"
if os.path.exists(stock_file):
    s0_df = pd.read_csv(stock_file)
    s0 = s0_df["Stock"].tolist()
else:
    s0 = [5000000] * M  # default for first run

# Daily refill budget (can be dynamic if desired)
B = [25000000] * T

# ----------------- Parameters -----------------
refill_cost_per_unit = 0.01
penalty_out_of_cash = 10000
N = max(C) * 2
epsilon = 1e-3

# ----------------- Optimization Model -----------------
model = LpProblem("ATM_Refill_Optimization", LpMinimize)

x = [[LpVariable(f"x_{m}_{t}", lowBound=0, cat='Continuous') for t in range(T)] for m in range(M)]
y = [[LpVariable(f"y_{m}_{t}", cat='Binary') for t in range(T)] for m in range(M)]
s = [[LpVariable(f"s_{m}_{t}", lowBound=0, upBound=C[m], cat='Continuous') for t in range(T)] for m in range(M)]
z = [[LpVariable(f"z_{m}_{t}", cat='Binary') for t in range(T)] for m in range(M)]

model += lpSum(refill_cost_per_unit * x[m][t] + penalty_out_of_cash * z[m][t] for m in range(M) for t in range(T))

for m in range(M):
    model += lpSum(y[m][t] for t in range(T)) <= 3
    for t in range(T):
        if t == 0:
            model += s[m][t] == s0[m] + x[m][t] - w[m][t]
        else:
            model += s[m][t] == s[m][t - 1] + x[m][t] - w[m][t]

        model += x[m][t] <= N * y[m][t]
        model += x[m][t] >= epsilon * y[m][t]
        model += s[m][t] <= C[m]
        model += s[m][t] >= 0 - N * z[m][t]
        model += s[m][t] <= 0 + N * (1 - z[m][t])

for t in range(T):
    model += lpSum(x[m][t] for m in range(M)) <= B[t]

# ----------------- Solve -----------------
status = model.solve(PULP_CBC_CMD(msg=1))
print(f"Solver status: {LpStatus[status]}")
if LpStatus[status] != "Optimal":
    print("Warning: Problem is not optimal or feasible!")

# ----------------- Collect Results -----------------
refill_data = []
total_cost_all = 0
for m in range(M):
    total_cost = 0
    for t in range(T):
        refill_amount = value(x[m][t])
        stock_amount = value(s[m][t])
        refilled_binary = int(value(y[m][t]))
        out_of_cash_binary = int(value(z[m][t]))
        day_cost = refill_cost_per_unit * refill_amount + penalty_out_of_cash * out_of_cash_binary
        total_cost += day_cost
        refill_data.append({
            "ATM": m + 1,
            "Day": t + 1,
            "Refill_Amount": refill_amount,
            "Stock": stock_amount,
            "Refilled": refilled_binary,
            "Out_of_Cash": out_of_cash_binary,
            "Day_Cost": day_cost
        })
    total_cost_all += total_cost

print(f"Total system-wide cost this week: {total_cost_all:,.2f}")

# ----------------- Append to Results -----------------
df_refill = pd.DataFrame(refill_data)

# Shift Day column to continue from previous file
output_file = "atm_refill_results_total.csv"
if os.path.exists(output_file):
    prev = pd.read_csv(output_file)
    last_day = prev["Day"].max()
    df_refill["Day"] += last_day
    df_refill = pd.concat([prev, df_refill], ignore_index=True)

# Save updated CSV
df_refill.to_csv(output_file, index=False)
print(f"Results appended to {output_file}")

# Save readable version
with open("atm_refill_results_readable.txt", "w", encoding="utf-8") as f:
    f.write(df_refill.to_string(index=False, float_format='{:,.0f}'.format))
print("Readable results saved to atm_refill_results_readable.txt")

# Save last stock values for next week's initial stock
last_stock = df_refill.groupby("ATM").last().reset_index()[["ATM", "Stock"]]
last_stock.to_csv(stock_file, index=False)

# ----------------- Plotting -----------------
for m in range(M):
    atm_data = df_refill[df_refill["ATM"] == m + 1]
    days = atm_data["Day"]
    refill_amt = atm_data["Refill_Amount"]
    stock_amt = atm_data["Stock"]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_title(f"ATM {m + 1} - Refill Amount and Stock Over Time")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Refill Amount", color='tab:blue')
    ax1.bar(days, refill_amt, color='tab:blue', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Stock Level", color='tab:orange')
    ax2.plot(days, stock_amt, color='tab:orange', marker='o')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f"atm_{m + 1}_refill_plot_all_weeks.png")
    plt.close()

