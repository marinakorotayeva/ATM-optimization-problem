import  pandas as pd 
import numpy as np



# Load your Excel file
file_path = 'data1.csv'  # or 'data.csv' if it's CSV
df = pd.read_csv(file_path)

# Clean numeric columns (convert European number formats)
df['WITHDRWLS'] = df['WITHDRWLS'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
df['Avg daily withdrawal per ATM'] = df['Avg daily withdrawal per ATM'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
df['Fixed cost per refill'] = df['Fixed cost per refill'].astype(str).str.replace(',', '.', regex=False).astype(float)
df['Daily payment for the insurance'] = df['Daily payment for the insurance'].astype(str).str.replace(',', '.', regex=False).astype(float)

# Constants
annual_interest_rate = 0.05  # 5% per year
daily_interest_rate = annual_interest_rate / 365

# Rename for clarity
df['T'] = df['Avg daily withdrawal per ATM']
df['b'] = df['Fixed cost per refill']
df['i'] = daily_interest_rate
df['insurance_cost'] = df['Daily payment for the insurance']

# Apply Baumol model: Optimal refill size C*
df['C_opt'] = np.sqrt((2 * df['b'] * df['T']) / df['i'])

# Number of refills per day
df['n_refills'] = df['T'] / df['C_opt']

# Holding cost: (C / 2) * i
df['holding_cost'] = (df['C_opt'] / 2) * df['i']

# Replenishment cost: n_refills * b
df['replenishment_cost'] = df['n_refills'] * df['b']

# Total cost
df['total_cost'] = df['holding_cost'] + df['replenishment_cost'] + df['insurance_cost']

# Save results to new Excel file
output_columns = [
    'CASHP_ID', 'DATE', 'T', 'C_opt', 'n_refills',
    'holding_cost', 'replenishment_cost', 'insurance_cost', 'total_cost'
]

df[output_columns].to_excel('atm_cost_optimization_results.xlsx', index=False)

print("✅ Optimization results saved to 'atm_cost_optimization_results.xlsx'")
