import json
import pandas as pd
import sys # Import sys to exit on error

# --- 1. Load Labels ---
try:
    df = pd.read_csv(r'C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS\gaia_resplit.csv')
except FileNotFoundError:
    print("Error: 'gaia_resplit.csv' not found.")
    print("Please make sure it's in the same directory as test.py")
    sys.exit() # Stop the script

# --- 2. Load Metric Data ---
try:
    # Use 'with' to safely open the file
    with open(r'C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS\anomalies\demo_metric.json') as f:
        m = json.load(f)
except FileNotFoundError:
    print("Error: 'demo_metric.json' not found.")
    print("Please make sure it's in the same directory as test.py")
    sys.exit() # Stop the script
except json.JSONDecodeError:
    print("Error: 'demo_metric.json' is corrupted or not a valid JSON file.")
    sys.exit() # Stop the script


# --- 3. Print Row and Keys (as before) ---
first_row = df.iloc[0]
print(first_row) 
keys_to_check = ['st_time', 'ed_time']
print(keys_to_check)
print("---------------------------------") # Added for clarity

# --- 4. Get Case ID and Check Metric ---

# Get the case_id from the first row.
# first_row['case_id'] is 0 (int)
# The json keys are strings, so convert to string
k = str(first_row['case_id']) # k is now '0'

print(f"Checking data for case_id '{k}' in demo_metric.json...")

# --- NEW, more detailed check ---
if k not in m:
    print(f"  (Error: Key '{k}' was NOT found in demo_metric.json)")
    
elif not m[k]:
    # This catches if m[k] is an empty list: []
    print(f"  (Skipping: Data for key '{k}' is an empty list.)")
    print(f"    Actual data found: {m[k]}") # This will likely print []

elif not m[k][0]:
    # This catches if m[k] is a list with an empty list inside: [[]]
    print(f"  (Skipping: Data for key '{k}' is a nested empty list.)")
    print(f"    Actual data found: {m[k]}") # This will likely print [[]]
    
else:
    # If we get here, the data exists, so m[k][0][0] is safe
    print(f"  Success! Sample metric timestamp (for key '{k}'):", m[k][0][0])