# tools/split_stratification_logs.py
import argparse
import numpy as np
import os

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to stratification_logs.npy")
    ap.add_argument("--outdir", required=True, help="Directory to save train/test files")
    ap.add_argument("--split", type=float, default=0.8, help="Train split ratio (e.g., 0.8 for 80%)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading logs from: {args.input}")
    
    # Load the .npy file, which is an array (list) of lists
    logs = np.load(args.input, allow_pickle=True)

    print(f"Total cases found: {len(logs)}")

    train_logs = []
    test_logs = []

    # Iterate through the array. 
    # Each 'log_list' is a list of log records for a single case
    for log_list in logs:
        if not isinstance(log_list, (list, np.ndarray)):
            continue # Skip if the item isn't a list (e.g., empty cases)
        
        # Calculate the split point for this case's logs
        split_idx = int(len(log_list) * args.split)
        
        # Add the first 80% to train_logs
        train_logs.extend(log_list[:split_idx])
        # Add the last 20% to test_logs
        test_logs.extend(log_list[split_idx:])

    print(f"Total log records found: {len(train_logs) + len(test_logs)}")
    print(f" - Training records: {len(train_logs)}")
    print(f" - Testing records:  {len(test_logs)}")

    train_txt_path = os.path.join(args.outdir, "train_logs.txt")
    test_txt_path = os.path.join(args.outdir, "test_logs.txt")

    # Write the training file
    # ... (rest of the script) ...

# Write the training file
with open(train_txt_path, 'w', encoding='utf-8') as f_train:
    # This loop MUST be INDENTED inside the 'with' block
    for record in train_logs:
        timestamp, service, event_id = record
        f_train.write(f"{event_id}\n")

# This block should be separate and also have its loop indented
with open(test_txt_path, 'w', encoding='utf-8') as f_test:
    # This loop MUST be INDENTED inside this 'with' block
    for record in test_logs:
        timestamp, service, event_id = record
        f_test.write(f"{event_id}\n")

print(f"✅ Saved train_logs.txt and test_logs.txt")
