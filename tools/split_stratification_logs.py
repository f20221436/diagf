# tools/split_logs.py (Corrected Version)
import argparse
import numpy as np
import os

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--split", type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    logs = np.load(args.input, allow_pickle=True)
    if isinstance(logs, np.ndarray) and logs.shape == ():
        logs = logs.item()

    print(f"Total services: {len(logs)}")

    train_logs = {}
    test_logs = {}

    for key, log_list in logs.items():
        split_idx = int(len(log_list) * args.split)
        train_logs[key] = log_list[:split_idx]
        test_logs[key] = log_list[split_idx:]

    train_txt_path = os.path.join(args.outdir, "train_logs.txt")
    test_txt_path = os.path.join(args.outdir, "test_logs.txt")

    # --- START OF FIX ---
    # We now unpack the tuple and write only the event_id.
    # The 'EventId' is the second item in the tuple from the .npy file.
    with open(train_txt_path, 'w', encoding='utf-8') as f_train:
        for key, log_list in train_logs.items():
            for timestamp, event_id, service_name in log_list:
                # Write just the event ID for each line
                f_train.write(f"{event_id}\n")

    with open(test_txt_path, 'w', encoding='utf-8') as f_test:
        for key, log_list in test_logs.items():
            for timestamp, event_id, service_name in log_list:
                # Write just the event ID for each line
                f_test.write(f"{event_id}\n")
    # --- END OF FIX ---

    print(f"✅ Saved train_logs.txt and test_logs.txt (services: {len(train_logs)})")