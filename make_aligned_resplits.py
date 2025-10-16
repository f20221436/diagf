# make_aligned_resplit.py
import os, pickle, pandas as pd, numpy as np

# paths (adjust if you want)
xs_path = r"C:/Users/DEVESH PALO/projects/Diagf/diagf/data/gaia/demo/demo_1100/anomalies/sentence_embedding.pkl"
# If you prefer to overwrite the original, change out_path to the original path after backing it up
out_path = r"C:/Users/DEVESH PALO/projects/Diagf/diagf/data/gaia/demo/demo_1100/gaia_resplit_aligned.csv"

# split parameters
train_frac = 0.8   # 80% train, 20% test — change if you want
seed = 2

# load the metadata dictionary
with open(xs_path, "rb") as f:
    meta_data = pickle.load(f)

# --- THE FIX IS HERE ---
# Instead of getting the length of the object, we get the value from the 'total_items' key.
n = meta_data['total_items']
print("Loaded metadata. Total items to generate:", n)

# Create minimal run_table aligned to embeddings.
# We'll create an index and a data_type column; you can add other columns later if needed.
np.random.seed(seed)
perm = np.random.permutation(n)
n_train = int(n * train_frac)
train_idx = set(perm[:n_train])

rows = []
for i in range(n):
    dtype = 'train' if i in train_idx else 'test'
    # FIX: Fill instance and anomaly_type with actual values
    rows.append({
        "service": "unknown",  # or leave empty if preferred
        "instance": str(i),    # <- CRITICAL: Use index as instance ID
        "message": "", 
        "anomaly_type": "normal",  # <- CRITICAL: Provide default anomaly type
        "st_time": "", "ed_time": "", "duration": "", 
        "data_type": dtype, 
        "source_file": ""
    })

df = pd.DataFrame(rows)
df.index.name = 'index'
df.to_csv(out_path, index=False)
print("Wrote aligned run_table to:", out_path)
print("train count:", (df['data_type']=='train').sum(), "test count:", (df['data_type']=='test').sum())