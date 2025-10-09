# make_aligned_resplit.py
import os, pickle, pandas as pd, numpy as np

# paths (adjust if you want)
xs_path = r"C:/Users/DEVESH PALO/projects/Diagf/diagf/data/gaia/demo/demo_1100/anomalies/sentence_embedding.pkl"
# If you prefer to overwrite the original, change out_path to the original path after backing it up
out_path = r"C:/Users/DEVESH PALO/projects/Diagf/diagf/data/gaia/demo/demo_1100/gaia_resplit_aligned.csv"

# split parameters
train_frac = 0.8   # 80% train, 20% test — change if you want
seed = 2

# load embeddings to know length
with open(xs_path, "rb") as f:
    Xs = pickle.load(f)
n = len(Xs)
print("Loaded sentence_embedding.pkl length:", n)

# Create minimal run_table aligned to embeddings.
# We'll create an index and a data_type column; you can add other columns later if needed.
np.random.seed(seed)
perm = np.random.permutation(n)
n_train = int(n * train_frac)
train_idx = set(perm[:n_train])

rows = []
for i in range(n):
    dtype = 'train' if i in train_idx else 'test'
    # keep placeholders for other required columns if code expects them
    rows.append({"service": "", "instance": "", "message": "", "anomaly_type": "", 
                 "st_time": "", "ed_time": "", "duration": "", "data_type": dtype, 
                 "source_file": ""})

df = pd.DataFrame(rows)
df.index.name = 'index'
df.to_csv(out_path)
print("Wrote aligned run_table to:", out_path)
print("train count:", (df['data_type']=='train').sum(), "test count:", (df['data_type']=='test').sum())
