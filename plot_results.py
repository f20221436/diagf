# plot_results.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  # seaborn optional — easy confusion matrix display

# ------------ CONFIG -------------
RESULT_DIR = "dgl_processed_data/9"   # change if your extracted folder is elsewhere
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# helper loaders
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def try_read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Could not read CSV {path}: {e}")
        return None

# ------------ 1) Label distributions -------------
def plot_label_distribution(train_pkl, test_pkl, label_name):
    train = load_pickle(train_pkl)
    test = load_pickle(test_pkl)
    train = np.asarray(train)
    test = np.asarray(test)
    unique_train, counts_train = np.unique(train, return_counts=True)
    unique_test, counts_test = np.unique(test, return_counts=True)

    df = pd.DataFrame({
        'label': unique_train,
        'train_count': counts_train
    }).set_index('label')

    # merge test counts
    test_map = dict(zip(unique_test, counts_test))
    df['test_count'] = [test_map.get(l, 0) for l in df.index]

    # Plot
    ax = df.sort_index().plot(kind='bar', figsize=(12,6))
    ax.set_title(f"Label counts for '{label_name}' (train vs test)")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{label_name}_label_distribution.png")
    plt.savefig(out)
    print(f"Saved {out}")
    plt.close()

# ------------ 2) Anomaly predictions: confusion matrix & classification report -------------
def analyze_anomaly_preds(pred_csv):
    df = try_read_csv(pred_csv)
    if df is None:
        print("No anomaly predictions CSV found.")
        return
    # Expect columns like ['Pred', 'GroundTruth'] or first column is Pred and last is GroundTruth
    if 'Pred' in df.columns and 'GroundTruth' in df.columns:
        y_pred = df['Pred'].to_numpy().astype(int)
        y_true = df['GroundTruth'].to_numpy().astype(int)
    else:
        # Try fallback: last column ground truth, first prediction
        y_pred = df.iloc[:,0].to_numpy()
        y_true = df.iloc[:,-1].to_numpy()
        try:
            y_pred = y_pred.astype(int); y_true = y_true.astype(int)
        except:
            print("Could not coerce pred/gt to int.")
    if len(y_true) == 0:
        print("Anomaly predictions CSV contains no rows.")
        return

    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # plot cm
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("GroundTruth")
    plt.title("Anomaly-type Confusion Matrix")
    out = os.path.join(OUT_DIR, "anomaly_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(out)
    print(f"Saved {out}")
    plt.close()

    # classification report
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    rep_df = pd.DataFrame(rep).transpose()
    rep_df.to_csv(os.path.join(OUT_DIR, "anomaly_classification_report.csv"))
    print("Saved classification report to anomaly_classification_report.csv")
    print(rep_df)

# ------------ 3) Top-k accuracy CSVs (plot) -------------
def plot_topk_accuracy(acc_csv, title_prefix):
    df = try_read_csv(acc_csv)
    if df is None:
        print(f"No {acc_csv}")
        return
    # expect columns (top_k, accuracy) or similar
    if set(['top_k', 'accuracy']).issubset(df.columns):
        x = df['top_k']
        y = df['accuracy']
    else:
        # fallback: use first two columns
        x = df.iloc[:,0]
        y = df.iloc[:,1]
    # drop NaN
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        print(f"{acc_csv} contains only NaNs.")
        return
    plt.figure(figsize=(6,4))
    plt.plot(x[mask], y[mask], marker='o')
    plt.xlabel("top_k")
    plt.ylabel("accuracy")
    plt.title(f"{title_prefix} Top-K Accuracy")
    plt.grid(True)
    out = os.path.join(OUT_DIR, f"{title_prefix.lower().replace(' ','_')}_topk_accuracy.png")
    plt.savefig(out)
    print(f"Saved {out}")
    plt.close()

# ------------ run everything -------------
if __name__ == "__main__":
    # change paths below if needed
    train_a = os.path.join(RESULT_DIR, "train_ys_anomaly_type.pkl")
    test_a  = os.path.join(RESULT_DIR, "test_ys_anomaly_type.pkl")
    train_s = os.path.join(RESULT_DIR, "train_ys_service.pkl")
    test_s  = os.path.join(RESULT_DIR, "test_ys_service.pkl")

    # Plot label distributions
    if os.path.exists(train_a) and os.path.exists(test_a):
        plot_label_distribution(train_a, test_a, "anomaly_type")
    else:
        print("anomaly label pickles missing; cannot plot.")

    if os.path.exists(train_s) and os.path.exists(test_s):
        plot_label_distribution(train_s, test_s, "service")
    else:
        print("service label pickles missing; cannot plot.")

    # Analyze anomaly preds
    analyze_anomaly_preds(os.path.join(RESULT_DIR, "preds", "multitask_seed42_anomaly_pred_multi_v0.csv"))

    # Plot top-k accuracy files
    plot_topk_accuracy(os.path.join(RESULT_DIR, "evaluations", "anomaly", "seed42_anomaly_acc_multi_v0.csv"), "Anomaly")
    plot_topk_accuracy(os.path.join(RESULT_DIR, "evaluations", "instance", "seed42_instance_acc_multi_v0.csv"), "Instance")

    print("Done. Check the 'plots' directory for output PNGs and the CSV report.")
