#!/usr/bin/env python3
"""
evaluate_or_infer_service.py

1) Looks for CSV preds for service / anomaly_type / instance in preds/
2) If service CSV missing, tries to infer service preds from instance preds
   using service_label_list.pkl (heuristic-safe).
3) Prints accuracy / precision / recall / f1 and classification reports.

Place under diagf/tools and run:
(venvDiagFusion) PS ...\diagf> python tools/evaluate_or_infer_service.py
"""
import os
import glob
import pickle
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

BASE_DIR = r"C:\Users\DEVESH PALO\projects\DiagFusionWorking\diagf\dgl_processed_data\9"
PRED_DIR = os.path.join(BASE_DIR, "preds")

# candidate filenames / patterns
SERVICE_LABEL_PKL = os.path.join(BASE_DIR, "service_label_list.pkl")
SERVICE_LABELS_ALTERNATIVE = os.path.join(BASE_DIR, "service_label_list.pickle")

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def find_csvs():
    if not os.path.isdir(PRED_DIR):
        return []
    return sorted(glob.glob(os.path.join(PRED_DIR, "*.csv")))

def find_label_pkl(kind):
    # looks for test_ys_{kind}.pkl
    pats = [f"**/*test*ys*{kind}*.pkl", f"**/*test_ys_{kind}*.pkl"]
    for pat in pats:
        matches = glob.glob(os.path.join(BASE_DIR, pat), recursive=True)
        if matches:
            return sorted(matches)[0]
    return None

def load_labels(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    # support dict(id->label), list-like, numpy array, pandas Series
    if isinstance(obj, dict):
        # try to preserve order if possible
        return list(obj.values()), list(obj.keys())
    if isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
        return list(obj), list(range(len(obj)))
    # fallback: try to iterate
    try:
        arr = list(obj)
        return arr, list(range(len(arr)))
    except Exception:
        raise ValueError("Unsupported label pkl format: {}".format(type(obj)))

def compute_metrics(y_true, y_pred, name):
    y_true = [str(x) for x in y_true]
    y_pred = [str(x) for x in y_pred]
    n = min(len(y_true), len(y_pred))
    if n == 0:
        print(f"[WARN] zero-length for {name}")
        return
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print("="*80)
    print(f"TASK: {name}  samples={n}")
    print(f"Accuracy: {acc:.4f}  Precision(macro):{p:.4f}  Recall(macro):{r:.4f}  F1(macro):{f1:.4f}\n")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("="*80 + "\n")

def pick_pred_column(df, kind):
    # prefer explicit columns
    cname_candidates = {
        "service": ["service", "svc", "service_pred", "pred_service"],
        "anomaly_type": ["anomaly", "anomaly_type", "anomaly_pred"],
        "instance": ["instance", "instance_pred", "pred_instance", "pred"]
    }
    cols_lower = {c.lower(): c for c in df.columns}
    for token in cname_candidates[kind]:
        for lower, orig in cols_lower.items():
            if token in lower:
                return orig
    # fallback heuristics:
    if "Pred" in df.columns:
        return "Pred"
    if df.shape[1] == 1:
        return df.columns[0]
    # if two columns and second looks like a label column use second
    if df.shape[1] >= 2:
        return df.columns[-1]
    return None

def build_instance2service_map():
    """
    Try to build a mapping instance_label -> service_label using service_label_list.pkl.
    Heuristics:
    - If pickle is a dict {service_label: [instance_indices...]}, invert it.
    - If pickle is a list where index==service_index and values are instance lists, invert.
    - If pickle is a list/array mapping instance_index -> service_index, use direct mapping.
    """
    p = SERVICE_LABEL_PKL if os.path.exists(SERVICE_LABEL_PKL) else (SERVICE_LABELS_ALTERNATIVE if os.path.exists(SERVICE_LABELS_ALTERNATIVE) else None)
    if p is None:
        return None, "service_label_list.pkl not found"
    try:
        obj = load_pickle(p)
    except Exception as e:
        return None, f"Failed to load {p}: {e}"

    # Case A: dict service->list(instance ids)
    if isinstance(obj, dict):
        inst2svc = {}
        for svc, inst_list in obj.items():
            for inst in inst_list:
                inst2svc[int(inst)] = svc
        return inst2svc, f"mapped from dict in {p}"

    # Case B: list of lists: each element is list of instance ids for a service index
    if isinstance(obj, (list, tuple)):
        # If elements are lists/tuples of instance ids
        if len(obj) > 0 and isinstance(obj[0], (list, tuple)):
            inst2svc = {}
            for svc_idx, inst_list in enumerate(obj):
                for inst in inst_list:
                    inst2svc[int(inst)] = svc_idx
            return inst2svc, f"mapped from list-of-lists in {p}"
        # If obj is a list mapping instance_index -> service_index (len == num_instances)
        if all(not isinstance(x, (list, tuple, dict)) for x in obj):
            # each index maps to a service label
            inst2svc = {idx: obj[idx] for idx in range(len(obj))}
            return inst2svc, f"mapped from list(index->service) in {p}"

    return None, f"Unhandled format for {p}: {type(obj)}"

def main():
    print("Base dir:", BASE_DIR)
    csvs = find_csvs()
    print("CSV files found in preds/:")
    for c in csvs:
        print(" ", os.path.basename(c))
    print()

    # find labels
    label_service = find_label_pkl("service")
    label_anom = find_label_pkl("anomaly_type")
    label_inst = find_label_pkl("instance")

    # load any preds available
    csv_map = {}
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
            csv_map[csv] = df
        except Exception as e:
            print("Failed to read", csv, e)

    # Evaluate anomaly_type if possible (choose CSV and column)
    if label_anom and csv_map:
        # try to prefer CSVs that have 'anomaly' in filename or columns
        an_csv = None
        for csv, df in csv_map.items():
            if "anomaly" in os.path.basename(csv).lower():
                an_csv = csv; break
        if an_csv is None:
            an_csv = list(csv_map.keys())[0]
        col = pick_pred_column(csv_map[an_csv], "anomaly_type")
        print("Anomaly CSV:", an_csv, "col:", col)
        if col:
            labels, _ = load_labels(label_anom)
            preds = csv_map[an_csv][col].tolist()
            compute_metrics(labels, preds, "anomaly_type")
    else:
        print("No anomaly labels or CSVs to evaluate anomaly.")

    # Evaluate instance if possible
    if label_inst and csv_map:
        inst_csv = None
        for csv, df in csv_map.items():
            if "instance" in os.path.basename(csv).lower():
                inst_csv = csv; break
        if inst_csv is None:
            # fallback: if only one csv that is not 'anomaly' pick it
            for csv in csv_map.keys():
                if "anomaly" not in os.path.basename(csv).lower():
                    inst_csv = csv; break
        if inst_csv:
            col = pick_pred_column(csv_map[inst_csv], "instance")
            print("Instance CSV:", inst_csv, "col:", col)
            if col:
                labels, _ = load_labels(label_inst)
                preds = csv_map[inst_csv][col].tolist()
                compute_metrics(labels, preds, "instance")
    else:
        print("No instance labels or CSVs to evaluate instance.")

    # SERVICE: try to find a service CSV. If not present, attempt to infer from instance preds.
    service_csv = None
    for csv, df in csv_map.items():
        if "service" in os.path.basename(csv).lower() or "svc" in os.path.basename(csv).lower():
            service_csv = csv; break
        # also if column contains 'service'
        for c in df.columns:
            if "service" in c.lower() or "svc" in c.lower():
                service_csv = csv; break
        if service_csv:
            break

    if service_csv:
        col = pick_pred_column(csv_map[service_csv], "service")
        print("Service CSV found:", service_csv, "col:", col)
        if col and label_service:
            labels, _ = load_labels(label_service)
            preds = csv_map[service_csv][col].tolist()
            compute_metrics(labels, preds, "service")
        else:
            print("Service labels missing or could not pick column.")
            return

    else:
        print("No explicit service CSV found. Attempting to infer service preds from instance preds.")
        # need instance preds + service_label_list.pkl + service labels
        if label_service is None:
            print("[ERROR] service label PKL (test_ys_service.pkl) not found. Cannot compute service metrics.")
            return
        if not csv_map:
            print("[ERROR] No CSV preds found at all in preds/.")
            return
        # pick an instance CSV
        inst_csv = None
        for csv in csv_map:
            if "instance" in os.path.basename(csv).lower():
                inst_csv = csv; break
        if inst_csv is None:
            # fallback: take a csv that's not anomaly
            for csv in csv_map:
                if "anomaly" not in os.path.basename(csv).lower():
                    inst_csv = csv; break
        if inst_csv is None:
            print("[ERROR] No candidate instance CSV to infer service from.")
            return
        df_inst = csv_map[inst_csv]
        col_inst = pick_pred_column(df_inst, "instance")
        if col_inst is None:
            print("[ERROR] Could not pick instance pred column in", inst_csv)
            return
        instance_preds = df_inst[col_inst].tolist()
        # build mapping
        inst2svc_map, info = build_instance2service_map()
        if inst2svc_map is None:
            print("[ERROR] Could not build instance->service map:", info)
            print("Please re-run evaluation to save service predictions or provide a mapping file.")
            return
        print("Built instance->service map:", info)
        # map instance preds to service preds
        service_preds = []
        missing = 0
        for inst in instance_preds:
            try:
                inst_i = int(inst)
            except Exception:
                # maybe it's string representing int
                try:
                    inst_i = int(float(inst))
                except Exception:
                    inst_i = None
            if inst_i is None or inst_i not in inst2svc_map:
                service_preds.append("UNK")
                missing += 1
            else:
                service_preds.append(inst2svc_map[inst_i])
        if missing > 0:
            print(f"[WARN] {missing} instance preds could not be mapped to a service; they are marked 'UNK' and will lower metrics.")
        # compute metrics using service labels
        svc_labels, _ = load_labels(label_service)
        compute_metrics(svc_labels, service_preds, "service (inferred-from-instance)")

if __name__ == "__main__":
    main()
