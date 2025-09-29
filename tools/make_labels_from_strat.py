# tools/make_labels_from_strat.py
import os, json, numpy as np, pandas as pd, argparse, pickle

def load_traces_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_traces(tr):
    """
    Try to produce a dict: case_id -> dict of metadata (must include 'anomaly_type' or similar).
    This function is forgiving about traces.json structure.
    """
    # If it's a dict mapping id->meta:
    if isinstance(tr, dict):
        return tr
    # If it's a list of dict entries, try to index by 'case_id' or 'trace_id'
    out = {}
    if isinstance(tr, list):
        for item in tr:
            if isinstance(item, dict):
                key = item.get('case_id') or item.get('trace_id') or item.get('id') or item.get('name')
                if key is None:
                    # fallback to auto-increment
                    key = str(len(out))
                out[str(key)] = item
    return out

def extract_label(item):
    # Try common fields that indicate anomaly label
    for k in ('anomaly_type', 'label', 'anomaly', 'fault_type', 'fault'):
        if isinstance(item, dict) and k in item:
            return item[k]
    # fallback: if item has 'tags' or 'meta'
    if isinstance(item, dict):
        if 'meta' in item and isinstance(item['meta'], dict):
            for k in ('anomaly_type','label'):
                if k in item['meta']:
                    return item['meta'][k]
    return '[normal]'

def main(traces_path, strat_path, out_path):
    if not os.path.exists(traces_path):
        raise FileNotFoundError(traces_path)
    if not os.path.exists(strat_path):
        raise FileNotFoundError(strat_path)
    traces_raw = load_traces_json(traces_path)
    traces = normalize_traces(traces_raw)  # dict-like
    strat = np.load(strat_path, allow_pickle=True)
    # strat can be a dict, array, or structured array. Try to interpret:
    case_ids = None
    labels = None
    splits = None

    # If strat is dict with lists:
    if isinstance(strat, dict):
        # try keys ["case_ids", "split"] ...
        # use heuristics:
        possible_id_keys = [k for k in strat.keys() if 'id' in k.lower() or 'case' in k.lower()]
        if possible_id_keys:
            case_ids = list(strat[possible_id_keys[0]])
        # split flags
        possible_split_keys = [k for k in strat.keys() if 'split' in k.lower() or 'type' in k.lower()]
        if possible_split_keys:
            splits = list(strat[possible_split_keys[0]])
    elif hasattr(strat, 'dtype') and strat.dtype.names:
        # structured numpy array
        names = strat.dtype.names
        for n in names:
            if 'id' in n.lower() or 'case' in n.lower():
                case_ids = strat[n].tolist()
            if 'split' in n.lower() or 'type' in n.lower():
                splits = strat[n].tolist()
    else:
        # If it's a 1D list-like (maybe same order as traces), assume case ids are the keys of traces
        try:
            arr = list(strat)
            # if each element is a tuple (id, split)
            if arr and isinstance(arr[0], (list, tuple)) and len(arr[0]) >= 2:
                case_ids = [str(x[0]) for x in arr]
                splits = [x[1] for x in arr]
            else:
                # fallback: case ids = traces keys order
                case_ids = list(traces.keys())
                # decide splits as 0/1 depending on numeric values in arr
                splits = arr
        except Exception:
            case_ids = list(traces.keys())
            splits = [ 'train' ] * len(case_ids)

    # Build DataFrame
    rows = []
    for i, cid in enumerate(case_ids):
        cid_s = str(cid)
        item = traces.get(cid_s, {}) if isinstance(traces, dict) else {}
        anomaly = extract_label(item)
        # interpret split value
        sp = splits[i] if (splits is not None and i < len(splits)) else 'train'
        # convert common numeric split markers: 0/1 or 'train'/'test'
        if isinstance(sp, (int, np.integer)):
            data_type = 'train' if int(sp) == 0 else 'test'
        elif isinstance(sp, str):
            if sp.lower() in ('train','tr','0'):
                data_type = 'train'
            elif sp.lower() in ('test','te','1'):
                data_type = 'test'
            else:
                # unknown string -> treat as train
                data_type = 'train'
        else:
            data_type = 'train'
        rows.append({'case_id': cid_s, 'anomaly_type': anomaly, 'data_type': data_type})

    df = pd.DataFrame(rows).set_index('case_id')
    # Save pickle
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(df, f)
    print("Saved labels.pkl to", out_path, " shape:", df.shape)
    print(df['data_type'].value_counts())
    print("example anomaly types:", df['anomaly_type'].unique()[:10])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', default='data/gaia/traces.json')
    ap.add_argument('--strat', default='data/gaia/stratification_logs.npy')
    ap.add_argument('--out', default='data/gaia/labels.pkl')
    args = ap.parse_args()
    main(args.traces, args.strat, args.out)
