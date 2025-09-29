# tools/create_minimal_stratification_texts.py
import os, sys, argparse, pickle
import pandas as pd

# Attempt to import repo helpers to read config. If not available, fallback to a default path.
try:
    import public_function as pf
    use_pf = True
except Exception:
    use_pf = False

def main(labels_pkl, out_path=None, node_name=None, placeholder_mode='caseid'):
    # load labels DataFrame
    if not os.path.exists(labels_pkl):
        raise FileNotFoundError(labels_pkl)
    labels = pd.read_pickle(labels_pkl)
    # choose node name
    if node_name is None:
        # try to read from config if available
        node_name = 'n1'
        if use_pf:
            try:
                cfg = pf.get_config()
                ft = pf.deal_config(cfg, 'fasttext')
                if 'nodes' in ft and isinstance(ft['nodes'], str):
                    node_name = ft['nodes'].split()[0]
            except Exception:
                pass

    # Build minimal stratification dict
    # expected repo shape: data[case_id] -> dict keyed by node_info where node_info[0]=node, node_info[1]=anomaly
    data = {}
    for cid, row in labels.iterrows():
        cid_s = str(cid)
        anomaly = row.get('anomaly_type', '[normal]')
        key = (node_name, anomaly)
        # placeholder tokens: either the case_id token or a simple fixed sequence
        if placeholder_mode == 'caseid':
            seq = f"case_{cid_s}"
        else:
            seq = "none_event"
        data[cid_s] = { key: seq }

    # decide output path: if out_path provided use it, else attempt to get from config or use default
    if out_path is None:
        if use_pf:
            cfg = pf.get_config()
            ft = pf.deal_config(cfg, 'fasttext')
            out_path = ft.get('text_path')  # expected to be the stratification_texts.pkl path
        if out_path is None:
            out_path = os.path.join('data','gaia','parse','stratification_texts.pkl')
    # ensure directory exists
    od = os.path.dirname(out_path)
    if od and not os.path.exists(od):
        os.makedirs(od, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    print("Wrote minimal stratification_texts.pkl ->", out_path)
    print("cases written:", len(data))
    print("example key (first case):", next(iter(data.items())))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', required=True, help='path to labels.pkl')
    ap.add_argument('--out', default=None, help='optional output path for stratification_texts.pkl (overrides config)')
    ap.add_argument('--node', default=None, help='node name to use (default from config or n1)')
    ap.add_argument('--placeholder', choices=['caseid','none'], default='caseid',
                    help='what tokens to use as placeholder text; caseid will create "case_<id>" token')
    args = ap.parse_args()
    main(args.labels, args.out, args.node, args.placeholder)
