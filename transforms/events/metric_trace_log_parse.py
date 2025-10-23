import json
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys

# Note: 'public_function as pf' is imported later in the __main__ block

def metric_trace_log_parse(trace, metric, logs, labels, save_path, nodes, pf_module):
    """
    Parses and combines trace, metric, and log data into event sequences.
    """
    print('Processing metric, trace, and log data...')
    
    if not metric is None: # Remove np.inf values from metrics
        print('Cleaning metric data...')
        for k, v in metric.items():
            # Check for v being a list and x[3] being numeric
            metric[k] = [x for x in v if isinstance(x, (list, tuple)) and len(x) > 3 and isinstance(x[3], (int, float)) and not math.isinf(x[3])]

    if not logs is None:
        print('Processing log data...')
        logs = list(logs)
        log = {x: [] for x in labels.index}
        if not logs:
             print("[WARNING] Log data is present but empty.")
        elif labels.index.is_integer() and labels.index.max() < len(logs):
            # Use direct integer indexing if possible
            for k in log.keys():
                log[k] = logs[k]
        else:
            # Fallback to slower sequential count (safer)
            print("[INFO] Using sequential mapping for logs.")
            count = 0
            for k in log.keys():
                if count < len(logs):
                    log[k] = logs[count]
                    count += 1
                else:
                    break # Stop if we run out of logs

    service_name = nodes.split()
    anomaly_service = list(labels['instance'])
    anomaly_type = list(labels['anomaly_type'])

    demo_metric = {x: {} for x in labels.index}
    
    print(f'Parsing {len(demo_metric)} cases...')
    for case_id in tqdm(demo_metric.keys(), desc="Parsing events"):
        # Ensure we're accessing the correct row by index
        try:
            current_case = labels.loc[case_id]
        except KeyError:
            print(f"[Warning] Skipping case_id {case_id} not found in labels file.")
            continue
            
        anomaly_service_name = current_case['instance']
        anomaly_service_type = current_case['anomaly_type']
        
        inner_dict_key = [(x, anomaly_service_type) if x == anomaly_service_name else (x, "[normal]") for x in
                          service_name]
        
        # Metrics
        if not metric is None and str(case_id) in metric:
            demo_metric[case_id] = {x: [[y[0], "{}_{}_{}".format(y[1], y[2], "+" if y[3] > 0 else "-")] for y in metric[str(case_id)] if
                                    isinstance(y, (list, tuple)) and len(y) > 1 and y[1] == x[0]] for x in inner_dict_key}
        else:
            demo_metric[case_id] = {x : [] for x in inner_dict_key}
        
        # Traces
        if not trace is None and str(case_id) in trace:
            for inner_key in inner_dict_key:
                demo_metric[case_id][inner_key].extend(
                    [[y[0], "{}_{}".format(y[1], y[2])] for y in trace[str(case_id)] 
                     if isinstance(y, (list, tuple)) and len(y) > 2 and (y[1] == inner_key[0] or y[2] == inner_key[0])])
        
        # Logs
        if not logs is None and case_id in log:
            for inner_key in inner_dict_key:
                demo_metric[case_id][inner_key].extend([[y[0], y[2]] for y in log[case_id] 
                                                        if isinstance(y, (list, tuple)) and len(y) > 2 and y[1] == inner_key[0]])
        
        # Sort and join all events into a single string
        for inner_key in inner_dict_key:
            temp = demo_metric[case_id][inner_key]
            sort_list = sorted(temp, key=lambda x: x[0])
            temp_list = [x[1] for x in sort_list]
            demo_metric[case_id][inner_key] = ' '.join(temp_list)

    print('Saving parsed data...')
    pf_module.save(save_path, demo_metric)


def run_parse(config, labels, pf_module):
    """
    Loads all data files and runs the main parsing function.
    """
    trace = None
    metric = None
    logs = None
    
    if config['log_path'] and os.path.exists(config['log_path']):
        print(f"Loading logs from {config['log_path']}...")
        logs = np.load(config['log_path'], allow_pickle=True)
    else:
        print(f"[WARNING] Log file not found: {config['log_path']}. Skipping logs.")

    if config['metric_path'] and os.path.exists(config['metric_path']):
        print(f"Loading metrics from {config['metric_path']}...")
        with open(config['metric_path'], 'r', encoding='utf8') as fp:
            metric = json.load(fp)
    else:
        print(f"[WARNING] Metric file not found: {config['metric_path']}. Skipping metrics.")
        
    if config['trace_path'] and os.path.exists(config['trace_path']):
        print(f"Loading traces from {config['trace_path']}...")
        with open(config['trace_path'], 'r', encoding='utf8') as fp:
            trace = json.load(fp)
    else:
        print(f"[WARNING] Trace file not found: {config['trace_path']}. Skipping traces.")
        
    metric_trace_log_parse(trace, metric, logs, labels, config['save_path'], config['nodes'], pf_module)


# This block makes the script runnable from the command line
if __name__ == "__main__":
    
    # --- 1. Find the 'public_function.py' helper ---
    # We are in 'diagf/transforms/events', so we go up two levels to 'diagf'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    diagf_root_dir = os.path.dirname(os.path.dirname(current_dir))
    
    # Add the 'diagf' root directory to the Python path
    sys.path.insert(0, diagf_root_dir)
    
    try:
        import public_function as pf
    except ImportError:
        print(f"[ERROR] Could not find 'public_function.py'.")
        print(f"        Expected it to be in: {diagf_root_dir}")
        sys.exit(1)

    # --- 2. DEFINE ABSOLUTE PATHS to your data ---
    # This is the root folder where you saved all your preprocessed data
    data_root = r'C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS'
    anomalies_dir = os.path.join(data_root, 'anomalies')

    # --- 3. Configuration ---
    # Define paths for all the files
    log_path = os.path.join(anomalies_dir, 'stratification_logs.npy')
    metric_path = os.path.join(anomalies_dir, 'demo_metric.json')
    trace_path = os.path.join(anomalies_dir, 'demo_trace.json')
    save_path = os.path.join(anomalies_dir, 'metric_trace_text.pkl')
    labels_path = os.path.join(data_root, 'gaia_resplit.csv')
    
    # 4. Load the labels file (gaia_resplit.csv)
    try:
        labels_df = pd.read_csv(labels_path)
        if 'case_id' in labels_df.columns:
            labels_df = labels_df.set_index('case_id')
        print(f"Successfully loaded labels from {labels_path}")
    except FileNotFoundError:
        print(f"[ERROR] Could not find labels file: '{labels_path}'.")
        sys.exit(1)

    # 5. Dynamically get the list of nodes (services) from the labels file
    nodes_list = sorted(list(set(labels_df['service'])))
    nodes_string = ' '.join(nodes_list)
    print(f"Found {len(nodes_list)} nodes: {nodes_string}")

    # 6. Create the config dictionary
    config = {
        'log_path': log_path,
        'metric_path': metric_path,
        'trace_path': trace_path,
        'save_path': save_path,
        'nodes': nodes_string
    }

    # --- Run the Parser ---
    print("\nStarting parser...")
    run_parse(config, labels_df, pf)
    print(f"\n[SUCCESS] Successfully created: {save_path}")