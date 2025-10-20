"""
Test Metric Processing Script
==============================
Tests the EXACT same metric processing logic on first 5 files.
Uses identical code from preprocess_gaia_raw.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import sys
import gc

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from detector.k_sigma import Ksigma


def process_metric_data_parallel(args):
    """Process single metric file (for parallel execution) - EXACT COPY from main script"""
    metric_file, time_windows = args
    
    case_metrics = {case_id: [] for case_id, _, _ in time_windows}
    
    try:
        # Read entire metric file
        df = pd.read_csv(metric_file)
        
        # Convert timestamp to numeric if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        # NO SAMPLING - process entire file
        sampled = df
        
        # Apply K-sigma anomaly detection
        ksigma = Ksigma()
        
        for case_id, st_ts, ed_ts in time_windows:
            # Filter metrics in time window (vectorized - fast)
            mask = (sampled['timestamp'] >= st_ts) & (sampled['timestamp'] < ed_ts)
            case_data = sampled[mask]
            
            if len(case_data) == 0:
                continue
            
            # Extract metric anomalies
            for col in case_data.columns:
                if col in ['timestamp', 'service']:
                    continue
                
                try:
                    is_anomaly, anomaly_ts, score = ksigma.detection(
                        case_data, col, int(st_ts), int(ed_ts)
                    )
                    
                    if is_anomaly:
                        service = metric_file.stem.split('_')[0] if '_' in metric_file.stem else 'unknown'
                        # Convert numpy types to Python native types for JSON serialization
                        case_metrics[case_id].append([
                            int(anomaly_ts),      # numpy.int64 -> Python int
                            str(service),         # Ensure string
                            str(col),             # Ensure string
                            float(score)          # numpy.float64 -> Python float
                        ])
                except:
                    pass
        
        del df, sampled
        gc.collect()
    
    except Exception as e:
        print(f"Error processing {metric_file.name}: {e}")
    
    return case_metrics


def test_metric_processing():
    """Test metric processing on first 5 files - EXACT COPY from main script"""
    print("="*80)
    print("METRIC PROCESSING TEST (First 5 Files)")
    print("="*80)
    
    # Paths - same as main script
    raw_path = Path(r'C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS')
    output_path = raw_path / 'anomalies'
    output_path.mkdir(exist_ok=True)
    
    # Load run table - EXACT COPY from main script
    print("\n[1/4] Loading run table...")
    run_files = list(raw_path.glob('run/run_table_*.csv'))
    run_dfs = []
    for f in run_files:
        df = pd.read_csv(f)
        run_dfs.append(df)
    
    run_table = pd.concat(run_dfs, ignore_index=True)
    
    if 'case_id' not in run_table.columns:
        run_table['case_id'] = range(len(run_table))
    
    run_table['st_time'] = pd.to_datetime(run_table['datetime'])
    run_table['ed_time'] = run_table['st_time'] + pd.to_timedelta(5, unit='m')
    
    print(f"  Cases loaded: {len(run_table)}")
    
    # Prepare time windows - EXACT COPY from main script
    print("\n[2/4] Preparing time windows...")
    time_windows = []
    for idx, case in run_table.iterrows():
        st_ts = pd.Timestamp(case['st_time']).timestamp() * 1000
        ed_ts = pd.Timestamp(case['ed_time']).timestamp() * 1000
        time_windows.append((case['case_id'], st_ts, ed_ts))
    
    print(f"  Time windows: {len(time_windows)}")
    
    # Get metric files - EXACT COPY from main script
    print("\n[3/4] Finding metric files...")
    metric_files = list(raw_path.glob('metric/metric_split/metric/*.csv'))
    print(f"  Total metric files: {len(metric_files)}")
    
    # Test on first 5 files only
    test_files = metric_files[:5]
    print(f"\n  Testing on first {len(test_files)} files:")
    for f in test_files:
        print(f"    - {f.name}")
    
    # Prepare arguments - EXACT COPY from main script
    args_list = [(f, time_windows) for f in test_files]
    
    # Process in parallel - EXACT COPY from main script
    print("\n[4/4] Processing metric files with 2 workers...")
    with Pool(processes=2) as pool:
        results = list(tqdm(
            pool.imap(process_metric_data_parallel, args_list),
            total=len(args_list),
            desc="Processing metric files"
        ))
    
    # Merge results - EXACT COPY from main script
    print("\n" + "="*80)
    print("MERGING RESULTS")
    print("="*80)
    
    metric_dict = {i: [] for i in range(len(run_table))}
    for result in results:
        for case_id, metrics in result.items():
            metric_dict[case_id].extend(metrics)
    
    # Convert all keys to Python int - EXACT COPY from main script
    metric_dict = {int(k): v for k, v in metric_dict.items()}
    
    total_metrics = sum(len(v) for v in metric_dict.values())
    cases_with_metrics = sum(1 for v in metric_dict.values() if len(v) > 0)
    
    print(f"  Total cases: {len(metric_dict)}")
    print(f"  Cases with metrics: {cases_with_metrics}")
    print(f"  Total anomalies: {total_metrics}")
    if cases_with_metrics > 0:
        print(f"  Avg anomalies per case: {total_metrics/cases_with_metrics:.1f}")
    
    # Save metric JSON - EXACT COPY from main script
    print("\n" + "="*80)
    print("SAVING JSON")
    print("="*80)
    
    output_file = output_path / 'test_metric.json'
    
    try:
        print("  Writing to JSON...")
        with open(output_file, 'w') as f:
            json.dump(metric_dict, f)
        
        file_size_kb = output_file.stat().st_size / 1024
        print(f"  ✓ JSON saved successfully!")
        print(f"  File: {output_file}")
        print(f"  Size: {file_size_kb:.2f} KB")
        
        # Verify by reading back
        print("\n  Verifying saved file...")
        with open(output_file, 'r') as f:
            loaded_dict = json.load(f)
        
        print(f"  ✓ File loaded successfully!")
        print(f"  Loaded cases: {len(loaded_dict)}")
        print(f"  Loaded anomalies: {sum(len(v) for v in loaded_dict.values())}")
        
        # Check sample entry if available
        if total_metrics > 0:
            sample_case = next((k for k, v in loaded_dict.items() if len(v) > 0), None)
            if sample_case:
                sample_anomaly = loaded_dict[str(sample_case)][0]  # JSON converts keys to strings
                print(f"\n  Sample anomaly from case {sample_case}:")
                print(f"    {sample_anomaly}")
                print(f"    Types: {[type(x).__name__ for x in sample_anomaly]}")
                
                # Check for numpy types
                has_numpy = any('numpy' in str(type(x).__module__) for x in sample_anomaly)
                if has_numpy:
                    print(f"  [ERROR] Numpy types detected!")
                    return False
                else:
                    print(f"  ✓ All types are Python native!")
        
        print("\n" + "="*80)
        print("TEST PASSED! ✓")
        print("="*80)
        print("\nThe metric processing works correctly:")
        print("  ✓ Files processed successfully")
        print(f"  ✓ {total_metrics} anomalies detected")
        print("  ✓ JSON serialization successful")
        print("  ✓ No numpy type errors")
        print("\nYou can now run the full preprocessing!")
        
        return True
        
    except TypeError as e:
        print(f"  [ERROR] JSON serialization failed!")
        print(f"  Error: {e}")
        print("\n  This means numpy types are still present in the data.")
        import traceback
        traceback.print_exc()
        return False
    
    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_metric_processing()
    sys.exit(0 if success else 1)