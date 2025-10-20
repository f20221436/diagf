"""
GAIA Raw Dataset Preprocessing Pipeline
========================================
This script preprocesses raw MicroSS dataset from GAIA and converts it to DiagFusion format.

Processing Strategy:
- Trace data: 100% (all trace files)
- Metric data: 30% sampling (stratified by service and time)
- Business/Log data: 30% sampling (stratified by service and time)
- Multi-core processing: 2-4 CPU cores
- Memory-efficient: Chunked CSV reading for large files

Output Format:
- anomalies/demo_metric.json
- anomalies/demo_trace.json
- anomalies/stratification_logs.npy (processed business data)
- gaia_resplit.csv (fault injection cases)
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datetime import datetime
import time
import gc

# Import DiagFusion modules
from detector.k_sigma import Ksigma

# Import Drain log parser
try:
    from log.logparser.logparser.Drain.Drain import LogParser as DrainParser
except ImportError:
    print("[WARNING] Drain log parser not available. Will use raw log messages.")
    DrainParser = None


class GAIAPreprocessor:
    def __init__(self, raw_data_path, output_path, n_workers=2, metric_sample_rate=0.3):
        """
        Initialize GAIA preprocessor
        
        Args:
            raw_data_path: Path to MicroSS directory
            output_path: Path to save preprocessed data
            n_workers: Number of CPU cores to use (2-4)
            metric_sample_rate: Sampling rate for metric/business data (0.3 = 30%)
        """
        self.raw_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.n_workers = min(n_workers, cpu_count())
        self.sample_rate = metric_sample_rate
        
        # Create output directories
        self.anomalies_dir = self.output_path / 'anomalies'
        self.anomalies_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temp directory for Drain
        self.drain_temp_dir = self.output_path / 'drain_temp'
        self.drain_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Service names in MicroSS
        self.services = [
            'business', 'docker001', 'docker002', 'docker003', 'docker004',
            'docker005', 'docker006', 'docker007', 'docker008', 'docker009'
        ]
        
        print(f"[Init] Using {self.n_workers} CPU cores")
        print(f"[Init] Metric/Business sampling rate: {self.sample_rate*100}%")
        print(f"[Init] Trace data: 100% (no sampling)")
        print(f"[Init] Drain log parsing: {'Enabled' if DrainParser else 'Disabled (using raw messages)'}")
    
    def load_run_table(self):
        """Load fault injection metadata from run/ folder"""
        print("\n" + "="*80)
        print("STEP 1: Loading Fault Injection Metadata (run_table)")
        print("="*80)
        
        run_files = list(self.raw_path.glob('run/run_table_*.csv'))
        print(f"Found {len(run_files)} run table files")
        
        run_dfs = []
        for f in run_files:
            print(f"  - Loading {f.name}")
            df = pd.read_csv(f)
            run_dfs.append(df)
        
        run_table = pd.concat(run_dfs, ignore_index=True)
        print(f"\n[Run Table] Total fault injection cases: {len(run_table)}")
        print(f"[Run Table] Columns: {list(run_table.columns)}")
        print(f"[Run Table] Services affected: {run_table['service'].nunique()}")
        
        # Create gaia_resplit.csv with proper format
        gaia_resplit = run_table.copy()
        if 'case_id' not in gaia_resplit.columns:
            gaia_resplit['case_id'] = range(len(gaia_resplit))
        
        # Ensure required columns exist
        required_cols = ['case_id', 'datetime', 'service', 'message']
        for col in required_cols:
            if col not in gaia_resplit.columns:
                print(f"[Warning] Missing column: {col}")
        
        # Add time window columns (5-minute window after fault injection)
        gaia_resplit['st_time'] = pd.to_datetime(gaia_resplit['datetime'])
        gaia_resplit['ed_time'] = gaia_resplit['st_time'] + pd.to_timedelta(5, unit='m')
        
        # Extract anomaly type from message
        gaia_resplit['anomaly_type'] = gaia_resplit['message'].str.extract(r'(pod-failure|node-failure|network-delay|network-loss|cpu-load|mem-load)')
        gaia_resplit['instance'] = gaia_resplit['service']
        gaia_resplit['data_type'] = 'fault'
        
        # Save gaia_resplit.csv
        output_file = self.output_path / 'gaia_resplit.csv'
        gaia_resplit.to_csv(output_file, index=False)
        print(f"\n[Saved] {output_file}")
        print(f"[Saved] Shape: {gaia_resplit.shape}")
        
        return gaia_resplit
    
    def parse_logs_with_drain(self, log_df):
        """
        Parse log messages using Drain algorithm to extract templates
        
        Args:
            log_df: DataFrame with 'message' column containing raw log messages
            
        Returns:
            DataFrame with added 'EventId' and 'EventTemplate' columns
        """
        if DrainParser is None:
            print("[WARNING] Drain not available, using raw messages as EventId")
            log_df['EventId'] = log_df['message'].astype(str).str[:50]
            log_df['EventTemplate'] = log_df['message']
            return log_df
        
        print(f"[Drain] Parsing {len(log_df)} log messages...")
        
        # Save logs to temp file for Drain
        temp_log_file = self.drain_temp_dir / 'temp_logs.csv'
        
        # Prepare log data in Drain-compatible format
        drain_input = log_df[['datetime', 'service', 'message']].copy()
        drain_input.columns = ['timestamp', 'cmdb_id', 'Content']
        drain_input['log_id'] = range(len(drain_input))
        drain_input['log_name'] = 'business'
        
        # Reorder columns for Drain format: <log_id>,<timestamp>,<cmdb_id>,<log_name>,<Content>
        drain_input = drain_input[['log_id', 'timestamp', 'cmdb_id', 'log_name', 'Content']]
        drain_input.to_csv(temp_log_file, index=False, header=False)
        
        # Configure Drain parser
        log_format = '<log_id>,<timestamp>,<cmdb_id>,<log_name>,<Content>'
        regex = [
            r'blk_(|-)[0-9]+',  # block id
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',  # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        
        try:
            parser = DrainParser(
                log_format=log_format,
                indir=str(self.drain_temp_dir),
                outdir=str(self.drain_temp_dir / 'output'),
                depth=4,
                st=0.5,
                rex=regex
            )
            
            # Parse logs
            parser.parse('temp_logs.csv')
            
            # Read parsed results
            structured_file = self.drain_temp_dir / 'output' / 'temp_logs.csv_structured.csv'
            if structured_file.exists():
                parsed = pd.read_csv(structured_file)
                
                # Merge EventId and EventTemplate back to original dataframe
                log_df['EventId'] = parsed['EventId'].values
                log_df['EventTemplate'] = parsed['EventTemplate'].values
                
                print(f"[Drain] Extracted {log_df['EventId'].nunique()} unique log templates")
            else:
                print("[WARNING] Drain parsing failed, using raw messages")
                log_df['EventId'] = log_df['message'].astype(str).str[:50]
                log_df['EventTemplate'] = log_df['message']
        
        except Exception as e:
            print(f"[WARNING] Drain parsing error: {e}, using raw messages")
            log_df['EventId'] = log_df['message'].astype(str).str[:50]
            log_df['EventTemplate'] = log_df['message']
        
        # Cleanup temp files
        try:
            if temp_log_file.exists():
                temp_log_file.unlink()
        except:
            pass
        
        return log_df
    
    @staticmethod
    def parse_logs_simple(log_df):
        """Simple log parsing without full Drain (for parallel processing)"""
        if 'message' in log_df.columns:
            # Extract event patterns by replacing numbers/IPs
            import re
            log_df['EventId'] = log_df['message'].astype(str).apply(
                lambda x: re.sub(r'\d+', '<NUM>', re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', x))[:100]
            )
        else:
            log_df['EventId'] = 'unknown'
        return log_df
    
    def process_trace_data(self, run_table):
        """Process trace data (100% - no sampling)"""
        print("\n" + "="*80)
        print("STEP 2: Processing Trace Data (100%)")
        print("="*80)
        
        trace_files = list(self.raw_path.glob('trace/trace_table_*.csv'))
        print(f"Found {len(trace_files)} trace files")
        
        if len(trace_files) == 0:
            print(f"[ERROR] No trace files found in: {self.raw_path / 'trace'}")
            print(f"[DEBUG] Checking if directory exists...")
            trace_dir = self.raw_path / 'trace'
            if trace_dir.exists():
                print(f"  Directory exists. Contents:")
                for f in trace_dir.iterdir():
                    print(f"    - {f.name}")
            else:
                print(f"  [ERROR] Directory does not exist!")
            return {}
        
        print(f"Processing {len(run_table)} fault cases")
        print(f"This may take 10-20 minutes... Please be patient!\n")
        
        # Result dict: {case_id: [[timestamp, caller, callee], ...]}
        trace_dict = {i: [] for i in range(len(run_table))}
        
        # Pre-compute time windows as numpy arrays for vectorized operations
        case_ids = run_table['case_id'].values
        st_times = pd.to_datetime(run_table['st_time']).values.astype('int64') / 1e6  # Convert to milliseconds
        ed_times = pd.to_datetime(run_table['ed_time']).values.astype('int64') / 1e6
        
        # Track statistics
        total_traces_found = 0
        files_processed = 0
        
        for trace_file in tqdm(trace_files, desc="Processing trace files"):
            service_name = trace_file.stem.replace('trace_table_', '')
            
            try:
                # Read trace file
                df = pd.read_csv(trace_file)
                
                if len(df) == 0:
                    continue
                
                # Convert timestamp column to milliseconds
                if 'timestamp' in df.columns:
                    # The timestamp column is already a datetime string, convert to milliseconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    df = df.dropna(subset=['timestamp'])
                    # Convert to milliseconds since epoch
                    df['timestamp'] = df['timestamp'].astype('int64') / 1e6
                
                if len(df) == 0:
                    print(f"  [SKIP] {service_name}: No valid timestamps")
                    continue
                
                # Sort by timestamp for faster binary search
                df = df.sort_values('timestamp').reset_index(drop=True)
                timestamps = df['timestamp'].values
                
                file_traces_found = 0
                files_processed += 1
                
                # Process in batches for memory efficiency
                batch_size = 1000
                for batch_start in range(0, len(run_table), batch_size):
                    batch_end = min(batch_start + batch_size, len(run_table))
                    
                    batch_case_ids = case_ids[batch_start:batch_end]
                    batch_st_times = st_times[batch_start:batch_end]
                    batch_ed_times = ed_times[batch_start:batch_end]
                    
                    # Use searchsorted for fast time window matching
                    for i, case_id in enumerate(batch_case_ids):
                        st_idx = np.searchsorted(timestamps, batch_st_times[i], side='left')
                        ed_idx = np.searchsorted(timestamps, batch_ed_times[i], side='right')
                        
                        if ed_idx > st_idx:
                            case_traces = df.iloc[st_idx:ed_idx]
                            
                            for _, row in case_traces.iterrows():
                                timestamp = row['timestamp']
                                caller = row.get('service_name', service_name)
                                callee = row.get('called_service', 'unknown')
                                
                                trace_dict[case_id].append([timestamp, caller, callee])
                                file_traces_found += 1
                                total_traces_found += 1
                
                del df
                gc.collect()
                
            except Exception as e:
                print(f"    [ERROR] {service_name}: {e}")
        
        # Validation statistics
        print(f"\n[Trace Processing Summary]")
        print(f"  Files processed: {files_processed}/{len(trace_files)}")
        print(f"  Total trace records extracted: {total_traces_found}")
        cases_with_traces = sum(1 for v in trace_dict.values() if len(v) > 0)
        print(f"  Cases with trace data: {cases_with_traces}/{len(run_table)}")
        if cases_with_traces > 0:
            print(f"  Average traces per case: {total_traces_found / cases_with_traces:.1f}")
        
        # Check if we're missing data
        if cases_with_traces < len(run_table) * 0.5:
            print(f"\n[WARNING] Only {cases_with_traces}/{len(run_table)} cases have trace data!")
            print("  Possible causes:")
            print("  1. Time window mismatch between run_table and trace files")
            print("  2. Service name mismatch")
            print("  3. Timestamp format issues")
        
        # Save trace JSON
        output_file = self.anomalies_dir / 'demo_trace.json'
        with open(output_file, 'w') as f:
            json.dump(trace_dict, f)
        
        trace_size_kb = output_file.stat().st_size / 1024
        print(f"\n[Saved] {output_file}")
        print(f"  Size: {trace_size_kb:.1f} KB")
        print(f"  Cases: {len(trace_dict)}")
        print(f"  Total traces: {total_traces_found}")
        
        # Compare with demo if it exists
        demo_trace = Path('data/gaia/demo/demo_1100/anomalies/demo_trace.json')
        if demo_trace.exists():
            demo_size_kb = demo_trace.stat().st_size / 1024
            print(f"\n[COMPARISON] Original demo trace: {demo_size_kb:.1f} KB")
            print(f"[COMPARISON] Your processed trace: {trace_size_kb:.1f} KB")
            ratio = (trace_size_kb / demo_size_kb) * 100
            print(f"[COMPARISON] Size ratio: {ratio:.1f}%")
            
            if ratio < 50:
                print("\n[WARNING] Your trace file is significantly smaller than demo!")
                print("  This might indicate data loss during processing")
        
        return trace_dict
    
    def process_metric_data_parallel(self, args):
        """Process single metric file (for parallel execution)"""
        metric_file, time_windows = args
        #print(f"[Worker] Starting to process: {metric_file.name}", flush=True)
        case_metrics = {case_id: [] for case_id, _, _ in time_windows}
        
        try:
            # Read entire metric file
            df = pd.read_csv(metric_file)
            
            # Convert timestamp to numeric if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df.dropna(inplace=True)
            
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
                # ADD THIS NEW BLOCK IN ITS PLACE
                service = metric_file.stem.split('_')[0] if '_' in metric_file.stem else 'unknown'

                for col in case_data.columns:
                    if col in ['timestamp', 'service']:
                        continue
                    
                    # First, detect if an anomaly exists in this window for this metric
                    is_anomaly_present, anomaly_ts, score = ksigma.detection(
                        case_data, col, int(st_ts), int(ed_ts)
                    )
                    
                    # Now, iterate through EVERY data point in the window for this metric
                    for index, row in case_data.iterrows():
                        timestamp = row['timestamp']
                        metric_value = row[col]
                        
                        # Label the data point: 1 if it's the anomaly, 0 otherwise
                        is_anomaly_label = 1 if (is_anomaly_present and timestamp == anomaly_ts) else 0
                        
                        # Append the labeled data point
                        case_metrics[case_id].append([
                            int(timestamp),
                            str(service),
                            str(col),             # The metric name (e.g., 'cpu_usage')
                            float(metric_value),  # The actual value of the metric
                            is_anomaly_label      # 0 for normal, 1 for anomaly
                        ])
            
            del df, sampled
            gc.collect()
        
        except Exception as e:
            print(f"Error processing {metric_file.name}: {e}")
        
        return case_metrics
    
    def process_metric_data(self, run_table):
        """Process metric data (30% sampling with K-sigma anomaly detection)"""
        print("\n" + "="*80)
        print("STEP 3: Processing Metric Data (30% of Files FULLY)")
        print("="*80)

        metric_files = list(self.raw_path.glob('metric/metric_split/metric/*.csv'))
        print(f"Found {len(metric_files)} metric files")

        import random
        random.seed(42)
        num_files_to_sample = int(len(metric_files) * 0.3)
        if num_files_to_sample == 0 and len(metric_files) > 0:
            num_files_to_sample = 1
            
        sampled_files = random.sample(metric_files, num_files_to_sample)
        print(f"Randomly selected 30% of files: {len(sampled_files)} files")
        print(f"Using {self.n_workers} parallel workers")

        time_windows = []
        for idx, case in run_table.iterrows():
            st_ts = pd.Timestamp(case['st_time']).timestamp() * 1000
            ed_ts = pd.Timestamp(case['ed_time']).timestamp() * 1000
            time_windows.append((case['case_id'], st_ts, ed_ts))

        metric_dict = {i: [] for i in range(len(run_table))}
        chunk_size = 200
        
        with Pool(processes=self.n_workers) as pool:
            # Outer progress bar for chunks
            for i in tqdm(range(0, len(sampled_files), chunk_size), desc="Processing Chunks", position=0):
                chunk_files = sampled_files[i:i + chunk_size]
                args_list = [(f, time_windows) for f in chunk_files]
                
                # Get an iterator for the results
                results_iterator = pool.imap_unordered(self.process_metric_data_parallel, args_list)
                
                # Inner progress bar for files within the current chunk
                inner_pbar = tqdm(results_iterator, total=len(chunk_files), desc="Files in Chunk", position=1, leave=False)
                
                # Process results, driving the inner progress bar
                for result in inner_pbar:
                    for case_id, metrics in result.items():
                        if case_id in metric_dict:
                            metric_dict[case_id].extend(metrics)
                
                del results_iterator, args_list
                gc.collect()

        metric_dict = {int(k): v for k, v in metric_dict.items()}

        output_file = self.anomalies_dir / 'demo_metric.json'
        print(f"\n[Saving] Writing metric data to JSON...")
        with open(output_file, 'w') as f:
            json.dump(metric_dict, f)

        total_metrics = sum(len(v) for v in metric_dict.values())
        print(f"\n[Metric Summary]")
        print(f"  Total cases: {len(metric_dict)}")
        print(f"  Total metric anomalies detected: {total_metrics:,}")
        if len(metric_dict) > 0 and total_metrics > 0:
            print(f"  Avg anomalies per case: {total_metrics/len(metric_dict):.1f}")
        print(f"[Saved] {output_file}")

        return metric_dict
    
    def process_business_logs_parallel(self, args):
        """Process single business/log file (for parallel execution)"""
        log_file, time_windows, sample_rate, use_drain = args
        
        case_logs = {case_id: [] for case_id, _, _ in time_windows}
        
        try:
            # Read entire log file
            df = pd.read_csv(log_file)
            
            # Convert datetime to proper format if needed
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                df = df.dropna(subset=['datetime'])
            
            if len(df) == 0:
                return case_logs
            
            # Sample 30% of rows
            sampled = df.sample(frac=sample_rate, random_state=42)
            
            # Parse logs (extract event templates)
            if 'message' in sampled.columns:
                # Simple pattern extraction: replace numbers and IPs
                import re
                sampled['EventId'] = sampled['message'].astype(str).apply(
                    lambda x: re.sub(r'\d+', '<NUM>', re.sub(r'\d+\.\d+\.\d+\.\d+', '<IP>', str(x)))[:100]
                )
            else:
                sampled['EventId'] = 'unknown'
            
            # Process each time window
            for case_id, st_ts, ed_ts in time_windows:
                # Convert milliseconds to datetime
                st_time = pd.Timestamp(st_ts / 1000, unit='s')
                ed_time = pd.Timestamp(ed_ts / 1000, unit='s')
                
                # Filter logs in time window
                mask = (sampled['datetime'] >= st_time) & (sampled['datetime'] <= ed_time)
                case_data = sampled[mask]
                
                if len(case_data) == 0:
                    continue
                
                # Extract log events (timestamp, service, EventId)
                for _, row in case_data.iterrows():
                    timestamp = pd.Timestamp(row['datetime']).timestamp() * 1000
                    service = row.get('service', 'unknown')
                    event_id = row.get('EventId', str(row.get('message', ''))[:50])
                    
                    case_logs[case_id].append([timestamp, service, event_id])
            
            del df, sampled
            gc.collect()
        
        except Exception as e:
            print(f"Error processing {log_file.name}: {e}")
        
        return case_logs
    
    def process_business_logs(self, run_table):
        """Process business/log data (30% sampling)"""
        print("\n" + "="*80)
        print("STEP 4: Processing Business/Log Data (30% Sampling)")
        print("="*80)
        
        log_files = list(self.raw_path.glob('business/*.csv'))
        print(f"Found {len(log_files)} business log files")
        print(f"Using {self.n_workers} parallel workers")
        
        # Pre-compute time windows as list (not dict!)
        time_windows = []
        for idx, case in run_table.iterrows():
            st_ts = pd.Timestamp(case['st_time']).timestamp() * 1000
            ed_ts = pd.Timestamp(case['ed_time']).timestamp() * 1000
            time_windows.append((case['case_id'], st_ts, ed_ts))
        
        # Prepare arguments for parallel processing
        use_drain = (DrainParser is not None)
        args_list = [(f, time_windows, self.sample_rate, use_drain) for f in log_files]
        
        # Process in parallel
        with Pool(processes=self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(self.process_business_logs_parallel, args_list),
                total=len(args_list),
                desc="Processing business log files"
            ))
        
        # Merge results
        log_list = []
        for case_id in range(len(run_table)):
            case_logs = []
            for result in results:
                case_logs.extend(result[case_id])
            log_list.append(case_logs)
        
        # Save as NPY array (DiagFusion format)
        output_file = self.anomalies_dir / 'stratification_logs.npy'
        np.save(output_file, log_list, allow_pickle=True)
        
        total_logs = sum(len(logs) for logs in log_list)
        print(f"\n[Business Log Summary]")
        print(f"  Total cases: {len(log_list)}")
        print(f"  Total log records: {total_logs:,}")
        if len(log_list) > 0:
            print(f"  Avg logs per case: {total_logs/len(log_list):.1f}")
        print(f"[Saved] {output_file}")
        
        return log_list
    
    def run_preprocessing(self):
        """Execute complete preprocessing pipeline"""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("GAIA Raw Dataset Preprocessing Pipeline")
        print("="*80)
        print(f"Raw data path: {self.raw_path}")
        print(f"Output path: {self.output_path}")
        print(f"Workers: {self.n_workers}")
        print(f"Sampling rate: {self.sample_rate*100}%")
        
        try:
            # Step 1: Load fault injection metadata
            run_table = self.load_run_table()
            
            # Step 2: Process trace data (100%) - COMMENTED OUT - Already completed
            # trace_dict = self.process_trace_data(run_table)
            print("\n" + "="*80)
            print("STEP 2: Trace Data (SKIPPED - Already processed)")
            print("="*80)
            print("Using existing trace file from previous run")
            trace_file = self.anomalies_dir / 'demo_trace.json'
            if trace_file.exists():
                print(f"[Found] {trace_file}")
                print(f"  Size: {trace_file.stat().st_size / 1024:.1f} KB")
            else:
                print(f"[WARNING] Trace file not found: {trace_file}")
                print("  Run full preprocessing if you need to regenerate trace data")
            
            # Step 3: Process metric data (30% with anomaly detection)
            metric_dict = self.process_metric_data(run_table)
            
            # Step 4: Process business/log data (30%)
            log_list = self.process_business_logs(run_table)
            
            # Summary
            elapsed_time = time.time() - start_time
            print("\n" + "="*80)
            print("PREPROCESSING COMPLETE!")
            print("="*80)
            print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
            print(f"\nOutput files created:")
            print(f"  1. {self.output_path}/gaia_resplit.csv")
            print(f"  2. {self.anomalies_dir}/demo_trace.json (from previous run)")
            print(f"  3. {self.anomalies_dir}/demo_metric.json")
            print(f"  4. {self.anomalies_dir}/stratification_logs.npy")
            print(f"\nNext steps:")
            print(f"  1. Update config/gaia_config.yaml to point to: {self.output_path}")
            print(f"  2. Run: python main.py --config gaia_config.yaml")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point"""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess raw GAIA MicroSS dataset')
    parser.add_argument(
        '--raw-path',
        type=str,
        default=r'C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS',
        help='Path to raw MicroSS directory'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=r'C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS',
        help='Path to save preprocessed data'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of CPU cores to use (2-4 recommended)'
    )
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=0.3,
        help='Sampling rate for metric/business data (0.3 = 30%%)'
    )
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = GAIAPreprocessor(
        raw_data_path=args.raw_path,
        output_path=args.output_path,
        n_workers=args.workers,
        metric_sample_rate=args.sample_rate
    )
    
    # Run preprocessing
    success = preprocessor.run_preprocessing()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
