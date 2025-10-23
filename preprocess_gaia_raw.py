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
import re

# Import DiagFusion modules
from detector.k_sigma import Ksigma

# Import Drain log parser
try:
    from log.logparser.logparser.Drain.Drain import LogParser as DrainParser
except ImportError as e:
    print(f"\n[DEBUG] The real import error is: {e}\n")
    print("[WARNING] Drain log parser not available. Will use raw log messages.")
    DrainParser = None


class GAIAPreprocessor:
    def __init__(self, raw_data_path, output_path, n_workers=2, metric_sample_rate=1):
        """
        Initialize GAIA preprocessor
        
        Args:
            raw_data_path: Path to MicroSS directory
            output_path: Path to save preprocessed data
            n_workers: Number of CPU cores to use (2-4)
            metric_sample_rate: Sampling rate for metric/business data (1 = 100%)
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
        

        BRACKET_RE = re.compile(r'\[([^\]]+)\]')

        # Ordered patterns (highest priority first). Add more patterns here as needed.
        ANOMALY_PATTERNS = [
            (re.compile(r'\bpod[-_\s]?failure\b|\bpodfailure\b', re.I), 'pod_failure'),
            (re.compile(r'\bnode[-_\s]?failure\b|\bnodfailure\b', re.I), 'node_failure'),
            (re.compile(r'\bnetwork[-_\s]?delay\b', re.I), 'network_delay'),
            (re.compile(r'\bnetwork[-_\s]?loss\b|\bpacket[-_\s]?loss\b', re.I), 'network_loss'),
            (re.compile(r'\blogin[-_\s]?failure\b', re.I), 'login_failure'),
            (re.compile(r'\bauthentication failed\b|\bauth failed\b|\binvalid credentials\b|\bfailed to authenticate\b', re.I), 'login_failure'),
            (re.compile(r'\bcpu[-_\s]?load\b|\bhigh cpu\b|\bcpu anomalies?\b', re.I), 'cpu_load'),
            (re.compile(r'\bmem(?:ory)?[-_\s]?load\b|\bmemory pressure\b|\bmemory anomalies?\b', re.I), 'mem_load'),
            (re.compile(r'\btimeout\b|\btimed out\b', re.I), 'timeout'),
            (re.compile(r'\berror\b|\bexception\b', re.I), 'error'),
            (re.compile(r'\bfailure\b', re.I), 'failure'),
        ]

        # Helper: get the "human" message (last pipe-separated field if present)
        def _take_message_field(msg: str) -> str:
            if not isinstance(msg, str):
                return ''
            if '|' in msg:
                return msg.split('|')[-1].strip()
            return msg.strip()

        # Helper: normalized text for keyword searching (lowercased, noisy tokens removed)
        _ip_re = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        _uuid_re = re.compile(r'\b[0-9a-f]{8,}\b', re.I)
        _date_re = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
        _time_re = re.compile(r'\b\d{2}:\d{2}:\d{2}(?:,\d+)?\b')
        _number_re = re.compile(r'\b\d+\b')
        _ts_prefix_re = re.compile(r'^\s*\d{4}-\d{2}-\d{2}.*?\|\s*')

        def _normalize_for_search(msg: str) -> str:
            s = str(msg or '')
            s = _ts_prefix_re.sub('', s)            # drop leading date/header before '|'
            s = _take_message_field(s)              # take last pipe-separated field (human text)
            s = _ip_re.sub('<IP>', s)
            s = _uuid_re.sub('<ID>', s)
            s = _date_re.sub('<DATE>', s)
            s = _time_re.sub('<TIME>', s)
            s = _number_re.sub('<NUM>', s)
            s = re.sub(r'\s+', ' ', s).strip().lower()
            return s

        def _detect_from_bracket_content(content: str):
            """
            Inspect bracket content like "pod_failure_id_011e_..." or "id_f23d_11eb_b91a".
            Return canonical anomaly name if it can be inferred, else None.
            """
            s = (content or '').strip().lower()

            # 1) Look for known anomaly keywords inside bracket content
            for pat, canonical in ANOMALY_PATTERNS:
                if pat.search(s):
                    return canonical

            # 2) Heuristic: id_... patterns (e.g. id_f23d_11eb_b91a or ..._id_011e_11ec...) => pod_failure
            if re.search(r'\bid_[0-9a-f]{3,}(_[0-9a-f]{2,})+\b', s) or re.match(r'^id_[0-9a-f_]+$', s):
                return 'pod_failure'

            # 3) If bracket content looks short and readable (no long uuids), treat as cleaned anomaly token
            if len(s) <= 40 and re.search(r'[a-z]', s) and not re.search(r'[a-f0-9]{12,}', s):
                cleaned = re.sub(r'[\s\-]+', '_', s)
                cleaned = re.sub(r'_id_.*$', '', cleaned)
                # only return if looks like an anomaly-like token (contains letters)
                if re.search(r'[a-z]', cleaned):
                    return cleaned

            return None

        def find_anomaly_type(message):
            """
            Extract canonical anomaly type from a single log message.
            Returns a string like 'pod_failure', 'network_delay', etc., or None if no anomaly found.
            """
            if not isinstance(message, str):
                return None

            normalized = _normalize_for_search(message)

            # 1) Specific keyword patterns (highest priority)
            for pat, canonical in ANOMALY_PATTERNS:
                if pat.search(normalized):
                    return canonical

            # 2) Quoted tokens (e.g., "pod-failure", 'pod_failure') — check them for known patterns
            quoted = re.findall(r'["\']([^"\']+)["\']', normalized)
            for q in quoted:
                for pat, canonical in ANOMALY_PATTERNS:
                    if pat.search(q):
                        return canonical
                # some quoted tokens may be exact anomaly names (pod-failure / network-delay)
                if re.match(r'^[a-z0-9_\-]+[-_]?failure$', q) or re.match(r'^[a-z0-9_\-]+[-_]?delay$', q) or re.match(r'^[a-z0-9_\-]+[-_]?loss$', q):
                    return q.replace('-', '_')

            # 3) id_ anywhere (explicit request): treat id_... as pod_failure
            #    This catches both bracketed and non-bracketed id_ tokens like id_f23d_11eb_b91a
            if re.search(r'\bid_[0-9a-f]{3,}(_[0-9a-f]{2,})*\b', normalized):
                return 'pod_failure'

            # 4) Bracketed content as LAST RESORT (only accept if it yields anomaly-like content)
            bracket_matches = BRACKET_RE.findall(message)
            for b in bracket_matches:
                candidate = _detect_from_bracket_content(b)
                if candidate:
                    return candidate

            # 5) Extra fallback for explicit simulate phrases (rare)
            if re.search(r'simulate.*pod[-_\s]?failure', normalized):
                return 'pod_failure'
            if re.search(r'simulate.*node[-_\s]?failure', normalized):
                return 'node_failure'

            # No anomaly detected
            return None
        # ------------------------------------------------------------------

        gaia_resplit['anomaly_type'] = gaia_resplit['message'].apply(find_anomaly_type)
        
        gaia_resplit['st_time'] = pd.to_datetime(gaia_resplit['datetime'])
        gaia_resplit['ed_time'] = gaia_resplit['st_time'] + pd.to_timedelta(5, unit='m')
        
        # Extract anomaly type from message
              
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
        """
        Truly vectorized version. Processes the file once and uses an efficient merge 
        to map data to the correct time windows.
        """
        metric_file, time_windows_list, temp_dir = args
        
        pid = os.getpid()
        timestamp = int(time.time() * 1000)
        temp_output_path = temp_dir / f"metric_result_{pid}_{timestamp}.json"
        
        try:
            # 1. Prepare the time windows DataFrame for efficient merging
            time_windows_df = pd.DataFrame(time_windows_list, columns=['case_id', 'st_time', 'ed_time'])
            
            # === FIX: Ensure timestamp data types match ===
            # Convert the float timestamps to integers to match the metric file's timestamp type
            time_windows_df['st_time'] = time_windows_df['st_time'].astype('int64')
            time_windows_df['ed_time'] = time_windows_df['ed_time'].astype('int64')
            # ===============================================
            
            time_windows_df = time_windows_df.sort_values('st_time')

            # 2. Read and prepare the metric data file
            df = pd.read_csv(metric_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df.dropna(inplace=True)
            if df.empty:
                return None
            
            # 3. Melt the DataFrame ONCE to get a long format
            id_vars = ['timestamp']
            value_vars = [col for col in df.columns if col not in ['timestamp', 'service']]
            if not value_vars:
                return None
            
            long_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                              var_name='metric_name', value_name='metric_value')
            long_df = long_df.sort_values('timestamp')

            # 4. Use a powerful and fast 'merge_asof' to map every metric row to its corresponding case_id
            merged_df = pd.merge_asof(
                left=long_df,
                right=time_windows_df,
                left_on='timestamp',
                right_on='st_time',
                direction='backward'
            )
            
            merged_df = merged_df[merged_df['timestamp'] < merged_df['ed_time']]
            merged_df.dropna(subset=['case_id'], inplace=True)
            if merged_df.empty:
                return None

            # 5. Efficiently detect anomalies for each metric group
            ksigma = Ksigma()
            merged_df['is_anomaly_label'] = 0
            
            for (case_id, metric_name), group in merged_df.groupby(['case_id', 'metric_name']):
                st_ts = group['st_time'].iloc[0]
                ed_ts = group['ed_time'].iloc[0]
                
                pivot_for_detection = group[['timestamp', 'metric_value']].set_index('timestamp')
                pivot_for_detection.columns = [metric_name]

                is_anomaly_present, anomaly_ts, score = ksigma.detection(
                    pivot_for_detection.reset_index(), metric_name, int(st_ts), int(ed_ts)
                )

                if is_anomaly_present:
                    anomaly_index = group[group['timestamp'] == anomaly_ts].index
                    merged_df.loc[anomaly_index, 'is_anomaly_label'] = 1

            # 6. Format and save the final results to a temporary JSON
            service = metric_file.stem.split('_')[0] if '_' in metric_file.stem else 'unknown'
            merged_df['service'] = service
            
            case_metrics = {}
            key_cols = ['timestamp', 'service', 'metric_name', 'metric_value', 'is_anomaly_label']
            
            for case_id, group in merged_df.groupby('case_id'):
                case_metrics[str(int(case_id))] = group[key_cols].values.tolist()

            with open(temp_output_path, 'w') as f:
                json.dump(case_metrics, f)
                
            return temp_output_path
        
        except Exception as e:
            print(f"Error processing {metric_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def process_metric_data(self, run_table):
        """Process metric data using a vectorized worker and a JSON-based file shuffle."""
        print("\n" + "="*80)
        print("STEP 3: Processing Metric Data (30% of Files FULLY)")
        print("="*80)

        import shutil
        temp_results_dir = self.output_path / "temp_metric_results"
        if temp_results_dir.exists():
            shutil.rmtree(temp_results_dir)
        temp_results_dir.mkdir(exist_ok=True)
        print(f"Using temporary directory for worker results: {temp_results_dir}")

        metric_files = list(self.raw_path.glob('metric/metric_split/metric/*.csv'))
        print(f"Found {len(metric_files)} metric files")

        import random
        random.seed(42)
        num_files_to_sample = int(len(metric_files) * self.sample_rate)
        if num_files_to_sample == 0 and len(metric_files) > 0:
            num_files_to_sample = 1
            
        sampled_files = random.sample(metric_files, num_files_to_sample)
        print(f"Randomly selected {self.sample_rate*100:.0f}% of files: {len(sampled_files)} files")
        print(f"Using {self.n_workers} parallel workers")

        time_windows = []
        for idx, case in run_table.iterrows():
            st_ts = pd.Timestamp(case['st_time']).timestamp() * 1000
            ed_ts = pd.Timestamp(case['ed_time']).timestamp() * 1000
            time_windows.append((case['case_id'], st_ts, ed_ts))

        metric_dict = {i: [] for i in range(len(run_table))}
        chunk_size = 200
        
        with Pool(processes=self.n_workers) as pool:
            for i in tqdm(range(0, len(sampled_files), chunk_size), desc="Processing Chunks", position=0):
                chunk_files = sampled_files[i:i + chunk_size]
                args_list = [(f, time_windows, temp_results_dir) for f in chunk_files]
                
                results_iterator = pool.imap_unordered(self.process_metric_data_parallel, args_list)
                
                inner_pbar = tqdm(results_iterator, total=len(chunk_files), desc="Files in Chunk", position=1, leave=False)
                
                # The result is a path to a temporary JSON file
                for temp_file_path in inner_pbar:
                    if temp_file_path:
                        with open(temp_file_path, 'r') as f:
                            worker_data = json.load(f)
                        
                        # Merge the data from the temp file
                        for case_id_str, metrics in worker_data.items():
                            case_id_int = int(case_id_str)
                            if case_id_int in metric_dict:
                                metric_dict[case_id_int].extend(metrics)
                        
                        # Delete the temporary file after merging
                        os.remove(temp_file_path)
                
                del results_iterator, args_list
                gc.collect()

        print("[Cleanup] Deleting temporary directory...")
        shutil.rmtree(temp_results_dir)

        output_file = self.anomalies_dir / 'demo_metric.json'
        print(f"\n[Saving] Writing final aggregated metric data to JSON...")
        with open(output_file, 'w') as f:
            json.dump(metric_dict, f)

        total_metrics = sum(len(v) for v in metric_dict.values())
        print(f"\n[Metric Summary]")
        print(f"  Total cases: {len(metric_dict)}")
        print(f"  Total metric records generated: {total_metrics:,}")
        print(f"[Saved] {output_file}")

        return metric_dict
    
    def process_business_logs_parallel(self, args):
        """
        Final robust version that includes a dedicated progress bar for file reading.
        """
        import re
        import _csv
        log_file, time_windows_list, temp_dir, worker_id = args # Unpack new worker_id
        
        pid = os.getpid()
        timestamp = int(time.time() * 1000)
        temp_output_path = temp_dir / f"log_result_{pid}_{timestamp}.json"

        try:
            time_windows_df = pd.DataFrame(time_windows_list, columns=['case_id', 'st_time', 'ed_time'])
            time_windows_df['st_time'] = pd.to_datetime(time_windows_df['st_time'], unit='ms', utc=True)
            time_windows_df['ed_time'] = pd.to_datetime(time_windows_df['ed_time'], unit='ms', utc=True)
            time_windows_df = time_windows_df.sort_values('st_time')
            
            all_results_dfs = []

            iterator = pd.read_csv(log_file, chunksize=50000, engine='python', on_bad_lines='warn')
            
            try:
                # === FIX: Add a tqdm progress bar for reading chunks within this worker ===
                pbar_chunk = tqdm(iterator, desc=f"Worker {worker_id} Reading {log_file.name}", position=worker_id + 1, leave=False)
                
                for df_chunk in pbar_chunk:
                # =========================================================================
                    if 'message' not in df_chunk.columns or df_chunk['message'].isnull().all():
                        continue

                    df_chunk['datetime'] = pd.to_datetime(
                        df_chunk['message'].str.extract(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})', expand=False),
                        errors='coerce'
                    )
                    df_chunk['datetime'] = df_chunk['datetime'].dt.tz_localize('UTC')

                    df_chunk.dropna(subset=['datetime'], inplace=True)
                    if df_chunk.empty:
                        continue
                    
                    df_chunk = df_chunk.sort_values('datetime')
                    
                    merged_df = pd.merge_asof(
                        left=df_chunk, right=time_windows_df,
                        left_on='datetime', right_on='st_time',
                        direction='backward'
                    )
                    
                    merged_df = merged_df[merged_df['datetime'] < merged_df['ed_time']]
                    merged_df.dropna(subset=['case_id'], inplace=True)

                    if merged_df.empty:
                        continue

                    merged_df['EventId'] = merged_df['message'].astype(str).str.replace(r'\d+', '<NUM>', regex=True).str[:100]
                    all_results_dfs.append(merged_df)

            except _csv.Error as e:
                print(f"\n[WARNING] CSV parsing error in {log_file.name}: '{e}'. File is corrupt. "
                    f"Processing with data recovered so far.\n")
            
            if not all_results_dfs:
                return None

            full_df = pd.concat(all_results_dfs, ignore_index=True)
            
            case_logs = {}
            full_df['timestamp_ms'] = (full_df['datetime'].astype(np.int64) / 1_000_000).astype(np.int64)
            key_cols = ['timestamp_ms', 'service', 'EventId']

            for case_id, group in full_df.groupby('case_id'):
                case_logs[str(int(case_id))] = group[key_cols].values.tolist()

            with open(temp_output_path, 'w') as f:
                json.dump(case_logs, f)
            
            return temp_output_path

        except Exception as e:
            print(f"An error occurred in process_business_logs_parallel for {log_file.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def process_business_logs(self, run_table):
        """Process business/log data with detailed progress tracking for each worker."""
        print("\n" + "="*80)
        print("STEP 4: Processing Business/Log Data")
        print("="*80)

        import shutil
        temp_results_dir = self.output_path / "temp_log_results"
        if temp_results_dir.exists():
            shutil.rmtree(temp_results_dir)
        temp_results_dir.mkdir(exist_ok=True)
        print(f"Using temporary directory for worker results: {temp_results_dir}")

        log_files = list(self.raw_path.glob('business/*.csv'))
        print(f"Found {len(log_files)} business log files")
        if not log_files:
            print("[INFO] No business log files found to process.")
            return []
        
        print(f"Using {self.n_workers} parallel workers")
        
        time_windows = []
        for idx, case in run_table.iterrows():
            st_ts = pd.Timestamp(case['st_time']).timestamp() * 1000
            ed_ts = pd.Timestamp(case['ed_time']).timestamp() * 1000
            time_windows.append((case['case_id'], st_ts, ed_ts))
        
        # === FIX: Add 'enumerate' to pass a unique worker_id to each job ===
        args_list = [(f, time_windows, temp_results_dir, i) for i, f in enumerate(log_files)]
        # ===================================================================
        
        log_list_dict = {i: [] for i in range(len(run_table))}

        with Pool(processes=self.n_workers) as pool:
            # This main progress bar will track the overall file completion
            pbar_main = tqdm(total=len(args_list), desc="Overall Log File Progress", position=0)
            
            for temp_file_path in pool.imap_unordered(self.process_business_logs_parallel, args_list):
                if temp_file_path:
                    with open(temp_file_path, 'r') as f:
                        worker_data = json.load(f)
                    
                    for case_id_str, logs in worker_data.items():
                        case_id_int = int(case_id_str)
                        if case_id_int in log_list_dict:
                            log_list_dict[case_id_int].extend(logs)
                    
                    os.remove(temp_file_path)
                
                pbar_main.update(1) # Manually update the main progress bar
            
            pbar_main.close()

        print("\n[Cleanup] Deleting temporary directory...")
        shutil.rmtree(temp_results_dir)
        
        log_list = [log_list_dict[i] for i in sorted(log_list_dict.keys())]

        output_file = self.anomalies_dir / 'stratification_logs.npy'
        np.save(output_file, log_list, allow_pickle=True)
        
        total_logs = sum(len(logs) for logs in log_list)
        print(f"\n[Business Log Summary]")
        print(f"  Total cases: {len(log_list)}")
        print(f"  Total log records generated: {total_logs:,}")
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
            #metric_dict = self.process_metric_data(run_table)
            
            # Step 4: Process business/log data (30%)
            #log_list = self.process_business_logs(run_table)
            
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
        default=1,
        help='Sampling rate for metric/business data (1 = 100%%)'
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
