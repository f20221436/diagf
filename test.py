#!/usr/bin/env python3
"""
discover_anomalies.py

Reads run_table_*.csv files from <raw-path>/run and prints ONLY the unique anomaly types
detected across all messages.
"""

import re
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# --------------------
# Normalization helpers
# --------------------
_ts_prefix_re = re.compile(r'^\s*\d{4}-\d{2}-\d{2}.*?\|\s*')  # trims leading timestamp + header up to '|'
_ip_re = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
_uuid_re = re.compile(r'\b[0-9a-f]{8,}\b', re.I)
_date_re = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
_time_re = re.compile(r'\b\d{2}:\d{2}:\d{2}(?:,\d+)?\b')
_number_re = re.compile(r'\b\d+\b')

def _take_message_field(msg: str) -> str:
    """Many logs are pipe-separated; assume the human-readable message is the last field."""
    if '|' in msg:
        return msg.split('|')[-1].strip()
    return msg.strip()

def _normalize_for_search(msg: str) -> str:
    """
    Lowercase copy suitable for keyword regex searching.
    Keep underscores and hyphens (they are meaningful), but replace IP/IDs/dates/numbers to reduce noise.
    """
    s = msg.strip()
    s = _ts_prefix_re.sub('', s)          # remove leading timestamp/header
    s = _take_message_field(s)            # take last pipe-separated field
    # replace noisy tokens (preserve underscores/hyphens)
    s = _ip_re.sub('<IP>', s)
    s = _uuid_re.sub('<ID>', s)
    s = _date_re.sub('<DATE>', s)
    s = _time_re.sub('<TIME>', s)
    s = _number_re.sub('<NUM>', s)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

# --------------------
# Ordered anomaly patterns (priority matters)
# --------------------
ANOMALY_PATTERNS = [
    (re.compile(r'\bpod[-_\s]?failure\b', re.I), 'pod_failure'),
    (re.compile(r'\bnode[-_\s]?failure\b', re.I), 'node_failure'),
    (re.compile(r'\bnetwork[-_\s]?delay\b', re.I), 'network_delay'),
    (re.compile(r'\bnetwork[-_\s]?loss\b', re.I), 'network_loss'),
    (re.compile(r'\bpacket[-_\s]?loss\b', re.I), 'network_loss'),
    (re.compile(r'\blogin[-_\s]?failure\b', re.I), 'login_failure'),
    (re.compile(r'\bauthentication failed\b|\bauth failed\b|\binvalid credentials\b|\bfailed to authenticate\b', re.I), 'login_failure'),
    (re.compile(r'\bcpu[-_\s]?load\b|\bhigh cpu\b', re.I), 'cpu_load'),
    (re.compile(r'\bmem(?:ory)?[-_\s]?load\b|\bmemory pressure\b', re.I), 'mem_load'),
    (re.compile(r'\btimeout\b|\btimed out\b', re.I), 'timeout'),
    (re.compile(r'\berror\b|\bexception\b', re.I), 'error'),
    (re.compile(r'\bfailure\b', re.I), 'failure'),
]

BRACKET_RE = re.compile(r'\[([^\]]+)\]')

def _detect_from_bracket_content(content: str):
    """
    Given a bracket content like "pod_failure_id_011e_11ec_ad8a_docker001" or "id_f23d_11eb_b91a",
    try to extract a known anomaly prefix (pod_failure, node_failure, network_delay...).
    If a known keyword is found inside the bracket content, return its canonical name;
    additionally treat short id-like tokens (id_xxx_yyy) as pod_failure heuristically.
    """
    s = content.strip().lower()

    # 1) If a known anomaly keyword appears inside the bracket content, return it.
    for pat, canonical in ANOMALY_PATTERNS:
        if pat.search(s):
            return canonical

    # 2) Heuristic: if bracket content is an id-like token such as "id_f23d_11eb_b91a"
    #    or contains an "_id_" suffix with hex-like segments, consider it a pod_failure.
    #    This covers cases where the log stores only the unique id and we should infer pod failure.
    if re.search(r'\bid_[0-9a-f]{3,}(_[0-9a-f]{2,})+\b', s) or re.match(r'^id_[0-9a-f_]+$', s):
        return 'pod_failure'

    # 3) If the bracket looks like a short anomaly name (no long uuid-like parts), return cleaned:
    if len(s) <= 40 and re.search(r'[a-z]', s) and not re.search(r'[a-f0-9]{12,}', s):
        cleaned = re.sub(r'[\s\-]+', '_', s)
        cleaned = re.sub(r'_id_.*$', '', cleaned)  # drop trailing _id_...
        return cleaned

    return None

def extract_anomaly_type(raw_message: str):
    """
    Return a canonical anomaly type (string) if detected, else None.
    Order:
      1) Look for specific anomaly keyword patterns (ordered)
      2) Look for quoted tokens containing anomaly keywords (e.g., "pod-failure")
      3) As LAST RESORT, check bracketed tokens and try to extract anomaly name from them
      4) If nothing found, return None (we only want to show anomalies)
    """
    if not isinstance(raw_message, str):
        return None

    normalized = _normalize_for_search(raw_message)

    # 1) Check specific keyword patterns (highest priority)
    for pat, canonical in ANOMALY_PATTERNS:
        if pat.search(normalized):
            return canonical

    # 2) Check quoted tokens: e.g., "... simulate \"pod-failure\" anomaly ..."
    quoted = re.findall(r'"([^"]+)"', normalized)
    for q in quoted:
        for pat, canonical in ANOMALY_PATTERNS:
            if pat.search(q):
                return canonical
        if re.match(r'^[a-z0-9_\-]+[-_]?failure$', q) or re.match(r'^[a-z0-9_\-]+[-_]?delay$', q) or re.match(r'^[a-z0-9_\-]+[-_]?loss$', q):
            return q.replace('-', '_')

    # 3) LAST RESORT: bracket tokens (only if they contain anomaly-like text or id-like token)
    bracket_matches = BRACKET_RE.findall(raw_message)
    for b in bracket_matches:
        candidate = _detect_from_bracket_content(b)
        if candidate:
            return candidate

    # Nothing matched => not an anomaly we care about
    return None

# --------------------
# Main CLI and flow
# --------------------
def main():
    parser = argparse.ArgumentParser(description='Discover anomaly types from GAIA run_table_*.csv files (only anomalies).')
    parser.add_argument(
        '--raw-path',
        type=str,
        default=r'C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS',
        help='Path to MicroSS directory (where the /run folder is)'
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    run_folder = raw_path / 'run'

    if not run_folder.exists():
        print(f"Error: run folder not found at: {run_folder}")
        return

    run_files = sorted(run_folder.glob('run_table_*.csv'))
    if not run_files:
        print(f"Error: No run_table_*.csv files found in {run_folder}")
        return

    print(f"Scanning {len(run_files)} files in {run_folder}...\n")

    all_dfs = []
    for f in run_files:
        print(f" - Reading {f.name} ...")
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"   Skipping {f.name} due to read error: {e}")
            continue
        if 'message' not in df.columns:
            print(f"   Warning: {f.name} has no 'message' column — skipping")
            continue
        all_dfs.append(df[['message']])
        print(f"   Loaded {len(df)} rows")

    if not all_dfs:
        print("No valid data to process.")
        return

    run_table = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal log messages to scan: {len(run_table)}")

    # collect only anomaly types
    found_types = set()
    example_map = {}

    print("\nDetecting anomaly types (only anomalies will be reported)...")
    for msg in tqdm(run_table['message'].astype(str), desc='Scanning messages'):
        a = extract_anomaly_type(msg)
        if a:
            found_types.add(a)
            if a not in example_map:
                example_map[a] = msg

    if not found_types:
        print("\nNo anomalies found by the configured detectors.")
        return

    print("\n" + "="*50)
    print(" DETECTED ANOMALY TYPES (canonical names)")
    print("="*50)
    for t in sorted(found_types):
        print(f"- {t}")
    print("\n(Only anomaly types are printed. Non-anomalous messages are ignored.)")

    print("\nExamples (first message that produced each anomaly type):")
    for t in sorted(found_types):
        print(f"\n--- {t} ---")
        print(example_map[t])

if __name__ == '__main__':
    main()
