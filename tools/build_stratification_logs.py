# tools/build_stratification_logs.py
import os
import argparse
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

def to_epoch_ms(series):
    """Convert datetime-like series to epoch milliseconds."""
    s = pd.to_datetime(series, errors='coerce', utc=True)
    if s.isna().all():
        s = pd.to_datetime(series, errors='coerce')
    return (s.astype('int64', errors='ignore') // 1_000_000).astype('float64')

def process_structured_csv(path, timestamp_col='timestamp',
                           eventcol='EventId', servicecol='log_name',
                           chunksize=200_000):
    """
    Read one structured CSV (Drain output) and return list of (ts_ms, eventId, service).
    Uses chunked reading + tqdm progress.
    """
    events = []
    total_lines = sum(1 for _ in open(path, 'r', encoding='utf-8', errors='ignore'))
    chunk_iter = pd.read_csv(
        path, chunksize=chunksize, iterator=True,
        encoding='utf-8', low_memory=True
    )

    pbar = tqdm(total=total_lines, desc=f"Parsing {os.path.basename(path)}",
                unit="lines", ncols=100, leave=False)

    for chunk in chunk_iter:
        chunk.columns = [c.strip() for c in chunk.columns]

        # Ensure timestamp column
        if timestamp_col not in chunk.columns:
            for alt in ['time', 'Time', 'timestamp_ms']:
                if alt in chunk.columns:
                    chunk[timestamp_col] = chunk[alt]
                    break

        # Ensure event column
        if eventcol not in chunk.columns:
            for alt in ['EventId', 'EventID', 'eventid', 'event_id']:
                if alt in chunk.columns:
                    chunk[eventcol] = chunk[alt]
                    break

        # Ensure service column
        if servicecol not in chunk.columns:
            chunk[servicecol] = os.path.basename(path)

        # Parse timestamps
        epoch_ms = to_epoch_ms(chunk[timestamp_col])

        # Build events
        for i, row in chunk.iterrows():
            ts = epoch_ms.loc[i]
            if pd.isna(ts):
                continue
            evt = row.get(eventcol, "")
            svc = row.get(servicecol, "")
            events.append((float(ts), str(evt), str(svc)))

        pbar.update(len(chunk))

    pbar.close()
    return events

def build_logs(input_dir, output_file, templates_map_file=None,
               timestamp_col='timestamp', eventcol='EventId',
               servicecol='log_name', n_jobs=4):
    """
    Walk input_dir for '*_structured.csv' files → numpy file with compact logs.
    """
    files = []
    for root, _, filenames in os.walk(input_dir):
        for f in filenames:
            if f.endswith('_structured.csv'):
                files.append(os.path.join(root, f))

    if not files:
        raise RuntimeError(f"No '*_structured.csv' files found in {input_dir}")

    print(f"Processing {len(files)} structured files with {n_jobs} workers...\n")

    # Process in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_structured_csv)(
            path, timestamp_col, eventcol, servicecol
        ) for path in tqdm(files, desc="Overall progress", unit="file", ncols=100)
    )

    # Build dictionary
    logs = OrderedDict()
    all_event_ids = set()
    for path, events in zip(files, results):
        key = os.path.basename(path).replace('_structured.csv', '')
        events.sort(key=lambda x: x[0])
        logs[key] = events
        for _, evt, _ in events:
            all_event_ids.add(evt)

    # Map EventId -> int
    event_list = sorted(all_event_ids)
    event_map = {eid: i for i, eid in enumerate(event_list)}

    logs_compact = {}
    for key, evs in logs.items():
        logs_compact[key] = [(ts, event_map.get(eid, -1), svc) for (ts, eid, svc) in evs]

    # Save
    np.save(output_file, logs_compact, allow_pickle=True)
    print(f"✅ Saved stratification logs → {output_file}")

    if templates_map_file:
        with open(templates_map_file, 'w', encoding='utf-8') as f:
            json.dump(event_map, f, indent=2)
        print(f"✅ Saved EventId → index map → {templates_map_file}")

    return logs_compact, event_map

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Dir with *_structured.csv (Drain output)")
    ap.add_argument("--output", required=True, help="Path for stratification_logs.npy")
    ap.add_argument("--map", required=False, help="Optional JSON map file")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--event-col", default="EventId")
    ap.add_argument("--service-col", default="log_name")
    ap.add_argument("--n_jobs", type=int, default=4, help="Number of parallel workers")
    args = ap.parse_args()

    build_logs(args.input, args.output, templates_map_file=args.map,
               timestamp_col=args.timestamp_col,
               eventcol=args.event_col,
               servicecol=args.service_col,
               n_jobs=args.n_jobs)
