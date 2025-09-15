import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..')))

import json
import glob
import pandas as pd
from detector.k_sigma import Ksigma
import datetime as dt
import pytz
import time
from tqdm import tqdm
import public_function as pf
import numbers
import math
from typing import Optional
import multiprocessing

# timezone used for ts->date conversion (kept the original Asia/Shanghai)
tz = pytz.timezone('Asia/Shanghai')


def ts_to_date(timestamp: int) -> str:
    """
    Convert epoch seconds OR epoch milliseconds to a timezone-aware datetime string.
    Returns string 'YYYY-MM-DD HH:MM:SS'
    """
    try:
        # handle milliseconds
        if timestamp is None:
            return ''
        ts = int(timestamp)
        if abs(ts) > 1e11:
            ts = ts // 1000
        return dt.datetime.fromtimestamp(ts, tz).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        try:
            # fallback treating as ms
            return dt.datetime.fromtimestamp(int(timestamp)//1000, tz).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return ''


def time_to_ts(ctime) -> Optional[int]:
    """
    Convert various time representations to integer epoch seconds.
    Accepts:
    
    - floats/ints: epoch seconds or epoch milliseconds
    - numpy floats/ints
    - strings: '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', or numeric epoch strings
    - datetime.datetime objects
    Returns:
    - int(seconds since epoch) or None if input is None/NaN/unparseable
    """
    if ctime is None:
        return None

    # handle numeric types (including numpy types)
    if isinstance(ctime, numbers.Number):
        # check for NaN / inf
        if math.isnan(ctime) or math.isinf(ctime):
            return None
        # heuristics: if very large (>1e11) it's ms, else seconds
        if abs(ctime) > 1e11:         # e.g. 1.63e12 -> milliseconds
            return int(ctime // 1000)
        else:
            return int(ctime)

    # datetime objects
    if isinstance(ctime, dt.datetime):
        try:
            return int(ctime.timestamp())
        except Exception:
            return None

    # strings: try common formats, then try numeric parse
    if isinstance(ctime, str):
        s = ctime.strip()
        if not s:
            return None
        # try common datetime formats
        for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
            try:
                d = dt.datetime.strptime(s, fmt)
                # assume naive dt is in the same tz as ts_to_date usage (but return epoch seconds)
                return int(d.replace(tzinfo=tz).timestamp())
            except Exception:
                pass
        # if string is numeric epoch (seconds or ms)
        try:
            f = float(s)
            if abs(f) > 1e11:
                return int(f // 1000)
            else:
                return int(f)
        except Exception:
            return None

    # anything else: give up
    return None


def read_metric_file(data_dir: str, metric) -> pd.DataFrame:
    """
    Robustly open a metric CSV file.
    - metric may be:
      * a full filename (with .csv) (preferred)
      * a name without .csv
      * a clean_name that is a prefix of actual files
      * a pandas Series/row with fields 'filename' or 'name'
    - returns DataFrame or raises FileNotFoundError with helpful message
    """
    # Normalize metric into a string candidate
    if metric is None:
        raise FileNotFoundError("No metric name provided.")

    if isinstance(metric, (pd.Series, dict)):
        # prefer explicit 'filename' field if available
        if 'filename' in metric and pd.notna(metric['filename']):
            metric = str(metric['filename'])
        elif 'name' in metric and pd.notna(metric['name']):
            metric = str(metric['name'])
        elif 'clean_name' in metric and pd.notna(metric['clean_name']):
            metric = str(metric['clean_name'])
        else:
            # fallback to first available column
            metric = str(metric.iloc[0]) if isinstance(metric, pd.Series) else str(list(metric.values())[0])

    metric = str(metric)

    candidate_paths = []

    # 1) exact join
    candidate_paths.append(os.path.join(data_dir, metric))

    # 2) with .csv appended if missing
    if not metric.lower().endswith('.csv'):
        candidate_paths.append(os.path.join(data_dir, metric + '.csv'))

    # 3) try wildcard match: files starting with metric
    candidate_paths += glob.glob(os.path.join(data_dir, f"{metric}*.csv"))
    # 4) try any file containing metric substring
    candidate_paths += glob.glob(os.path.join(data_dir, f"*{metric}*.csv"))

    # Remove duplicates while preserving order
    seen = set(); candidates = []
    for p in candidate_paths:
        if p not in seen:
            seen.add(p); candidates.append(p)

    # Use first existing candidate
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                # If there's a timestamp column we'll index with converted readable times
                if 'timestamp' in df.columns:
                    try:
                        df.index = [ts_to_date(ts) for ts in df['timestamp']]
                    except Exception:
                        pass
                return df
            except Exception as e:
                raise IOError(f"Found file {p} but failed to read as CSV: {e}")

    # If none found, raise detailed error listing samples in folder
    sample_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))[:20]
    raise FileNotFoundError(
        f"No metric file found for '{metric}' in {data_dir}.\n"
        f"Tried candidates: {candidates}\n"
        f"Sample files in folder (first 20): {sample_files}\n"
        "Tip: ensure your metric index CSV 'name' column contains the exact filename (including .csv),\n"
        "or update config to point to the correct metric folder."
    )


class MetricEvent:
    def __init__(self, cases: pd.DataFrame, metric_path: Optional[str], data_dir: str,
                dataset: str = 'gaia', config: Optional[dict] = None):
        self.cases = cases
        self.periods = ['2021-07-01_2021-07-15', '2021-07-15_2021-07-31']
        self.data_dir = data_dir
        self.dataset = dataset

        # load metric list depending on dataset
        if dataset == 'gaia':
            if metric_path is None:
                raise ValueError("metric_path must be provided for dataset 'gaia'")
            # metric_path can be a CSV; allow either 'name' column storing filename or cleaned 'clean_name'
            metrics_info = pd.read_csv(metric_path)
            # construct metric list: prefer explicit filename if present (with .csv),
            # otherwise try to reconstruct filenames by appending periods for each period
            metrics = []
            for _, row in metrics_info.iterrows():
                # if the index CSV already contains full filename values in 'name' or 'filename', use them:
                if 'filename' in row and pd.notna(row['filename']):
                    metrics.append(row['filename'])
                    continue
                if 'name' in row and pd.notna(row['name']) and str(row['name']).lower().endswith('.csv'):
                    metrics.append(str(row['name']))
                    continue
                # otherwise, try to build using row['name'] (clean name) + period(s)
                base_name = None
                if 'name' in row and pd.notna(row['name']):
                    base_name = str(row['name'])
                elif 'clean_name' in row and pd.notna(row['clean_name']):
                    base_name = str(row['clean_name'])
                else:
                    # fallback to first column value
                    base_name = str(row.iloc[0])
                # try both periods, building filenames: base_name + '_' + period + '.csv'
                for p in self.periods:
                    metrics.append(f"{base_name}_{p}.csv")
            self.metrics = metrics
        elif dataset == '20aiops':
            # for 20aiops, data_dir contains metric files directly
            self.metrics = sorted(os.listdir(data_dir))
        else:
            raise Exception(f'Unknown dataset {dataset}')

        # keep original detector for backward compatibility in single-threaded mode
        self.detector = Ksigma()
        if config is None:
            config = {}
        # ensure required defaults exist even if config was provided by pf.get_config()
        # this prevents KeyError: 'minute' when users pass a project config that doesn't include these keys
        config.setdefault('minute', 60000)
        config.setdefault('MIN_TEST_LENGTH', 5)
        # keep n_jobs optional; it's resolved when used below
        self.config = config

        # results structure keyed by case index -> list of detections
        self.res = dict(zip(list(cases.index), [[] for _ in range(len(cases))]))

    def read(self, metric):
        """Wrapper to call the robust reader"""
        return read_metric_file(self.data_dir, metric)

    def get_metric_events(self):
        """
        Main loop parallelized across metrics. Behavior is preserved:
        - metrics are processed in the original order
        - for each metric, cases are iterated in the original order
        - detection results are appended to self.res preserving the order they would have in serial run
        """
        # number of worker processes to use
        n_jobs = int(self.config.get('n_jobs', max(1, multiprocessing.cpu_count())))

        # If n_jobs == 1, keep serial behavior (and reuse the existing detector instance)
        if n_jobs <= 1:
            for metric in tqdm(self.metrics, desc="metrics"):
                try:
                    metric_data = self.read(metric)
                except FileNotFoundError as e:
                    print(f"[WARN] Metric file not found for '{metric}': {e}")
                    continue
                except Exception as e:
                    print(f"[ERROR] Failed to load metric '{metric}': {e}")
                    continue

                # iterate through all cases
                for case_id, case in self.cases.iterrows():
                    try:
                        if self.dataset == 'gaia':
                            start_ts = time_to_ts(case.get('st_time'))
                            if start_ts is None:
                                # no start time for this case -> skip
                                continue
                            # subtract minutes (config['minute'] is in ms by default in this repo)
                            start_ts = start_ts - (self.config['minute'] * 40)

                            end_ts = time_to_ts(case.get('ed_time'))
                            if end_ts is None:
                                # if end time missing, skip this case
                                continue
                        elif self.dataset == '20aiops':
                            interval = int(metric.split('-')[-1].replace('.csv', ''))
                            before_min = interval * 65
                            after_min = interval * 2
                            start_ts = time_to_ts(case['st_time']) - self.config['minute'] * before_min
                            end_ts = time_to_ts(case['ed_time']) + self.config['minute'] * after_min
                        else:
                            raise Exception(f'Unknown dataset {self.dataset}')

                        res = self.detector.detection(metric_data, 'value', start_ts, end_ts)
                        if res is None:
                            continue
                        if isinstance(res, (list, tuple)) and len(res) >= 1 and res[0] is True:
                            metric_splits = metric.split('_')
                            if self.dataset == 'gaia':
                                if len(metric_splits) >= 4:
                                    name = '_'.join(metric_splits[2:-1]).replace('.csv', '')
                                    service = metric_splits[0]
                                    address = metric_splits[1]
                                else:
                                    name = os.path.splitext(os.path.basename(metric))[0]
                                    service = ''
                                    address = ''
                            elif self.dataset == '20aiops':
                                try:
                                    name = metric_splits[2].replace('.csv', '')
                                    service = metric_splits[1]
                                    address = metric_splits[0]
                                except Exception:
                                    name = os.path.splitext(os.path.basename(metric))[0]
                                    service = ''
                                    address = ''
                            else:
                                raise Exception(f'Unknown dataset {self.dataset}')

                            try:
                                ts_detect = int(res[1])
                            except Exception:
                                ts_detect = int(time.time())
                            combined = f'{address}_{service}' if address or service else os.path.splitext(os.path.basename(metric))[0]
                            extras = res[2] if len(res) > 2 else None
                            self.res[case_id].append((ts_detect, combined, name, extras))

                    except Exception as e:
                        print(f"[WARN] error processing case {case_id} for metric {metric}: {e}")
                        continue
            return

        # --- Parallel path ---
        # Prepare lightweight serializable structures for worker processes
        cases_dict = self.cases.to_dict('index')
        case_ids = list(self.cases.index)
        # build input arguments for each metric so worker has everything it needs
        worker_inputs = []
        for metric in self.metrics:
            worker_inputs.append((metric, self.data_dir, self.dataset, case_ids, cases_dict, self.config, self.periods))

        # Worker function must be defined at module top-level (see below)
        with multiprocessing.Pool(processes=n_jobs) as pool:
            # use imap to preserve input order while allowing a progress bar
            for per_metric_result in tqdm(pool.imap(_process_metric_worker, worker_inputs), total=len(worker_inputs), desc="metrics"):
                # per_metric_result is a list of tuples (case_id, detection_tuple)
                if not per_metric_result:
                    continue
                for case_id, detection in per_metric_result:
                    # append preserving metric order (we iterated metrics in original order)
                    self.res[case_id].append(detection)
                    
    def save_res(self, savepath: str):
        """Persist self.res to savepath as JSON (creates dir if needed)."""
        # ensure the dir exists
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, 'w', encoding='utf-8') as f:
            json.dump(self.res, f)
        print(f'Save successfully to {savepath}!')



def _process_metric_worker(args):
    """
    Worker function for processing a single metric. It returns a list of (case_id, detection_tuple).
    args is a tuple: (metric, data_dir, dataset, case_ids, cases_dict, config, periods)
    """
    metric, data_dir, dataset, case_ids, cases_dict, config, periods = args
    per_case_detections = []

    # read the metric file
    try:
        metric_data = read_metric_file(data_dir, metric)
    except FileNotFoundError as e:
        print(f"[WARN] Metric file not found for '{metric}': {e}")
        return per_case_detections
    except Exception as e:
        print(f"[ERROR] Failed to load metric '{metric}': {e}")
        return per_case_detections

    # create a detector instance per worker to avoid pickling issues
    detector = Ksigma()

    for case_id in case_ids:
        try:
            case = cases_dict.get(case_id, {})
            if dataset == 'gaia':
                start_ts = time_to_ts(case.get('st_time'))
                if start_ts is None:
                    continue
                start_ts = start_ts - (config['minute'] * 40)

                end_ts = time_to_ts(case.get('ed_time'))
                if end_ts is None:
                    continue
            elif dataset == '20aiops':
                try:
                    interval = int(metric.split('-')[-1].replace('.csv', ''))
                except Exception:
                    interval = 1
                before_min = interval * 65
                after_min = interval * 2
                start_ts = time_to_ts(case.get('st_time')) - config['minute'] * before_min
                end_ts = time_to_ts(case.get('ed_time')) + config['minute'] * after_min
            else:
                print(f"[ERROR] Unknown dataset {dataset} in worker")
                continue

            res = detector.detection(metric_data, 'value', start_ts, end_ts)
            if res is None:
                continue
            if isinstance(res, (list, tuple)) and len(res) >= 1 and res[0] is True:
                metric_splits = metric.split('_')
                if dataset == 'gaia':
                    if len(metric_splits) >= 4:
                        name = '_'.join(metric_splits[2:-1]).replace('.csv', '')
                        service = metric_splits[0]
                        address = metric_splits[1]
                    else:
                        name = os.path.splitext(os.path.basename(metric))[0]
                        service = ''
                        address = ''
                elif dataset == '20aiops':
                    try:
                        name = metric_splits[2].replace('.csv', '')
                        service = metric_splits[1]
                        address = metric_splits[0]
                    except Exception:
                        name = os.path.splitext(os.path.basename(metric))[0]
                        service = ''
                        address = ''
                else:
                    # shouldn't happen
                    name = os.path.splitext(os.path.basename(metric))[0]
                    service = ''
                    address = ''

                try:
                    ts_detect = int(res[1])
                except Exception:
                    ts_detect = int(time.time())
                combined = f'{address}_{service}' if address or service else os.path.splitext(os.path.basename(metric))[0]
                extras = res[2] if len(res) > 2 else None
                per_case_detections.append((case_id, (ts_detect, combined, name, extras)))

        except Exception as e:
            print(f"[WARN] error processing case {case_id} for metric {metric} in worker: {e}")
            continue

    return per_case_detections


# --------------------
# CLI / module entry
# --------------------
if __name__ == '__main__':
    config = pf.get_config()
    project_root_dir = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..'))

    # determine label_path: support both he_dgl.run_table and default config layout
    if 'he_dgl' in config and 'run_table' in config['he_dgl']:
        label_path = config['he_dgl']['run_table']
    else:
        label_path = os.path.abspath(os.path.join(project_root_dir,
                                                config['base_path'],
                                                config['demo_path'],
                                                config['label'], 'demo.csv'))

    labels = pd.read_csv(label_path)
    metric_info_path = config.get('metric_info_path')
    metric_data_dir = config.get('metric_data_dir')
    dataset = config.get('dataset', 'gaia')

    # instantiate MetricEvent and run
    metric_event = MetricEvent(labels, metric_info_path, metric_data_dir, dataset, config)
    metric_event.get_metric_events()

    save_path = os.path.abspath(os.path.join(project_root_dir,
                                            pf.deal_config(config, 'parse')['metric_path']))
    metric_event.save_res(save_path)
