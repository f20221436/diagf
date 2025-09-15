import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..')))
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pytz
import json
from detector.k_sigma import Ksigma
import public_function as pf
from tqdm import tqdm

tz = pytz.timezone('Asia/Shanghai')

def ts_to_date(timestamp):
    """
    Convert integer milliseconds timestamp to formatted string with timezone.
    """
    try:
        return datetime.fromtimestamp(int(timestamp) // 1000, tz).strftime('%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        return datetime.fromtimestamp(int(timestamp) // 1000, tz).strftime('%Y-%m-%d %H:%M:%S')

def time_to_ts(ctime):
    """
    Robust conversion to milliseconds since epoch.
    Accepts:
      - str like '2021-07-01' or '2021-07-01 12:34:56' or '2021-07-01 12:34:56.123'
      - numeric seconds (e.g. 1625097600) or milliseconds (1625097600000)
      - pandas.Timestamp or datetime
      - numpy.datetime64
    Returns:
      int milliseconds since epoch
    """
    import pandas as _pd

    if ctime is None or (isinstance(ctime, float) and np.isnan(ctime)):
        raise ValueError(f"Invalid/empty time value: {ctime}")

    # datetime or pandas Timestamp
    if isinstance(ctime, datetime):
        return int(ctime.timestamp() * 1000)
    if isinstance(ctime, _pd.Timestamp):
        return int(ctime.value // 10**6)
    if isinstance(ctime, np.datetime64):
        return int(ctime.astype('datetime64[ms]').astype('int64'))

    # numeric types (seconds or milliseconds)
    if isinstance(ctime, (int, float, np.integer, np.floating)):
        t = int(ctime)
        if t > 10**12:      # probably already ms
            return t
        if t > 10**9:       # seconds -> ms
            return int(t * 1000)
        raise ValueError(f"Numeric timestamp too small/unexpected: {ctime}")

    # strings: try several formats, then pandas fallback
    if isinstance(ctime, str):
        s = ctime.strip()
        for fmt in ('%Y-%m-%d %H:%M:%S.%f',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d'):
            try:
                dt = datetime.strptime(s, fmt)
                return int(dt.timestamp() * 1000)
            except Exception:
                pass
        try:
            ts = _pd.to_datetime(s, errors='raise')
            return int(ts.value // 10**6)
        except Exception:
            pass

    raise ValueError(f"Unsupported time format/value: {repr(ctime)}")


class TraceUtils:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pairs = self.getPairs()

    def getPairs(self):
        services_pairs = {'webservice': ['mobservice', 'redisservice'], 'mobservice': ['redisservice'],
                          'logservice': ['dbservice', 'redisservice'], 'dbservice': ['redisservice']}
        pairs = []
        for caller in services_pairs:
            for callee in services_pairs[caller]:
                for i in [1, 2]:
                    for j in [1, 2]:
                        pairs.append((caller + str(i), callee + str(j)))
        pairs.extend([('logservice1', 'logservice2'), ('logservice2', 'logservice1')])
        return pairs

    def get_trace_by_day(self, day: int):
        """
        Original code expected per-service subfolders with files named like:
          {service}/trace_{service}_2021-07-{day}.csv
        But GAIA can be shipped with flat filenames (e.g., trace_table_dbservice1_2021-07.csv).
        This helper tries both patterns.
        """
        day = f'0{day}' if day < 10 else str(day)
        temp = []
        for fname in os.listdir(self.data_dir):
            # try both potential patterns
            candidate1 = os.path.join(self.data_dir, fname, f'trace_{fname}_2021-07-{day}.csv')  # folder style
            candidate2 = os.path.join(self.data_dir, f'trace_table_{fname}_2021-07.csv')  # some GAIA names
            candidate3 = os.path.join(self.data_dir, f'{fname}_2021-07-{day}.csv')
            candidate4 = os.path.join(self.data_dir, f'trace_table_{fname}_2021-07-{day}.csv')
            tried = [candidate1, candidate2, candidate3, candidate4]
            found = False
            for filepath in tried:
                if os.path.exists(filepath):
                    temp.append(pd.read_csv(filepath, index_col=None))
                    found = True
                    break
            # also if the listing itself is a csv file matching the day pattern, include it
            flat_path = os.path.join(self.data_dir, fname)
            if not found and os.path.isfile(flat_path) and flat_path.endswith('.csv'):
                # check if this file's name includes the desired month/day (loose)
                if '2021-07' in fname:
                    temp.append(pd.read_csv(flat_path, index_col=None))
        if not temp:
            return pd.DataFrame()
        return pd.concat(temp, ignore_index=True)

    def data_process(self, day: int):
        data = self.get_trace_by_day(day)
        if data.empty:
            return data
        cdata = data[['parent_id', 'service_name']].rename(columns={'parent_id': 'span_id', 'service_name': 'cservice_name'})
        return pd.merge(data, cdata, on='span_id', how='left')

    def turn_to_timeseries(self, day, savepath):
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        df = self.data_process(day)
        if df.empty:
            print(f"No data for day {day}")
            return
        # ensure timestamp -> ms
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).copy()
        df['timestamp'] = df['timestamp'].astype('int64') // 10**6  # ms
        date = ts_to_date(int(df['timestamp'].iloc[0])).split()[0]
        start_ts = time_to_ts(date)
        delta = 30000  # 30s
        # reasonable default points_count (can be large) - we will use 24h worth of 30s points
        points_count = (24 * 60 * 60 * 1000) // delta
        ts = [start_ts + delta * i for i in range(1, int(points_count) + 1)]
        for caller, callee in self.pairs:
            print("Processing pair:", caller, "->", callee)
            temp = df.loc[(df['service_name'] == caller) & (df['cservice_name'] == callee)]
            info = {'timestamp': ts, '200': [], '500': [], 'other': [], 'lagency': []}
            for k in range(int(points_count)):
                window_start = start_ts + k * delta
                window_end = start_ts + (k + 1) * delta
                chosen = temp.loc[(temp['timestamp'] >= window_start) & (temp['timestamp'] < window_end)]
                cur_lagency = 0
                if not chosen.empty and 'lagency' in chosen.columns:
                    try:
                        cur_lagency = max(0, float(np.mean(chosen['lagency'].values)))
                    except Exception:
                        cur_lagency = 0
                cur_200 = len(chosen.loc[chosen['status_code'] == 200]) if 'status_code' in chosen.columns else 0
                cur_500 = len(chosen.loc[chosen['status_code'] == 500]) if 'status_code' in chosen.columns else 0
                cur_other = len(chosen) - cur_200 - cur_500
                info['lagency'].append(cur_lagency)
                info['200'].append(cur_200)
                info['500'].append(cur_500)
                info['other'].append(cur_other)
            out_fname = os.path.join(savepath, f'{caller}_{callee}.csv')
            pd.DataFrame(info).to_csv(out_fname, index=False)
            print(" -> saved", out_fname)


class InvocationEvent:
    def __init__(self, cases, data_dir, dataset, trace_pairs_path=None, config=None):
        self.cases = cases
        self.data_dir = data_dir
        self.dataset = dataset
        self.trace_pairs_path = trace_pairs_path
        self.detector = Ksigma()
        self.pairs = self.getPairs()
        if config is None:
            config = {}
            config['minute'] = 60000
            config['MIN_TEST_LENGTH'] = 5
        self.config = config
        # initialize result dict keyed by case index
        self.res = dict(zip(list(cases.index), [[] for _ in range(len(cases))]))

    def getPairs(self):
        if self.dataset == 'gaia':
            services_pairs = {'webservice': ['mobservice', 'redisservice'], 'mobservice': ['redisservice'],
                              'logservice': ['dbservice', 'redisservice'], 'dbservice': ['redisservice']}
            pairs = []
            for caller in services_pairs:
                for callee in services_pairs[caller]:
                    for i in [1, 2]:
                        for j in [1, 2]:
                            pairs.append((caller + str(i), callee + str(j)))
            pairs.extend([('logservice1', 'logservice2'), ('logservice2', 'logservice1')])
            return pairs
        elif self.dataset == '20aiops':
            if self.trace_pairs_path and os.path.exists(self.trace_pairs_path):
                with open(self.trace_pairs_path, 'r') as f:
                    pairs = [eval(line.rstrip('\n')) for line in f]
                return pairs
            else:
                # fallback: return empty list if file not found
                return []
        else:
            raise Exception("Unknown dataset")

    def read(self, day, caller, callee):
        # allow either data_dir/day/<pair>.csv or data_dir/<pair>.csv depending on how trace_timeseries is produced
        filepath1 = os.path.join(self.data_dir, str(day), f'{caller}_{callee}.csv')
        filepath2 = os.path.join(self.data_dir, f'{caller}_{callee}.csv')
        if os.path.exists(filepath1):
            filepath = filepath1
        elif os.path.exists(filepath2):
            filepath = filepath2
        else:
            raise FileNotFoundError(f"Trace timeseries file not found for {caller}_{callee} (day={day})")
        data = pd.read_csv(filepath)
        # convert index to human-readable timestamps for detection api (detector expects index like 'YYYY-mm-dd HH:MM:SS.fff')
        if 'timestamp' in data.columns:
            data.index = [ts_to_date(int(ts)) for ts in data['timestamp'].astype('int64')]
        return data

    def get_invocation_events(self):
        # latency anomalies and count-of-500 anomalies
        if self.dataset == 'gaia':
            for case_id, case in tqdm(self.cases.iterrows()):
                day = int(str(case['datetime']).split('-')[-1])
                for caller, callee in self.pairs:
                    try:
                        invocation_data = self.read(day, caller, callee)
                    except FileNotFoundError:
                        continue
                    # before-one-minute to after-one-minute window (scaled by minute config)
                    start_ts = time_to_ts(case['st_time']) - int(self.config.get('minute', 60000)) * 31
                    end_ts = time_to_ts(case['ed_time']) + int(self.config.get('minute', 60000)) * 1
                    res1 = self.detector.detection(invocation_data, 'lagency', start_ts, end_ts)
                    res2 = self.detector.detection(invocation_data, '500', start_ts, end_ts)
                    if not (res1[0] or res2[0]):
                        continue
                    ts = None
                    if res1[0]:
                        ts = res1[1]
                        score = res1[2]
                    if res2[0]:
                        if ts is None:
                            ts = res2[1]
                        else:
                            ts = min(ts, res2[1])
                        if ts == res2[1]:
                            score = res2[2]
                    self.res[case_id].append((int(ts), caller, callee, score))
        elif self.dataset == '20aiops':
            for case_id, case in tqdm(self.cases.iterrows()):
                day = case['st_time'].split(" ")[0]
                for caller, callee in self.pairs:
                    temp_csv = os.path.join(self.data_dir, day, f'{caller}_{callee}.csv')
                    if not os.path.exists(temp_csv):
                        continue
                    invocation_data = self.read(day, caller, callee)
                    start_ts = time_to_ts(case['st_time']) - int(self.config.get('minute', 60000)) * 31
                    end_ts = time_to_ts(case['ed_time']) + int(self.config.get('minute', 60000)) * 1
                    res1 = self.detector.detection(invocation_data, 'lagency', start_ts, end_ts)
                    res2 = self.detector.detection(invocation_data, 'other', start_ts, end_ts)
                    if not (res1[0] or res2[0]):
                        continue
                    ts = None
                    if res1[0]:
                        ts = res1[1]
                        score = res1[2]
                    if res2[0]:
                        if ts is None:
                            ts = res2[1]
                        else:
                            ts = min(ts, res2[1])
                        if ts == res2[1]:
                            score = res2[2]
                    self.res[case_id].append((int(ts), caller, callee, score))
        else:
            raise Exception("Unknown dataset for invocation event processing")

    def save_res(self, savepath):
        # save JSON with ensure_ascii=False so non-ascii characters are readable
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, 'w', encoding='utf-8') as f:
            json.dump(self.res, f, ensure_ascii=False)
        print('Save successfully!')


if __name__ == '__main__':
    config = pf.get_config()
    project_root_dir = os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0],
                                                    '../..'))

    # --- patched label_path handling ---
    label_path = None
    # Prefer he_dgl.run_table if provided in config
    if isinstance(config.get('he_dgl'), dict) and config['he_dgl'].get('run_table'):
        candidate = config['he_dgl']['run_table']
        if os.path.isabs(candidate) and os.path.exists(candidate):
            label_path = candidate
        else:
            maybe1 = os.path.join(project_root_dir, config.get('base_path', '.'), config.get('demo_path', ''), candidate)
            maybe2 = os.path.join(project_root_dir, config.get('base_path', '.'), candidate)
            for p in (maybe1, maybe2):
                if os.path.exists(p):
                    label_path = p
                    break

    # fallback: look for demo.csv in demo_path/label/
    if label_path is None:
        label_path = os.path.abspath(os.path.join(project_root_dir,
                                                  config.get('base_path', '.'),
                                                  config.get('demo_path', ''),
                                                  config.get('label', ''), 'demo.csv'))

    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Label/run table not found. Tried: {label_path} — "
            f"set he_dgl.run_table in gaia_config.yaml or create demo.csv under demo_path."
        )
    # --- end patch ---

    # Trace input directory (must be set in your YAML)
    trace_data_dir = config.get('trace_data_dir')
    if not trace_data_dir:
        raise KeyError("Please set 'trace_data_dir' in gaia_config.yaml to point to your trace (or trace_timeseries) folder")

    # Load labels/run table
    labels = pd.read_csv(label_path)

    # Build invocation events
    trace_pairs_path = config.get('trace_pairs_path')  # may be None
    invocation_event = InvocationEvent(labels, trace_data_dir,
                                       config.get('dataset', 'gaia'), trace_pairs_path)
    invocation_event.get_invocation_events()

    # Save output - use parse.trace_path
    parse_cfg = pf.deal_config(config, 'parse')
    save_path = os.path.abspath(os.path.join(project_root_dir, parse_cfg.get('trace_path', parse_cfg.get('trace_data_dir'))))
    invocation_event.save_res(save_path)

