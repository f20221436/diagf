# tools/prepare_logs_for_drain.py
import os, re, argparse, csv
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm

# common timestamp regex
TS_RE = re.compile(r'(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)')

def guess_timestamp(line):
    m = TS_RE.search(line)
    if m:
        return m.group(1)
    return None

def prepare_file(inpath, outpath, service_name=None, chunk_size=10000):
    """Process one log file → CSV"""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    lid = 1
    buffer = []
    total_lines = sum(1 for _ in open(inpath, 'r', encoding='utf-8', errors='ignore'))

    with open(outpath, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout)
        with open(inpath, 'r', encoding='utf-8', errors='ignore') as fin:
            for line in tqdm(fin, total=total_lines, desc=f"Processing {os.path.basename(inpath)}", leave=False):
                line = line.rstrip('\n\r')
                if not line:
                    continue
                ts = guess_timestamp(line)
                if ts is None:
                    try:
                        ts = datetime.fromtimestamp(os.path.getmtime(inpath)).strftime('%Y-%m-%d %H:%M:%S.%f')
                    except:
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                cmdb = ''
                log_name = service_name if service_name else os.path.splitext(os.path.basename(inpath))[0]
                content = line.replace(',', ' ')
                buffer.append([str(lid), ts, cmdb, log_name, content])
                lid += 1

                if len(buffer) >= chunk_size:
                    writer.writerows(buffer)
                    buffer.clear()

        if buffer:
            writer.writerows(buffer)

def prepare_dir(input_root, output_root, n_jobs=1):
    os.makedirs(output_root, exist_ok=True)
    tasks = []
    for entry in sorted(os.listdir(input_root)):
        full = os.path.join(input_root, entry)
        if os.path.isfile(full):
            outf = os.path.join(output_root, entry + '.prepped')
            tasks.append((full, outf, os.path.splitext(entry)[0]))
        elif os.path.isdir(full):
            out_dir_service = os.path.join(output_root, entry)
            os.makedirs(out_dir_service, exist_ok=True)
            for fname in sorted(os.listdir(full)):
                inf = os.path.join(full, fname)
                outf = os.path.join(out_dir_service, fname + '.prepped')
                tasks.append((inf, outf, entry))

    print(f"Preparing {len(tasks)} log files with {n_jobs} workers...")
    Parallel(n_jobs=n_jobs)(
        delayed(prepare_file)(inf, outf, svc) for inf, outf, svc in tasks
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="MicroSS/business root dir")
    ap.add_argument("--output", required=True, help="Prepared output dir (mid/...)")
    ap.add_argument("--n_jobs", type=int, default=1, help="Number of parallel workers")
    args = ap.parse_args()

    prepare_dir(args.input, args.output, n_jobs=args.n_jobs)
    print("Prepared logs under", args.output)
