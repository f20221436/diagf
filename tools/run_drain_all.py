# tools/run_drain_all.py
"""
Run Drain log parser across a directory of prepared log files.

Features:
- Keeps core Drain usage unchanged (instantiates LogParser and calls parser.parse(fname)).
- Per-file timeout via running parse() in a child process.
- Optionally process very large files sequentially first (threshold controlled by --sequential_size_bytes).
- Monitor CPU/memory while parsing (uses psutil if available).
- Subsample input via --data_fraction (default 0.3) with two modes:
    - --fraction_by size: choose files until cumulative bytes ≈ fraction * total_bytes (random shuffle for fairness).
    - --fraction_by count: randomly select fraction * total_files files (uniform).
- Logging to console and file (default run_drain_all.log).
"""

import os
import sys
import argparse
import logging
import time
import multiprocessing as mp
import threading
import random
from joblib import Parallel, delayed

# --- Fix sys.path so local Drain module is importable ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "log", "logparser", "logparser"))
sys.path.insert(0, os.path.join(project_root, "log", "logparser"))

# Try imports in a forgiving way
try:
    from Drain.Drain import LogParser
except Exception:
    try:
        from logparser.Drain.Drain import LogParser
    except Exception as e:
        print("Failed to import local LogParser from Drain.py:", e)
        raise

# Optional psutil for better monitoring; fallback if unavailable
try:
    import psutil
except Exception:
    psutil = None

# -----------------------
# Logging setup
# -----------------------
logger = logging.getLogger("run_drain_all")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(log_formatter)
logger.addHandler(ch)

# File handler will be added after args are parsed.

# -----------------------
# Monitoring helpers
# -----------------------
def get_basic_system_stats():
    """Return a dict with some basic system stats (works without psutil)."""
    try:
        load = os.getloadavg() if hasattr(os, "getloadavg") else None
    except Exception:
        load = None
    try:
        du = os.statvfs(".")
        free_bytes = du.f_bavail * du.f_frsize
    except Exception:
        free_bytes = None
    return {"loadavg": load, "free_disk_bytes": free_bytes}

def sample_system_stats(stop_event, interval, out_list):
    """If psutil available, sample cpu and memory every `interval` seconds and append to out_list."""
    if psutil is None:
        return
    while not stop_event.is_set():
        try:
            cpu = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            out_list.append({"time": time.time(), "cpu": cpu, "mem_percent": vm.percent, "mem_used": vm.used})
        except Exception:
            pass
        # sleep in small increments so stop_event is responsive
        for _ in range(int(max(1, interval * 10))):
            if stop_event.is_set():
                break
            time.sleep(interval / 10.0)

# -----------------------
# Worker that runs parser.parse() in a separate process
# -----------------------
def _parse_worker(indir, outdir, fname, st, depth, regex, keep_para, result_queue):
    """Child process entrypoint: instantiate LogParser and call parse()."""
    try:
        parser = LogParser(
            '<log_id>,<timestamp>,<cmdb_id>,<log_name>,<Content>',
            indir=indir,
            outdir=outdir,
            depth=depth,
            st=st,
            rex=regex,
            keep_para=keep_para,
        )
        parser.path = indir
        parser.savePath = outdir
        parser.parse(fname)
        result_queue.put({"status": "ok"})
    except Exception as e:
        # Send minimal error info back to parent; avoid complex objects.
        result_queue.put({"status": "error", "error": str(e)})

def parse_one_file(prepared_root, out_root, fname, st, depth, regex, idx, total, timeout=None, keep_para=True):
    """
    Run parser.parse for a single file in a child process, monitor resources,
    and handle timeout / errors gracefully.
    """
    indir = prepared_root
    outdir = out_root
    file_path = os.path.join(indir, fname)
    logger.info(f"[{idx}/{total}] ▶ Starting: {file_path}")

    monitor_samples = []
    stop_monitor = threading.Event()
    monitor_thread = None
    if psutil is not None:
        monitor_thread = threading.Thread(target=sample_system_stats, args=(stop_monitor, 1.0, monitor_samples))
        monitor_thread.daemon = True
        monitor_thread.start()

    start = time.time()
    q = mp.Queue()
    p = mp.Process(target=_parse_worker, args=(indir, outdir, fname, st, depth, regex, keep_para, q))
    p.start()
    p_pid = p.pid

    try:
        p.join(timeout=timeout)
        if p.is_alive():
            logger.warning(f"[{idx}/{total}] ⚠ Timeout after {timeout}s. Terminating parser for {fname} (pid={p_pid})")
            p.terminate()
            p.join(5)
            result = {"status": "timeout"}
        else:
            try:
                result = q.get_nowait()
            except Exception:
                result = {"status": "ok"}
    except Exception as e:
        logger.exception(f"[{idx}/{total}] ✖ Exception while running parser for {fname}: {e}")
        result = {"status": "error", "error": str(e)}
        if p.is_alive():
            p.terminate()
            p.join(5)
    finally:
        stop_monitor.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)

    elapsed = time.time() - start

    # attempt to report lines processed if readable
    try:
        total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8", errors="ignore"))
        speed = total_lines / elapsed if elapsed > 0 else 0
        lines_info = f"{total_lines} lines in {elapsed:.2f}s ({speed:.1f} lines/s)"
    except Exception:
        lines_info = f"in {elapsed:.2f}s (line count unavailable)"

    # monitoring summary
    monitor_summary = ""
    if monitor_samples:
        try:
            max_cpu = max(s["cpu"] for s in monitor_samples)
            max_mem = max(s["mem_percent"] for s in monitor_samples)
            monitor_summary = f" | peak_cpu={max_cpu:.1f}% peak_mem={max_mem:.1f}% samples={len(monitor_samples)}"
        except Exception:
            monitor_summary = ""
    else:
        basic = get_basic_system_stats()
        if basic["loadavg"] is not None or basic["free_disk_bytes"] is not None:
            monitor_summary = f" | basic_stats={basic}"

    status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
    if status == "ok":
        logger.info(f"[{idx}/{total}] ✔ Done {os.path.basename(file_path)}: {lines_info}{monitor_summary}")
    elif status == "timeout":
        logger.error(f"[{idx}/{total}] ✖ Timeout/terminated {os.path.basename(file_path)} after {timeout}s{monitor_summary}")
    elif status == "error":
        err = result.get("error", "<no-error-info>")
        logger.error(f"[{idx}/{total}] ✖ Error parsing {os.path.basename(file_path)}: {err}{monitor_summary}")
    else:
        logger.warning(f"[{idx}/{total}] ? Unknown result for {os.path.basename(file_path)}: {result}{monitor_summary}")

# -----------------------
# Subsampling helpers
# -----------------------
def pick_fraction_by_count(file_entries, fraction, seed=None):
    """Select fraction of files by count (random subset, reproducible via seed)."""
    total = len(file_entries)
    k = max(1, int(round(total * fraction)))
    rng = random.Random(seed)
    chosen = rng.sample(file_entries, k)
    chosen_set = set(chosen)
    # preserve original ordering
    return [fe for fe in file_entries if fe in chosen_set]

def pick_fraction_by_size(file_entries, fraction, seed=None):
    """
    Select files by cumulative size until reaching fraction * total_bytes.
    Shuffles entries to avoid always picking the same large files.
    """
    entries_with_size = []
    total_bytes = 0
    for indir, fname in file_entries:
        p = os.path.join(indir, fname)
        try:
            s = os.path.getsize(p)
        except Exception:
            s = 0
        entries_with_size.append((indir, fname, s))
        total_bytes += s

    target = total_bytes * fraction
    if target <= 0:
        return pick_fraction_by_count([(i, f) for (i, f, _) in entries_with_size], fraction, seed)

    rng = random.Random(seed)
    rng.shuffle(entries_with_size)

    chosen = []
    cum = 0
    for indir, fname, s in entries_with_size:
        if cum >= target:
            break
        chosen.append((indir, fname))
        cum += s

    if not chosen and entries_with_size:
        # fallback: pick first file if everything else zero-sized
        chosen = [(entries_with_size[0][0], entries_with_size[0][1])]

    return chosen

# -----------------------
# Main orchestration
# -----------------------
def run_all(prepared_root, out_root, st=0.5, depth=4, n_jobs=1, timeout=None, sequential_size_bytes=0, keep_para=True, data_fraction=0.3, fraction_by="size", seed=42):
    os.makedirs(out_root, exist_ok=True)
    regex = []

    # collect files (flatten nested service directories)
    file_list = []
    for entry in sorted(os.listdir(prepared_root)):
        full = os.path.join(prepared_root, entry)
        if os.path.isfile(full):
            file_list.append((prepared_root, entry))
        elif os.path.isdir(full):
            for fname in sorted(os.listdir(full)):
                file_list.append((os.path.join(prepared_root, entry), fname))

    total_files_all = len(file_list)
    if total_files_all == 0:
        logger.warning("No files found in %s", prepared_root)
        return

    # apply data_fraction subsampling if requested
    if data_fraction is not None and 0.0 < data_fraction < 1.0:
        logger.info("Selecting %.1f%% of data using method '%s' (seed=%s)", data_fraction * 100.0, fraction_by, seed)
        if fraction_by == "count":
            subset = pick_fraction_by_count(file_list, data_fraction, seed=seed)
        else:
            subset = pick_fraction_by_size(file_list, data_fraction, seed=seed)
        file_list = subset
        logger.info("Selected %d files out of %d available (%.1f%%).", len(file_list), total_files_all, 100.0 * len(file_list) / max(1, total_files_all))
    else:
        logger.info("Processing all files (data_fraction=%s).", data_fraction)

    total_files = len(file_list)
    if total_files == 0:
        logger.warning("After sampling there are no files to process. Exiting.")
        return

    # Optionally treat very large files sequentially first
    if sequential_size_bytes and sequential_size_bytes > 0:
        large = []
        small = []
        for indir, fname in file_list:
            try:
                p = os.path.join(indir, fname)
                size = os.path.getsize(p)
                if size >= sequential_size_bytes:
                    large.append((indir, fname, size))
                else:
                    small.append((indir, fname, size))
            except Exception:
                small.append((indir, fname, 0))
        large_sorted = sorted(large, key=lambda x: x[2], reverse=True)
        small_sorted = sorted(small, key=lambda x: x[2], reverse=False)
        file_list = [(i, f) for (i, f, s) in large_sorted] + [(i, f) for (i, f, s) in small_sorted]
        logger.info("Will process %d large files sequentially first (threshold=%d bytes).", len(large_sorted), sequential_size_bytes)
    else:
        file_list = [(i, f) for (i, f) in file_list]

    logger.info(f"Found {total_files} files to process — running Drain with {n_jobs} worker(s)...")

    if n_jobs == 1:
        for idx, (indir, fname) in enumerate(file_list, start=1):
            parse_one_file(indir, out_root, fname, st, depth, regex, idx, total_files, timeout=timeout, keep_para=keep_para)
    else:
        if sequential_size_bytes and sequential_size_bytes > 0:
            idx_split = 0
            for idx_tmp, (indir, fname) in enumerate(file_list):
                p = os.path.join(indir, fname)
                try:
                    if os.path.getsize(p) < sequential_size_bytes:
                        break
                except Exception:
                    break
                idx_split += 1
            # run large ones sequentially
            for idx, (indir, fname) in enumerate(file_list[:idx_split], start=1):
                parse_one_file(indir, out_root, fname, st, depth, regex, idx, total_files, timeout=timeout, keep_para=keep_para)

            remainder = file_list[idx_split:]
            if remainder:
                Parallel(n_jobs=n_jobs)(
                    delayed(parse_one_file)(indir, out_root, fname, st, depth, regex, idx + idx_split, total_files, timeout, keep_para)
                    for idx, (indir, fname) in enumerate(remainder, start=1)
                )
        else:
            Parallel(n_jobs=n_jobs)(
                delayed(parse_one_file)(indir, out_root, fname, st, depth, regex, idx, total_files, timeout, keep_para)
                for idx, (indir, fname) in enumerate(file_list, start=1)
            )

    logger.info("\n✅ Selected files parsed.")

# -----------------------
# CLI entrypoint
# -----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Prepared input directory (e.g. mid/business_prepared)")
    ap.add_argument("--output", required=True, help="Output directory for Drain results")
    ap.add_argument("--st", default=0.5, type=float, help="Similarity threshold")
    ap.add_argument("--depth", default=4, type=int, help="Tree depth")
    ap.add_argument("--n_jobs", default=1, type=int, help="Number of parallel workers")
    ap.add_argument("--timeout", default=None, type=float, help="Per-file timeout in seconds (useful to kill hanging parse calls). Default: none")
    ap.add_argument("--sequential_size_bytes", default=0, type=int, help="Files >= this size (bytes) will be processed sequentially first. Default 0 (disabled).")
    ap.add_argument("--log_file", default=None, help="Optional log file path. If not provided, run_drain_all.log in current dir will be used.")
    ap.add_argument("--keep_para", action="store_true", help="Set keep_para=True for LogParser (keeps parameters). Default is False unless flag provided.")
    ap.add_argument("--data_fraction", default=0.3, type=float, help="Fraction of input data to process (0..1). Default 0.3 (30%%).")
    ap.add_argument("--fraction_by", choices=["size", "count"], default="size", help="How to interpret data_fraction: 'size' (by bytes) or 'count' (by number of files). Default 'size'.")
    ap.add_argument("--seed", default=42, type=int, help="Random seed for reproducible subsampling (when fraction_by=count or shuffle is used).")

    args = ap.parse_args()

    # Setup file logging now that args are parsed
    log_path = args.log_file or os.path.join(os.getcwd(), "run_drain_all.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(log_formatter)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info("Starting run_drain_all.py")
    logger.info("Input=%s Output=%s n_jobs=%s timeout=%s seq_size=%s data_fraction=%s fraction_by=%s seed=%s",
                args.input, args.output, args.n_jobs, args.timeout, args.sequential_size_bytes, args.data_fraction, args.fraction_by, args.seed)

    try:
        run_all(
            args.input,
            args.output,
            args.st,
            args.depth,
            args.n_jobs,
            timeout=args.timeout,
            sequential_size_bytes=args.sequential_size_bytes,
            keep_para=args.keep_para,
            data_fraction=args.data_fraction,
            fraction_by=args.fraction_by,
            seed=args.seed,
        )
    except Exception:
        logger.exception("Fatal error running run_all")
        raise
