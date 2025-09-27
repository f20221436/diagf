# tools/run_drain_all.py
import os
import sys
import argparse
from tqdm import tqdm

# --- Fix sys.path so local Drain module is importable ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
drain_path = os.path.join(project_root, "log", "logparser")
sys.path.insert(0, drain_path)

from Drain.Drain import LogParser  # import local Drain


def run_all(prepared_root, out_root, st=0.5, depth=4, sample_ratio=0.3):
    """
    Run Drain parser on all prepared log files, but only process `sample_ratio` (default 30%) of lines.
    """
    os.makedirs(out_root, exist_ok=True)
    log_format = '<log_id>,<timestamp>,<cmdb_id>,<log_name>,<Content>'
    regex = []

    # Collect files
    file_list = []
    for f in sorted(os.listdir(prepared_root)):
        fp = os.path.join(prepared_root, f)
        if os.path.isdir(fp):
            for subf in sorted(os.listdir(fp)):
                file_list.append((f, subf))
        else:
            file_list.append((None, f))

    print(f"Parsing {len(file_list)} files with Drain (only {int(sample_ratio*100)}% of each)...\n")

    parser = LogParser(
        log_format,
        indir=prepared_root,
        outdir=out_root,
        depth=depth,
        st=st,
        rex=regex,
        keep_para=True,
    )

    for idx, item in enumerate(file_list, start=1):
        if item[0] is None:
            f = item[1]
            parser.path = prepared_root
            parser.savePath = out_root
            file_path = os.path.join(prepared_root, f)
        else:
            subdir, f = item
            parser.path = os.path.join(prepared_root, subdir)
            parser.savePath = os.path.join(out_root, subdir)
            os.makedirs(parser.savePath, exist_ok=True)
            file_path = os.path.join(parser.path, f)

        print(f"[{idx}/{len(file_list)}] Parsing file: {file_path}")

        # Count total lines
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fin:
            total_lines = sum(1 for _ in fin)

        # Decide number of lines to keep
        keep_lines = int(total_lines * sample_ratio)
        if keep_lines == 0:
            print(f"⚠ Skipping {file_path}, no lines to process.")
            continue

        # Create a temp sampled file
        sampled_path = file_path + ".sampled"
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fin, \
             open(sampled_path, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i >= keep_lines:
                    break
                fout.write(line)

        print(f"   → Using {keep_lines}/{total_lines} lines ({sample_ratio*100:.0f}%)")

        # Parse only the sampled file
        parser.parse(os.path.basename(sampled_path))

        # Cleanup sampled file to save space
        os.remove(sampled_path)

    print("\n✅ Drain parsing completed (sampled data).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--st", default=0.5, type=float)
    ap.add_argument("--depth", default=4, type=int)
    ap.add_argument("--sample_ratio", default=0.3, type=float, help="Fraction of data to use (default 0.3 = 30%)")
    args = ap.parse_args()

    run_all(args.input, args.output, args.st, args.depth, args.sample_ratio)
