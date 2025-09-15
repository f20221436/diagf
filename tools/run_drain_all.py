# tools/run_drain_all.py
import os
import sys
import argparse

# --- Fix sys.path so local Drain module is importable ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
drain_path = os.path.join(project_root, "log", "logparser", "logparser")
sys.path.insert(0, drain_path)

# Now import your Drain
from Drain.Drain import LogParser


def run_all(prepared_root, out_root, st=0.5, depth=4, chunk_size=200000):
    os.makedirs(out_root, exist_ok=True)
    log_format = '<log_id>,<timestamp>,<cmdb_id>,<log_name>,<Content>'
    regex = []

    # collect all files
    file_list = []
    for entry in sorted(os.listdir(prepared_root)):
        full = os.path.join(prepared_root, entry)
        if os.path.isdir(full):
            for fname in sorted(os.listdir(full)):
                file_list.append((entry, fname))
        else:
            file_list.append((None, entry))

    print(f"Found {len(file_list)} files to parse with Drain.\n")

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
        subdir, fname = item
        if subdir is None:
            parser.path = prepared_root
            parser.savePath = out_root
        else:
            parser.path = os.path.join(prepared_root, subdir)
            parser.savePath = os.path.join(out_root, subdir)
            os.makedirs(parser.savePath, exist_ok=True)

        file_path = os.path.join(parser.path, fname)
        print(f"[{idx}/{len(file_list)}] Parsing: {file_path}")

        parser.parse(fname, chunk_size=chunk_size)

        print(f"✔ Finished file {idx}/{len(file_list)}: {fname}\n")

    print("✅ All files parsed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--st", default=0.5, type=float)
    ap.add_argument("--depth", default=4, type=int)
    ap.add_argument("--chunk_size", default=200000, type=int)
    args = ap.parse_args()

    run_all(args.input, args.output, args.st, args.depth, args.chunk_size)

