import os
import glob
import pandas as pd

def generate_trace_timeseries(input_dir, output_dir, freq="30S"):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    print(f"Found {len(files)} raw trace files in {input_dir}")

    for f in files:
        base = os.path.basename(f)
        print(f"Processing {base} ...")
        try:
            df = pd.read_csv(f)

            # Ensure timestamp column
            if "timestamp" not in df.columns:
                raise ValueError("No 'timestamp' column in trace file: " + base)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp")

            # Map parent_id -> service_name
            span_to_service = dict(zip(df["span_id"], df["service_name"]))

            # Build caller→callee pairs
            def get_pair(row):
                parent_span = row["parent_id"]
                child_service = row["service_name"]
                parent_service = span_to_service.get(parent_span, "ROOT")
                return f"{parent_service}->{child_service}"

            df["pair"] = df.apply(get_pair, axis=1)

            # Resample counts per pair
            out_frames = []
            for pair, g in df.groupby("pair"):
                g = g.set_index("timestamp")
                counts = g["pair"].resample(freq).count()
                out_df = counts.reset_index().rename(columns={"pair": "count"})
                out_df["pair"] = pair
                out_frames.append(out_df)

            if out_frames:
                result = pd.concat(out_frames, ignore_index=True)
                out_path = os.path.join(output_dir, base)
                result.to_csv(out_path, index=False)
                print(f"  -> Saved {out_path} ({len(result)} rows)")
            else:
                print(f"  -> No pairs found in {base}")

        except Exception as e:
            print(f"Failed to process {base}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw trace CSV dir")
    parser.add_argument("--output", required=True, help="Output directory for timeseries")
    parser.add_argument("--freq", default="30S", help="Resample frequency (default: 30S)")
    args = parser.parse_args()

    generate_trace_timeseries(args.input, args.output, args.freq)
