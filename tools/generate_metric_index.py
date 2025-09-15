# generate_metric_index.py
import os, glob, csv

metric_dir = r"C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS\metric"
out = os.path.join(metric_dir, "metric.csv")

def parse_name(name):
    # name = filename without .csv
    parts = name.split('_')
    service = parts[0] if len(parts) >= 1 else ''
    address = parts[1] if len(parts) >= 2 else ''
    # assume last two parts are start/end dates like 2021-07-01
    if len(parts) >= 4 and parts[-2].count('-') == 2 and parts[-1].count('-') == 2:
        start_date, end_date = parts[-2], parts[-1]
        metric_name = "_".join(parts[2:-2])
        clean_name = "_".join(parts[:-2])
    else:
        start_date = end_date = ''
        metric_name = "_".join(parts[2:]) if len(parts) > 2 else ''
        clean_name = name
    return clean_name, service, address, metric_name, start_date, end_date

rows = []
for f in sorted(glob.glob(os.path.join(metric_dir, "*.csv"))):
    base = os.path.basename(f)                # e.g. 'redis_database_0.0.0.3_..._2021-07-01_2021-07-15.csv'
    name_no_ext = os.path.splitext(base)[0]   # without .csv
    clean_name, service, address, metric_name, start_date, end_date = parse_name(name_no_ext)
    rows.append({
        # IMPORTANT: write `name` as the actual filename so metric_event can open it directly
        "name": base,           # store the actual file name (with .csv)
        "filename": base,       # duplicate column (helps other code)
        "clean_name": clean_name,
        "service": service,
        "address": address,
        "metric_name": metric_name,
        "start_date": start_date,
        "end_date": end_date
    })

fieldnames = ["name","filename","clean_name","service","address","metric_name","start_date","end_date"]
with open(out, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print("Wrote", out, "with", len(rows), "rows.")
