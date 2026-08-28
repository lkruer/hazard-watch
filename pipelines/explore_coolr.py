"""Quick profile of the COOLR catalog: stdlib only, no numpy/pandas needed."""
import csv, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW

rows = list(csv.DictReader((RAW / "coolr_events_points.csv").open(encoding="utf-8")))
print(f"total rows: {len(rows):,}\n")

def vc(field, n=12):
    c = Counter((r.get(field) or "(blank)") for r in rows)
    print(f"--- {field} (top {n} of {len(c)}) ---")
    for k, v in c.most_common(n):
        print(f"  {v:7,}  {k}")
    print()

dates = sorted(r["event_date_iso"] for r in rows if r.get("event_date_iso"))
print(f"--- event_date_iso ---\n  non-null: {len(dates):,}   range: {dates[0]} .. {dates[-1]}\n")
yrs = Counter(d[:4] for d in dates)
print("  events per year:")
for y in sorted(yrs):
    print(f"    {y}  {yrs[y]:6,}  {'#' * min(60, yrs[y] // 60)}")
print()

for f in ("landslide_category", "landslide_trigger", "country_name", "source_name"):
    vc(f)

# valid coordinates
def fl(x):
    try: return float(x)
    except (TypeError, ValueError): return None

good = [r for r in rows
        if fl(r["latitude"]) is not None and fl(r["longitude"]) is not None
        and -90 <= fl(r["latitude"]) <= 90 and -180 <= fl(r["longitude"]) <= 180
        and r.get("event_date_iso")]
print(f"rows with valid lat/lon AND a date: {len(good):,}")
