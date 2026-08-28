"""Ingest NASA COOLR landslide layers from the Earthdata GIS FeatureServer.

The legacy data.nasa.gov Socrata export (`dd9e-wu2v`) is dead as of 2026
(verified 404), and the gis05 host is token-gated; gis01 serves anonymously.

Two layers matter, and they are NOT interchangeable:

  COOLR_Events_Points  (~40k) -- satellite-mapped event *inventories*. Dense in
      space, extremely clustered in time (Vietnam: 12,566 events on 11 dates).
      Spatially near-complete within a mapped footprint, so absence inside a
      footprint is informative. Use for the SUSCEPTIBILITY layer.
  COOLR_Reports_Points (~15k) -- media/report-derived catalog. Sparse in space,
      spread across many distinct dates. Use for the TRIGGER layer.

Dependency-light on purpose: stdlib + requests only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW  # noqa: E402
from pipelines.common import SESSION  # noqa: E402

HOST = "https://gis.earthdata.nasa.gov/gis01/rest/services/Landslides"
PAGE = 2000

LAYERS = {
    "events": "COOLR_Events_Points",
    "reports": "COOLR_Reports_Points",
}


def _iso(ms):
    """Epoch-ms -> ISO date. COOLR carries a few out-of-range/sentinel values
    (pre-1970 and year-9999 style) that raise OSError on Windows."""
    if not isinstance(ms, (int, float)):
        return None
    try:
        d = dt.datetime.fromtimestamp(ms / 1000, dt.UTC)
    except (OSError, OverflowError, ValueError):
        return None
    if not (1900 <= d.year <= 2100):
        return None
    return d.strftime("%Y-%m-%d")


def layer_url(service: str) -> str:
    return f"{HOST}/{service}/FeatureServer/0"


def describe(service: str) -> tuple[list[str], set[str]]:
    """Return (field names, names of esriFieldTypeDate fields)."""
    r = SESSION.get(f"{layer_url(service)}?f=json", timeout=90)
    r.raise_for_status()
    meta = r.json()
    if "error" in meta:
        raise RuntimeError(meta["error"])
    names, dates = [], set()
    for f in meta.get("fields", []):
        if f["type"] == "esriFieldTypeGeometry":
            continue
        names.append(f["name"])
        if f["type"] == "esriFieldTypeDate":
            dates.add(f["name"])
    return names, dates


def count(service: str) -> int:
    r = SESSION.get(f"{layer_url(service)}/query",
                    params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
                    timeout=90)
    r.raise_for_status()
    return int(r.json()["count"])


def fetch_page(service: str, fields: list[str], offset: int) -> list[dict]:
    params = {
        "where": "1=1",
        "outFields": ",".join(fields),
        "returnGeometry": "false",
        "orderByFields": f"{fields[0]} ASC",
        "resultOffset": offset,
        "resultRecordCount": PAGE,
        "f": "json",
    }
    r = SESSION.get(f"{layer_url(service)}/query", params=params, timeout=180)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS error at offset {offset}: {payload['error']}")
    return [f["attributes"] for f in payload.get("features", [])]


def ingest(key: str) -> Path:
    service = LAYERS[key]
    fields, date_fields = describe(service)
    total = count(service)
    print(f"[{key}] {service}: {total:,} records, {len(fields)} fields")
    print(f"[{key}] date fields: {sorted(date_fields) or 'none'}")

    rows: list[dict] = []
    offset = 0
    while offset < total:
        page = fetch_page(service, fields, offset)
        if not page:
            print(f"[{key}]   empty page at {offset}, stopping")
            break
        rows.extend(page)
        offset += PAGE
        print(f"[{key}]   {len(rows):,}/{total:,}", end="\r", flush=True)
    print()

    # epoch-ms -> ISO date, as a sidecar column so the raw value is preserved
    out_cols = list(fields)
    for df in sorted(date_fields):
        iso = f"{df}_iso"
        out_cols.append(iso)
        bad = 0
        for r in rows:
            r[iso] = _iso(r.get(df))
            if r[iso] is None and r.get(df) is not None:
                bad += 1
        if bad:
            print(f"[{key}]   {bad:,} unparseable values in {df}")

    RAW.mkdir(parents=True, exist_ok=True)
    nd = RAW / f"coolr_{key}_points.ndjson"
    with nd.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    cs = RAW / f"coolr_{key}_points.csv"
    with cs.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[{key}] wrote {len(rows):,} rows -> {cs.name}, {nd.name}")
    return cs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("layers", nargs="*", default=list(LAYERS), choices=list(LAYERS))
    a = ap.parse_args()
    for k in (a.layers or list(LAYERS)):
        ingest(k)
