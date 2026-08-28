"""Shared paths and study-region config."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
REPORTS = ROOT / "reports"

for _p in (RAW, INTERIM, PROCESSED, CACHE, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# Study region is chosen empirically in pipelines/select_region.py and written
# here. bbox = (min_lon, min_lat, max_lon, max_lat)
STUDY_BBOX = None
STUDY_NAME = None
