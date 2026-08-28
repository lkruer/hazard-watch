"""HTTP session with retry/backoff plus a tiny on-disk cache."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE  # noqa: E402

USER_AGENT = "multi-hazard-risk-platform/0.1 (research; contact: local)"


def make_session(total_retries: int = 5) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": USER_AGENT})
    return s


SESSION = make_session()


def _key(url: str, params: dict | None) -> str:
    raw = url + "|" + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def get_json(url: str, params: dict | None = None, cache: bool = True,
             timeout: int = 90, subdir: str = "http") -> dict:
    """GET returning parsed JSON, cached on disk by (url, params)."""
    cdir = CACHE / subdir
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"{_key(url, params)}.json"
    if cache and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if cache:
        path.write_text(json.dumps(data), encoding="utf-8")
    return data


def polite(seconds: float = 0.2) -> None:
    time.sleep(seconds)
