"""Per-request logging: latency, tokens, estimated cost -> logs/requests.jsonl"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict

import numpy as np

from . import config as C

_lock = threading.Lock()
LOG_FILE = C.LOG_DIR / "requests.jsonl"


def cost_usd(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * C.PRICE_IN + tokens_out * C.PRICE_OUT) / 1e6


def log_request(endpoint: str, latency_ms: float, **fields) -> dict:
    rec = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), endpoint=endpoint, latency_ms=round(latency_ms, 1), **fields)
    C.LOG_DIR.mkdir(exist_ok=True)
    with _lock, open(LOG_FILE, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def summarize() -> dict:
    if not LOG_FILE.exists():
        return {}
    rows = [json.loads(l) for l in LOG_FILE.read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["endpoint"]].append(r)
    out = {}
    for ep, rs in by.items():
        lat = [r["latency_ms"] for r in rs]
        out[ep] = dict(requests=len(rs), latency_ms_p50=round(float(np.percentile(lat, 50)), 1),
                       latency_ms_p95=round(float(np.percentile(lat, 95)), 1),
                       input_tokens=sum(r.get("input_tokens", 0) for r in rs),
                       output_tokens=sum(r.get("output_tokens", 0) for r in rs),
                       est_cost_usd=round(sum(r.get("est_cost_usd", 0.0) for r in rs), 5))
    return out
