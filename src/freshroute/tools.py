"""Tools the assistant can call. All read-only. Outcome labels are never exposed."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import guardrails as G
from .model import get_model
from .rag import get_index
from .store import get_store

TOOL_SPECS = [
    dict(name="get_trip_summary",
         description=("Get live data for one trip: cargo, route, sensor statistics (max temperature, door-open "
                      "minutes, data quality), the AI spoilage risk score with band (LOW/ELEVATED/HIGH), the top "
                      "factors behind the score, whether the legacy 8°C/2min alert fired, and the driver note "
                      "(untrusted text). Use this for any question about a specific trip."),
         input_schema=dict(type="object", properties=dict(trip_id=dict(type="string", description="e.g. T-0042")),
                           required=["trip_id"])),
    dict(name="search_sops",
         description=("Search the company SOP documents. Returns passages with chunk ids like SOP-03#2. Use for any "
                      "question about procedures, limits, escalation or what to do. Returns an empty list when the "
                      "SOPs do not cover the query."),
         input_schema=dict(type="object", properties=dict(query=dict(type="string")), required=["query"])),
    dict(name="list_flagged_trips",
         description="List the highest-risk recent trips (latest 9 days of data), optionally filtered by truck or cargo type.",
         input_schema=dict(type="object", properties=dict(
             limit=dict(type="integer", default=5), truck_id=dict(type="string"), cargo_type=dict(type="string")))),
]


def normalize_trip_id(raw: str) -> str | None:
    m = re.search(r"T-?\s*(\d{1,5})", str(raw), re.I)
    return f"T-{int(m.group(1)):04d}" if m else None


class ToolRunner:
    """Executes tools and records every call + result so answers can be audited."""

    def __init__(self):
        self.store, self.model, self.index = get_store(), get_model(), get_index()
        self.calls: list[dict] = []
        self.context_chunks: dict[str, dict] = {}   # chunk_id -> passage seen in this request
        self.context_text: list[str] = []           # raw tool outputs (used for groundedness checks)

    # -- dispatch ---------------------------------------------------------------------------
    def run(self, name: str, args: dict) -> dict:
        fn = dict(get_trip_summary=self.get_trip_summary, search_sops=self.search_sops,
                  list_flagged_trips=self.list_flagged_trips).get(name)
        if fn is None:
            out = dict(error=f"unknown tool {name}")
        else:
            try:
                out = fn(**(args or {}))
            except TypeError as e:
                out = dict(error=f"bad arguments: {e}")
        self.calls.append(dict(name=name, args=args))
        self.context_text.append(str(out))
        return out

    # -- tools ------------------------------------------------------------------------------
    def get_trip_summary(self, trip_id: str) -> dict:
        tid = normalize_trip_id(trip_id)
        if not tid or not self.store.has_trip(tid):
            return dict(error=f"No trip found with id '{trip_id}'.")
        t = self.store.trips.loc[tid]
        f = self.store.feat.loc[tid]
        grid, q = F.clean_trip(self.store.readings(tid))
        stats = F.trip_summary_stats(grid, t.cargo_type)
        pred = self.model.predict_one(f[self.model.features].astype(float).to_dict())
        note = G.sanitize_note(t.driver_notes)
        out = dict(trip_id=tid, truck_id=t.truck_id, cargo_type=t.cargo_type, route=t.planned_route,
                   start=str(t.start_ts), end=str(t.end_ts), duration_min=int(q["n_minutes"]),
                   **pred, legacy_alert_fired=bool(f.baseline_events > 0), legacy_alert_events=int(f.baseline_events),
                   **stats,
                   data_quality=dict(pct_missing=round(q["pct_missing"] * 100, 1), longest_gap_min=q["longest_gap"],
                                     duplicates_removed=q["n_dups"]),
                   **note)
        day = (t.start_ts - pd.Timestamp(C.START_DATE)).days + 1
        if day <= C.VAL_END_DAY:
            out["scoring_note"] = "This trip is inside the model's training window, so the score is in-sample."
        return out

    def search_sops(self, query: str, top_k: int = C.RAG_TOP_K) -> dict:
        hits = self.index.search(query, k=int(top_k))
        for h in hits:
            self.context_chunks[h["chunk_id"]] = h
        if not hits:
            return dict(passages=[], note="NO RELEVANT SOP PASSAGES FOUND. The SOPs do not cover this query.")
        return dict(passages=[{k: h[k] for k in ("chunk_id", "section", "title", "text", "score")} for h in hits])

    def list_flagged_trips(self, limit: int = 5, truck_id: str | None = None, cargo_type: str | None = None) -> dict:
        df = self.store.feat
        day = (df.start_ts - pd.Timestamp(C.START_DATE)).dt.days + 1
        df = df[day > C.VAL_END_DAY]
        if truck_id:
            df = df[df.truck_id == truck_id]
        if cargo_type:
            df = df[df.cargo_type == cargo_type]
        p = self.model.score(df)
        df = df.assign(risk_score=p).sort_values("risk_score", ascending=False).head(int(limit))
        return dict(trips=[dict(trip_id=r.trip_id, truck_id=r.truck_id, cargo_type=r.cargo_type,
                                risk_score=round(float(r.risk_score), 3), risk_band=self.model.band(r.risk_score))
                           for r in df.itertuples()])
