"""FastAPI service: POST /predict (risk score) and POST /ask (assistant). Run: uvicorn freshroute.api:app"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from . import config as C
from . import features as F
from .assistant import Assistant, pick_backend
from .logging_utils import log_request, summarize
from .model import get_model
from .rag import get_index
from .store import get_store
from .tools import ToolRunner, normalize_trip_id

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    st = get_store(); get_model(); get_index()
    st.readings(st.trips.index[0])           # warm the readings cache (~1s) so the first request is fast
    state["assistant"] = Assistant(pick_backend())
    yield


app = FastAPI(title="FreshRoute Cold-Chain Intelligence", version="1.0", lifespan=lifespan)


class Reading(BaseModel):
    ts: datetime
    temp_c: float | None = None
    humidity: float | None = None
    door_open: int = 0
    battery_pct: float | None = None
    lat: float | None = None
    lon: float | None = None


class PredictRequest(BaseModel):
    trip_id: str | None = Field(None, description="Score a known trip, e.g. T-2555")
    readings: list[Reading] | None = Field(None, description="...or score raw readings (>= 30 minutes)")
    cargo_type: str | None = Field(None, description="required with readings: vaccine|dairy|seafood|produce|meat")
    truck_id: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if not self.trip_id and not self.readings:
            raise ValueError("provide trip_id or readings")
        if self.readings and not self.trip_id:
            if self.cargo_type not in C.CARGO:
                raise ValueError(f"cargo_type must be one of {list(C.CARGO)}")
            if len(self.readings) < 30:
                raise ValueError("need at least 30 readings")
        return self


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    trip_id: str | None = None


@app.get("/health")
def health():
    return dict(status="ok", llm_backend=state["assistant"].backend.name, flag_threshold=get_model().threshold)


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    model, store = get_model(), get_store()
    if req.trip_id:
        tid = normalize_trip_id(req.trip_id)
        if not tid or not store.has_trip(tid):
            raise HTTPException(404, f"trip {req.trip_id} not found")
        row = store.feat.loc[tid]
        out = model.predict_one(row[model.features].astype(float).to_dict())
        out.update(trip_id=tid, legacy_alert_fired=bool(row.baseline_events > 0), source="stored trip")
    else:
        df = pd.DataFrame([r.model_dump() for r in req.readings]).rename(columns={})
        df["trip_id"] = "adhoc"
        grid, q = F.clean_trip(df)
        offset = store.latest_offset(req.truck_id) if req.truck_id else 0.0
        feat = F.trip_features(grid, q, req.cargo_type, offset)
        out = model.predict_one(feat)
        out.update(legacy_alert_fired=F.baseline_alerts(grid["temp_raw"].to_numpy()) > 0, source="raw readings",
                   note="Model was trained on complete trips; scores on partial trips are cumulative-so-far.")
    lat = (time.perf_counter() - t0) * 1000
    out["latency_ms"] = round(lat, 1)
    log_request("/predict", lat, source=out["source"], risk_score=out["risk_score"])
    return out


@app.post("/ask")
def ask(req: AskRequest):
    try:
        res = state["assistant"].ask(req.question, req.trip_id)
    except Exception as e:  # LLM/network failure must not crash the service
        log_request("/ask", 0, error=str(e)[:200], status="error")
        raise HTTPException(502, f"assistant backend error: {type(e).__name__}")
    res.pop("context_text", None)
    log_request("/ask", res["latency_ms"], backend=res["backend"], input_tokens=res["usage"]["input_tokens"],
                output_tokens=res["usage"]["output_tokens"], est_cost_usd=res["usage"]["est_cost_usd"],
                tool_calls=[c["name"] for c in res["tool_calls"]], idk=res["idk"],
                injection=res["guardrails"], status="ok")
    return res


@app.get("/trips/flagged")
def flagged(limit: int = 10):
    return ToolRunner().list_flagged_trips(limit=limit)


@app.get("/stats")
def stats():
    return summarize()
