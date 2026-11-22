"""Central configuration. Everything tunable lives here or in env vars."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SOP_DIR = ROOT / "sops"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"

# ---- data generation -------------------------------------------------------
SEED = 42
N_TRUCKS = 50
N_DAYS = 30
START_DATE = "2026-10-01"          # day 1
TRIPS_PER_TRUCK_PER_DAY = 2        # morning + afternoon trip
TRIP_MIN_MINUTES, TRIP_MAX_MINUTES = 180, 420

# Cargo physics / spoilage definition. Spoilage is defined from the TRUE
# (latent) temperature: degree-minutes above the cargo's upper limit.
CARGO = {
    #            setpoint  upper_limit  dose_limit (degC*min above limit)  share
    "vaccine": dict(setpoint=5.0, limit=8.0, dose=20.0, share=0.15),
    "dairy":   dict(setpoint=3.0, limit=6.0, dose=120.0, share=0.30),
    "seafood": dict(setpoint=1.0, limit=4.0, dose=100.0, share=0.15),
    "produce": dict(setpoint=6.0, limit=10.0, dose=200.0, share=0.25),
    "meat":    dict(setpoint=2.0, limit=6.0, dose=100.0, share=0.15),
}
CARGO_TYPES = list(CARGO)

# ---- baseline rule (current production rule) ------------------------------
BASELINE_TEMP_C = 8.0
BASELINE_MINUTES = 2

# ---- modelling -------------------------------------------------------------
TRAIN_END_DAY = 14      # days 1..14  -> fit
VAL_END_DAY = 21        # days 15..21 -> choose threshold ; then refit on 1..21
# days 22..30 -> untouched test set
COST_MISSED_SPOILAGE = 400.0   # USD, cargo + claim (assumption, see docs)
COST_FALSE_ALERT = 20.0        # USD, ~15 min of dispatcher time (assumption)
RISK_BAND_HIGH = 0.60          # bands used in SOP-01

# ---- retrieval -------------------------------------------------------------
RAG_TOP_K = 4
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.12"))
RAG_MIN_COVERAGE = float(os.getenv("RAG_MIN_COVERAGE", "0.40"))

# ---- LLM -------------------------------------------------------------------
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
PRICE_IN = float(os.getenv("PRICE_INPUT_PER_MTOK", "1.0"))
PRICE_OUT = float(os.getenv("PRICE_OUTPUT_PER_MTOK", "5.0"))
MAX_TOOL_ROUNDS = 6
CANARY = "CANARY-9d41"   # planted in the system prompt; must never appear in output
