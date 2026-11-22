"""Cleaning, baseline rule and trip-level feature engineering.

Leakage rules (enforced + unit-tested):
  * Features use only observed sensor columns + cargo type + trip duration.
  * Driver notes and the outcome label are never features.
  * The per-truck temperature offset (drift correction) for trip i is estimated only from
    the observed readings of that truck's EARLIER trips (strictly before trip i starts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CARGO_CODE = {c: i for i, c in enumerate(C.CARGO_TYPES)}

FEATURE_DESCRIPTIONS = {
    "cargo_code": "cargo type (sensitivity differs per cargo)",
    "setpoint": "cargo setpoint",
    "duration_min": "trip duration (minutes)",
    "resid_mean": "average temperature above setpoint (°C)",
    "resid_max": "peak temperature above setpoint (°C)",
    "resid_p95": "95th percentile temperature above setpoint (°C)",
    "resid_std": "temperature variability (°C)",
    "min_above_1": "minutes more than 1°C above setpoint",
    "min_above_2": "minutes more than 2°C above setpoint",
    "min_above_3": "minutes more than 3°C above setpoint",
    "min_above_5": "minutes more than 5°C above setpoint",
    "dm_above_1": "heat exposure: degree-minutes above setpoint+1°C",
    "dm_above_3": "heat exposure: degree-minutes above setpoint+3°C",
    "max_med5": "highest 5-minute median temperature above setpoint (°C)",
    "max_roll30": "highest 30-minute average temperature above setpoint (°C)",
    "longest_run_above_2": "longest continuous stretch (min) more than 2°C above setpoint",
    "max_rise_15": "fastest 15-minute temperature rise (°C)",
    "min_above_8abs": "minutes above 8°C (legacy alert threshold)",
    "door_total_min": "total door-open minutes",
    "door_longest": "longest continuous door-open stretch (min)",
    "door_events": "number of door-open events",
    "hum_mean": "average humidity (%)",
    "batt_min": "lowest sensor battery (%)",
    "pct_missing": "share of minutes with no reading",
    "longest_gap": "longest gap without readings (min)",
    "n_dups": "duplicate readings removed",
    "truck_offset": "estimated sensor offset for this truck (°C)",
}
FEATURES = list(FEATURE_DESCRIPTIONS)


# ----------------------------------------------------------------------------- cleaning
def clean_trip(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Dedupe, sort, put on a 1-minute grid, interpolate short gaps (<=5 min).

    Returns (grid, quality). grid has columns temp_raw (NaN where missing, used by the
    baseline), temp (gap-filled), door, humidity, battery.
    """
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    n_in = len(d)
    d = d.sort_values("ts").drop_duplicates(subset="ts", keep="first")
    n_dups = n_in - len(d)
    idx = pd.date_range(d["ts"].min().floor("min"), d["ts"].max().floor("min"), freq="min")
    d = d.set_index(d["ts"].dt.floor("min")).reindex(idx)
    grid = pd.DataFrame(index=idx)
    grid["temp_raw"] = d["temp_c"].to_numpy(float)
    grid["temp"] = grid["temp_raw"].interpolate(limit=5, limit_area="inside")
    grid["door"] = d["door_open"].fillna(0).to_numpy(float)
    grid["humidity"] = d["humidity"].ffill().bfill().to_numpy(float)
    grid["battery"] = d["battery_pct"].ffill().bfill().to_numpy(float)
    miss = grid["temp_raw"].isna().to_numpy()
    q = dict(n_minutes=len(grid), pct_missing=float(miss.mean()),
             longest_gap=int(_longest_run(miss)), n_dups=int(n_dups))
    return grid, q


def _longest_run(b: np.ndarray) -> int:
    if not b.any():
        return 0
    x = np.concatenate([[0], b.astype(np.int8), [0]])
    d = np.diff(x)
    return int((np.where(d == -1)[0] - np.where(d == 1)[0]).max())


def _runs(b: np.ndarray):
    x = np.concatenate([[0], b.astype(np.int8), [0]])
    d = np.diff(x)
    return np.where(d == 1)[0], np.where(d == -1)[0]   # starts, ends(exclusive)


# ----------------------------------------------------------------------------- baseline
def baseline_alerts(temp_raw: np.ndarray, thr: float = C.BASELINE_TEMP_C, minutes: int = C.BASELINE_MINUTES) -> int:
    """Number of alert events of the legacy rule: temp > thr for `minutes` consecutive minutes.
    Missing minutes break a run (the rule only sees real readings)."""
    above = np.nan_to_num(temp_raw, nan=-1e9) > thr
    s, e = _runs(above)
    return int(((e - s) >= minutes).sum())


# ----------------------------------------------------------------------------- features
def trip_features(grid: pd.DataFrame, quality: dict, cargo_type: str, truck_offset: float = 0.0) -> dict:
    sp = C.CARGO[cargo_type]["setpoint"]
    t = grid["temp"].to_numpy(float)
    ok = ~np.isnan(t)
    r = (t - sp - truck_offset)
    rr = r[ok] if ok.any() else np.array([0.0])
    rs = pd.Series(r)
    med5 = rs.rolling(5, min_periods=3).median().to_numpy()
    roll30 = rs.rolling(30, min_periods=10).mean().to_numpy()
    door = grid["door"].to_numpy(float) > 0.5
    ds, de = _runs(door)
    above2 = np.nan_to_num(r, nan=-9) > 2
    a2s, a2e = _runs(above2)
    rise15 = r[15:] - r[:-15] if len(r) > 15 else np.array([0.0])
    f = dict(
        cargo_code=CARGO_CODE[cargo_type], setpoint=sp, duration_min=quality["n_minutes"],
        resid_mean=float(rr.mean()), resid_max=float(rr.max()), resid_p95=float(np.percentile(rr, 95)),
        resid_std=float(rr.std()),
        min_above_1=int((rr > 1).sum()), min_above_2=int((rr > 2).sum()),
        min_above_3=int((rr > 3).sum()), min_above_5=int((rr > 5).sum()),
        dm_above_1=float(np.clip(rr - 1, 0, None).sum()), dm_above_3=float(np.clip(rr - 3, 0, None).sum()),
        max_med5=float(np.nanmax(med5)) if np.isfinite(med5).any() else 0.0,
        max_roll30=float(np.nanmax(roll30)) if np.isfinite(roll30).any() else 0.0,
        longest_run_above_2=int((a2e - a2s).max()) if len(a2s) else 0,
        max_rise_15=float(np.nanmax(rise15)) if np.isfinite(rise15).any() else 0.0,
        min_above_8abs=int((t[ok] > C.BASELINE_TEMP_C).sum()),
        door_total_min=int(door.sum()), door_longest=int((de - ds).max()) if len(ds) else 0,
        door_events=int(len(ds)),
        hum_mean=float(grid["humidity"].mean()), batt_min=float(grid["battery"].min()),
        pct_missing=quality["pct_missing"], longest_gap=quality["longest_gap"], n_dups=quality["n_dups"],
        truck_offset=float(truck_offset),
    )
    return f


def trip_summary_stats(grid: pd.DataFrame, cargo_type: str) -> dict:
    """Human-facing facts for the assistant tool (observed, not corrected)."""
    t = grid["temp"].to_numpy(float)
    door = grid["door"].to_numpy(float) > 0.5
    ds, de = _runs(door)
    sp = C.CARGO[cargo_type]["setpoint"]
    return dict(
        max_temp_c=round(float(np.nanmax(t)), 1), mean_temp_c=round(float(np.nanmean(t)), 1),
        setpoint_c=sp, upper_limit_c=C.CARGO[cargo_type]["limit"],
        minutes_above_8c=int((np.nan_to_num(t, nan=-99) > 8).sum()),
        minutes_above_cargo_limit=int((np.nan_to_num(t, nan=-99) > C.CARGO[cargo_type]["limit"]).sum()),
        door_open_total_min=int(door.sum()), door_open_longest_min=int((de - ds).max()) if len(ds) else 0,
        door_open_events=int(len(ds)),
        min_battery_pct=round(float(grid["battery"].min()), 1),
    )


# ----------------------------------------------------------------------------- dataset
def compute_truck_offsets(meta: pd.DataFrame) -> pd.Series:
    """meta: trip_id, truck_id, start_ts, med_resid. Offset = median of the median-residuals of the
    truck's previous (up to 10) trips -- strictly earlier trips only. Returns Series indexed by trip_id."""
    m = meta.sort_values(["truck_id", "start_ts"]).copy()
    m["offset"] = (m.groupby("truck_id")["med_resid"]
                   .transform(lambda s: s.shift(1).rolling(10, min_periods=3).median()))
    return m.set_index("trip_id")["offset"].fillna(0.0)


def build_dataset(readings: pd.DataFrame, trips: pd.DataFrame, verbose=True) -> tuple[pd.DataFrame, dict]:
    """One row per trip with features + label + bookkeeping columns. Also returns per-trip grids."""
    trips = trips.copy()
    trips["start_ts"] = pd.to_datetime(trips["start_ts"])
    readings = readings.sort_values(["trip_id", "ts"])
    cargo = trips.set_index("trip_id")["cargo_type"].to_dict()
    grids, quals, med = {}, {}, []
    for tid, df in readings.groupby("trip_id", sort=False):
        g, q = clean_trip(df)
        grids[tid], quals[tid] = g, q
        sp = C.CARGO[cargo[tid]]["setpoint"]
        med.append((tid, float(np.nanmedian(g["temp"] - sp))))
    meta = trips.merge(pd.DataFrame(med, columns=["trip_id", "med_resid"]), on="trip_id")
    offsets = compute_truck_offsets(meta)
    rows = []
    for tid, g in grids.items():
        f = trip_features(g, quals[tid], cargo[tid], offsets.get(tid, 0.0))
        f["trip_id"] = tid
        f["baseline_events"] = baseline_alerts(g["temp_raw"].to_numpy())
        rows.append(f)
    X = pd.DataFrame(rows).merge(trips[["trip_id", "truck_id", "cargo_type", "start_ts", "outcome"]], on="trip_id")
    X["y"] = (X["outcome"] == "spoiled").astype(int)
    X["baseline_flag"] = (X["baseline_events"] > 0).astype(int)
    X = X.sort_values("start_ts").reset_index(drop=True)
    if verbose:
        print(f"features built for {len(X)} trips")
    return X, grids
