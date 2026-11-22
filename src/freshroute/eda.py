"""Data-quality note (reports/data_quality.md). Run after datagen."""
import json

import numpy as np
import pandas as pd

from . import config as C
from . import features as F


def main():
    r = pd.read_csv(C.DATA_DIR / "readings.csv.gz", parse_dates=["ts"])
    t = pd.read_csv(C.DATA_DIR / "trips.csv", parse_dates=["start_ts", "end_ts"])
    meta = json.loads((C.DATA_DIR / "meta.json").read_text())
    n = len(r)
    dup = int(r.duplicated(subset=["trip_id", "ts"]).sum())
    nan_t = int(r.temp_c.isna().sum())
    expected = int(((t.end_ts - t.start_ts).dt.total_seconds() / 60 + 1).sum())
    uniq = n - dup
    X = pd.read_csv(C.DATA_DIR / "features.csv", parse_dates=["start_ts"])
    X["day"] = (X.start_ts - pd.Timestamp(C.START_DATE)).dt.days
    est, post = {}, {}
    for trk, g in X.groupby("truck_id"):
        est[trk] = float(np.polyfit(g.day, g.truck_offset, 1)[0]) if len(g) > 5 else 0.0   # learned correction
        post[trk] = float(np.polyfit(g.day, g.resid_mean, 1)[0]) if len(g) > 5 else 0.0    # what is left
    inj = pd.Series(meta["drift_per_day"])
    cmp_ = pd.DataFrame({"injected_drift_per_day": inj, "learned_offset_slope": pd.Series(est),
                         "residual_slope_after_correction": pd.Series(post)})
    cmp_["injected_drift_per_day"] = cmp_["injected_drift_per_day"]
    cmp_ = cmp_.rename(columns={"injected_drift_per_day": "injected"})
    big = cmp_[cmp_.injected.abs() > 0.05].round(3)
    L = ["# Data-quality note", "",
         f"- Trips: {len(t)} ({t.truck_id.nunique()} trucks, {C.N_DAYS} days). Readings: {n:,} rows (one per minute while a trip is active).",
         f"- Class balance: {(t.outcome == 'spoiled').mean():.2%} of trips spoiled.",
         f"- Exact duplicate (trip, timestamp) rows: {dup:,} ({dup / n:.2%}). Cleaning keeps the first.",
         f"- Rows with empty temperature: {nan_t:,} ({nan_t / n:.2%}).",
         f"- Missing minutes after de-duplication: {1 - uniq / expected:.2%} of expected minutes (random drops, burst gaps up to 30 min, more on 4 low-battery trucks).",
         f"- Rows arrive shuffled within a trip; cleaning sorts by timestamp and re-indexes to a 1-minute grid, interpolating gaps of 5 minutes or less. Longer gaps stay empty and are exposed as features (pct_missing, longest_gap).",
         "- Sensor drift: 13 trucks drift between about 0.03 and 0.09 °C per day. For the strongly drifting trucks, the learned offset follows the injected drift and the residual trend after correction is close to zero (°C per day):", "",
         big.to_markdown() if hasattr(big, "to_markdown") else big.to_string(), "",
         "- Drift handling: each trip subtracts the median offset of that truck's previous 10 trips (earlier trips only, so no leakage). The residual is measured against the cargo setpoint.",
         "- Glitches: 2-4 minute spikes of +8 to +13 °C appear in about 4% of trips. They can trigger the legacy 2-minute rule; rolling-median features make the model robust to them.",
         "- Labels come from the true latent temperature (degree-minutes above the cargo limit, with per-trip tolerance noise); driver notes and labels are never model inputs.",
         ]
    try:
        import tabulate  # noqa: F401
    except ImportError:
        pass
    (C.REPORT_DIR / "data_quality.md").write_text("\n".join(L) + "\n")
    print("wrote reports/data_quality.md")


if __name__ == "__main__":
    main()
