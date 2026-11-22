import json

import numpy as np
import pandas as pd

from freshroute import config as C
from freshroute import features as F


def test_class_balance_about_5_percent():
    t = pd.read_csv(C.DATA_DIR / "trips.csv")
    assert 0.03 < (t.outcome == "spoiled").mean() < 0.08
    assert t.truck_id.nunique() == C.N_TRUCKS


def test_data_has_quality_problems_and_cleaning_fixes_them():
    r = pd.read_csv(C.DATA_DIR / "readings.csv.gz", parse_dates=["ts"])
    assert r.duplicated(subset=["trip_id", "ts"]).sum() > 0          # duplicates exist
    assert r.temp_c.isna().sum() > 0                                  # missing values exist
    tid = r.trip_id.iloc[0]
    raw = r[r.trip_id == tid]
    grid, q = F.clean_trip(raw)
    assert grid.index.is_monotonic_increasing and not grid.index.duplicated().any()
    assert q["n_dups"] == raw.duplicated(subset="ts").sum()


def test_baseline_rule():
    t = np.array([5, 9, 9, 5, 5, 9, 5, 9, np.nan, 9, 5.0])
    assert F.baseline_alerts(t) == 1                     # only the 2-minute run counts; NaN breaks the 2nd run
    assert F.baseline_alerts(np.array([5, 9, 5, 9, 5.0])) == 0
    assert F.baseline_alerts(np.array([9, 9, 9, 9, 5, 9, 9.0])) == 2


def test_time_split_is_strictly_chronological():
    X = pd.read_csv(C.DATA_DIR / "features.csv", parse_dates=["start_ts"])
    day = (X.start_ts - pd.Timestamp(C.START_DATE)).dt.days + 1
    tr, va, te = X[day <= C.TRAIN_END_DAY], X[(day > C.TRAIN_END_DAY) & (day <= C.VAL_END_DAY)], X[day > C.VAL_END_DAY]
    assert tr.start_ts.max() < va.start_ts.min() < va.start_ts.max() < te.start_ts.min()
    assert len(te) > 500 and te.y.sum() > 20
