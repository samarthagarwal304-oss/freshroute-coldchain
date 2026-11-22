import json

import pandas as pd

from freshroute import config as C
from freshroute import features as F


def test_no_label_or_notes_in_features():
    bad = {"y", "outcome", "driver_notes", "dose", "scenario", "baseline_flag"}
    assert not (set(F.FEATURES) & bad)


def test_truck_offset_uses_only_earlier_trips():
    t0 = pd.Timestamp("2026-10-01")
    meta = pd.DataFrame(dict(trip_id=[f"T{i}" for i in range(8)], truck_id="A",
                             start_ts=[t0 + pd.Timedelta(hours=i) for i in range(8)],
                             med_resid=[0.5] * 8))
    base = F.compute_truck_offsets(meta)
    meta2 = meta.copy(); meta2.loc[5, "med_resid"] = 99.0         # change trip 5's own data...
    changed = F.compute_truck_offsets(meta2)
    assert base["T5"] == changed["T5"]                              # ...its own offset must not move
    assert base["T0"] == 0.0 and base["T2"] == 0.0                   # no history -> 0
    assert base["T7"] == 0.5


def test_model_beats_baseline_on_test_set():
    rep = json.loads((C.REPORT_DIR / "model_report.json").read_text())["models"]
    b, m = rep["baseline_rule_8C_2min"], rep["lightgbm"]
    assert m["precision"] > 2 * b["precision"]
    assert m["false_alerts_per_truck_week"] < 0.5 * b["false_alerts_per_truck_week"]
    assert m["recall"] >= b["recall"] - 0.10
    assert m["cost"] < b["cost"]


def test_threshold_not_tuned_on_test():
    art = __import__("joblib").load(C.MODEL_DIR / "model.joblib")
    assert art["trained_on_days"] == f"1-{C.VAL_END_DAY}"
