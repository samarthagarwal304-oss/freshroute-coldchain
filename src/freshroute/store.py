"""Serving-time data access: trips, cleaned readings, features and model scores."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from . import features as F


class TripStore:
    def __init__(self):
        self.trips = pd.read_csv(C.DATA_DIR / "trips.csv", parse_dates=["start_ts", "end_ts"])
        self.trips = self.trips.set_index("trip_id", drop=False)
        self.feat = pd.read_csv(C.DATA_DIR / "features.csv", parse_dates=["start_ts"]).set_index("trip_id", drop=False)
        self.meta = json.loads((C.DATA_DIR / "meta.json").read_text())
        self._readings = None

    def readings(self, trip_id: str) -> pd.DataFrame:
        if self._readings is None:   # lazy: ~870k rows, loaded once
            r = pd.read_csv(C.DATA_DIR / "readings.csv.gz", parse_dates=["ts"])
            self._readings = {k: v for k, v in r.groupby("trip_id")}
        return self._readings[trip_id]

    def has_trip(self, trip_id: str) -> bool:
        return trip_id in self.trips.index

    def latest_offset(self, truck_id: str) -> float:
        f = self.feat[self.feat.truck_id == truck_id]
        return float(f.sort_values("start_ts").truck_offset.iloc[-1]) if len(f) else 0.0


@lru_cache(maxsize=1)
def get_store() -> TripStore:
    return TripStore()
