"""Model artifact loading and scoring (used by the API and the assistant tools)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURES, FEATURE_DESCRIPTIONS

ARTIFACT = C.MODEL_DIR / "model.joblib"


class RiskModel:
    def __init__(self, artifact: dict):
        self.art = artifact
        self.model = artifact["model"]
        self.features = artifact["features"]
        self.threshold = float(artifact["threshold"])

    def band(self, p: float) -> str:
        if p >= C.RISK_BAND_HIGH:
            return "HIGH"
        return "ELEVATED" if p >= self.threshold else "LOW"

    def score(self, feat_rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(feat_rows[self.features])[:, 1]

    def explain(self, feat_row: pd.Series, k: int = 3) -> list[dict]:
        """Top-k features pushing the score UP (SHAP-style contributions from LightGBM)."""
        X = feat_row[self.features].to_frame().T.astype(float)
        contrib = self.model.booster_.predict(X, pred_contrib=True)[0][:-1]
        hidden = {"cargo_code", "setpoint", "duration_min", "truck_offset", "hum_mean"}   # not actionable for a dispatcher
        order = [i for i in np.argsort(-contrib) if self.features[i] not in hidden][:k]
        out = []
        for i in order:
            if contrib[i] <= 0:
                continue
            name = self.features[i]
            out.append(dict(feature=name, description=FEATURE_DESCRIPTIONS[name],
                            value=(int(feat_row[name]) if float(feat_row[name]).is_integer() else round(float(feat_row[name]), 2)), contribution=round(float(contrib[i]), 2)))
        return out

    def predict_one(self, feat: dict) -> dict:
        row = pd.Series(feat)
        p = float(self.score(row.to_frame().T.astype(float))[0])
        return dict(risk_score=round(p, 3), flagged=bool(p >= self.threshold), risk_band=self.band(p),
                    threshold=round(self.threshold, 3), top_factors=self.explain(row))


@lru_cache(maxsize=1)
def get_model() -> RiskModel:
    return RiskModel(joblib.load(ARTIFACT))
