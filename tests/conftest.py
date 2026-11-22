import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freshroute import config as C


def pytest_sessionstart(session):
    """Generate data + train the model if they are missing (first run only, ~40 s)."""
    if not (C.DATA_DIR / "readings.csv.gz").exists():
        from freshroute import datagen
        datagen.generate()
    if not (C.MODEL_DIR / "model.joblib").exists() or not (C.DATA_DIR / "features.csv").exists():
        from freshroute import train
        train.main()
