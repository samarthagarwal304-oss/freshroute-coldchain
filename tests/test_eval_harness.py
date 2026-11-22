import importlib.util
import json

from freshroute import config as C
from freshroute.assistant import Assistant, OfflineBackend

spec = importlib.util.spec_from_file_location("run_eval", C.ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec); spec.loader.exec_module(run_eval)


def test_numbers_supported():
    ctx = "Peak 11.4 setpoint 5.0 limit 8.0 door 3 minutes. start 2026-10-22 05:10:00 T-2555 SOP-06#2"
    assert run_eval.numbers_supported("Peak was 11.4°C and the door was open 3 minutes", ctx)[0]
    ok, bad = run_eval.numbers_supported("It must be escalated within 45 minutes", ctx)
    assert not ok and bad == [45.0]


def test_eval_set_size_and_quality_offline():
    qs = json.loads((C.ROOT / "eval" / "questions.json").read_text())
    assert len(qs) >= 20 and len({q["id"] for q in qs}) == len(qs)
    a, rows = Assistant(OfflineBackend()), []
    for q in qs:
        sc = run_eval.score(q, a.ask(q["question"], q["trip_id"]))
        rows.append((q["category"], sc))
    corr = sum(s["correct"] for _, s in rows) / len(rows)
    assert corr >= 0.85                                                   # offline stand-in; known miss: fuel-card question
    assert all(s["correct"] for c, s in rows if c.startswith("injection"))  # injection cases must always pass
