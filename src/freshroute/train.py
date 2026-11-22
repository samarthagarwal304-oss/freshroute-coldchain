"""Part A pipeline: features -> time-based split -> models -> evaluation vs baseline.

Split (by trip start day, 30 days):
    days  1-14  train          (fit candidates)
    days 15-21  validation     (choose alert threshold by business cost)
    days 22-30  test           (touched once, for the report)
Final model is refit on days 1-21 with the same hyper-parameters and the validation threshold.
"""
from __future__ import annotations

import json
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import features as F

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=15,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=C.SEED, verbose=-1)


def make_lgb():
    return lgb.LGBMClassifier(**LGB_PARAMS)


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))


def confusion(y, flag):
    y, flag = np.asarray(y), np.asarray(flag).astype(bool)
    return int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()), int((~flag & (y == 1)).sum())


def metrics(y, flag, n_trucks, weeks, scores=None):
    tp, fp, fn = confusion(y, flag)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    out = dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec,
               f1=(2 * prec * rec / (prec + rec)) if (tp and prec + rec) else 0.0,
               false_alerts_per_truck_week=fp / (n_trucks * weeks),
               cost=fn * C.COST_MISSED_SPOILAGE + fp * C.COST_FALSE_ALERT)
    if scores is not None:
        out["pr_auc"] = float(average_precision_score(y, scores))
    return out


def pick_threshold(y, p):
    """Minimise expected business cost on the validation set."""
    best = (None, 1e18)
    for t in np.round(np.arange(0.02, 0.95, 0.01), 2):
        _, fp, fn = confusion(y, p >= t)
        cost = fn * C.COST_MISSED_SPOILAGE + fp * C.COST_FALSE_ALERT
        if cost < best[1] - 1e-9:
            best = (float(t), cost)
    return best[0]


def bootstrap_ci(y, flag, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, flag = np.asarray(y), np.asarray(flag)
    P, R = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        tp, fp, fn = confusion(y[i], flag[i])
        if tp + fp: P.append(tp / (tp + fp))
        if tp + fn: R.append(tp / (tp + fn))
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return dict(precision_ci95=q(P), recall_ci95=q(R))


def day_of(ts):
    return (ts - pd.Timestamp(C.START_DATE)).dt.days + 1


def main():
    readings = pd.read_csv(C.DATA_DIR / "readings.csv.gz", parse_dates=["ts"])
    trips = pd.read_csv(C.DATA_DIR / "trips.csv", parse_dates=["start_ts", "end_ts"])
    X, _ = F.build_dataset(readings, trips)
    X.to_csv(C.DATA_DIR / "features.csv", index=False)

    X["day"] = day_of(X["start_ts"])
    tr, va, te = X[X.day <= C.TRAIN_END_DAY], X[(X.day > C.TRAIN_END_DAY) & (X.day <= C.VAL_END_DAY)], X[X.day > C.VAL_END_DAY]
    assert tr.start_ts.max() < va.start_ts.min() and va.start_ts.max() < te.start_ts.min(), "time split violated"
    weeks_te = (C.N_DAYS - C.VAL_END_DAY) / 7
    weeks_va = (C.VAL_END_DAY - C.TRAIN_END_DAY) / 7
    feats = F.FEATURES
    print(f"trips train/val/test = {len(tr)}/{len(va)}/{len(te)}  positives = {tr.y.sum()}/{va.y.sum()}/{te.y.sum()}")

    report = dict(split=dict(train=len(tr), val=len(va), test=len(te),
                             positives=dict(train=int(tr.y.sum()), val=int(va.y.sum()), test=int(te.y.sum())),
                             test_days=C.N_DAYS - C.VAL_END_DAY),
                  assumptions=dict(cost_missed_spoilage=C.COST_MISSED_SPOILAGE, cost_false_alert=C.COST_FALSE_ALERT),
                  models={})

    # ---- baseline (no fitting) ----
    base_te = metrics(te.y, te.baseline_flag, C.N_TRUCKS, weeks_te)
    base_te["alert_events_per_truck_week"] = float(te.baseline_events.sum() / (C.N_TRUCKS * weeks_te))
    base_te.update(bootstrap_ci(te.y, te.baseline_flag))
    report["models"]["baseline_rule_8C_2min"] = base_te

    final = {}
    for name, maker in [("logistic_regression", make_lr), ("lightgbm", make_lgb)]:
        m = maker().fit(tr[feats], tr.y)
        thr = pick_threshold(va.y.values, m.predict_proba(va[feats])[:, 1])
        trva = pd.concat([tr, va])
        m = maker().fit(trva[feats], trva.y)             # refit on train+val, same hyper-params
        p_te = m.predict_proba(te[feats])[:, 1]
        r = metrics(te.y, p_te >= thr, C.N_TRUCKS, weeks_te, p_te)
        r["threshold"] = thr
        r.update(bootstrap_ci(te.y, p_te >= thr))
        # recall at the SAME false-alert volume as the baseline (oracle on test; comparison aid only)
        order = np.argsort(-p_te)
        fp_budget, fps, rec_at = base_te["fp"], 0, 0
        for k, i in enumerate(order):
            fps += te.y.values[i] == 0
            if fps > fp_budget: break
            rec_at = te.y.values[order[:k + 1]].sum() / max(1, te.y.sum())
        r["recall_at_baseline_false_alert_volume"] = float(rec_at)
        report["models"][name] = r
        final[name] = (m, thr)
        print(f"{name}: thr={thr} P={r['precision']:.2f} R={r['recall']:.2f} FA/truck/wk={r['false_alerts_per_truck_week']:.2f} PR-AUC={r['pr_auc']:.3f}")
    print(f"baseline: P={base_te['precision']:.2f} R={base_te['recall']:.2f} FA/truck/wk={base_te['false_alerts_per_truck_week']:.2f}")

    # ---- save serving artifact (LightGBM) ----
    m, thr = final["lightgbm"]
    imp = sorted(zip(feats, m.booster_.feature_importance("gain")), key=lambda x: -x[1])
    report["lightgbm_top_features_by_gain"] = [(k, round(float(v), 1)) for k, v in imp[:10]]
    joblib.dump(dict(model=m, features=feats, threshold=thr, trained_on_days=f"1-{C.VAL_END_DAY}",
                     test_metrics=report["models"]["lightgbm"]), C.MODEL_DIR / "model.joblib")

    # ---- error analysis on test (uses hidden scenario truth, analysis only) ----
    hidden = pd.read_csv(C.DATA_DIR / "trips_hidden_truth.csv")
    te2 = te.assign(p=m.predict_proba(te[feats])[:, 1]).merge(hidden, on="trip_id")
    te2["flag"] = te2.p >= thr
    ea = {}
    for nm, sub in [("model_false_negatives", te2[(te2.y == 1) & ~te2.flag]),
                    ("model_false_positives", te2[(te2.y == 0) & te2.flag]),
                    ("baseline_false_positives", te2[(te2.y == 0) & (te2.baseline_flag == 1)]),
                    ("baseline_false_negatives", te2[(te2.y == 1) & (te2.baseline_flag == 0)])]:
        ea[nm] = dict(count=len(sub), by_scenario=sub.scenario.value_counts().to_dict(),
                      by_cargo=sub.cargo_type.value_counts().to_dict())
    report["error_analysis_test"] = ea
    (C.REPORT_DIR / "model_report.json").write_text(json.dumps(report, indent=2, default=float))
    write_markdown(report)
    try:
        plot_pr(te2, m, feats, base_te, thr)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


def write_markdown(r):
    L = ["# Model report: spoilage risk vs baseline", "",
         f"Time-based split: train days 1-14, validation days 15-21 (threshold choice), test days 22-30. "
         f"Trips train/val/test = {r['split']['train']}/{r['split']['val']}/{r['split']['test']}; "
         f"spoiled trips = {r['split']['positives']['train']}/{r['split']['positives']['val']}/{r['split']['positives']['test']}.", "",
         f"Business cost assumption: missed spoilage = ${C.COST_MISSED_SPOILAGE:.0f}, false alert = ${C.COST_FALSE_ALERT:.0f}. "
         "Threshold chosen on validation to minimise expected cost.", "",
         "| Model | Precision | Recall | False alerts / truck / week | PR-AUC | Cost on test ($) | Recall at baseline's alert volume |",
         "|---|---|---|---|---|---|---|"]
    for k, v in r["models"].items():
        L.append(f"| {k} | {v['precision']:.2f} {v['precision_ci95']} | {v['recall']:.2f} {v['recall_ci95']} | "
                 f"{v['false_alerts_per_truck_week']:.2f} | {v.get('pr_auc', float('nan')):.3f} | {v['cost']:.0f} | "
                 f"{v.get('recall_at_baseline_false_alert_volume', float('nan')):.2f} |")
    b = r["models"]["baseline_rule_8C_2min"]
    L += ["", f"Baseline raw alert events (not trip flags): {b['alert_events_per_truck_week']:.2f} per truck per week. "
          "Trip-level numbers count a trip as flagged if the rule fired at least once.", "",
          "Brackets are 95% bootstrap intervals over test trips; with few positives they are wide, so read differences accordingly.", "",
          "## Top LightGBM features (gain)", ""]
    L += [f"- {k}: {v}" for k, v in r["lightgbm_top_features_by_gain"]]
    L += ["", "## Error analysis on test (by hidden generator scenario)", ""]
    for k, v in r["error_analysis_test"].items():
        L.append(f"- **{k}**: {v['count']}  scenarios={v['by_scenario']}  cargo={v['by_cargo']}")
    (C.REPORT_DIR / "model_report.md").write_text("\n".join(L) + "\n")


def plot_pr(te2, m, feats, base, thr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(te2.y, te2.p)
    plt.figure(figsize=(5, 4))
    plt.plot(r, p, label="LightGBM")
    plt.scatter([base["recall"]], [base["precision"]], c="red", label="baseline rule")
    plt.xlabel("recall"); plt.ylabel("precision"); plt.title("Test PR curve"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(C.REPORT_DIR / "pr_curve.png", dpi=120)


if __name__ == "__main__":
    sys.exit(main())
