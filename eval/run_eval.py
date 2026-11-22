"""Scores the assistant on eval/questions.json for correctness and groundedness.
   python eval/run_eval.py [--backend offline|anthropic|auto]

correctness  : every must_include group matched (any alternative, case-insensitive), no must_not_include hit,
               and "I don't know" exactly when expect_idk.
groundedness : (a) cites at least one retrieved chunk and none that were not retrieved, (b) at least one citation comes
               from an expected SOP, (c) every number-with-unit in the answer appears in the retrieved SOP text or tool data.
               For "I don't know" answers: grounded = no citations.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freshroute import config as C
from freshroute.assistant import Assistant, pick_backend

NUM_UNIT = re.compile(r"(?<![\w.#-])(\d+(?:\.\d+)?)\s?(?:°C|°|minutes?|mins?|hours?|%)", re.I)
NUM_ANY = re.compile(r"(?<![\w#-])\d+(?:\.\d+)?")
STRIP = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}|T-\d{4}|SOP-\d+#?\d*|§\d+")


def numbers_supported(answer: str, context: str) -> tuple[bool, list]:
    ctx = {float(x) for x in NUM_ANY.findall(STRIP.sub(" ", context))}
    ans = [float(x) for x in NUM_UNIT.findall(STRIP.sub(" ", answer))]
    unsupported = [x for x in ans if x not in ctx]
    return not unsupported, unsupported


def has(text, alts):
    return any(a.lower() in text.lower() for a in alts)


def score(q, res):
    a = res["answer"]
    if q["expect_idk"]:
        return dict(correct=res["idk"], grounded=not res["citations"], source_hit=None, unsupported=[])
    ok = all(has(a, g) for g in q["must_include"]) and not any(has(a, g) for g in q["must_not_include"]) and not res["idk"]
    cited = {c["chunk_id"].split("#")[0] for c in res["citations"]}
    src_hit = None
    if q["expected_sources"]:
        src_hit = bool(cited & set(q["expected_sources"]))
    nums_ok, bad = numbers_supported(a, res["context_text"])
    grounded = not res["invalid_citations"] and nums_ok and (src_hit if src_hit is not None else True) \
        and (bool(res["citations"]) if q["expected_sources"] else True)
    return dict(correct=ok, grounded=grounded, source_hit=src_hit, unsupported=bad)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backend", default=None); ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    qs = json.loads((ROOT / "eval" / "questions.json").read_text())[: args.limit]
    asst = Assistant(pick_backend(args.backend))
    rows = []
    for q in qs:
        try:
            res = asst.ask(q["question"], q["trip_id"])
        except Exception as e:
            res = dict(answer=f"ERROR {e}", citations=[], invalid_citations=[], idk=False, context_text="", latency_ms=0,
                       usage=dict(input_tokens=0, output_tokens=0, est_cost_usd=0), tool_calls=[], retrieved_chunk_ids=[])
        sc = score(q, res)
        rows.append(dict(q=q, res={k: v for k, v in res.items() if k != "context_text"}, score=sc))
        print(f"{q['id']} {q['category']:<18} correct={sc['correct']!s:<5} grounded={sc['grounded']!s:<5} {res['latency_ms']:>7.0f} ms")
    cat = defaultdict(list)
    for r in rows:
        cat[r["q"]["category"]].append(r)
    pct = lambda xs: 100 * sum(xs) / max(1, len(xs))
    n = len(rows)
    summary = dict(backend=asst.backend.name, n=n, correctness=pct([r["score"]["correct"] for r in rows]),
                   groundedness=pct([r["score"]["grounded"] for r in rows]),
                   by_category={k: dict(n=len(v), correctness=pct([x["score"]["correct"] for x in v]),
                                        groundedness=pct([x["score"]["grounded"] for x in v])) for k, v in cat.items()},
                   idk_accuracy=pct([r["res"]["idk"] == r["q"]["expect_idk"] for r in rows]),
                   latency_ms_avg=sum(r["res"]["latency_ms"] for r in rows) / n,
                   input_tokens=sum(r["res"]["usage"]["input_tokens"] for r in rows),
                   output_tokens=sum(r["res"]["usage"]["output_tokens"] for r in rows),
                   est_cost_usd=sum(r["res"]["usage"]["est_cost_usd"] for r in rows))
    (C.REPORT_DIR / "eval_results.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=2, default=str))
    write_report(summary, rows)
    print(json.dumps(summary, indent=2))


def write_report(s, rows):
    L = ["# Assistant evaluation report", "", f"Backend: **{s['backend']}**, questions: {s['n']}.", ""]
    if s["backend"] == "offline":
        L += ["> The offline backend is a rule-based stand-in, not an LLM. These numbers verify retrieval, tools, guardrails and "
              "scoring plumbing. Re-run with `ANTHROPIC_API_KEY` set (`make eval-llm`) for LLM quality numbers.", ""]
    L += [f"- Correctness: **{s['correctness']:.0f}%**", f"- Groundedness: **{s['groundedness']:.0f}%**",
          f"- 'I don't know' accuracy: **{s['idk_accuracy']:.0f}%**",
          f"- Avg latency {s['latency_ms_avg']:.0f} ms, tokens in/out {s['input_tokens']}/{s['output_tokens']}, est. cost ${s['est_cost_usd']:.4f}", "",
          "| Category | n | Correctness | Groundedness |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['correctness']:.0f}% | {v['groundedness']:.0f}% |" for k, v in s["by_category"].items()]
    fails = [r for r in rows if not (r["score"]["correct"] and r["score"]["grounded"])]
    L += ["", f"## Failures ({len(fails)})", ""]
    for r in fails:
        L += [f"### {r['q']['id']} [{r['q']['category']}] {r['q']['question']}",
              f"- correct={r['score']['correct']} grounded={r['score']['grounded']} source_hit={r['score']['source_hit']} "
              f"unsupported_numbers={r['score']['unsupported']} retrieved={r['res']['retrieved_chunk_ids']}",
              f"- Expected: {r['q']['must_include'] or ('I don' + chr(39) + 't know' if r['q']['expect_idk'] else '')}",
              "- Answer (first 400 chars): " + r["res"]["answer"][:400].replace("\n", " "),
              "- Root cause / fix: *(fill in after inspecting the retrieved passages)*", ""]
    (C.REPORT_DIR / "eval_report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
