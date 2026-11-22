# Design note (one page)

## Architecture
```
 sensors (synthetic) ──► clean (dedupe, 1-min grid, gap fill ≤5 min, per-truck drift offset)
                         └► trip features (27) ──► LightGBM ──► risk score + band + top factors
 SOP .txt (§ sections) ─► TF-IDF index ──► search_sops ┐
                                                       ├─► LLM (tool loop) ─► answer + [SOP-xx#n] citations
 trip data + score ─────► get_trip_summary ────────────┘          │
 driver notes ─► guardrails (screen, withhold, wrap) ──────────────┘ ─► output filter ─► FastAPI /ask
                                            FastAPI /predict ◄── same model     logs: latency, tokens, cost
```

## Model choice
LightGBM on trip-level windowed features (minutes above setpoint+k, degree-minutes, rolling-median peaks, door-open
totals, data-quality flags, cargo). Chosen because it handles small tabular data with few positives (about 5%), is
fast, and gives per-prediction contributions for the assistant's "why". Logistic regression is reported as a sanity
check. Labels come from the latent true temperature (degree-minutes above the cargo limit with per-trip tolerance
noise plus 0.4% unexplained spoilage), so there is irreducible error by design.

**Leakage controls:** strict chronological split (train 1-14, validation 15-21, test 22-30; asserted in code and tests);
the threshold is chosen on validation only; the drift offset for a trip uses only that truck's earlier trips; driver notes
and labels are never features; the assistant tool never returns the outcome label.

## Retrieval
One chunk per numbered SOP section (38 chunks), TF-IDF with light stemming and bigrams, cosine score plus an
IDF-weighted query-coverage check; below threshold the tool returns "no relevant passages" and the assistant must answer
"I don't know". Citations are chunk ids that the code verifies were actually retrieved in that request.

## Prompt-injection defence (defence in depth)
1. Regex screen on driver notes; matches are withheld from the model. 2. Remaining note text is wrapped as untrusted
data and the system prompt says it is never instructions. 3. Tools are read-only; the score is computed by code.
4. Output filter blocks system-prompt/canary leakage; invalid citations are flagged. SOP-10 gives the human policy.

## Three key trade-offs
1. **Trip-level vs streaming scoring.** Trip-level is simpler to evaluate leak-free and explain, but it warns later than a
   minute-by-minute model would. Chosen for the time-box; checkpoint training is the first next step.
2. **Alert threshold by cost, not F1.** A missed spoilage ($400 assumed) costs 20x a false alert ($20), which gives a low
   threshold (0.09). The dispatcher-trust cost of noisy alerts is real, so the report also shows false alerts per truck-week.
3. **TF-IDF over embeddings.** No model download, deterministic, auditable, 1:1 citations; costs recall on paraphrases and
   admits lexical false positives (fuel card). Mitigation: LLM relevance judgement now, embeddings/reranker later.
