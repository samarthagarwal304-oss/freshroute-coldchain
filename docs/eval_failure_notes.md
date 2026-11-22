# Assistant failure cases (offline run, 28 questions)

The real-LLM run (`make eval-llm`) is not included because no API key was available when this was built. After running it,
add its failures here. The three cases below are real findings from building and tuning the offline run.

## 1. Q15 "What is the daily fuel card limit for drivers?" answered instead of "I don't know"  (open)
* **What happened:** retrieval returned SOP-05#2 ("checks the unit fuel or power ... The driver reports...") and the offline
  backend, which cannot judge relevance, quoted it.
* **Root cause:** TF-IDF matches on the shared words "fuel" and "driver". Cosine score (0.22) and query coverage (0.50) are
  both inside the range of legitimate paraphrased questions, so no threshold separates them cleanly.
* **Mitigation:** the real LLM sees the passage and the system prompt tells it to say "I don't know" when passages do not answer
  the question. Longer term: a reranker or cross-encoder relevance score, and an "unanswerable" set in the eval to tune it.

## 2. Q05 "Who handles a Level 3 escalation ...?" retrieved the Level 1/2 sections  (fixed)
* **What happened:** the first answer cited SOP-06#1 and #2, which do not mention the Quality manager, so correctness failed.
* **Root cause:** the tokenizer dropped single-character tokens, so "3" vanished and "Level 3" matched every Level heading equally.
* **Fix:** keep numeric tokens. Q05 now retrieves SOP-06#3 first. A regression test (`test_retrieval_finds_right_passage`) covers it.

## 3. Threshold knife-edge between "paraphrased but covered" and "not covered"  (managed)
* **What happened:** during tuning, a query-coverage cutoff of 0.55 rejected the fuel-card question (coverage 0.50) but also
  rejected valid paraphrased questions such as "how soon must the customer be *told* ..." (coverage 0.51).
* **Root cause:** users paraphrase ("told" vs "notified"), which lowers lexical coverage just like off-topic questions do.
* **Decision:** use a lenient tool threshold (0.40) so valid questions work, and rely on the LLM for the final relevance
  call. Cost: the offline stand-in lets the fuel-card case through (failure 1). Not tuning thresholds further avoids
  overfitting 28 questions.

## Model-side failure analysis (from `reports/model_report.md`)
* Baseline false alerts (301 trips on test) are dominated by produce loads (179; limit 10°C but the rule alerts at 8°C) and also come from drifting sensors, 2-4 minute sensor glitches and short door-open spikes.
* Baseline misses (8 on test) are most often seafood (4 of 8): slow warming that stays under 8°C but above its 4°C limit.
* Model misses are mostly spoilage with no sensor signal (the 0.4% unexplained cases and high-tolerance products): irreducible here.
