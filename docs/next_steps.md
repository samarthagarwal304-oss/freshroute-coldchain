# Next steps before production

**Modelling**
1. Validate on real historical trips; run in shadow mode for 4-8 weeks and compare with dispatcher decisions and claims.
2. Move from trip-level to checkpoint scoring (every 15 min on the readings so far, label = "spoilage already occurred / will occur") so warnings arrive early.
3. Calibrate probabilities (isotonic) per cargo class; re-derive the threshold from real costs, per cargo and per customer.
4. Replace the median-offset drift correction with a per-sensor state estimate that also triggers recalibration work orders.

**Monitoring and drift**
5. Input drift: PSI/KS on features per week, share of missing minutes, per-truck offset trend. Alert on thresholds.
6. Outcome monitoring: weekly precision/recall on trips with known outcomes, false alerts per truck-week, alert acknowledgement rate (a falling rate means dispatchers stopped trusting alerts).
7. Retrain on a schedule with a time-based holdout and champion/challenger promotion; keep the previous model for rollback.
8. Assistant: log every question, tool call and citation; sample 50 answers a week for human grading; track "I don't know" rate and invalid-citation rate.

**Assistant and security**
9. Embeddings plus reranker with the unanswerable eval set; version SOPs and re-index automatically on change.
10. Wider injection defence: classifier on all untrusted text (driver notes, uploads), strict tool allow-list per user role, red-team set in CI.
11. Authentication, rate limits, PII handling for driver data, audit trail for who asked what.

**Platform**
12. Streaming ingestion (Kafka or MQTT), feature store, model server behind a queue, SLOs and on-call. At 100x volume: cache SOP answers, route simple questions to a small model, batch scoring, prompt caching for the system prompt and SOP passages.
