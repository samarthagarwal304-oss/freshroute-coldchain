# Assistant evaluation report

Backend: **offline**, questions: 28.

> The offline backend is a rule-based stand-in, not an LLM. These numbers verify retrieval, tools, guardrails and scoring plumbing. Re-run with `ANTHROPIC_API_KEY` set (`make eval-llm`) for LLM quality numbers.

- Correctness: **96%**
- Groundedness: **96%**
- 'I don't know' accuracy: **96%**
- Avg latency 41 ms, tokens in/out 19296/3472, est. cost $0.0000

| Category | n | Correctness | Groundedness |
|---|---|---|---|
| sop | 14 | 100% | 100% |
| not_covered | 6 | 83% | 83% |
| trip_combined | 4 | 100% | 100% |
| trip_data | 1 | 100% | 100% |
| injection_note | 1 | 100% | 100% |
| injection_question | 2 | 100% | 100% |

## Failures (1)

### Q15 [not_covered] What is the daily fuel card limit for drivers?
- correct=False grounded=False source_hit=None unsupported_numbers=[] retrieved=['SOP-05#2']
- Expected: I don't know
- Answer (first 400 chars): According to the SOPs: - The driver stops safely, checks the unit fuel or power, and power-cycles the unit once. The driver reports the result to the dispatcher. [SOP-05#2]
- Root cause / fix: *(fill in after inspecting the retrieved passages)*

