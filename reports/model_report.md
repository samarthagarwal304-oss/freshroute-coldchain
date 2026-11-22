# Model report: spoilage risk vs baseline

Time-based split: train days 1-14, validation days 15-21 (threshold choice), test days 22-30. Trips train/val/test = 1400/700/900; spoiled trips = 76/37/47.

Business cost assumption: missed spoilage = $400, false alert = $20. Threshold chosen on validation to minimise expected cost.

| Model | Precision | Recall | False alerts / truck / week | PR-AUC | Cost on test ($) | Recall at baseline's alert volume |
|---|---|---|---|---|---|---|
| baseline_rule_8C_2min | 0.11 [0.082, 0.148] | 0.83 [0.708, 0.932] | 4.68 | nan | 9220 | nan |
| logistic_regression | 0.47 [0.365, 0.577] | 0.81 [0.684, 0.921] | 0.65 | 0.789 | 4440 | 0.91 |
| lightgbm | 0.68 [0.556, 0.797] | 0.87 [0.762, 0.956] | 0.30 | 0.846 | 2780 | 0.94 |

Baseline raw alert events (not trip flags): 11.32 per truck per week. Trip-level numbers count a trip as flagged if the rule fired at least once.

Brackets are 95% bootstrap intervals over test trips; with few positives they are wide, so read differences accordingly.

## Top LightGBM features (gain)

- dm_above_3: 2607.4
- min_above_3: 2334.9
- max_roll30: 1579.5
- resid_p95: 1098.0
- cargo_code: 710.4
- longest_run_above_2: 627.6
- min_above_8abs: 355.9
- setpoint: 327.6
- duration_min: 263.6
- dm_above_1: 254.3

## Error analysis on test (by hidden generator scenario)

- **model_false_negatives**: 6  scenarios={'normal': 4, 'door_benign': 1, 'cooling_partial': 1}  cargo={'produce': 4, 'vaccine': 1, 'meat': 1}
- **model_false_positives**: 19  scenarios={'door_long': 8, 'door_benign': 4, 'cooling_failure': 4, 'cooling_partial': 3}  cargo={'dairy': 7, 'vaccine': 6, 'seafood': 5, 'meat': 1}
- **baseline_false_positives**: 301  scenarios={'normal': 124, 'door_benign': 79, 'glitch': 38, 'door_long': 29, 'cooling_partial': 20, 'cooling_failure': 11}  cargo={'produce': 179, 'dairy': 51, 'vaccine': 48, 'meat': 12, 'seafood': 11}
- **baseline_false_negatives**: 8  scenarios={'cooling_failure': 4, 'normal': 2, 'cooling_partial': 1, 'door_long': 1}  cargo={'seafood': 4, 'dairy': 1, 'meat': 1, 'vaccine': 1, 'produce': 1}
