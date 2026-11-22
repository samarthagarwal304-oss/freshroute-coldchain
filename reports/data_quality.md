# Data-quality note

- Trips: 3000 (50 trucks, 30 days). Readings: 870,873 rows (one per minute while a trip is active).
- Class balance: 5.33% of trips spoiled.
- Exact duplicate (trip, timestamp) rows: 4,356 (0.50%). Cleaning keeps the first.
- Rows with empty temperature: 3,492 (0.40%).
- Missing minutes after de-duplication: 3.29% of expected minutes (random drops, burst gaps up to 30 min, more on 4 low-battery trucks).
- Rows arrive shuffled within a trip; cleaning sorts by timestamp and re-indexes to a 1-minute grid, interpolating gaps of 5 minutes or less. Longer gaps stay empty and are exposed as features (pct_missing, longest_gap).
- Sensor drift: 13 trucks drift between about 0.03 and 0.09 °C per day. For the strongly drifting trucks, the learned offset follows the injected drift and the residual trend after correction is close to zero (°C per day):

|        |   injected |   learned_offset_slope |   residual_slope_after_correction |
|:-------|-----------:|-----------------------:|----------------------------------:|
| TRK-09 |     -0.087 |                 -0.077 |                            -0.002 |
| TRK-22 |      0.083 |                  0.086 |                            -0.012 |
| TRK-38 |      0.074 |                  0.078 |                            -0     |

- Drift handling: each trip subtracts the median offset of that truck's previous 10 trips (earlier trips only, so no leakage). The residual is measured against the cargo setpoint.
- Glitches: 2-4 minute spikes of +8 to +13 °C appear in about 4% of trips. They can trigger the legacy 2-minute rule; rolling-median features make the model robust to them.
- Labels come from the true latent temperature (degree-minutes above the cargo limit, with per-trip tolerance noise); driver notes and labels are never model inputs.
