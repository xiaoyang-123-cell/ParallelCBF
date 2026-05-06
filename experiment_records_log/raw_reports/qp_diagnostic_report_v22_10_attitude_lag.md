# V22.10 Attitude-Lag Layer 1 Diagnostic

Cap offset applied: `1.5`
D_t cap ratio anchors: `[5.5, 5.5, 6.0, 6.5, 7.5]`

| Scenario | u_cmd | Wind | Feasibility | Hard Cap Binding | Target | Soft Transition | D_t/|h| | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hover_low_wind | 1.0 | 0.05 N | 100.00% | 13.87% | 15.00% | 18.02% | 2.062 | PASS |
| approach_high_wind | 2.0 | 0.20 N | 100.00% | 12.01% | 35.00% | 15.53% | 2.247 | PASS |
| aggressive_high_wind | 5.0 | 0.20 N | 100.00% | 36.28% | 50.00% | 45.63% | 3.885 | PASS |

## D_t Breakdown

| Scenario | D_t Used | D_t Curr | Wind Steady | Attitude Lag | Mass | Param Rate | Cap Ratio | Wind Radial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hover_low_wind | 6.4429 | 6.9241 | 0.5591 | 5.8160 | 0.5489 | 0.0000 | 5.50 | 0.55 |
| approach_high_wind | 6.9284 | 7.3815 | 1.0166 | 5.8160 | 0.5489 | 0.0000 | 6.50 | 0.25 |
| aggressive_high_wind | 13.5926 | 16.4716 | 1.0166 | 14.5401 | 0.9149 | 0.0000 | 6.50 | 0.25 |

## Gate

Layer 1 **PASS**. V22.10 is cleared for Step 3 data collection/BC/PPO.
