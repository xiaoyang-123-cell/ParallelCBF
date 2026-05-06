# V22.10 Attitude-Lag Layer 1 Diagnostic

Cap offset applied: `1.5`
D_t cap ratio anchors: `[5.0, 5.0, 5.5, 6.0, 7.0]`

| Scenario | u_cmd | Wind | Feasibility | Hard Cap Binding | Target | Soft Transition | D_t/|h| | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hover_low_wind | 1.0 | 0.05 N | 100.00% | 16.02% | 15.00% | 20.70% | 2.013 | FAIL |
| approach_high_wind | 2.0 | 0.20 N | 100.00% | 13.67% | 35.00% | 17.58% | 2.206 | PASS |
| aggressive_high_wind | 5.0 | 0.20 N | 100.00% | 40.23% | 50.00% | 50.78% | 3.729 | PASS |

## D_t Breakdown

| Scenario | D_t Used | D_t Curr | Wind Steady | Attitude Lag | Mass | Param Rate | Cap Ratio | Wind Radial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hover_low_wind | 6.3789 | 6.9241 | 0.5591 | 5.8160 | 0.5489 | 0.0000 | 5.00 | 0.55 |
| approach_high_wind | 6.8789 | 7.3815 | 1.0166 | 5.8160 | 0.5489 | 0.0000 | 6.00 | 0.25 |
| aggressive_high_wind | 13.2570 | 16.4715 | 1.0166 | 14.5401 | 0.9149 | 0.0000 | 6.00 | 0.25 |

## Gate

Layer 1 **FAIL**. Do not launch Step 3.
