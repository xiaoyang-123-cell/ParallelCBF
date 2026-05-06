# V16 Layer 1 Offline QP Verification

## Scope

V16 keeps the V15 D_t formula unchanged and updates only the two requested constants:

- `wind_radial_factor = 0.85`
- `D_t_max_ratio = 3.0`

No PPO training was launched. Layer 2 sanity and Layer 3 full training remain blocked unless all Layer 1 criteria pass.

## Diagnostic Runs

- 0.05 N report: `qp_diagnostic_report_v16_wind005.md`
- 0.20 N report: `qp_diagnostic_report_v16_wind020.md`
- Envs: `512`
- Steps: `160`
- Samples per wind level: `81920`
- Source checkpoint: `logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/model_99.pt`

## Gate Results

| Criterion | Required | 0.05 N | 0.20 N | Status |
|---|---:|---:|---:|---|
| Feasibility rate | 0.05 N > 97%, 0.20 N > 95% | 99.2554% | 99.2041% | PASS |
| Box residual rate | < 2% | 0.7446% | 0.7959% | PASS |
| D_t / \|h\| ratio | 0.05 N [1.3, 1.8], 0.20 N [1.8, 2.5] | 1.147247 | 2.130240 | FAIL at 0.05 N |
| CLF cost p95 when active | < 30 | 20.625994 | 20.690535 | PASS |
| Cap-binding rate | 0.05 N < 20%, 0.20 N < 40% | 28.1384% | 63.5205% | FAIL |

## Additional Metrics

| Metric | 0.05 N | 0.20 N |
|---|---:|---:|
| Infeasibility rate | 0.7446% | 0.7959% |
| Mean D_t used | 1.465886 | 3.319060 |
| Mean D_t_curr | 1.609366 | 4.321888 |
| Mean D_t_floor | 0.143438 | 0.143542 |
| CLF cost mean when active | 8.153342 | 8.251240 |
| Mean t_clf | 6.568158 | 6.672807 |
| Max t_clf | 98.494019 | 102.445168 |

## Verdict

Layer 1: FAIL.

V16 improved the high-wind D_t ratio into the target band: 0.20 N moved from V15's `1.574871` to `2.130240`. Feasibility, box residual, and CLF p95 remain healthy.

However, the calibration is still not precise enough for the Director's gate:

- 0.05 N D_t / |h| is `1.147247`, below the required `[1.3, 1.8]`.
- 0.05 N cap-binding rate is `28.1384%`, above the required `<20%`.
- 0.20 N cap-binding rate is `63.5205%`, above the required `<40%`.

Per protocol, V16 sanity training and the 1500-iteration full run were not started.

## Targeted Retune Notes

The ratio and cap gates are now pulling in different directions at low wind:

- Raising `wind_radial_factor` would help the 0.05 N ratio but may worsen high-wind cap pressure.
- Raising `D_t_max_ratio` should reduce cap-binding and may raise the 0.05 N ratio, but could push 0.20 N ratio above the upper bound if too large.
- A wind-dependent cap or cap based on hard `h` instead of soft `h` may be needed if a single global cap cannot satisfy both cap-binding and ratio bands.
