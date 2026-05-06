# V17 Layer 1 Offline QP Verification

## Scope

V17 replaces the static D_t cap with an adaptive shield:

- `D_t_cap_wind_anchors = [0.00, 0.05, 0.10, 0.20, 0.30]`
- `D_t_cap_ratio_anchors = [2.0, 2.0, 2.5, 3.5, 4.0]`
- `soft_clamp_smoothness = 0.2`
- `wind_radial_factor = 0.85`

No PPO training was launched. Layer 2 sanity and Layer 3 full training remain blocked unless all Layer 1 criteria pass.

## Diagnostic Runs

- 0.05 N report: `qp_diagnostic_report_v17_wind005.md`
- 0.20 N report: `qp_diagnostic_report_v17_wind020.md`
- Envs: `512`
- Steps: `160`
- Samples per wind level: `81920`
- Source checkpoint: `logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/model_99.pt`

## Gate Results

| Criterion | Required | 0.05 N | 0.20 N | Status |
|---|---:|---:|---:|---|
| Feasibility rate | 0.05 N > 97%, 0.20 N > 95% | 99.2566% | 99.2517% | PASS |
| Box residual rate | < 2% | 0.7434% | 0.7483% | PASS |
| D_t / \|h\| ratio | 0.05 N [1.3, 1.8], 0.20 N [1.8, 2.5] | 1.087984 | 1.844338 | FAIL at 0.05 N |
| CLF cost p95 when active | < 30 | 20.629066 | 20.700581 | PASS |
| Hard-cap binding rate | < 25% | 34.0039% | 73.9929% | FAIL |
| Soft-clamp transition rate | [15%, 50%] | 39.5007% | 81.0742% | FAIL at 0.20 N |

## Additional Metrics

| Metric | 0.05 N | 0.20 N |
|---|---:|---:|
| Infeasibility rate | 0.7434% | 0.7483% |
| Mean D_t used | 1.425476 | 3.053849 |
| Mean D_t_curr | 1.605575 | 4.317452 |
| Mean D_t_floor | 0.143100 | 0.143395 |
| Mean D_t cap ratio current | 1.989526 | 1.989648 |
| CLF cost mean when active | 8.148935 | 8.235651 |
| Mean t_clf | 6.564410 | 6.636575 |

## Verdict

Layer 1: FAIL.

The adaptive shield preserved the core wins: feasibility remains above 99%, box residual remains below 1%, and CLF p95 remains comfortably below 30. The 0.20 N D_t ratio is now inside the requested band.

However, the cap-ratio telemetry indicates the piecewise interpolator is effectively stuck near `~2.0` at both wind levels, so the 0.20 N cap is not using the intended 3.5 ratio. That explains the high clamp rates:

- 0.05 N: D_t ratio is too low (`1.087984`) and hard-cap binding is too high (`34.0039%`).
- 0.20 N: D_t ratio passes (`1.844338`), but soft transition (`81.0742%`) and hard-cap binding (`73.9929%`) are far above target.

Per protocol, V17 sanity training and the 1500-iteration full run were not started.

## Targeted Retune Notes

The immediate suspect is interpolation input scale. `wind_force_bound_n` appears to be per-env current wind after curriculum/DR handling, but the cap ratio remains near the 0.05 N anchor even when the diagnostic wind is 0.20 N. Before tuning anchors, verify the exact tensor passed to `_interpolate_cap_ratio` and whether it represents max wind, sampled wind magnitude, or randomized instantaneous wind.
