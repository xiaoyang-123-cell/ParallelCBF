# V17.5 Layer 1 Offline QP Verification

## Scope

V17.5 applies the emergency pivot requested by the Director:

- Fixed the wind reference bug by passing the true stage-limit wind `_wind_max_current` into QP D_t scaling instead of `_wind_max_current * randomization_scale_episode`.
- Replaced the tanh soft clamp with the Softplus soft-min formula.
- Updated cap anchors to `[2.2, 2.2, 2.8, 3.8, 4.5]`.

No PPO training was launched. Layer 2 sanity and Layer 3 full training remain blocked unless all Layer 1 criteria pass.

## Variable Audit

Confirmed bug:

- Old QP input: `wind_force_bound_n = self._wind_max_current * self._randomization_scale_episode`
- At diagnostic `dr_scale = 0.20`, the 0.20 N run was effectively passed as 0.04 N to the cap-ratio interpolator.
- This explains why V17 `D_t_cap_ratio_current` stayed near `~1.99`.

Patch:

- New QP input: `wind_force_bound_n = self._wind_max_current`
- `disturbance_scale = self._randomization_scale_episode` remains separate for DR/mass scaling.

## Diagnostic Runs

- 0.05 N report: `qp_diagnostic_report_v17_5_wind005.md`
- 0.20 N report: `qp_diagnostic_report_v17_5_wind020.md`
- Envs: `512`
- Steps: `160`
- Samples per wind level: `81920`
- Source checkpoint: `logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/model_99.pt`

## Gate Results

| Criterion | Required | 0.05 N | 0.20 N | Status |
|---|---:|---:|---:|---|
| Feasibility rate | 0.05 N > 97%, 0.20 N > 95% | 99.2566% | 99.2456% | PASS |
| Box residual rate | < 2% | 0.7434% | 0.7544% | PASS |
| D_t / \|h\| ratio | 0.05 N [1.3, 1.8], 0.20 N [1.8, 2.5] | 1.761607 | 3.311014 | FAIL at 0.20 N |
| CLF cost p95 when active | < 30 | 20.654736 | 21.054308 | PASS |
| Hard-cap binding rate | < 15% | 78.0908% | 95.9900% | FAIL |
| Soft-clamp transition rate | [15%, 50%] | 84.3958% | 97.8479% | FAIL |

## Additional Metrics

| Metric | 0.05 N | 0.20 N |
|---|---:|---:|
| Infeasibility rate | 0.7434% | 0.7544% |
| Mean D_t used | 3.098072 | 6.788087 |
| Mean D_t_curr | 5.216573 | 18.761320 |
| Mean D_t_floor | 0.143292 | 0.143381 |
| Mean D_t cap ratio current | 2.188559 | 3.780334 |
| CLF cost mean when active | 8.202323 | 8.348593 |
| Mean t_clf | 6.612422 | 6.758467 |

## Verdict

Layer 1: FAIL.

V17.5 successfully fixed the interpolation input bug. The cap ratio now tracks the intended wind stage:

- 0.05 N cap ratio: `2.188559`, near the 2.2 anchor.
- 0.20 N cap ratio: `3.780334`, near the 3.8 anchor.

The 0.05 N D_t ratio now passes at `1.761607`, matching the expected target. However, fixing the wind reference exposed a high-wind D_t magnitude problem:

- 0.20 N D_t ratio is `3.311014`, above the `[1.8, 2.5]` target band.
- Soft-clamp and hard-cap rates are extremely high at both winds, so the current clamp metrics are not compatible with this Softplus formulation and/or the unclamped D_t is too far above cap.

Per protocol, V17.5 sanity training and the 1500-iteration full run were not started.

## Targeted Retune Notes

The remaining issue is no longer interpolation. The next likely retune is to reduce high-wind D_t magnitude or make the radial factor wind-dependent:

- Keep the stage-limit wind input fix.
- Consider reducing high-wind cap anchors or `wind_radial_factor` at 0.20 N.
- Revisit the hard-cap metric definition for Softplus, because `uncapped > cap * (1 + smoothness)` now fires often even when feasibility and box residual are healthy.
