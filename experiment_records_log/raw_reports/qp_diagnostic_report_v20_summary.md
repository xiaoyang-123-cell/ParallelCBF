# QP Diagnostic Report V20 Summary

## V20 Patch Applied

- Added `include_alpha_rate_in_Dt = False` to the environment and QP config.
- Added `_last_commanded_u_xy_norm` and reset it to `0.20 * geometric_accel_limit_xy_max`.
- Passed `recent_u_xy_norm` into the QP and used it for the mass mismatch term.
- Replaced the mass disturbance term with `mass_max_perturbation * disturbance_scale * recent_u_xy_norm * grad_h_norm`.
- Added D_t component telemetry for wind, mass, parameter-rate, and recent xy command norm.

## Layer 1 Results

| Metric | 0.05N Target | 0.05N Result | 0.20N Target | 0.20N Result |
| --- | ---: | ---: | ---: | ---: |
| Feasibility | > 98% | 99.2615% PASS | > 98% | 99.2346% PASS |
| Box residual rate | < 2% | 0.7385% PASS | < 2% | 0.7654% PASS |
| D_t / \|h\| expected | [0.5, 0.7] | 1.927740 OUT | [1.2, 1.7] | 2.860369 OUT |
| CLF cost p95 | < 30 | 20.770609 PASS | < 30 | 20.907175 PASS |
| Hard-cap binding | < 25% | 50.5066% FAIL | < 25% | 61.0779% FAIL |
| Soft-clamp transition | [15%, 50%] diagnostic | 58.3496% OUT | [15%, 50%] diagnostic | 69.4824% OUT |

## D_t Component Breakdown

| Component | 0.05N | 0.20N |
| --- | ---: | ---: |
| Mean D_t used | 2.965334 | 4.580350 |
| Mean D_t_curr | 3.905867 | 6.535540 |
| Mean D_t wind term | 3.723479 | 6.356608 |
| Mean D_t mass term | 0.182388 | 0.178932 |
| Mean D_t parameter-rate term | 0.000000 | 0.000000 |
| Mean recent commanded u_xy norm | 5.867522 | 5.785247 |

## Verdict

V20 Deep Extraction succeeded at removing the two targeted phantoms: the mass term dropped to about `0.18` and the parameter-rate term is exactly `0.0`. However, Layer 1 still fails because the wind term now dominates D_t by itself.

The correct protocol action is to stop before PPO. No sanity or full training run was launched.

## Artifacts

- Low wind report: `qp_diagnostic_report_v20_wind005.md`
- High wind report: `qp_diagnostic_report_v20_wind020.md`
