# V14 Layer 1 Offline QP Verification

## Scope

V14 applies constant-only changes on top of the V13 CLF-as-cost architecture:

- `rho_clf_cost = 8.0`
- `u_xy_min/max = +/-15.0`
- `u_z_min = -11.0`
- `u_z_max = 12.0`

No PPO training was launched. The Layer 2 sanity gate remains blocked unless all Layer 1 criteria pass.

## Diagnostic Runs

- 0.05 N report: `qp_diagnostic_report_v14_wind005.md`
- 0.20 N report: `qp_diagnostic_report_v14_wind020.md`
- Envs: `512`
- Steps: `160`
- Samples per wind level: `81920`
- Source checkpoint: `logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/model_99.pt`

## Gate Results

| Criterion | Required | 0.05 N | 0.20 N | Status |
|---|---:|---:|---:|---|
| Feasibility rate | 0.05 N > 97%, 0.20 N > 90% | 99.2310% | 99.1650% | PASS |
| Box residual rate | < 2% | 0.7690% | 0.8350% | PASS |
| D_t / |h| ratio | [1.4, 1.8] | 1.564866 | 4.234174 | FAIL at 0.20 N |
| CLF cost active rate | < 50% | 80.8252% | 82.7734% | FAIL |

## Additional Metrics

| Metric | 0.05 N | 0.20 N |
|---|---:|---:|
| Infeasibility rate | 0.7690% | 0.8350% |
| Mean D_t used | 1.781700 | 5.083265 |
| Mean t_clf | 6.694502 | 7.477281 |
| Max t_clf | 98.932968 | 102.168930 |
| Soft CBF residual rate | 4.8218% | 5.3064% |
| CLF residual rate | 30.2991% | 30.3333% |

## Verdict

Layer 1: FAIL.

V14 successfully released actuator box pressure: feasibility is now above target at both 0.05 N and 0.20 N, and box residual rate dropped below 1% in both diagnostics. However, two required criteria still fail:

- CLF cost active rate remains chronically high at ~81-83%, above the <50% target.
- D_t / |h| is calibrated at 0.05 N but rises to 4.23 at 0.20 N, outside the [1.4, 1.8] band.

Per protocol, V14 sanity training and the 1500-iteration full run were not started.

## Targeted Retune Notes

The action box is no longer the main blocker. The remaining bottleneck is cost pressure and disturbance scaling:

- `rho_clf_cost = 8.0` is still not enough to bring CLF cost active rate below 50%.
- The curriculum/floor D_t formula scales too aggressively at 0.20 N relative to physical h in the visited-state distribution.
