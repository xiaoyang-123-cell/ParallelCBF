# QP Diagnostic Report V21 Summary

## Phase 0 Wind Model Verification

The wind disturbance is sampled once during reset in `apply_reset_environment_randomization` and stored in `_external_force_w`. Runtime kinematics then apply the stored force each step; the disturbance is not re-sampled at high frequency.

V21 therefore uses `wind_drag_compensation_factor = 0.70`.

## Layer 1 Offline Math Gate

| Metric | 0.05N Result | 0.20N Result |
| --- | ---: | ---: |
| Feasibility | 99.2273% PASS | 99.2517% PASS |
| Box residual rate | 0.7727% PASS | 0.7483% PASS |
| D_t / \|h\| | 0.763456 diagnostic | 1.247097 IN-BAND |
| CLF cost p95 | 20.587656 PASS | 20.708012 PASS |
| Hard-cap binding | 15.2161% PASS | 19.2737% PASS |
| Soft-clamp transition | 18.1433% IN-BAND | 22.9346% IN-BAND |

## Layer 2 Sanity Snapshot

Run: `logs/rsl_rl/uav_safe_3B1_v21_drag_calibrated_sanity/2026-04-27_17-50-34`

| Metric | Iter 50 | Iter 99 |
| --- | ---: | ---: |
| Train/mean_episode_length | 149.27 | 134.59 |
| Train/mean_reward | 272.95172 | 191.54846 |
| Policy/mean_std | 0.792632 | 0.802519 |
| safety/fallback_activation_rate | 1.4923% | 1.4211% |
| qp/hard_cap_binding_rate | 11.8317% | 12.5275% |
| qp/D_t_to_h_ratio | 0.731581 | 0.746036 |
| qp/D_t_wind_term_mean | 0.969350 | 0.993404 |
| qp/D_t_mass_term_mean | 0.103636 | 0.219844 |
| qp/D_t_param_rate_term_mean | 0.000000 | 0.000000 |
| hover/goal_dist_mean | 1.987973 | 1.995034 |
| dr/wind_max_current | 0.050000 | 0.050000 |

## Verdict

V21 clears the offline math gate at both 0.05N and 0.20N. The 100-iteration sanity run also keeps hard-cap binding below 20%, fallback below 2%, and reward strongly positive. Per the V21 automatic protocol, the full 1500-iteration run is authorized.

## Artifacts

- Low wind report: `qp_diagnostic_report_v21_wind005.md`
- High wind report: `qp_diagnostic_report_v21_wind020.md`
