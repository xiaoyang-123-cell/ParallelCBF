# V23 Micro-Logger Audit

Generated: `2026-04-29T18:25:32`

Raw log: `data/v23_micro_logger_audit_10ep.pt`
Validation JSON: `data/v23_micro_logger_audit_validation.json`

## Pre-Registered Clinical Trial

| Quantity | Claude's Prediction | Observed Value |
| --- | ---: | ---: |
| In open_space: % timesteps with >=1 arena barrier active | >=70% | 0.00% |
| u_delta_norm distribution mode (when active) | 0.05-0.15 | 0.5250 |
| Empirical dv/dt when CBF active & u_ff requests >1 m/s^2 | 0.2-0.4 m/s^2 | 0.9247 m/s^2 |
| Active-barrier h value mode | 0.1-0.3 m | 0.0500 raw QP h |
| Active-barrier physical clearance mode | context | 0.4500 m |

## Summary

```json
{
  "active_barrier_clearance_mode_m": 0.45,
  "active_barrier_h_mode": 0.05,
  "active_barrier_h_units": "raw_qp_h",
  "active_timesteps": 1349,
  "all_predictions_match": false,
  "arena_diagnostic_active_rate": 0.0,
  "empirical_dv_dt_when_active_and_ff_gt_1": 0.9246714690517648,
  "episodes_completed": 10,
  "episodes_requested": 10,
  "ff_gt_1_active_timesteps": 847,
  "generated": "2026-04-29T18:25:32",
  "internal_active_rate": 0.5114899925871016,
  "open_space_arena_barrier_active_rate": 0.0,
  "open_space_timesteps": 454,
  "prediction_match": {
    "empirical_dv_dt_02_04": false,
    "h_mode_01_03": false,
    "open_space_arena_active_ge_70pct": false,
    "u_delta_mode_005_015": false
  },
  "scene_distribution": {
    "open": 5,
    "single_static": 5
  },
  "success_by_scene": {
    "open": 0.6,
    "single_static": 0.0
  },
  "success_rate": 0.30000001192092896,
  "total_timesteps": 1349,
  "u_delta_norm_mode_when_active": 0.525
}
```
