# V22.8 Nominal Engine Audit Report

Run timestamp: `2026-04-27T22:58:08`
TensorBoard log dir: `logs/v22_8_nominal_engine_audit/2026-04-27_22-57-39`

## Executive Verdict

V22.8 strategic gate **PASS**: zero-residual nominal control is cleared for V22.9.

## Test Results

| Test | Result | Key Metrics | Notes |
| --- | --- | --- | --- |
| Hover & Gravity Sanity | PASS | `u_nominal_norm_first5=0.0043743`<br>`actual_acc_norm_mean=0.00531554`<br>`actual_acc_z_mean=9.7458e-05`<br>`gravity_body_z_mean=-9.81`<br>`thrust_total_mean=0.264943` | - |
| X-Axis Basis Vector | PASS | `u_nominal_x_step0=0.9`<br>`u_nominal_y_abs_step0=0`<br>`actual_acc_x_mean_steps5_20=0.631672`<br>`actual_acc_y_abs_mean_steps5_20=1.37597e-11`<br>`qp_delta_norm_step0=8.9407e-07` | - |
| Yaw Invariance | PASS | `u_nominal_world_diff_norm=0`<br>`u_safe_world_diff_norm=0`<br>`u_nominal_yaw0_x=0.9`<br>`u_nominal_yaw90_x=0.9` | - |
| Z-Axis Climb | PASS | `u_nominal_z_step0=0.9`<br>`actual_acc_z_mean_steps5_20=0.225747`<br>`thrust_total_step0=0.28917`<br>`hover_force_expected=0.26487` | - |
| Closed-Loop Convergence | PASS | `initial_goal_dist_mean=0.500003`<br>`final_goal_dist_mean=0.107765`<br>`min_goal_dist_mean=0.107765`<br>`monotonic_nonincrease_rate=1`<br>`teacher_success_rate=1`<br>`final_speed_mean=0.049004` | - |
| Acceleration Realization | PASS | `x_acc_to_nominal_ratio=1.06311`<br>`z_acc_to_nominal_ratio=0.399237`<br>`x_actual_acc_mean=0.631672`<br>`z_actual_acc_mean=0.225747`<br>`x_cross_axis_acc_mean=0.0214088`<br>`z_cross_axis_acc_mean=1.42305e-19` | - |

## Protocol State

- PPO training remains suspended unless the strategic gate is PASS.
- BC pre-training remains suspended unless the strategic gate is PASS.
- Residual RL V22.9 may only resume after teacher success exceeds 90% in the clear environment.
