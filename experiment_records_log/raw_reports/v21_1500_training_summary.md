# V21 1500-Iteration Training Summary

Run: `logs/rsl_rl/uav_safe_3B1_v21_drag_calibrated_full/2026-04-27_17-55-47`

Training completed normally at iteration `1499/1500`.

## Final Snapshot

| Metric | Final Value |
| --- | ---: |
| Train/mean_reward | 210.2255 |
| Train/mean_episode_length | 132.7100 |
| Mean action std | 0.70 |
| safety/fallback_activation_rate | 0.6856% |
| qp/hard_cap_binding_rate | 21.9198% |
| safety/collision_rate | 0.6205% |
| safety/transient_infeasibility_rate | 0.6744% |
| qp/persistent_terminations | 0.0112% |
| hover/goal_dist_mean | 1.9021 |
| dr/wind_max_current | 0.2000 N |
| dr/curriculum_scale | 1.0000 |

## Key Checkpoints

| Iter | Reward | Episode Len | Fallback | Hard Cap | Collision | Goal Dist | Wind Max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 29.92 | 16.62 | 0.006% | 15.437% | 0.212% | 2.07 | 0.050 N |
| 100 | 173.30 | 139.43 | 1.491% | 11.979% | 0.598% | 1.99 | 0.050 N |
| 300 | 188.81 | 134.25 | 1.392% | 15.216% | 0.665% | 1.90 | 0.050 N |
| 700 | 235.23 | 135.02 | 0.803% | 21.157% | 0.580% | 1.97 | 0.100 N |
| 1000 | 191.55 | 119.85 | 0.649% | 21.248% | 0.586% | 1.90 | 0.160 N |
| 1200 | 148.92 | 119.69 | 0.723% | 20.276% | 0.606% | 1.91 | 0.200 N |
| 1499 | 210.23 | 132.71 | 0.686% | 21.920% | 0.621% | 1.90 | 0.200 N |

## Late-Run Stability

| Window | Reward Mean | Episode Len Mean | Fallback Mean | Hard Cap Mean | Goal Dist Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Iter 1000-1499 | 212.0250 | 133.2104 | 0.7297% | 20.5993% | 1.9109 |
| Iter 1200-1499 | 210.8870 | 133.3377 | 0.7425% | 20.8209% | 1.9109 |
| Iter 1400-1499 | 220.3591 | 133.0602 | 0.7562% | 20.9132% | 1.8928 |

## QP And Safety Diagnostics

| Metric | Final | Iter 1200-1499 Mean | Interpretation |
| --- | ---: | ---: | --- |
| qp/D_t_to_h_ratio | 1.3787 | 1.3899 | In the intended V21 low-conservatism band. |
| qp/D_t_used | 2.0586 | 2.0949 | Stable after 0.20 N wind ramp. |
| qp/D_t_wind_term_mean | 1.7478 | n/a | Wind remains dominant but controlled. |
| qp/D_t_mass_term_mean | 0.5640 | n/a | V20 commanded-action mass term remains bounded. |
| qp/D_t_param_rate_term_mean | 0.0000 | n/a | Alpha-rate phantom remains evicted. |
| qp/soft_clamp_transition_rate | 25.9419% | n/a | Soft clamp actively shaping margin, not fully saturated. |
| qp/hard_cap_binding_rate | 21.9198% | 20.8209% | Below the 25% alert line for the full run. |
| qp/binding_hard_rate | 0.0010% | n/a | Hard CBF residuals are essentially non-binding. |
| qp/binding_box_rate | 0.6846% | n/a | Actuator box pressure is low. |
| qp/clf_cost_p95_when_active | 22.4506 | 22.8491 | Below the p95 target of 30. |

## Best And Worst Points

| Metric | Best / Worst |
| --- | --- |
| Max reward | Iter 1305, `312.5612` |
| Max episode length | Iter 138, `151.1500` |
| Max hard-cap binding | Iter 899, `24.9023%` |
| Final hard-cap binding | Iter 1499, `21.9198%` |
| Min late goal distance | Iter 1400-1499 window mean `1.8928` |

## Verdict

V21 solved the structural safety/QP bottleneck. The full run completed without the hard-cap re-spike failure mode: hard-cap binding stayed below the 25% alarm threshold, fallback stayed below 1%, and persistent QP termination was nearly zero.

The remaining limitation is policy/task performance rather than QP feasibility. Episode length plateaued around `133`, goal distance stayed around `1.9 m`, and the policy did not break into long-horizon goal-reaching despite a stable safety layer.

Recommended next focus: keep the V21 QP math locked, then improve task performance through reward/curriculum/objective design or a staged navigation policy, not another D_t/QP rework.

