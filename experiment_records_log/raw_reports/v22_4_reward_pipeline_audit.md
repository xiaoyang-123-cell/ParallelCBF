# V22.4 Reward Pipeline Audit

Run: `logs/rsl_rl/uav_safe_3B1_v22_4_reward_pipeline_sanity/2026-04-27_20-26-14`

Status: 100-iteration sanity completed. Full 1500-iteration run was not launched because the sanity gate failed goal-reaching.

## Tier 1 PBS Diagnostic

Root cause found: PBS arithmetic is correct. The negative `r_pbs_mean` anomaly was not caused by a buffer reset spike or sign inversion.

| Metric | Final |
| --- | ---: |
| `task/r_pbs_mean` | `+0.016202` |
| `debug/phi_now_mean` | `-0.749216` |
| `debug/phi_prev_mean` | `-0.742129` |
| `debug/r_pbs_at_step_0` | `+0.201394` |
| `debug/r_pbs_at_step_5` | `+0.216579` |
| `debug/r_pbs_when_hovering` | `+0.218232` |
| `debug/r_pbs_when_approaching` | `+0.641471` |

Interpretation:

- No massive step-0 PBS spike was observed, so there is no active `_prev_goal_dist` reset bug.
- Hovering PBS is positive, so there is no PBS sign inversion bug.
- Approaching PBS is strongly positive, so the reward correctly favors motion toward the goal when the policy actually approaches.

## Tier 2 / Tier 3 Fixes Applied

- Goal reaching is now a true termination condition in `_get_dones()`.
- Timeouts remain truncations rather than terminations.
- `bootstrap_truncations = True` is explicitly pinned in the PPO runner config.

## 100-Iteration Sanity Result

| Metric | Pass Criteria | Final | Last-10 Mean | Result |
| --- | ---: | ---: | ---: | --- |
| `task/goal_reached_rate` | `> 35%` | `0.3866%` | `0.4007%` | FAIL |
| `task/r_pbs_mean` | `> 0` | `+0.016202` | `+0.029242` | PASS |
| `safety/collision_rate` | monitor | `0.0020%` | `0.0019%` | PASS |
| `qp/hard_cap_binding_rate` | monitor | `0.0000%` | `0.0000%` | PASS |
| `Train/mean_reward` | monitor | `-8.2715` | `-2.8945` | OUT |
| `Train/mean_episode_length` | monitor | `110.3000` | `110.2590` | monitor |
| `Policy/mean_std` | monitor | `0.8025` | `0.8024` | monitor |
| `debug/goal_dist_3d_mean` | monitor | `0.7492` | `0.7373` | OUT |

## Verdict

V22.4 fixed the reward-pipeline accounting issue: PBS is now positive and behaves with the expected sign. However, the policy still does not reach goals in the 100-iteration gate. The remaining failure is not QP feasibility, collision safety, or PBS arithmetic; it is task learning / credit assignment.

Recommended next focus: keep V21 QP math frozen and keep the V22.4 termination/bootstrap fixes. The next patch should target early goal-reaching signal density, for example through stronger supervised-style warm start, temporary direct velocity/action guidance, or a short no-QP/no-obstacle imitation-style curriculum before reintroducing the shield costs.
