# V22.5 Signal-to-Noise Sanity Report

Run: `logs/rsl_rl/uav_safe_3B1_v22_5_signal_to_noise_sanity/2026-04-27_21-29-45`

Status: 100-iteration sanity completed. Full 1500-iteration run was not launched because the Layer 2 gate failed.

## Patch Applied

- Set `c_time_penalty = 0.0`.
- Increased `r_pbs_scale` from `40.0` to `60.0`.
- Added early velocity-alignment bootstrap reward:
  - `c_velocity_current = 1.5` through iter 200.
  - Linear decay to `0.0` by iter 800.
  - Reward term: `c_velocity_current * clamp(v_aligned, max=1.5)`.
- Added TensorBoard metrics:
  - `task/r_velocity_alignment_mean`
  - `task/v_aligned_mean`
  - `task/c_velocity_current`

## Layer 2 Result

| Metric | Pass Criteria | Final | Last-10 Mean | Result |
| --- | ---: | ---: | ---: | --- |
| `task/goal_reached_rate` | `> 50%` | `0.3866%` | `0.4007%` | FAIL |
| `task/r_pbs_mean` | `> +0.20` | `+0.024303` | `+0.043863` | FAIL |
| `Train/mean_episode_length` | `< 80` trend | `110.3000` | `110.2590` | FAIL |
| `task/r_velocity_alignment_mean` | diagnostic | `-0.552305` | `-0.519366` | OUT |
| `task/v_aligned_mean` | diagnostic | `-0.364018` | `-0.341440` | OUT |
| `task/c_velocity_current` | expected `1.5` | `1.5000` | `1.5000` | PASS |
| `task/r_time_mean` | expected `0.0` | `0.0000` | `0.0000` | PASS |
| `safety/collision_rate` | monitor | `0.0020%` | `0.0019%` | PASS |
| `qp/hard_cap_binding_rate` | monitor | `0.0000%` | `0.0000%` | PASS |
| `Policy/mean_std` | monitor | `0.8025` | `0.8018` | monitor |

## Verdict

V22.5 correctly removed the action-independent time tax and installed the velocity-alignment helper, but the sanity gate still failed. The key diagnostic is that `v_aligned_mean` is negative, meaning the current early policy distribution is still moving away from the goal on average. The new reward term is therefore penalizing the right behavior, but PPO did not discover the corrective action within 100 iterations.

Recommended next focus: keep V21 QP math and the V22.4/V22.5 reward accounting fixes. The next intervention likely needs a stronger exploration or action prior, such as a short nominal-controller warm start, behavior cloning from the nominal acceleration command, or temporarily blending policy actions with the goal-directed nominal command during the first curriculum stage.
