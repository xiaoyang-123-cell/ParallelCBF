# V22.10d Layer 3 Summary

Run: `logs/rsl_rl/uav_safe_3B1_v22_10d_bc_anchor_full/2026-04-28_19-35-33_layer3_1500`

Status: completed normally at iteration `1499/1500`.

## Logging Patch Verification

The old `task/goal_reached_rate` was renamed to `task/goal_reached_instantaneous_rate`.

New episode-level metrics were added and verified before Layer 3:

| Metric | Dummy Run Result |
| --- | ---: |
| `task/episode_success_rate` | reached `1.0000` by iter 2 |
| `task/final_dist_mean_successful` | about `0.448 m` at relaxed radius `0.45 m` |
| Old instantaneous occupancy | about `2.65%` |

This confirms the earlier 2% metric was a logging illusion, not a failure of the BC prior.

## Gate Checkpoints

| Iter | Episode Success | Final Dist Successful | Goal Radius | Goal Distance Curriculum | Obstacle Presence | Collision | Hard Cap | Wind Max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 90.03% | 0.4486 m | 0.45 m | 0.5 m | 0.0 | 0.0% | 0.0% | 0.05 N |
| 100 | 100.00% | 0.4314 m | 0.433 m | 0.8 m | 0.0 | 0.0% | 0.0% | 0.05 N |
| 500 | 67.15% | 0.2986 m | 0.30 m | 1.5 m | 1.0 | 0.0% | 2.49% | 0.075 N |
| 750 | 20.39% | 0.2991 m | 0.30 m | 2.5 m | 1.0 | 0.0% | 25.57% | 0.11 N |
| 1000 | 0.76% | 0.2992 m | 0.30 m | 2.5 m | 1.0 | 0.0% | 26.37% | 0.16 N |
| 1200 | 0.74% | 0.2989 m | 0.30 m | 2.5 m | 1.0 | 0.0% | 23.09% | 0.20 N |
| 1499 | 1.44% | 0.2993 m | 0.30 m | 2.5 m | 1.0 | 0.0% | 25.94% | 0.20 N |

## Final Snapshot

| Metric | Final Value |
| --- | ---: |
| `task/episode_success_rate` | 1.4388% |
| `task/final_dist_mean_successful` | 0.2993 m |
| `Train/mean_reward` | 727.1982 |
| `Train/mean_episode_length` | 987.4000 |
| `Policy/mean_std` | 0.3606 |
| `Loss/clip_fraction` | 0.0440 |
| `safety/collision_rate` | 0.0000% |
| `safety/fallback_activation_rate` | 0.0000% |
| `safety/transient_infeasibility_rate` | 0.0000% |
| `qp/persistent_terminations` | 0.0000% |
| `qp/hard_cap_binding_rate` | 25.9392% |
| `qp/D_t_to_h_ratio` | 1.9319 |
| `qp/D_t_used` | 7.7880 |
| `qp/D_t_attitude_lag_term_mean` | 7.3689 |
| `qp/D_t_wind_term_mean` | 1.1521 |
| `qp/D_t_mass_term_mean` | 0.3720 |
| `qp/soft_clamp_transition_rate` | 28.7529% |
| `qp/clf_cost_p95_when_active` | 9.5069 |
| `hover/goal_dist_mean` | 1.0789 m |
| `dr/wind_max_current` | 0.2000 N |

## Late-Run Means

| Window | Episode Success | Collision | Hard Cap | Goal Dist | Episode Len | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Iter 1000-1499 | 1.36% | 0.0% | 24.45% | 1.0535 m | 990.20 | 719.30 |
| Iter 1200-1499 | 1.45% | 0.0% | 24.20% | 1.0642 m | 990.07 | 727.29 |

## Verdict

The logging patch succeeded and V22.10d passed the early Layer 3 gates:

- Iter 100 target `episode_success_rate > 80%`: passed with `100%`.
- Iter 500 target `episode_success_rate > 50%`: passed with `67.15%`.

The final Layer 3 task gate did not pass:

- Final target `episode_success_rate > 75%`: failed with `1.44%`.
- Final target `collision_rate < 2%`: passed with `0.0%`.

Interpretation: the safety architecture is stable and conservative under full wind/obstacle curriculum, but the task policy does not recover after the curriculum reaches 2.5 m goals, full obstacle presence, strict 0.30 m radius, and 0.20 N wind. The successful episodes that do occur terminate at the correct precision, around `0.299 m`, so the remaining bottleneck is not precision geometry. It is long-horizon navigation/reaching under the final curriculum.

Recommended next move: keep V22.10d logging and V22.10 math locked, then address the task side with a staged curriculum or objective change focused on long-distance 2.5 m reaching under obstacles, rather than another CBF feasibility rewrite.
