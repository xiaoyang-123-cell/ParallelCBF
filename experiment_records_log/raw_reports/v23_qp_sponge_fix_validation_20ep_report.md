# V23 Controller Diagnostic 20-Episode Report

Generated: `2026-04-30T11:17:12`

Raw log: `data/v23_qp_sponge_fix_validation_20ep.pt`
Summary JSON: `data/v23_qp_sponge_fix_validation_20ep.json`
Plot dir: `plots/v23_qp_sponge_fix_validation_20ep`

## Summary

| Metric | Value |
| --- | ---: |
| `episodes_completed` | 20 |
| `overall_success_rate` | 0.7000 |
| `cbf_activation_rate` | 0.0000 |
| `action_delta_rate` | 0.0000 |
| `h_hard_violation_rate` | 0.000000 |
| `h_hard_violation_count` | 0 |
| `h_hard_checked_count` | 1727 |
| `mean_episode_length_steps` | 114.05 |
| `mean_final_dist_to_goal` | 0.6662 |
| `mean_min_dist_to_goal` | 0.6641 |
| `mean_speed` | 0.5887 |
| `max_speed_achieved` | 1.5721 |
| `mean_v_profile_cap_hit_pct` | 0.3903 |
| `mean_lat_acc_hit_pct` | 0.1580 |
| `mean_u_nominal_norm` | 1.0719 |

## Termination Reasons

```json
{
  "open_space": {
    "goal_reached": 10
  },
  "single_obstacle": {
    "goal_reached": 4,
    "timeout": 6
  }
}
```

## Episode Summaries

| Ep | Scene | Reason | Len | Final Dist | Min Dist | Mean Speed | Max Speed | v_cap Hit | lat_acc Hit | Mean Cmd | Final v_ref |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | open_space | goal_reached | 11 | 0.4483 | 0.4483 | 0.2591 | 0.3978 | 0.1429 | 0.1429 | 1.1086 | 0.3192 |
| 1 | open_space | goal_reached | 23 | 0.4435 | 0.4435 | 0.4587 | 0.7085 | 0.0370 | 0.0370 | 1.4453 | 0.6859 |
| 2 | open_space | goal_reached | 40 | 0.4468 | 0.4468 | 0.4372 | 0.7506 | 0.0625 | 0.0625 | 0.8689 | 0.6882 |
| 3 | open_space | goal_reached | 40 | 0.4485 | 0.4485 | 0.6141 | 1.0449 | 0.0000 | 0.0000 | 1.1469 | 1.0203 |
| 4 | open_space | goal_reached | 60 | 0.4341 | 0.4341 | 0.5895 | 1.0912 | 0.0930 | 0.0930 | 1.0809 | 1.0844 |
| 5 | open_space | goal_reached | 61 | 0.4378 | 0.4378 | 0.6917 | 1.2732 | 0.1224 | 0.1224 | 1.2806 | 1.0281 |
| 6 | open_space | goal_reached | 63 | 0.4393 | 0.4393 | 0.8163 | 1.3602 | 0.0926 | 0.0926 | 1.4965 | 1.0318 |
| 7 | open_space | goal_reached | 86 | 0.4289 | 0.4289 | 0.6913 | 1.4059 | 0.0833 | 0.0833 | 1.1623 | 1.1427 |
| 8 | open_space | goal_reached | 74 | 0.4407 | 0.4407 | 0.8380 | 1.4836 | 0.0000 | 0.0000 | 1.4810 | 0.8782 |
| 9 | open_space | goal_reached | 96 | 0.4463 | 0.4463 | 0.8216 | 1.5721 | 0.0986 | 0.0986 | 1.3244 | 1.1354 |
| 250 | single_obstacle | timeout | 179 | 1.2948 | 1.2948 | 0.1302 | 0.3841 | 1.0000 | 0.0000 | 0.1522 | 0.0500 |
| 251 | single_obstacle | timeout | 179 | 1.5944 | 1.5525 | 0.1060 | 0.3734 | 1.0000 | 0.0000 | 0.1367 | 0.0500 |
| 252 | single_obstacle | timeout | 179 | 1.2799 | 1.2799 | 0.3263 | 0.4062 | 0.9474 | 0.0211 | 0.3427 | 0.3000 |
| 253 | single_obstacle | goal_reached | 158 | 0.4494 | 0.4494 | 0.8053 | 1.1492 | 0.6190 | 0.3429 | 1.2859 | 0.9632 |
| 254 | single_obstacle | timeout | 179 | 1.3725 | 1.3725 | 0.2825 | 0.3901 | 0.9485 | 0.0412 | 0.2173 | 0.2999 |
| 255 | single_obstacle | goal_reached | 155 | 0.4454 | 0.4454 | 0.8196 | 1.1327 | 0.5702 | 0.3246 | 1.3239 | 0.9568 |
| 256 | single_obstacle | goal_reached | 169 | 0.4339 | 0.4339 | 0.8025 | 1.1564 | 0.6239 | 0.4103 | 1.4101 | 1.0000 |
| 257 | single_obstacle | timeout | 179 | 0.5198 | 0.5198 | 0.6807 | 1.0594 | 0.3136 | 0.2373 | 1.3486 | 0.5208 |
| 258 | single_obstacle | goal_reached | 171 | 0.4348 | 0.4348 | 0.9333 | 1.2566 | 0.6889 | 0.6889 | 1.4162 | 1.1150 |
| 259 | single_obstacle | timeout | 179 | 1.0842 | 1.0842 | 0.6695 | 1.0643 | 0.3617 | 0.3617 | 1.4100 | 0.6886 |

## Plots

- `plots/v23_qp_sponge_fix_validation_20ep/01_termination_reason_histogram.png`
- `plots/v23_qp_sponge_fix_validation_20ep/02_empirical_velocity_profile.png`
- `plots/v23_qp_sponge_fix_validation_20ep/03_failed_trajectory_goal_sphere_overlay.png`
- `plots/v23_qp_sponge_fix_validation_20ep/04_endpoint_scatter.png`
- `plots/v23_qp_sponge_fix_validation_20ep/05_command_distribution.png`
