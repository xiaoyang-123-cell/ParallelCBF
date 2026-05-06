# V23 Controller Diagnostic 20-Episode Report

Generated: `2026-04-30T00:13:22`

Raw log: `data/v23_controller_diagnostic_20ep.pt`
Summary JSON: `data/v23_controller_diagnostic_20ep.json`
Plot dir: `plots/v23_controller_diagnostic_20ep`

## Summary

| Metric | Value |
| --- | ---: |
| `episodes_completed` | 20 |
| `overall_success_rate` | 0.1500 |
| `cbf_activation_rate` | 0.1191 |
| `action_delta_rate` | 1.0000 |
| `mean_episode_length_steps` | 156.95 |
| `mean_final_dist_to_goal` | 1.1229 |
| `mean_min_dist_to_goal` | 1.0522 |
| `mean_speed` | 0.2479 |
| `max_speed_achieved` | 0.6750 |
| `mean_v_profile_cap_hit_pct` | 0.3903 |
| `mean_lat_acc_hit_pct` | 0.1580 |
| `mean_u_nominal_norm` | 0.9860 |

## Termination Reasons

```json
{
  "open_space": {
    "goal_reached": 3,
    "timeout": 7
  },
  "single_obstacle": {
    "timeout": 10
  }
}
```

## Episode Summaries

| Ep | Scene | Reason | Len | Final Dist | Min Dist | Mean Speed | Max Speed | v_cap Hit | lat_acc Hit | Mean Cmd | Final v_ref |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | open_space | goal_reached | 12 | 0.4430 | 0.4430 | 0.2578 | 0.3595 | 0.1429 | 0.1429 | 1.1358 | 0.3392 |
| 1 | open_space | goal_reached | 28 | 0.4467 | 0.4467 | 0.3662 | 0.4607 | 0.0370 | 0.0370 | 1.5293 | 0.7861 |
| 2 | open_space | goal_reached | 56 | 0.4440 | 0.4440 | 0.3071 | 0.4706 | 0.0625 | 0.0625 | 1.1933 | 0.9428 |
| 3 | open_space | timeout | 179 | 0.8288 | 0.4763 | 0.2260 | 0.5200 | 0.0000 | 0.0000 | 0.7529 | 0.0500 |
| 4 | open_space | timeout | 179 | 0.7842 | 0.5151 | 0.2449 | 0.5375 | 0.0930 | 0.0930 | 0.8304 | 0.0500 |
| 5 | open_space | timeout | 179 | 0.9117 | 0.6878 | 0.2278 | 0.5502 | 0.1224 | 0.1224 | 0.8106 | 0.0500 |
| 6 | open_space | timeout | 179 | 0.9792 | 0.7868 | 0.2393 | 0.6022 | 0.0926 | 0.0926 | 0.7881 | 0.0500 |
| 7 | open_space | timeout | 179 | 0.9604 | 0.8371 | 0.2408 | 0.6127 | 0.0833 | 0.0833 | 0.8526 | 0.0500 |
| 8 | open_space | timeout | 179 | 1.1147 | 0.9885 | 0.2215 | 0.6727 | 0.0000 | 0.0000 | 0.7522 | 0.0500 |
| 9 | open_space | timeout | 179 | 1.0875 | 1.0157 | 0.2768 | 0.6750 | 0.0986 | 0.0986 | 0.9174 | 0.0500 |
| 250 | single_obstacle | timeout | 179 | 1.4085 | 1.4085 | 0.0669 | 0.3749 | 1.0000 | 0.0000 | 0.1626 | 0.0500 |
| 251 | single_obstacle | timeout | 179 | 1.6087 | 1.5525 | 0.0492 | 0.3591 | 1.0000 | 0.0000 | 0.0965 | 0.0500 |
| 252 | single_obstacle | timeout | 179 | 1.4583 | 1.4583 | 0.1465 | 0.2985 | 0.9474 | 0.0211 | 0.4911 | 0.3000 |
| 253 | single_obstacle | timeout | 179 | 1.2343 | 1.2343 | 0.2960 | 0.5523 | 0.6190 | 0.3429 | 1.4210 | 0.6437 |
| 254 | single_obstacle | timeout | 179 | 1.5709 | 1.5709 | 0.1438 | 0.2862 | 0.9485 | 0.0412 | 0.4981 | 0.2999 |
| 255 | single_obstacle | timeout | 179 | 1.3339 | 1.3339 | 0.3071 | 0.5607 | 0.5702 | 0.3246 | 1.4960 | 0.6241 |
| 256 | single_obstacle | timeout | 179 | 1.4081 | 1.4081 | 0.3209 | 0.5688 | 0.6239 | 0.4103 | 1.5330 | 0.8555 |
| 257 | single_obstacle | timeout | 179 | 1.4150 | 1.4150 | 0.3182 | 0.4775 | 0.3136 | 0.2373 | 1.4212 | 0.5208 |
| 258 | single_obstacle | timeout | 179 | 1.4145 | 1.4145 | 0.3551 | 0.6324 | 0.6889 | 0.6889 | 1.5596 | 0.8911 |
| 259 | single_obstacle | timeout | 179 | 1.6062 | 1.6062 | 0.3456 | 0.5292 | 0.3617 | 0.3617 | 1.4788 | 0.6886 |

## Plots

- `plots/v23_controller_diagnostic_20ep/01_termination_reason_histogram.png`
- `plots/v23_controller_diagnostic_20ep/02_empirical_velocity_profile.png`
- `plots/v23_controller_diagnostic_20ep/03_failed_trajectory_goal_sphere_overlay.png`
- `plots/v23_controller_diagnostic_20ep/04_endpoint_scatter.png`
- `plots/v23_controller_diagnostic_20ep/05_command_distribution.png`
