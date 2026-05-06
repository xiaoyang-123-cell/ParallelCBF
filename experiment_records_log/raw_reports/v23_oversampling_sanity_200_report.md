# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_oversampling_sanity_200/2026-04-30_12-43-14`
Demo path: `data/v23_oversampling_sanity_200.pt`

```json
{
  "episodes_requested": 200,
  "episodes_completed": 200,
  "episodes_accepted": 122,
  "teacher_success_rate": 0.61,
  "total_timesteps": 29890,
  "accepted_total_timesteps": 15928,
  "waypoint_index_histogram": {
    "1": 18356,
    "2": 6317,
    "3": 3024,
    "4": 1360,
    "5": 625,
    "6": 155,
    "7": 50,
    "8": 3
  },
  "accepted_waypoint_index_histogram": {
    "1": 10157,
    "2": 3559,
    "3": 1233,
    "4": 535,
    "5": 334,
    "6": 82,
    "7": 25,
    "8": 3
  },
  "waypoint_index_ge2_timestep_fraction": 0.38588156574105054,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.3623179306880964,
  "success_by_scene": {
    "dynamic_obstacle": 0.325,
    "multi_obstacle_maze": 0.36538461538461536,
    "open": 1.0,
    "single_static": 0.75
  },
  "episodes_by_scene": {
    "dynamic_obstacle": 40,
    "multi_obstacle_maze": 52,
    "open": 36,
    "single_static": 72
  },
  "path_len_mean_by_scene": {
    "dynamic_obstacle": 6.9,
    "multi_obstacle_maze": 7.346153846153846,
    "open": 2.0,
    "single_static": 6.75
  },
  "obstacle_scene_success_rate": 0.524390243902439,
  "wp_inside_cbf_zone_start_rate": 0.012195121951219513,
  "astar_no_path_rate": 0.06097560975609756,
  "astar_tier_used_distribution": {
    "0": 36,
    "1": 29,
    "2": 79,
    "3": 46,
    "5": 10
  },
  "cbf_activation_rate": 0.004215456674473068,
  "action_delta_rate": 0.0051856808297089324,
  "mean_u_delta_norm_when_h_soft_active": 0.5956260272493911,
  "h_soft_active_timestep_count": 126,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 8537,
  "mean_lateral_overshoot": 0.13036603526468485,
  "cbf_conflict_discard_rate": 0.0,
  "pursuit_teacher_enabled": true,
  "astar_no_path_rate_by_scene": {
    "dynamic_obstacle": 0.1,
    "multi_obstacle_maze": 0.019230769230769232,
    "open": 0.0,
    "single_static": 0.06944444444444445
  },
  "wp_inside_cbf_zone_start_rate_by_scene": {
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.027777777777777776
  },
  "phase2_obstacle_scene_success_gate_30pct": true,
  "phase2_wp_inside_cbf_gate_2pct": true,
  "phase2_astar_no_path_gate_5pct": false,
  "teacher_success_gate_80pct": false,
  "waypoint_index_ge2_gate_30pct": true,
  "dry_run_status": "FAIL"
}
```
