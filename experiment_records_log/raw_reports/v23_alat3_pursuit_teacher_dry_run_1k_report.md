# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_alat3_pursuit_teacher_dry_run_1k_retry40/2026-04-30_11-35-00`
Demo path: `data/v23_alat3_pursuit_teacher_dry_run_1k.pt`

```json
{
  "episodes_requested": 1000,
  "episodes_completed": 1000,
  "episodes_accepted": 638,
  "teacher_success_rate": 0.638,
  "total_timesteps": 141515,
  "accepted_total_timesteps": 76717,
  "waypoint_index_histogram": {
    "1": 98208,
    "2": 24487,
    "3": 11023,
    "4": 4797,
    "5": 2277,
    "6": 618,
    "7": 102,
    "8": 3
  },
  "accepted_waypoint_index_histogram": {
    "1": 52703,
    "2": 12371,
    "3": 6114,
    "4": 3250,
    "5": 1617,
    "6": 569,
    "7": 90,
    "8": 3
  },
  "waypoint_index_ge2_timestep_fraction": 0.3060240963855422,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.31302058213955186,
  "success_by_scene": {
    "adversarial_start": 0.0,
    "dynamic_obstacle": 0.29,
    "multi_obstacle_maze": 0.415,
    "open": 1.0,
    "single_static": 0.69
  },
  "episodes_by_scene": {
    "adversarial_start": 50,
    "dynamic_obstacle": 100,
    "multi_obstacle_maze": 200,
    "open": 250,
    "single_static": 400
  },
  "path_len_mean_by_scene": {
    "adversarial_start": 2.0,
    "dynamic_obstacle": 6.88,
    "multi_obstacle_maze": 7.475,
    "open": 2.0,
    "single_static": 6.87
  },
  "obstacle_scene_success_rate": 0.5173333333333333,
  "wp_inside_cbf_zone_start_rate": 0.0,
  "astar_no_path_rate": 0.108,
  "astar_tier_used_distribution": {
    "0": 250,
    "1": 162,
    "2": 304,
    "4": 203,
    "5": 81
  },
  "cbf_activation_rate": 0.0017665971805108998,
  "action_delta_rate": 0.0027134932692647424,
  "mean_u_delta_norm_when_h_soft_active": 0.5842712834775448,
  "h_soft_active_timestep_count": 250,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 5018,
  "mean_lateral_overshoot": 0.13211381825173862,
  "cbf_conflict_discard_rate": 0.0,
  "pursuit_teacher_enabled": true,
  "astar_no_path_rate_by_scene": {
    "adversarial_start": 1.0,
    "dynamic_obstacle": 0.12,
    "multi_obstacle_maze": 0.025,
    "open": 0.0,
    "single_static": 0.035
  },
  "wp_inside_cbf_zone_start_rate_by_scene": {
    "adversarial_start": 0.0,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.0
  },
  "phase2_obstacle_scene_success_gate_30pct": true,
  "phase2_wp_inside_cbf_gate_2pct": true,
  "phase2_astar_no_path_gate_5pct": false,
  "teacher_success_gate_80pct": false,
  "waypoint_index_ge2_gate_30pct": true,
  "dry_run_status": "FAIL"
}
```
