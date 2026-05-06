# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_shape_micro_4env_5step/2026-04-29_23-35-51`
Demo path: `data/v23_shape_micro_4env_5step.pt`

```json
{
  "episodes_requested": 4,
  "episodes_completed": 4,
  "episodes_accepted": 0,
  "teacher_success_rate": 0.0,
  "total_timesteps": 20,
  "accepted_total_timesteps": 0,
  "waypoint_index_histogram": {
    "1": 20
  },
  "accepted_waypoint_index_histogram": {},
  "waypoint_index_ge2_timestep_fraction": 0.0,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.0,
  "success_by_scene": {
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.0
  },
  "episodes_by_scene": {
    "multi_obstacle_maze": 1,
    "open": 1,
    "single_static": 2
  },
  "path_len_mean_by_scene": {
    "multi_obstacle_maze": 8.0,
    "open": 2.0,
    "single_static": 3.5
  },
  "obstacle_scene_success_rate": 0.0,
  "wp_inside_cbf_zone_start_rate": 0.0,
  "astar_no_path_rate": 0.3333333333333333,
  "astar_tier_used_distribution": {
    "0": 1,
    "4": 2,
    "5": 1
  },
  "cbf_activation_rate": 1.0,
  "mean_u_delta_norm_when_h_soft_active": 0.0,
  "h_soft_active_timestep_count": 0,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 20,
  "mean_lateral_overshoot": 0.0038218340996536426,
  "cbf_conflict_discard_rate": 0.0,
  "pursuit_teacher_enabled": true,
  "astar_no_path_rate_by_scene": {
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.5
  },
  "wp_inside_cbf_zone_start_rate_by_scene": {
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.0
  },
  "phase2_obstacle_scene_success_gate_30pct": false,
  "phase2_wp_inside_cbf_gate_2pct": true,
  "phase2_astar_no_path_gate_5pct": false,
  "teacher_success_gate_80pct": false,
  "waypoint_index_ge2_gate_30pct": false,
  "dry_run_status": "FAIL"
}
```
