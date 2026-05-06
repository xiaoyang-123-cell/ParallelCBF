# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_broadcast_fix_validation_200/2026-04-29_23-37-36`
Demo path: `data/v23_broadcast_fix_validation_200.pt`

```json
{
  "episodes_requested": 200,
  "episodes_completed": 200,
  "episodes_accepted": 0,
  "teacher_success_rate": 0.0,
  "total_timesteps": 34572,
  "accepted_total_timesteps": 0,
  "waypoint_index_histogram": {
    "1": 26201,
    "2": 7450,
    "3": 696,
    "4": 225
  },
  "accepted_waypoint_index_histogram": {},
  "waypoint_index_ge2_timestep_fraction": 0.2421323614485711,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.0,
  "success_by_scene": {
    "adversarial_start": 0.0,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.18,
    "single_static": 0.0
  },
  "episodes_by_scene": {
    "adversarial_start": 10,
    "dynamic_obstacle": 20,
    "multi_obstacle_maze": 40,
    "open": 50,
    "single_static": 80
  },
  "path_len_mean_by_scene": {
    "adversarial_start": 2.0,
    "dynamic_obstacle": 6.65,
    "multi_obstacle_maze": 7.675,
    "open": 2.0,
    "single_static": 6.8
  },
  "obstacle_scene_success_rate": 0.0,
  "wp_inside_cbf_zone_start_rate": 0.0,
  "astar_no_path_rate": 0.1,
  "astar_tier_used_distribution": {
    "0": 50,
    "1": 35,
    "2": 67,
    "4": 33,
    "5": 15
  },
  "cbf_activation_rate": 0.9997107485826681,
  "mean_u_delta_norm_when_h_soft_active": 0.7852734052217923,
  "h_soft_active_timestep_count": 143,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 213,
  "mean_lateral_overshoot": 0.1954425115610264,
  "cbf_conflict_discard_rate": 0.045,
  "pursuit_teacher_enabled": true,
  "astar_no_path_rate_by_scene": {
    "adversarial_start": 1.0,
    "dynamic_obstacle": 0.15,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.025
  },
  "wp_inside_cbf_zone_start_rate_by_scene": {
    "adversarial_start": 0.0,
    "dynamic_obstacle": 0.0,
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
