# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_dual_barrier_pursuit_validation_200/2026-04-29_19-02-10`
Demo path: `data/v23_dual_barrier_pursuit_validation_200.pt`

```json
{
  "episodes_requested": 200,
  "episodes_completed": 200,
  "episodes_accepted": 1,
  "teacher_success_rate": 0.005,
  "total_timesteps": 34577,
  "accepted_total_timesteps": 12,
  "waypoint_index_histogram": {
    "1": 26115,
    "2": 7767,
    "3": 615,
    "4": 80
  },
  "accepted_waypoint_index_histogram": {
    "1": 12
  },
  "waypoint_index_ge2_timestep_fraction": 0.24472915521878705,
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
    "dynamic_obstacle": 6.55,
    "multi_obstacle_maze": 7.7,
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
  "cbf_activation_rate": 0.999855395204905,
  "mean_u_delta_norm_when_h_soft_active": 0.827631549184559,
  "h_soft_active_timestep_count": 131,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 167,
  "mean_lateral_overshoot": 0.20383579822544295,
  "cbf_conflict_discard_rate": 0.04,
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
