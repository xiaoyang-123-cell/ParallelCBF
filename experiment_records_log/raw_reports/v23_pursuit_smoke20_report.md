# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_pursuit_smoke20/2026-04-29_17-35-26`
Demo path: `data/v23_pursuit_smoke20.pt`

```json
{
  "episodes_requested": 20,
  "episodes_completed": 20,
  "episodes_accepted": 0,
  "teacher_success_rate": 0.0,
  "total_timesteps": 2001,
  "accepted_total_timesteps": 0,
  "waypoint_index_histogram": {
    "1": 1827,
    "2": 174
  },
  "accepted_waypoint_index_histogram": {},
  "waypoint_index_ge2_timestep_fraction": 0.08695652173913043,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.0,
  "success_by_scene": {
    "adversarial_start": 0.0,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.6,
    "single_static": 0.0
  },
  "episodes_by_scene": {
    "adversarial_start": 1,
    "dynamic_obstacle": 2,
    "multi_obstacle_maze": 4,
    "open": 5,
    "single_static": 8
  },
  "path_len_mean_by_scene": {
    "adversarial_start": 2.0,
    "dynamic_obstacle": 6.5,
    "multi_obstacle_maze": 8.75,
    "open": 2.0,
    "single_static": 7.25
  },
  "obstacle_scene_success_rate": 0.0,
  "wp_inside_cbf_zone_start_rate": 0.0,
  "astar_no_path_rate": 0.06666666666666667,
  "astar_tier_used_distribution": {
    "0": 5,
    "1": 4,
    "2": 4,
    "4": 6,
    "5": 1
  },
  "cbf_activation_rate": 1.0,
  "mean_lateral_overshoot": 0.1034766593206957,
  "cbf_conflict_discard_rate": 0.15,
  "pursuit_teacher_enabled": true,
  "astar_no_path_rate_by_scene": {
    "adversarial_start": 1.0,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.0
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
