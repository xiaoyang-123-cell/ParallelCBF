# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_phase3_debug8/2026-04-29_16-14-13`
Demo path: `data/v23_phase3_debug8.pt`

```json
{
  "episodes_requested": 8,
  "episodes_completed": 8,
  "episodes_accepted": 1,
  "teacher_success_rate": 0.125,
  "total_timesteps": 152,
  "accepted_total_timesteps": 12,
  "waypoint_index_histogram": {
    "1": 148,
    "2": 4
  },
  "accepted_waypoint_index_histogram": {
    "1": 12
  },
  "waypoint_index_ge2_timestep_fraction": 0.02631578947368421,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.0,
  "success_by_scene": {
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.5,
    "single_static": 0.0
  },
  "episodes_by_scene": {
    "dynamic_obstacle": 1,
    "multi_obstacle_maze": 2,
    "open": 2,
    "single_static": 3
  },
  "path_len_mean_by_scene": {
    "dynamic_obstacle": 7.0,
    "multi_obstacle_maze": 6.5,
    "open": 2.0,
    "single_static": 6.0
  },
  "obstacle_scene_success_rate": 0.0,
  "wp_inside_cbf_zone_start_rate": 0.3333333333333333,
  "astar_no_path_rate": 0.0,
  "astar_tier_used_distribution": {
    "0": 2,
    "2": 3,
    "4": 3
  },
  "astar_no_path_rate_by_scene": {
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.0
  },
  "wp_inside_cbf_zone_start_rate_by_scene": {
    "dynamic_obstacle": 1.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.3333333333333333
  },
  "phase2_obstacle_scene_success_gate_30pct": false,
  "phase2_wp_inside_cbf_gate_2pct": false,
  "phase2_astar_no_path_gate_5pct": true,
  "teacher_success_gate_80pct": false,
  "waypoint_index_ge2_gate_30pct": false,
  "dry_run_status": "FAIL"
}
```
