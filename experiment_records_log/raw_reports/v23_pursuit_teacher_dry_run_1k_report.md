# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_pursuit_teacher_dry_run_1k/2026-04-29_17-37-29`
Demo path: `data/v23_pursuit_teacher_dry_run_1k.pt`

```json
{
  "episodes_requested": 1000,
  "episodes_completed": 1000,
  "episodes_accepted": 0,
  "teacher_success_rate": 0.0,
  "total_timesteps": 173914,
  "accepted_total_timesteps": 0,
  "waypoint_index_histogram": {
    "1": 131749,
    "2": 35731,
    "3": 6229,
    "4": 205
  },
  "accepted_waypoint_index_histogram": {},
  "waypoint_index_ge2_timestep_fraction": 0.24244741653920904,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.0,
  "success_by_scene": {
    "adversarial_start": 0.0,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.152,
    "single_static": 0.0
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
  "obstacle_scene_success_rate": 0.0,
  "wp_inside_cbf_zone_start_rate": 0.008,
  "astar_no_path_rate": 0.108,
  "astar_tier_used_distribution": {
    "0": 250,
    "1": 162,
    "2": 304,
    "4": 203,
    "5": 81
  },
  "cbf_activation_rate": 0.9996090021504882,
  "mean_lateral_overshoot": 0.1944491251793912,
  "cbf_conflict_discard_rate": 0.038,
  "pursuit_teacher_enabled": true,
  "astar_no_path_rate_by_scene": {
    "adversarial_start": 1.0,
    "dynamic_obstacle": 0.12,
    "multi_obstacle_maze": 0.025,
    "open": 0.0,
    "single_static": 0.035
  },
  "wp_inside_cbf_zone_start_rate_by_scene": {
    "adversarial_start": 0.04,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.01
  },
  "phase2_obstacle_scene_success_gate_30pct": false,
  "phase2_wp_inside_cbf_gate_2pct": true,
  "phase2_astar_no_path_gate_5pct": false,
  "teacher_success_gate_80pct": false,
  "waypoint_index_ge2_gate_30pct": false,
  "dry_run_status": "FAIL"
}
```
