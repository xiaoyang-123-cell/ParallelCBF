# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_phase3_astar_cbf_fix_mini192/2026-04-29_16-18-28`
Demo path: `data/v23_phase3_astar_cbf_fix_mini192.pt`

```json
{
  "episodes_requested": 192,
  "episodes_completed": 192,
  "episodes_accepted": 8,
  "teacher_success_rate": 0.041666666666666664,
  "total_timesteps": 33953,
  "accepted_total_timesteps": 1017,
  "waypoint_index_histogram": {
    "1": 29634,
    "2": 4187,
    "3": 115,
    "4": 17
  },
  "accepted_waypoint_index_histogram": {
    "1": 1017
  },
  "waypoint_index_ge2_timestep_fraction": 0.1272052543221512,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.0,
  "success_by_scene": {
    "adversarial_start": 0.0,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.16666666666666666,
    "single_static": 0.0
  },
  "episodes_by_scene": {
    "adversarial_start": 10,
    "dynamic_obstacle": 19,
    "multi_obstacle_maze": 38,
    "open": 48,
    "single_static": 77
  },
  "path_len_mean_by_scene": {
    "adversarial_start": 2.0,
    "dynamic_obstacle": 6.2631578947368425,
    "multi_obstacle_maze": 7.842105263157895,
    "open": 2.0,
    "single_static": 6.740259740259741
  },
  "obstacle_scene_success_rate": 0.0,
  "wp_inside_cbf_zone_start_rate": 0.020833333333333332,
  "astar_no_path_rate": 0.09722222222222222,
  "astar_tier_used_distribution": {
    "0": 48,
    "1": 29,
    "2": 61,
    "4": 40,
    "5": 14
  },
  "astar_no_path_rate_by_scene": {
    "adversarial_start": 1.0,
    "dynamic_obstacle": 0.10526315789473684,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.025974025974025976
  },
  "wp_inside_cbf_zone_start_rate_by_scene": {
    "adversarial_start": 0.1,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.025974025974025976
  },
  "phase2_obstacle_scene_success_gate_30pct": false,
  "phase2_wp_inside_cbf_gate_2pct": false,
  "phase2_astar_no_path_gate_5pct": false,
  "teacher_success_gate_80pct": false,
  "waypoint_index_ge2_gate_30pct": false,
  "dry_run_status": "FAIL"
}
```
