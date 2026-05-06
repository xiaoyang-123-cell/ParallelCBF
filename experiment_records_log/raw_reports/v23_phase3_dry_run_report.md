# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_phase3_dry_run/2026-04-29_14-11-07`
Demo path: `data/v23_phase3_dry_run_1k.pt`

```json
{
  "episodes_requested": 1000,
  "episodes_completed": 1000,
  "episodes_accepted": 29,
  "teacher_success_rate": 0.029,
  "total_timesteps": 176992,
  "accepted_total_timesteps": 3183,
  "waypoint_index_histogram": {
    "1": 164902,
    "2": 11984,
    "3": 106
  },
  "accepted_waypoint_index_histogram": {
    "1": 3183
  },
  "waypoint_index_ge2_timestep_fraction": 0.06830817212077382,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.0,
  "success_by_scene": {
    "adversarial_start": 0.0,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.116,
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
    "adversarial_start": 2.64,
    "dynamic_obstacle": 6.28,
    "multi_obstacle_maze": 6.595,
    "open": 2.0,
    "single_static": 5.8825
  },
  "teacher_success_gate_80pct": false,
  "waypoint_index_ge2_gate_30pct": false,
  "dry_run_status": "FAIL"
}
```
