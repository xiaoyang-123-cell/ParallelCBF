# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_phase3_diag32_v2_dry_run/2026-04-29_13-58-49`
Demo path: `data/v23_phase3_diag32_v2_dry_run.pt`

```json
{
  "episodes_requested": 32,
  "episodes_completed": 32,
  "episodes_accepted": 2,
  "teacher_success_rate": 0.0625,
  "total_timesteps": 3653,
  "accepted_total_timesteps": 83,
  "waypoint_index_histogram": {
    "1": 3336,
    "2": 317
  },
  "accepted_waypoint_index_histogram": {
    "1": 83
  },
  "waypoint_index_ge2_timestep_fraction": 0.08677799069258144,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.0,
  "success_by_scene": {
    "adversarial_start": 0.0,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.25,
    "single_static": 0.0
  },
  "episodes_by_scene": {
    "adversarial_start": 2,
    "dynamic_obstacle": 3,
    "multi_obstacle_maze": 6,
    "open": 8,
    "single_static": 13
  },
  "path_len_mean_by_scene": {
    "adversarial_start": 3.0,
    "dynamic_obstacle": 6.333333333333333,
    "multi_obstacle_maze": 6.5,
    "open": 2.0,
    "single_static": 6.384615384615385
  }
}
```
