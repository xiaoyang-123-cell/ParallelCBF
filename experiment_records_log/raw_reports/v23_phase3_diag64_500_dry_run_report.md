# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_phase3_diag64_500_dry_run/2026-04-29_13-59-56`
Demo path: `data/v23_phase3_diag64_500_dry_run.pt`

```json
{
  "episodes_requested": 64,
  "episodes_completed": 64,
  "episodes_accepted": 19,
  "teacher_success_rate": 0.296875,
  "total_timesteps": 28353,
  "accepted_total_timesteps": 6126,
  "waypoint_index_histogram": {
    "1": 26099,
    "2": 2254
  },
  "accepted_waypoint_index_histogram": {
    "1": 6126
  },
  "waypoint_index_ge2_timestep_fraction": 0.0794977603780905,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.0,
  "success_by_scene": {
    "adversarial_start": 1.0,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 1.0,
    "single_static": 0.0
  },
  "episodes_by_scene": {
    "adversarial_start": 3,
    "dynamic_obstacle": 6,
    "multi_obstacle_maze": 13,
    "open": 16,
    "single_static": 26
  },
  "path_len_mean_by_scene": {
    "adversarial_start": 2.0,
    "dynamic_obstacle": 6.166666666666667,
    "multi_obstacle_maze": 6.923076923076923,
    "open": 2.0,
    "single_static": 5.923076923076923
  }
}
```
