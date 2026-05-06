# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_phase3_transition06_mini200/2026-04-29_15-12-43`
Demo path: `data/v23_phase3_transition06_mini200.pt`

```json
{
  "episodes_requested": 200,
  "episodes_completed": 200,
  "episodes_accepted": 7,
  "teacher_success_rate": 0.035,
  "total_timesteps": 35392,
  "accepted_total_timesteps": 845,
  "waypoint_index_histogram": {
    "1": 31231,
    "2": 4011,
    "3": 150
  },
  "accepted_waypoint_index_histogram": {
    "1": 845
  },
  "waypoint_index_ge2_timestep_fraction": 0.11756894213381555,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.0,
  "success_by_scene": {
    "adversarial_start": 0.0,
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.14,
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
    "adversarial_start": 2.4,
    "dynamic_obstacle": 6.3,
    "multi_obstacle_maze": 6.6,
    "open": 2.0,
    "single_static": 5.975
  },
  "teacher_success_gate_80pct": true,
  "waypoint_index_ge2_gate_30pct": false,
  "dry_run_status": "FAIL"
}
```
