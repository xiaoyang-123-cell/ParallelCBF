# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_hybrid_tune_pursuit_teacher_dry_run_1k/2026-04-30_12-04-40`
Demo path: `data/v23_hybrid_tune_pursuit_teacher_dry_run_1k.pt`

```json
{
  "episodes_requested": 1000,
  "episodes_completed": 1000,
  "episodes_accepted": 682,
  "teacher_success_rate": 0.682,
  "total_timesteps": 140647,
  "accepted_total_timesteps": 83725,
  "waypoint_index_histogram": {
    "1": 91121,
    "2": 25476,
    "3": 13743,
    "4": 6354,
    "5": 3095,
    "6": 714,
    "7": 142,
    "8": 2
  },
  "accepted_waypoint_index_histogram": {
    "1": 56206,
    "2": 13903,
    "3": 7313,
    "4": 3836,
    "5": 1796,
    "6": 574,
    "7": 96,
    "8": 1
  },
  "waypoint_index_ge2_timestep_fraction": 0.352129800137934,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.3286831890116453,
  "success_by_scene": {
    "dynamic_obstacle": 0.31,
    "multi_obstacle_maze": 0.40454545454545454,
    "open": 1.0,
    "single_static": 0.7255813953488373
  },
  "episodes_by_scene": {
    "dynamic_obstacle": 100,
    "multi_obstacle_maze": 220,
    "open": 250,
    "single_static": 430
  },
  "path_len_mean_by_scene": {
    "dynamic_obstacle": 7.01,
    "multi_obstacle_maze": 7.413636363636364,
    "open": 2.0,
    "single_static": 6.888372093023256
  },
  "obstacle_scene_success_rate": 0.576,
  "wp_inside_cbf_zone_start_rate": 0.005333333333333333,
  "astar_no_path_rate": 0.037333333333333336,
  "astar_tier_used_distribution": {
    "0": 250,
    "1": 176,
    "2": 325,
    "3": 221,
    "5": 28
  },
  "cbf_activation_rate": 0.0016850697135381487,
  "action_delta_rate": 0.0025098295733289727,
  "mean_u_delta_norm_when_h_soft_active": 0.7787665260429121,
  "h_soft_active_timestep_count": 237,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 4971,
  "mean_lateral_overshoot": 0.1353250796970091,
  "cbf_conflict_discard_rate": 0.0,
  "pursuit_teacher_enabled": true,
  "astar_no_path_rate_by_scene": {
    "dynamic_obstacle": 0.08,
    "multi_obstacle_maze": 0.022727272727272728,
    "open": 0.0,
    "single_static": 0.03488372093023256
  },
  "wp_inside_cbf_zone_start_rate_by_scene": {
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.009302325581395349
  },
  "phase2_obstacle_scene_success_gate_30pct": true,
  "phase2_wp_inside_cbf_gate_2pct": true,
  "phase2_astar_no_path_gate_5pct": true,
  "teacher_success_gate_80pct": false,
  "waypoint_index_ge2_gate_30pct": true,
  "dry_run_status": "FAIL"
}
```
