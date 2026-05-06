# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_hybrid_tune_mid_validation_200/2026-04-30_12-03-00`
Demo path: `data/v23_hybrid_tune_mid_validation_200.pt`

```json
{
  "episodes_requested": 200,
  "episodes_completed": 200,
  "episodes_accepted": 146,
  "teacher_success_rate": 0.73,
  "total_timesteps": 27695,
  "accepted_total_timesteps": 18029,
  "waypoint_index_histogram": {
    "1": 16586,
    "2": 5495,
    "3": 3313,
    "4": 1507,
    "5": 572,
    "6": 175,
    "7": 39,
    "8": 8
  },
  "accepted_waypoint_index_histogram": {
    "1": 11131,
    "2": 3567,
    "3": 1949,
    "4": 1004,
    "5": 247,
    "6": 106,
    "7": 24,
    "8": 1
  },
  "waypoint_index_ge2_timestep_fraction": 0.40111933562014807,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.38260580176382497,
  "success_by_scene": {
    "dynamic_obstacle": 0.45,
    "multi_obstacle_maze": 0.45454545454545453,
    "open": 1.0,
    "single_static": 0.7790697674418605
  },
  "episodes_by_scene": {
    "dynamic_obstacle": 20,
    "multi_obstacle_maze": 44,
    "open": 50,
    "single_static": 86
  },
  "path_len_mean_by_scene": {
    "dynamic_obstacle": 6.05,
    "multi_obstacle_maze": 7.5227272727272725,
    "open": 2.0,
    "single_static": 6.755813953488372
  },
  "obstacle_scene_success_rate": 0.64,
  "wp_inside_cbf_zone_start_rate": 0.006666666666666667,
  "astar_no_path_rate": 0.04666666666666667,
  "astar_tier_used_distribution": {
    "0": 50,
    "1": 35,
    "2": 70,
    "3": 38,
    "5": 7
  },
  "cbf_activation_rate": 0.002527532045495577,
  "action_delta_rate": 0.0031774688571944393,
  "mean_u_delta_norm_when_h_soft_active": 0.5956420028050031,
  "h_soft_active_timestep_count": 70,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 4553,
  "mean_lateral_overshoot": 0.1122741058769031,
  "cbf_conflict_discard_rate": 0.0,
  "pursuit_teacher_enabled": true,
  "astar_no_path_rate_by_scene": {
    "dynamic_obstacle": 0.2,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.03488372093023256
  },
  "wp_inside_cbf_zone_start_rate_by_scene": {
    "dynamic_obstacle": 0.0,
    "multi_obstacle_maze": 0.0,
    "open": 0.0,
    "single_static": 0.011627906976744186
  },
  "phase2_obstacle_scene_success_gate_30pct": true,
  "phase2_wp_inside_cbf_gate_2pct": true,
  "phase2_astar_no_path_gate_5pct": true,
  "teacher_success_gate_80pct": true,
  "waypoint_index_ge2_gate_30pct": true,
  "dry_run_status": "PASS"
}
```
