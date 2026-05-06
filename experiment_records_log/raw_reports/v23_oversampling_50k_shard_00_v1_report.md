# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_oversampling_50k_shards/shard_00_v1/2026-04-30_15-58-10`
Demo path: `data/v23_oversampling_50k_shards_20260430_155800/shard_00_v1.pt`

```json
{
  "episodes_requested": 10000,
  "episodes_completed": 10000,
  "episodes_accepted": 6030,
  "teacher_success_rate": 0.603,
  "total_timesteps": 1497692,
  "accepted_total_timesteps": 787062,
  "waypoint_index_histogram": {
    "1": 939162,
    "2": 282508,
    "3": 149367,
    "4": 78756,
    "5": 34142,
    "6": 9945,
    "7": 3121,
    "8": 637,
    "9": 54
  },
  "accepted_waypoint_index_histogram": {
    "1": 496091,
    "2": 141124,
    "3": 77460,
    "4": 42735,
    "5": 20403,
    "6": 6827,
    "7": 1957,
    "8": 422,
    "9": 43
  },
  "waypoint_index_ge2_timestep_fraction": 0.37292714389874554,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.36969260363224243,
  "success_by_scene": {
    "dynamic_obstacle": 0.3135,
    "multi_obstacle_maze": 0.40692307692307694,
    "open": 0.9838888888888889,
    "single_static": 0.715
  },
  "episodes_by_scene": {
    "dynamic_obstacle": 2000,
    "multi_obstacle_maze": 2600,
    "open": 1800,
    "single_static": 3600
  },
  "path_len_mean_by_scene": {
    "dynamic_obstacle": 6.996,
    "multi_obstacle_maze": 7.480384615384615,
    "open": 2.0,
    "single_static": 6.8436111111111115
  },
  "obstacle_scene_success_rate": 0.519390243902439,
  "wp_inside_cbf_zone_start_rate": 0.003292682926829268,
  "astar_no_path_rate": 0.04682926829268293,
  "astar_tier_used_distribution": {
    "0": 1800,
    "1": 1673,
    "2": 3645,
    "3": 2498,
    "5": 384
  },
  "cbf_activation_rate": 0.0018334877932178312,
  "action_delta_rate": 0.002702157720011858,
  "mean_u_delta_norm_when_h_soft_active": 0.8261249709375154,
  "h_soft_active_timestep_count": 2746,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 11821,
  "mean_lateral_overshoot": 0.13816472716654432,
  "cbf_conflict_discard_rate": 0.0,
  "pursuit_teacher_enabled": true,
  "attempt_mix": {
    "open": 0.18,
    "single_static": 0.36,
    "multi_obstacle": 0.26,
    "dynamic_obstacle": 0.2
  },
  "shard_id": 0,
  "shard_version": 1,
  "seed": 2303,
  "resource_start": {
    "label": "shard_00_start",
    "timestamp": "2026-04-30T15:58:12",
    "pid": 86224,
    "rss_mb": 6361.65625,
    "fd_count": 273,
    "gpu": {
      "available": true,
      "device": 0,
      "allocated_mb": 8.29541015625,
      "reserved_mb": 22.0,
      "free_mb": 18253.1875,
      "total_mb": 24101.75
    }
  },
  "astar_no_path_rate_by_scene": {
    "dynamic_obstacle": 0.095,
    "multi_obstacle_maze": 0.02423076923076923,
    "open": 0.0,
    "single_static": 0.03638888888888889
  },
  "wp_inside_cbf_zone_start_rate_by_scene": {
    "dynamic_obstacle": 0.0005,
    "multi_obstacle_maze": 0.0019230769230769232,
    "open": 0.0,
    "single_static": 0.005833333333333334
  },
  "phase2_obstacle_scene_success_gate_30pct": true,
  "phase2_wp_inside_cbf_gate_2pct": true,
  "phase2_astar_no_path_gate_5pct": true,
  "teacher_success_gate_80pct": false,
  "waypoint_index_ge2_gate_30pct": true,
  "dry_run_status": "FAIL",
  "resource_end": {
    "label": "shard_00_end",
    "timestamp": "2026-04-30T16:53:26",
    "pid": 86224,
    "rss_mb": 6767.40625,
    "fd_count": 273,
    "gpu": {
      "available": true,
      "device": 0,
      "allocated_mb": 9.06982421875,
      "reserved_mb": 24.0,
      "free_mb": 18186.9375,
      "total_mb": 24101.75
    }
  }
}
```
