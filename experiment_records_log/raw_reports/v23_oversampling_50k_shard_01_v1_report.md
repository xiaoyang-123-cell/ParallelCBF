# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_oversampling_50k_shards/shard_01_v1/2026-04-30_16-53-38`
Demo path: `data/v23_oversampling_50k_shards_20260430_155800/shard_01_v1.pt`

```json
{
  "episodes_requested": 10000,
  "episodes_completed": 10000,
  "episodes_accepted": 6334,
  "teacher_success_rate": 0.6334,
  "total_timesteps": 1485686,
  "accepted_total_timesteps": 829472,
  "waypoint_index_histogram": {
    "1": 895516,
    "2": 258731,
    "3": 159411,
    "4": 105746,
    "5": 47279,
    "6": 14098,
    "7": 3888,
    "8": 1004,
    "9": 13
  },
  "accepted_waypoint_index_histogram": {
    "1": 510787,
    "2": 136368,
    "3": 85294,
    "4": 56144,
    "5": 28112,
    "6": 9547,
    "7": 2443,
    "8": 764,
    "9": 13
  },
  "waypoint_index_ge2_timestep_fraction": 0.39723737048070723,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.384202239496933,
  "success_by_scene": {
    "dynamic_obstacle": 0.3455,
    "multi_obstacle_maze": 0.45615384615384613,
    "open": 1.0,
    "single_static": 0.7380555555555556
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
  "obstacle_scene_success_rate": 0.5529268292682927,
  "wp_inside_cbf_zone_start_rate": 0.003292682926829268,
  "astar_no_path_rate": 0.04682926829268293,
  "astar_tier_used_distribution": {
    "0": 1800,
    "1": 1673,
    "2": 3645,
    "3": 2498,
    "5": 384
  },
  "cbf_activation_rate": 0.0016517622162421938,
  "action_delta_rate": 0.002519374888098831,
  "mean_u_delta_norm_when_h_soft_active": 0.6966793804953957,
  "h_soft_active_timestep_count": 2454,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 11224,
  "mean_lateral_overshoot": 0.10486351662254359,
  "cbf_conflict_discard_rate": 0.0,
  "pursuit_teacher_enabled": true,
  "attempt_mix": {
    "open": 0.18,
    "single_static": 0.36,
    "multi_obstacle": 0.26,
    "dynamic_obstacle": 0.2
  },
  "shard_id": 1,
  "shard_version": 1,
  "seed": 1002306,
  "resource_start": {
    "label": "shard_01_start",
    "timestamp": "2026-04-30T16:53:40",
    "pid": 87348,
    "rss_mb": 6354.99609375,
    "fd_count": 272,
    "gpu": {
      "available": true,
      "device": 0,
      "allocated_mb": 8.29345703125,
      "reserved_mb": 22.0,
      "free_mb": 18197.5,
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
    "label": "shard_01_end",
    "timestamp": "2026-04-30T17:48:09",
    "pid": 87348,
    "rss_mb": 6753.05859375,
    "fd_count": 272,
    "gpu": {
      "available": true,
      "device": 0,
      "allocated_mb": 9.06982421875,
      "reserved_mb": 24.0,
      "free_mb": 18265.375,
      "total_mb": 24101.75
    }
  }
}
```
