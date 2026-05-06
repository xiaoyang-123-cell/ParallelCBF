# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_oversampling_50k_shards/shard_04_v1/2026-04-30_19-36-15`
Demo path: `data/v23_oversampling_50k_shards_20260430_155800/shard_04_v1.pt`

```json
{
  "episodes_requested": 10000,
  "episodes_completed": 10000,
  "episodes_accepted": 6339,
  "teacher_success_rate": 0.6339,
  "total_timesteps": 1484606,
  "accepted_total_timesteps": 829287,
  "waypoint_index_histogram": {
    "1": 884563,
    "2": 262246,
    "3": 161240,
    "4": 106307,
    "5": 49582,
    "6": 15153,
    "7": 4427,
    "8": 1037,
    "9": 51
  },
  "accepted_waypoint_index_histogram": {
    "1": 503560,
    "2": 140066,
    "3": 85109,
    "4": 56935,
    "5": 29777,
    "6": 10114,
    "7": 2873,
    "8": 817,
    "9": 36
  },
  "waypoint_index_ge2_timestep_fraction": 0.4041765963494692,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.3927795805312274,
  "success_by_scene": {
    "dynamic_obstacle": 0.3475,
    "multi_obstacle_maze": 0.4511538461538461,
    "open": 1.0,
    "single_static": 0.7419444444444444
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
  "obstacle_scene_success_rate": 0.5535365853658537,
  "wp_inside_cbf_zone_start_rate": 0.003292682926829268,
  "astar_no_path_rate": 0.04682926829268293,
  "astar_tier_used_distribution": {
    "0": 1800,
    "1": 1673,
    "2": 3645,
    "3": 2498,
    "5": 384
  },
  "cbf_activation_rate": 0.0025205340676246763,
  "action_delta_rate": 0.0025205340676246763,
  "mean_u_delta_norm_when_h_soft_active": 0.7085340314206785,
  "h_soft_active_timestep_count": 3742,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 14286,
  "mean_lateral_overshoot": 0.10333704278247197,
  "cbf_conflict_discard_rate": 0.0,
  "pursuit_teacher_enabled": true,
  "attempt_mix": {
    "open": 0.18,
    "single_static": 0.36,
    "multi_obstacle": 0.26,
    "dynamic_obstacle": 0.2
  },
  "shard_id": 4,
  "shard_version": 1,
  "seed": 4002315,
  "resource_start": {
    "label": "shard_04_start",
    "timestamp": "2026-04-30T19:36:18",
    "pid": 90024,
    "rss_mb": 6325.8203125,
    "fd_count": 272,
    "gpu": {
      "available": true,
      "device": 0,
      "allocated_mb": 8.29345703125,
      "reserved_mb": 22.0,
      "free_mb": 18064.5625,
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
    "label": "shard_04_end",
    "timestamp": "2026-04-30T20:33:00",
    "pid": 90024,
    "rss_mb": 6768.9453125,
    "fd_count": 272,
    "gpu": {
      "available": true,
      "device": 0,
      "allocated_mb": 9.06982421875,
      "reserved_mb": 24.0,
      "free_mb": 18005.9375,
      "total_mb": 24101.75
    }
  }
}
```
