# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_oversampling_50k_shards/shard_02_v1/2026-04-30_17-48-21`
Demo path: `data/v23_oversampling_50k_shards_20260430_155800/shard_02_v1.pt`

```json
{
  "episodes_requested": 10000,
  "episodes_completed": 10000,
  "episodes_accepted": 6364,
  "teacher_success_rate": 0.6364,
  "total_timesteps": 1484651,
  "accepted_total_timesteps": 833807,
  "waypoint_index_histogram": {
    "1": 885054,
    "2": 254784,
    "3": 158991,
    "4": 109856,
    "5": 53403,
    "6": 16526,
    "7": 4829,
    "8": 1136,
    "9": 65,
    "10": 7
  },
  "accepted_waypoint_index_histogram": {
    "1": 509241,
    "2": 136079,
    "3": 84141,
    "4": 57998,
    "5": 31468,
    "6": 10902,
    "7": 3082,
    "8": 832,
    "9": 57,
    "10": 7
  },
  "waypoint_index_ge2_timestep_fraction": 0.40386393839360224,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.3892579457836166,
  "success_by_scene": {
    "dynamic_obstacle": 0.3535,
    "multi_obstacle_maze": 0.4626923076923077,
    "open": 1.0,
    "single_static": 0.7372222222222222
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
  "obstacle_scene_success_rate": 0.5565853658536586,
  "wp_inside_cbf_zone_start_rate": 0.003292682926829268,
  "astar_no_path_rate": 0.04682926829268293,
  "astar_tier_used_distribution": {
    "0": 1800,
    "1": 1673,
    "2": 3645,
    "3": 2498,
    "5": 384
  },
  "cbf_activation_rate": 5.253759974566413e-05,
  "action_delta_rate": 0.0025015980186589306,
  "mean_u_delta_norm_when_h_soft_active": 0.45768832406984306,
  "h_soft_active_timestep_count": 78,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 8462,
  "mean_lateral_overshoot": 0.09633155808101335,
  "cbf_conflict_discard_rate": 0.0,
  "pursuit_teacher_enabled": true,
  "attempt_mix": {
    "open": 0.18,
    "single_static": 0.36,
    "multi_obstacle": 0.26,
    "dynamic_obstacle": 0.2
  },
  "shard_id": 2,
  "shard_version": 1,
  "seed": 2002309,
  "resource_start": {
    "label": "shard_02_start",
    "timestamp": "2026-04-30T17:48:23",
    "pid": 88506,
    "rss_mb": 6325.984375,
    "fd_count": 272,
    "gpu": {
      "available": true,
      "device": 0,
      "allocated_mb": 8.29345703125,
      "reserved_mb": 22.0,
      "free_mb": 18240.5625,
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
    "label": "shard_02_end",
    "timestamp": "2026-04-30T18:42:04",
    "pid": 88506,
    "rss_mb": 6772.66015625,
    "fd_count": 272,
    "gpu": {
      "available": true,
      "device": 0,
      "allocated_mb": 9.06982421875,
      "reserved_mb": 24.0,
      "free_mb": 18233.0625,
      "total_mb": 24101.75
    }
  }
}
```
