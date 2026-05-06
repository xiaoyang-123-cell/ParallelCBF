# V23 Phase 3 1K Dry Run Report

TensorBoard log dir: `logs/v23_oversampling_50k_shards/shard_03_v1/2026-04-30_18-42-17`
Demo path: `data/v23_oversampling_50k_shards_20260430_155800/shard_03_v1.pt`

```json
{
  "episodes_requested": 10000,
  "episodes_completed": 10000,
  "episodes_accepted": 6348,
  "teacher_success_rate": 0.6348,
  "total_timesteps": 1483344,
  "accepted_total_timesteps": 829636,
  "waypoint_index_histogram": {
    "1": 883717,
    "2": 259759,
    "3": 159402,
    "4": 105743,
    "5": 51576,
    "6": 16603,
    "7": 4996,
    "8": 1480,
    "9": 68
  },
  "accepted_waypoint_index_histogram": {
    "1": 502045,
    "2": 137261,
    "3": 85508,
    "4": 57247,
    "5": 32014,
    "6": 11076,
    "7": 3388,
    "8": 1052,
    "9": 45
  },
  "waypoint_index_ge2_timestep_fraction": 0.4042400144538286,
  "accepted_waypoint_index_ge2_timestep_fraction": 0.39486111981640143,
  "success_by_scene": {
    "dynamic_obstacle": 0.344,
    "multi_obstacle_maze": 0.4523076923076923,
    "open": 1.0,
    "single_static": 0.7455555555555555
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
  "obstacle_scene_success_rate": 0.5546341463414635,
  "wp_inside_cbf_zone_start_rate": 0.003292682926829268,
  "astar_no_path_rate": 0.04682926829268293,
  "astar_tier_used_distribution": {
    "0": 1800,
    "1": 1673,
    "2": 3645,
    "3": 2498,
    "5": 384
  },
  "cbf_activation_rate": 4.8538976798369093e-05,
  "action_delta_rate": 0.0025105437444045346,
  "mean_u_delta_norm_when_h_soft_active": 0.5608681626359208,
  "h_soft_active_timestep_count": 72,
  "h_hard_violation_rate": 0.0,
  "h_hard_checked_barrier_count": 8443,
  "mean_lateral_overshoot": 0.10248584892576297,
  "cbf_conflict_discard_rate": 0.0,
  "pursuit_teacher_enabled": true,
  "attempt_mix": {
    "open": 0.18,
    "single_static": 0.36,
    "multi_obstacle": 0.26,
    "dynamic_obstacle": 0.2
  },
  "shard_id": 3,
  "shard_version": 1,
  "seed": 3002312,
  "resource_start": {
    "label": "shard_03_start",
    "timestamp": "2026-04-30T18:42:19",
    "pid": 89128,
    "rss_mb": 6342.46484375,
    "fd_count": 273,
    "gpu": {
      "available": true,
      "device": 0,
      "allocated_mb": 8.29345703125,
      "reserved_mb": 22.0,
      "free_mb": 18269.0625,
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
    "label": "shard_03_end",
    "timestamp": "2026-04-30T19:36:03",
    "pid": 89128,
    "rss_mb": 6781.30078125,
    "fd_count": 273,
    "gpu": {
      "available": true,
      "device": 0,
      "allocated_mb": 9.06982421875,
      "reserved_mb": 24.0,
      "free_mb": 18180.75,
      "total_mb": 24101.75
    }
  }
}
```
