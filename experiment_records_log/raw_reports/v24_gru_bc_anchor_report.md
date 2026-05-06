# V24 GRU BC Anchor Report

```json
{
  "status": "FAIL",
  "demo_path": "data/v23_oversampling_50k_bc.pt",
  "checkpoint_path": "checkpoints/v24_gru_bc_anchor.pt",
  "episodes": 31415,
  "timesteps": 4109264,
  "final_bc_loss": 0.059459819820711426,
  "epochs_run": 30,
  "losses": [
    0.36826756730099075,
    0.19919441959037315,
    0.16363251606825654,
    0.1302697430165625,
    0.10978764412420101,
    0.09961764036455854,
    0.09346117678642758,
    0.08900012003064399,
    0.08557623327385384,
    0.08256967701093729,
    0.080090164903043,
    0.0778635522377224,
    0.07586367360777137,
    0.07394182059132397,
    0.07242199168550022,
    0.07092447603944606,
    0.06976748254081867,
    0.06833002973507232,
    0.06740129758772685,
    0.06634516058045832,
    0.06552293562840542,
    0.064550829827482,
    0.06367922294315884,
    0.06295124125759859,
    0.06235463477959458,
    0.061662200868676006,
    0.06093950697852492,
    0.060569337241032944,
    0.059866480961775825,
    0.059459819820711426
  ],
  "frozen_std_parameters": [
    "distribution.log_std_param"
  ],
  "device": "cuda",
  "step_1_0_dataset_audit": {
    "status": "PASS",
    "episodes": 31415,
    "timesteps": 4109264,
    "zero_action_fraction": 0.0,
    "zero_action_threshold": 0.0001,
    "zero_action_max_fraction": 0.4,
    "shape_error_count": 0,
    "nan_episode_count": 0,
    "bptt_sample_requested": 100,
    "bptt_sample_checked": 100,
    "bptt_max_abs_diff": 8.940696716308594e-08,
    "bptt_max_abs_tolerance": 1e-05,
    "attempt_distribution": {
      "open": 0.18,
      "single_static": 0.36,
      "multi_obstacle": 0.26,
      "dynamic_obstacle": 0.2
    },
    "attempt_target_mix": {
      "open": 0.18,
      "single_static": 0.36,
      "multi_obstacle": 0.26,
      "dynamic_obstacle": 0.2
    },
    "attempt_mix_max_abs_error": 0.0,
    "attempt_mix_tolerance": 0.025,
    "per_shard_attempt_mix_tolerance": 0.02,
    "per_shard_attempt_audit": [],
    "accepted_distribution": {
      "open": 0.2855642209135763,
      "single_static": 0.4214547190832405,
      "multi_obstacle": 0.18449785134489893,
      "dynamic_obstacle": 0.10848320865828426
    },
    "accepted_target_mix": {
      "open": 0.29654036243822074,
      "single_static": 0.42998352553542013,
      "multi_obstacle": 0.171334431630972,
      "dynamic_obstacle": 0.10214168039538715
    },
    "accepted_mix_max_abs_error": 0.013163419713926927,
    "accepted_mix_tolerance": 0.08,
    "prediction_match": {
      "zero_action_fraction_le_40pct": true,
      "bptt_hidden_continuity": true,
      "attempt_mix_matches": true,
      "accepted_mix_matches": true,
      "shape_and_nan_clean": true
    }
  },
  "demo_metadata": {
    "source": "v23_sharded_bc_rescue",
    "num_shards": 5,
    "episodes_accepted": 31415,
    "episodes_requested": 50000,
    "episodes_completed": 50000,
    "attempt_mix": {
      "open": 0.18,
      "single_static": 0.36,
      "multi_obstacle": 0.26,
      "dynamic_obstacle": 0.2
    },
    "episodes_by_scene": {
      "dynamic_obstacle": 10000,
      "multi_obstacle": 13000,
      "open": 9000,
      "single_static": 18000
    },
    "attempt_distribution": {
      "open": 0.18,
      "single_static": 0.36,
      "multi_obstacle": 0.26,
      "dynamic_obstacle": 0.2
    },
    "shard_paths": [
      "/data/uav_project/ParallelCBF/data/v23_oversampling_50k_shards_20260430_155800/shard_00_v1.pt",
      "/data/uav_project/ParallelCBF/data/v23_oversampling_50k_shards_20260430_155800/shard_01_v1.pt",
      "/data/uav_project/ParallelCBF/data/v23_oversampling_50k_shards_20260430_155800/shard_02_v1.pt",
      "/data/uav_project/ParallelCBF/data/v23_oversampling_50k_shards_20260430_155800/shard_03_v1.pt",
      "/data/uav_project/ParallelCBF/data/v23_oversampling_50k_shards_20260430_155800/shard_04_v1.pt"
    ],
    "shards": [
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
      },
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
      },
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
      },
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
      },
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
    ]
  },
  "architecture": "RNNModel(GRU hidden=128, mlp=[64], action_dim=3)"
}
```
