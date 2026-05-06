# V22.10d Reward Audit

```json
{
  "status": "FAIL",
  "goal_radius": 0.4336666763,
  "num_envs": 256,
  "steps": 160,
  "bc": {
    "kind": "bc",
    "checkpoint": "checkpoints/v22_10c_bc_warmstart.pt",
    "reward_mean": 73.40008544921875,
    "reward_std": 3.324979543685913,
    "success_rate": 1.0,
    "min_goal_dist_mean": 0.43460896611213684,
    "final_goal_dist_mean": 0.4696773290634155,
    "first_done_step_mean": 46.1875,
    "action_norm_mean": 0.05705304443836212
  },
  "failed_policy": {
    "kind": "failed_v22_10c_model_99",
    "checkpoint": "logs/rsl_rl/uav_safe_3B1_v22_10d_bc_anchor_sanity/2026-04-28_17-17-19_layer2_100/model_99.pt",
    "reward_mean": 73.43363952636719,
    "reward_std": 3.2184579372406006,
    "success_rate": 1.0,
    "min_goal_dist_mean": 0.43440771102905273,
    "final_goal_dist_mean": 0.4700840711593628,
    "first_done_step_mean": 46.87890625,
    "action_norm_mean": 0.05323443561792374
  },
  "bc_reward_higher": false,
  "reward_delta_bc_minus_failed": -0.0335540771484375
}
```
