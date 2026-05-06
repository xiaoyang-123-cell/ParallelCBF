# V22.10d Reward Audit

```json
{
  "status": "PASS",
  "goal_radius": 0.4336666763,
  "num_envs": 64,
  "steps": 160,
  "bc": {
    "kind": "bc",
    "checkpoint": "checkpoints/v22_10c_bc_warmstart.pt",
    "reward_mean": 73.2781982421875,
    "reward_std": 3.0054030418395996,
    "success_rate": 1.0,
    "min_goal_dist_mean": 0.4344334602355957,
    "final_goal_dist_mean": 0.46795982122421265,
    "first_done_step_mean": 50.53125,
    "action_norm_mean": 0.05546855926513672
  },
  "failed_policy": {
    "kind": "failed_v22_10c_model_99",
    "checkpoint": "logs/rsl_rl/uav_safe_3B1_v22_10c_curriculum_sanity/2026-04-28_16-15-35_layer2_100/model_99.pt",
    "reward_mean": 62.22265625,
    "reward_std": 4.230187892913818,
    "success_rate": 0.96875,
    "min_goal_dist_mean": 0.43419212102890015,
    "final_goal_dist_mean": 0.46753644943237305,
    "first_done_step_mean": 78.90625,
    "action_norm_mean": 0.8472893238067627
  },
  "bc_reward_higher": true,
  "reward_delta_bc_minus_failed": 11.0555419921875
}
```
