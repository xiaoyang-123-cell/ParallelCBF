# V22.10 BC Pipeline Report

```json
{
  "status": "PASS",
  "demo_path": "data/v22_10c_teacher_demos.pt",
  "checkpoint_path": "checkpoints/v22_10c_bc_warmstart.pt",
  "num_samples": 70000,
  "obs_shape": [
    70000,
    35
  ],
  "action_shape": [
    70000,
    3
  ],
  "demo_metadata": {
    "clear_teacher_success_rate": 1.0,
    "clear_policy_success_rate": 0.16015625,
    "teacher_push_through_fraction": 0.203125,
    "teacher_push_through_strict_success_rate": 0.7884615659713745,
    "teacher_relaxed_only_success_rate": 1.0,
    "obstacle_residual_action_norm_mean": 1.2559013104514232,
    "clear_fraction": 0.7142857142857143
  },
  "mode_clear_count": 50176,
  "mode_obstacle_count": 19824,
  "final_bc_loss": 0.0493038321180003,
  "epochs_run": 14,
  "losses": [
    0.1534472557050841,
    0.10778342762163708,
    0.08995569114174162,
    0.07851460916655405,
    0.07029115600245339,
    0.06432091476661818,
    0.05987215084689004,
    0.056942143397671834,
    0.05454252777355058,
    0.053071507279361996,
    0.05175341303859438,
    0.05101499493632998,
    0.05011454469391278,
    0.0493038321180003
  ],
  "frozen_std_parameters": [
    "distribution.log_std_param"
  ],
  "device": "cuda",
  "action_semantics": "residual_cartesian_acceleration_action_space"
}
```
