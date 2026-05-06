# V22.10 BC Pipeline Report

```json
{
  "status": "PASS",
  "demo_path": "data/v22_10_teacher_demos.pt",
  "checkpoint_path": "checkpoints/v22_10_bc_warmstart.pt",
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
    "clear_teacher_success_rate": 0.0,
    "obstacle_residual_action_norm_mean": 1.4048929691314698,
    "clear_fraction": 0.7142857142857143
  },
  "mode_clear_count": 50176,
  "mode_obstacle_count": 19824,
  "final_bc_loss": 0.04990049611244883,
  "epochs_run": 11,
  "losses": [
    0.16623323730060033,
    0.1057747232062476,
    0.08087341955729893,
    0.06866409374134881,
    0.0617564357817173,
    0.05717473540987287,
    0.05382883282644408,
    0.052233708969184334,
    0.05102681219577789,
    0.05042637011834553,
    0.04990049611244883
  ],
  "frozen_std_parameters": [
    "distribution.log_std_param"
  ],
  "device": "cuda",
  "action_semantics": "residual_cartesian_acceleration_action_space"
}
```
