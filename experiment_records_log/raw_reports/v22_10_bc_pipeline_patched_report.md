# V22.10 BC Pipeline Report

```json
{
  "status": "PASS",
  "demo_path": "data/v22_10_teacher_demos_patched.pt",
  "checkpoint_path": "checkpoints/v22_10_bc_warmstart_patched.pt",
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
    "clear_teacher_success_rate": 0.955078125,
    "obstacle_residual_action_norm_mean": 1.4417485475540162,
    "clear_fraction": 0.7142857142857143
  },
  "mode_clear_count": 50176,
  "mode_obstacle_count": 19824,
  "final_bc_loss": 0.04948481832231794,
  "epochs_run": 7,
  "losses": [
    0.16998579459530966,
    0.10721341137375151,
    0.08112010168177741,
    0.06758538463285992,
    0.05869136612330164,
    0.05286936131971223,
    0.04948481832231794
  ],
  "frozen_std_parameters": [
    "distribution.log_std_param"
  ],
  "device": "cuda",
  "action_semantics": "residual_cartesian_acceleration_action_space"
}
```
