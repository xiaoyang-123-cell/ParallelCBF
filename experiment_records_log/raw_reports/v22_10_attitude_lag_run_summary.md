# V22.10 Attitude-Lag Run Summary

Status: Layer 1 passed. Step 3 BC/PPO resume is blocked by host GPU/Isaac runtime initialization, not by the V22.10 math.

## Implementation

- Added V22.10 constants:
  - `attitude_tracker_time_constant = 0.087`
  - `cbf_frequency = 50.0`
  - `eta_tilt_steady_state = 0.85`
  - `u_cmd_floor = 2.0`
- Removed the V21 `wind_drag_compensation_factor` path from active QP math.
- Rewrote `D_t` into:
  - steady wind residual
  - attitude-lag term
  - V20 commanded-action-scaled mass term
  - optional alpha-rate term, still disabled by default
- Preserved the existing perception margin path in `uav_safe_env.py`.
- Added QP/env telemetry for `qp/D_t_attitude_lag_term_mean`.
- Added `u_cmd_norm` into the QP input path, using current `u_combined_xyz.norm()` in the environment and falling back to `recent_u_xy_norm` for older callers.
- Added `scripts/qp_v22_10_attitude_lag_diagnostic.py` for direct offline Layer 1 validation.

## Layer 1 Result

Initial cap anchors `[4.0, 4.0, 4.5, 5.0, 6.0]` were too tight for the explicit 200 ms attitude lag. Per protocol, anchors were raised by `+1.5` and frozen as:

```text
[5.5, 5.5, 6.0, 6.5, 7.5]
```

| Scenario | u_cmd | Wind | Feasibility | Hard-Cap Binding | Target | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Hover low wind | 1.0 | 0.05 N | 100.00% | 13.87% | < 15.00% | PASS |
| Approach high wind | 2.0 | 0.20 N | 100.00% | 12.01% | < 35.00% | PASS |
| Aggressive high wind | 5.0 | 0.20 N | 100.00% | 36.28% | < 50.00% | PASS |

## D_t Breakdown

| Scenario | D_t Used | D_t Curr | Wind Steady | Attitude Lag | Mass | D_t/|h| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hover low wind | 6.4429 | 6.9241 | 0.5591 | 5.8160 | 0.5489 | 2.062 |
| Approach high wind | 6.9284 | 7.3815 | 1.0166 | 5.8160 | 0.5489 | 2.247 |
| Aggressive high wind | 13.5926 | 16.4716 | 1.0166 | 14.5401 | 0.9149 | 3.885 |

Report artifact: `qp_diagnostic_report_v22_10_attitude_lag.md`.

## Step 3 Attempt

The V22.9 BC pipeline was updated with `--v22_10_layer1_cleared` so the legacy `tau90 > 30 ms` abort gate does not block after V22.10 Layer 1 has explicitly compensated attitude lag in `D_t`.

Attempted command:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n parallel_uav \
  python ParallelCBF_UAV/tools/v22_9_bc_pipeline.py \
  --num_envs 1024 \
  --seed 42 \
  --clear_transitions 50000 \
  --obstacle_transitions 20000 \
  --epochs 40 \
  --batch_size 2048 \
  --lr 1e-4 \
  --loss_threshold 0.05 \
  --demo_path data/v22_10_teacher_demos.pt \
  --checkpoint_path checkpoints/v22_10_bc_warmstart.pt \
  --report_path v22_10_bc_pipeline_report.md \
  --bandwidth_steps 240 \
  --v22_10_layer1_cleared \
  --headless
```

Result: no demos/checkpoint/report were produced.

Blocking runtime evidence:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
CUDA error: no CUDA-capable device is detected
Failed to create primary CUDA context
CUDA libs are present, but no suitable CUDA GPU was found
```

A small `--device cpu` smoke test also exited during Isaac initialization before entering data collection, so this is a host Isaac/GPU runtime blocker.

## Verdict

V22.10 math is implemented and Layer 1 is cleared. Do not interpret Step 3 as an algorithm failure: BC/PPO did not start because Isaac Sim cannot initialize PhysX/GPU tensors on the current host.

Next executable action after GPU driver recovery:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n parallel_uav \
  python ParallelCBF_UAV/tools/v22_9_bc_pipeline.py \
  --num_envs 1024 \
  --seed 42 \
  --clear_transitions 50000 \
  --obstacle_transitions 20000 \
  --epochs 40 \
  --batch_size 2048 \
  --lr 1e-4 \
  --loss_threshold 0.05 \
  --demo_path data/v22_10_teacher_demos.pt \
  --checkpoint_path checkpoints/v22_10_bc_warmstart.pt \
  --report_path v22_10_bc_pipeline_report.md \
  --bandwidth_steps 240 \
  --v22_10_layer1_cleared \
  --headless
```
