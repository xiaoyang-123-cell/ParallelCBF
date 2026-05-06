# V22.10 Recovery And Layer 2 Summary

Status: NVIDIA driver recovery succeeded. V22.10 BC warm-start succeeded. Layer 2 PPO sanity completed but failed the task-success gate.

## Runtime Recovery

The previous failure was caused by the host NVIDIA kernel module missing for the active kernel. After driver recovery, the system reported:

```text
GPU: NVIDIA GeForce RTX 3090
Driver Version: 580.142
CUDA Version: 13.0
/dev/nvidia* present
nvidia kernel modules loaded
```

The Isaac/PhysX startup blocker is resolved.

## Re-Run Target

Re-ran the previously blocked V22.10/V22.9 residual BC pipeline:

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

The script successfully entered Isaac, ran bandwidth sign-off, and collected demonstrations. It exited after actor/critic construction before running BC epochs, so BC was resumed from the saved demo dataset without re-running simulation.

## Phase -1 Bandwidth Sign-Off

| Metric | Value |
| --- | ---: |
| `tau90_ms` | `195.0 ms` |
| Max roll | `13.7798 deg` |
| Rotor saturation mean | `0.0` |
| Legacy `>30ms` abort | True |
| V22.10 override | Allowed because attitude lag is now explicitly in `D_t` |

## Demonstration Dataset

Artifact: `data/v22_10_teacher_demos.pt`

| Metric | Value |
| --- | ---: |
| Observations | `(70000, 35)` |
| Actions | `(70000, 3)` |
| Clear samples | `50176` |
| Obstacle samples | `19824` |
| Clear fraction | `71.43%` |
| Clear residual norm mean | `0.0` |
| Obstacle residual norm mean | `1.4115` |
| `clear_teacher_success_rate` | `0.0` |

Important caveat: the clear-path dataset contains the intended zero-residual labels, but the recorded clear teacher success metric stayed at `0.0`. This suggests the collection loop did not observe goal completion during the sampled clear windows, even though it still produced the planned zero-residual BC target distribution.

## Behavior Cloning

BC was resumed using `ParallelCBF_UAV/tools/v22_10_train_bc_from_demos.py`.

Artifact: `checkpoints/v22_10_bc_warmstart.pt`
Report: `v22_10_bc_pipeline_report.md`

| Metric | Value |
| --- | ---: |
| Final BC loss | `0.0499005` |
| Threshold | `< 0.05` |
| Epochs run | `11 / 40` |
| Frozen std params | `distribution.log_std_param` |
| Result | PASS |

## Layer 2 PPO Sanity

Command:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n parallel_uav \
  python ParallelCBF_UAV/train_ppo.py \
  --num_envs 4096 \
  --seed 42 \
  --max_iterations 100 \
  --cbf_enabled True \
  --bc_warmstart_path checkpoints/v22_10_bc_warmstart.pt \
  --v22_9_residual_overrides \
  --experiment_name uav_safe_3B1_v22_10_residual_sanity \
  --run_name layer2_100 \
  --headless
```

Run directory:

```text
logs/rsl_rl/uav_safe_3B1_v22_10_residual_sanity/2026-04-28_14-29-50_layer2_100
```

Training completed normally at iteration `99/100`.

## Layer 2 Final Metrics

| Metric | Final | Late-10 Mean | Gate |
| --- | ---: | ---: | --- |
| `task/goal_reached_rate` | `1.3969%` | `1.4058%` | FAIL, target `>80%` |
| `task/v_aligned_mean` | `0.1445` | `0.1441` | below target `>0.3` |
| `Policy/mean_std` | `0.3742` | n/a | PASS, target `0.25-0.45` |
| `Loss/clip_fraction` | `0.0215` | n/a | PASS, target `<0.3` |
| `Train/mean_episode_length` | `61.33` | `61.22` | slightly above `<60` target |
| `safety/collision_rate` | `0.0%` | `0.0%` | PASS |
| `safety/fallback_activation_rate` | `0.0%` | `0.0%` | PASS |
| `qp/hard_cap_binding_rate` | `0.0%` | `0.0%` | PASS |
| `qp/binding_box_rate` | `0.0%` | `0.0%` | PASS |

## Interpretation

The system is now technically runnable end-to-end again: driver, Isaac startup, demo collection, BC warm-start, and PPO sanity all execute.

The residual PPO gate did **not** pass. The failure mode is task-performance, not safety/QP feasibility:

- The safety layer is extremely clean in the empty-room Layer 2 phase.
- Policy exploration remains healthy and PPO updates are not clipping aggressively.
- The goal-reaching rate remains around `1.4%`, far below the required `>80%`.
- BC converged to the mixed target distribution, but the clear teacher success metric being `0.0` is a major warning that zero-residual nominal behavior still is not producing measured goal completions inside the data collection loop.

## Verdict

Do **not** auto-launch Layer 3. Layer 2 failed the task-success gate.

Recommended next action: audit the V22.9/V22.10 demo collection and goal-reaching measurement path before another PPO run. Specifically verify whether the clear-path zero-residual nominal rollout actually reaches the goal under the same reset/window logic used by `collect_demos()`, and whether `goal_reached_rate` is being computed on the intended termination/reward condition.
