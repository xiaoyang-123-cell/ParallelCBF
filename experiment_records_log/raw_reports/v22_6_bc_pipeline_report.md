# V22.6 Behavior Cloning Pipeline Report

Status: aborted at Phase 0 teacher sanity. No demos, BC checkpoint, PPO sanity run, or full run were launched.

## Implemented Infrastructure

- Added `ParallelCBF_UAV/tools/v22_6_bc_pipeline.py`.
- Added `ScriptedGoalTeacher` for the current executable CBF action interface.
- Added actor-only BC checkpoint loading through `--bc_warmstart_path` in `ParallelCBF_UAV/train_ppo.py`.
- Added `PPOWithClipFraction` so TensorBoard logs `Loss/clip_fraction` for the V22.6 PPO gate.

## Critical Interface Finding

The current CBF PPO actor does not output translational acceleration. The environment reports:

| Field | Value |
| --- | ---: |
| `cbf_enabled` | `True` |
| `num_actions` | `2` |
| `num_observations` | `35` |
| `expected_action_dim` | `2` |

The two actor actions are interpreted as `[alpha_dot, gamma_dot]` shield-parameter rates. The final physical acceleration `u_qp` is produced inside the environment by the nominal controller plus the CBF-QP. Therefore, a P-PD acceleration teacher cannot be directly cloned into the current PPO actor without changing the action interface.

## Phase 0 Teacher Sanity

Command:

```bash
PYTHONUNBUFFERED=1 conda run -n parallel_uav python ParallelCBF_UAV/tools/v22_6_bc_pipeline.py \
  --num_envs 1024 \
  --seed 42 \
  --transitions 50000 \
  --epochs 80 \
  --batch_size 2048 \
  --teacher_steps 125 \
  --headless
```

Result:

| Metric | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Teacher success in empty room at `d=0.5m` | `> 80%` | `0.0%` | FAIL |
| Teacher `v_aligned_mean` | positive | `-0.124130` | FAIL |
| Last logged goal reached rate | diagnostic | `0.5859%` | OUT |

## Protocol Decision

The BC pipeline was stopped before data collection. `data/teacher_demos.pt` and `checkpoints/bc_warmstart.pt` were not created.

This is the correct abort because cloning a teacher that cannot control the current 2D shield-action interface would produce a misleading warm start. The likely next architectural step is to expose a goal-directed action interface to PPO, such as residual acceleration around the nominal controller, or to run BC on a separate deployment actor that directly maps observations to `u_ref/u_cmd` before the CBF-QP.
