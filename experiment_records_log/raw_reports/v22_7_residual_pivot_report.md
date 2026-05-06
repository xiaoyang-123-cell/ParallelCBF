# V22.7 Residual Cartesian RL Pivot Report

Status: aborted before BC/PPO launch because the zero-residual nominal teacher failed Phase 0.

## Step 1 Audit Result

Current pre-V22.7 action semantics were **Interpretation A**.

Evidence:

- `cbf_action_space` was previously `2`.
- In `_pre_physics_step`, `actions[:, 0]` and `actions[:, 1]` were assigned to `alpha_dot` and `gamma_dot`.
- These rates updated `alpha` and `gamma`, then entered `dual_barrier_qp.py` as `rl_action_rates`.
- The physical acceleration reference came from the environment's nominal controller, not from PPO.

Conclusion: the old PPO actor tuned CBF parameters. It was not a polar or Cartesian acceleration actor.

## Implemented V22.7 Refactor

- Changed CBF PPO action space to 3D residual Cartesian acceleration.
- Added residual scaling:
  - XY residual scale: `3.0 m/s^2`
  - Z residual scale: `2.0 m/s^2`
- Added residual blend curriculum:
  - `lambda = 0.3` at iter 0
  - linear ramp to `1.0` by iter 800
- Froze PPO control over `alpha/gamma`; QP receives zero parameter rates.
- Replaced QP `u_ref` with:

```text
u_combined = clip(u_nominal + lambda * u_residual)
```

- Added telemetry:
  - `task/residual_lambda_current`
  - `action/residual_norm_mean`
  - `action/nominal_accel_norm_mean`
  - `action/combined_accel_norm_mean`
- Updated BC pipeline so the clear-path teacher target is zero residual.

## Phase 0 Teacher Gate

Command:

```bash
PYTHONUNBUFFERED=1 timeout 300s conda run --no-capture-output -n parallel_uav \
  python ParallelCBF_UAV/tools/v22_6_bc_pipeline.py \
  --num_envs 512 \
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
| Zero-residual nominal teacher success | `> 80%` | `0.0%` | FAIL |
| Teacher `v_aligned_mean` | positive | `-0.117286` | FAIL |
| Last logged `goal_reached_rate` | diagnostic | `0.5859%` | OUT |

Progress snapshots:

| Step | Success So Far | `v_aligned` |
| ---: | ---: | ---: |
| 1 | `0.0%` | `+0.0038` |
| 25 | `0.0%` | `-0.0337` |
| 50 | `0.0%` | `-0.3784` |
| 75 | `0.0%` | `-0.0512` |
| 100 | `0.0%` | `+0.5663` |
| 125 | `0.0%` | `-1.1217` |

## Protocol Decision

BC data collection was not performed. `data/teacher_demos.pt` and `checkpoints/bc_warmstart.pt` were not created. V22.7 PPO Layer 2 sanity was not launched.

This is the correct abort: V22.7 assumes the nominal P-PD base reaches the goal in clear paths. The gate proves that assumption is currently false under the existing nominal/QP/low-level dynamics chain.

## Recommended Next Step

Before training residual RL, run a direct nominal-control diagnostic that bypasses PPO residuals and measures:

- Whether `u_nominal` points toward the goal.
- Whether QP output `last_u_safe` tracks `u_combined`.
- Whether the geometric controller converts `u_safe` into the expected acceleration/velocity.
- Whether the sign convention in `goal_delta`, pitch/roll mapping, or force-world/body transform is inverted.

The residual RL architecture is now wired, but the nominal base must be fixed before BC or PPO can be meaningful.
