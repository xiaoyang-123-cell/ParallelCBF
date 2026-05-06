# V22.10d Critic Warmup & KL-to-BC Report

Status: Layer 2 completed. Layer 3 was not launched because the TensorBoard gate `task/goal_reached_rate > 65%` did not pass.

## Component C Reward Audit

Pre-flight audit passed: the frozen BC prior had higher expected reward than the failed V22.10c policy.

| Metric | Frozen BC | Failed V22.10c |
| --- | ---: | ---: |
| Reward mean | 73.2782 | 62.2227 |
| Success rate | 100.0000% | 96.8750% |
| First done step mean | 50.5312 | 78.9062 |
| Reward delta | +11.0555 | n/a |

Report: `v22_10d_reward_audit.md`

## Implementation Notes

- Added `--v22_10d_bc_anchor` to enable the V22.10d path.
- Injected V22.10d PPO-only kwargs through the runner config dictionary, not through Isaac Lab's strict config class.
- Added actor freeze for iterations `0-49`.
- Added KL-to-BC penalty with schedule:
  - Iter `<50`: beta `0.0`
  - Iter `50-199`: beta `1.0`
  - Iter `200-599`: linear decay to `0.0`
  - Iter `>=600`: beta `0.0`
- Captured a frozen BC actor immediately after loading `checkpoints/v22_10c_bc_warmstart.pt`.
- Fixed an implementation bug where restoring the full actor state during freeze tried to copy inference-mode normalizer buffers. The final implementation freezes actor updates by zeroing actor gradients and skips actor normalizer updates during warmup/KL anchoring.

## Layer 2 Run

Run: `logs/rsl_rl/uav_safe_3B1_v22_10d_bc_anchor_sanity/2026-04-28_17-17-19_layer2_100`

Command:

```bash
python ParallelCBF_UAV/train_ppo.py \
  --num_envs 4096 \
  --seed 54 \
  --max_iterations 100 \
  --cbf_enabled True \
  --bc_warmstart_path checkpoints/v22_10c_bc_warmstart.pt \
  --v22_9_residual_overrides \
  --v22_10d_bc_anchor \
  --experiment_name uav_safe_3B1_v22_10d_bc_anchor_sanity \
  --run_name layer2_100 \
  --headless
```

## Final Iteration 99 Metrics

| Metric | Value | Gate |
| --- | ---: | --- |
| `task/goal_reached_rate` | 2.1095% | FAIL, target >65% |
| `Policy/mean_std` | 0.3511 | PASS, target 0.25-0.45 |
| `Train/mean_episode_length` | 46.34 | PASS, target <60 |
| Actor drift `model_0 -> model_99` | 1.8374% | PASS, target <5% |
| `Loss/kl_bc` | 0.001327 | PASS, low drift |
| `Loss/kl_bc_beta` | 1.0000 | Expected |
| `Loss/clip_fraction` | 0.000378 | Stable |
| `safety/collision_rate` | 0.0000% | PASS |
| `qp/hard_cap_binding_rate` | 0.0000% | PASS |
| `safety/fallback_activation_rate` | 0.0000% | PASS |
| `hover/goal_dist_mean` | 0.4696 m | Still parked outside 0.4337 m radius |
| `task/v_aligned_mean` | 0.0734 | Low forward drive |
| `task/r_pbs_mean` | 0.3681 | Positive |

## Post-Run Deterministic Audit

Post-run deterministic audit compared frozen BC against the V22.10d `model_99.pt`.

| Metric | Frozen BC | V22.10d model_99 |
| --- | ---: | ---: |
| Reward mean | 73.4001 | 73.4336 |
| Success rate | 100.0000% | 100.0000% |
| First done step mean | 46.1875 | 46.8789 |
| Final goal distance mean | 0.4697 m | 0.4701 m |
| Action norm mean | 0.0571 | 0.0532 |

Report: `v22_10d_model99_reward_audit.md`

## Verdict

V22.10d successfully prevented catastrophic forgetting: actor drift dropped from the V22.10c diagnostic value of about `14.25%` to `1.84%`, while policy std stayed healthy and KL-to-BC remained small.

However, the official Layer 2 TensorBoard gate did not pass because `task/goal_reached_rate` remained around `2.1%`. Therefore Layer 3 was held.

Important diagnosis: deterministic evaluation shows both frozen BC and V22.10d `model_99` still achieve `100%` episode-level success under the audit script. This suggests the current TB `task/goal_reached_rate` is likely an instantaneous rollout occupancy metric under stochastic exploration, not the same as episode-level success. Before the next launch, we should separate `episode_goal_success_rate` from per-step `goal_reached_rate` so the Layer 2 gate measures the intended quantity.
