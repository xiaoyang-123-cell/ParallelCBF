# V22.10c Curriculum + BC Augmentation Summary

Status: Layer 2 completed, Layer 3 not launched.

## Implemented Changes

- Added a goal-success-radius curriculum in `UAVSafeEnv`:
  - Iter `< 50`: `0.45 m`
  - Iter `50-500`: linearly anneals `0.45 -> 0.30 m`
  - Iter `>= 500`: `0.30 m`
- Routed reward and termination goal checks through the current success radius.
- Added telemetry:
  - `task/goal_success_radius_current`
  - `task/goal_success_radius_mean`
- Added `last_goal_reached_mask` so demo collection can observe success events before Isaac resets the env.
- Added `ParallelCBF_UAV/tools/v22_10c_collect_demos.py` for V22.10c demo collection.
- Added 20% push-through lanes in clear-path demos. These lanes do not terminate at the relaxed boundary, allowing the teacher to continue toward the strict `0.30 m` region.
- Made `train_ppo.py` preflight optional via `--run_preflight_self_test`, because the diagnostic stepping path can close Isaac before PPO starts. PPO training now skips that diagnostic by default; the diagnostic remains available explicitly.

## Demo And BC Results

| Metric | Result |
| --- | ---: |
| Demo path | `data/v22_10c_teacher_demos.pt` |
| Samples | `70,000` |
| Clear / obstacle split | `50,176 / 19,824` |
| Clear relaxed teacher success | `100.00%` |
| Clear strict policy success | `16.02%` |
| Push-through fraction | `20.31%` |
| Push-through strict success | `78.85%` |
| Obstacle residual action norm mean | `1.2559` |
| BC checkpoint | `checkpoints/v22_10c_bc_warmstart.pt` |
| BC final loss | `0.049304` |
| BC epochs | `14` |

## Layer 2 Run

Run:

```bash
logs/rsl_rl/uav_safe_3B1_v22_10c_curriculum_sanity/2026-04-28_16-15-35_layer2_100
```

| Metric | Target | Final | Tail-10 Mean | Verdict |
| --- | ---: | ---: | ---: | --- |
| `task/goal_reached_rate` | `> 70%` | `0.3250%` | `0.4034%` | FAIL |
| `task/goal_success_radius_current` | approx `0.433 m` | `0.4337 m` | `0.4352 m` | PASS |
| `hover/goal_dist_mean` | `< current radius` | `0.4728 m` | `0.4729 m` | FAIL |
| `task/v_aligned_mean` | positive/strong | `0.0121` | `0.0126` | FAIL |
| `Train/mean_episode_length` | should drop | `173.60` | `125.14` | FAIL |
| `Policy/mean_std` | `0.25 - 0.45` | `0.3769` | `0.3761` | PASS |
| `Loss/clip_fraction` | `< 0.3` | `0.0220` | `0.0260` | PASS |
| `safety/collision_rate` | low | `0.0000%` | `0.0000%` | PASS |
| `qp/hard_cap_binding_rate` | low | `0.0000%` | `0.0000%` | PASS |
| `safety/fallback_activation_rate` | low | `0.0000%` | `0.0000%` | PASS |
| `task/r_pbs_mean` | positive | `0.2983` | `0.2990` | PASS |

## Verdict

The V22.10c bridge was implemented correctly, and the radius curriculum is active in PPO. The BC dataset now contains real strict-radius push-through evidence: about `20%` of clear lanes were assigned to push-through, and `78.85%` of those lanes reached the strict `0.30 m` radius at least once.

However, Layer 2 still fails. At iteration 99, the active success radius is `0.4337 m`, but the policy's mean goal distance is `0.4728 m`, leaving it about `3.9 cm` outside the current curriculum gate. Goal-reaching is still effectively absent.

This is not a CBF or safety bottleneck:

- Hard-cap binding is `0%`.
- Fallback is `0%`.
- Collision is `0%`.
- Policy std and PPO clip fraction are healthy.

Failure classification: **residual policy still does not inherit the teacher's fine-convergence behavior despite the 20% push-through augmentation**.

Layer 3 was not launched.

Recommended next step: inspect actor outputs near the goal shell. Specifically compare BC actor mean residuals for states at `0.45 m`, `0.40 m`, `0.35 m`, and `0.30 m`; if the residual mean remains near zero in the final shell, the zero-residual clear-path target is still overpowering the strict push-through signal.
