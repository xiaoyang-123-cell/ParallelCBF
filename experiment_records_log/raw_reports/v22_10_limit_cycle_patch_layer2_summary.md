# V22.10 Limit-Cycle Patch + Layer 2 Summary

Status: Layer 3 not launched. The physical/plumbing patch passed, but PPO Layer 2 did not meet the task-success gate.

## Patch Artifacts

| Artifact | Path |
| --- | --- |
| Unpatched limit-cycle trajectory | `data/v22_10_teacher_limit_cycle_unpatched.pt` |
| Patched convergence trajectory | `data/v22_10_teacher_convergence_patched.pt` |
| Patched demos | `data/v22_10_teacher_demos_patched.pt` |
| Patched BC checkpoint | `checkpoints/v22_10_bc_warmstart_patched.pt` |
| Layer 2 run | `logs/rsl_rl/uav_safe_3B1_v22_10_limit_cycle_patch_sanity/2026-04-28_15-28-36_layer2_100` |

## Patch Validation

| Gate | Result |
| --- | ---: |
| Step 1A direct +X acceleration | PASS, `dx_mean = 0.41056 m` |
| Step 1B direct +Z acceleration | PASS, `dz_mean = 0.49234 m` |
| Step 2 patched zero-residual teacher | PASS, `teacher_success_rate = 100.00%` |
| Patched clear demo success | PASS, `clear_teacher_success_rate = 95.51%` |
| BC final loss | PASS, `0.04948` |
| BC epochs | `7`, early stop |

## Layer 2 Final Metrics

| Metric | Target | Final | Tail-10 Mean | Verdict |
| --- | ---: | ---: | ---: | --- |
| `task/goal_reached_rate` | high sanity gate | `0.7645%` | `0.8261%` | FAIL |
| `task/v_aligned_mean` | `> 0.3` | `0.0791` | `0.0843` | FAIL |
| `Policy/mean_std` | `0.25 - 0.45` | `0.3862` | `0.3848` | PASS |
| `Loss/clip_fraction` | `< 0.3` | `0.0228` | `0.0215` | PASS |
| `Train/mean_episode_length` | `< 60` | `114.11` | `111.51` | FAIL |
| `safety/collision_rate` | low | `0.0000%` | `0.0000%` | PASS |
| `qp/hard_cap_binding_rate` | low | `0.0000%` | `0.0000%` | PASS |
| `safety/fallback_activation_rate` | low | `0.0000%` | `0.0000%` | PASS |
| `hover/goal_dist_mean` | should enter `0.30 m` policy radius | `0.3840 m` | `0.3859 m` | FAIL |
| `task/r_pbs_mean` | positive | `0.3251` | `0.3324` | PASS |

## Verdict

The 0.17 mm limit-cycle patch successfully repaired the teacher/plumbing path:

- The patched teacher reaches the relaxed teacher radius reliably.
- The BC dataset is no longer a pure-failure dataset.
- BC converged below the required loss threshold.
- QP/CBF is not suppressing motion: hard-cap binding, fallback, and collisions are all zero in Layer 2.

However, Layer 2 PPO did not pass because the policy still does not reliably enter the stricter `0.30 m` policy/evaluation radius. The final mean goal distance is approximately `0.384 m`, which is inside the teacher collection radius `0.45 m` but outside the policy success radius `0.30 m`.

Current failure classification: **teacher-policy radius mismatch / residual fine-convergence failure**, not a CBF or physics plumbing failure.

Recommended next action: do not launch Layer 3 yet. The next patch should preserve the V22.10 physical fix and address the final `0.45 m -> 0.30 m` convergence gap, either by staged teacher-radius annealing, an inner-zone braking controller, or collecting a second fine-convergence demo phase against the `0.30 m` policy radius.
