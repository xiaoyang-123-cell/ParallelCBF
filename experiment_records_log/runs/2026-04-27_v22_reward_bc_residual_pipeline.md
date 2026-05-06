# V22 Reward, BC, and Residual Pipeline

Status: `FAIL`

## Purpose

After V21 stabilized Layer-1 safety/QP, shift attention to task success: reward accounting, signal density, nominal controller plumbing, BC warm starts, and residual RL.

## Setup

- Reward/signal reports: `v22_4_reward_pipeline_audit.md`, `v22_5_signal_to_noise_sanity.md`
- BC/residual reports: `v22_6_bc_pipeline_report.md`, `v22_7_residual_pivot_report.md`
- Nominal engine audit: `v22_8_nominal_engine_audit_report.md`
- Attitude/plumbing/BC reports: `v22_9_phase_minus1_abort_report.md`, `v22_10_*`
- Later V22.10c/V22.10d reports: `v22_10c_*`, `v22_10d_*`

## Result

- V22.4 fixed PBS reward accounting, but goal reached rate was only `0.3866%`; collision and hard-cap binding were clean.
- V22.5 added velocity-alignment reward, but `v_aligned_mean` remained negative and the goal gate still failed.
- V22.6 aborted BC at teacher sanity: teacher success was `0.0%`, exposing an action-interface mismatch.
- V22.7 pivoted to residual Cartesian control but again aborted because zero-residual nominal teacher success was `0.0%`.
- V22.8 nominal engine audit passed; closed-loop convergence reached teacher success `1.0`.
- V22.9/V22.10 exposed attitude-lag and plumbing problems; V22.10 math later passed Layer 1, but BC/PPO startup was blocked by host GPU/Isaac runtime in one attempt.
- V22.10 limit-cycle patch repaired the teacher/plumbing path: patched clear demo success `95.51%`, BC final loss `0.04948`, but Layer-2 PPO still failed with `task/goal_reached_rate=0.7645%`.
- V22.10d prevented catastrophic forgetting with actor drift `1.8374%`, but final Layer-3 success was only `1.4388%` despite `0.0%` collision rate.

## Diagnosis

V22 confirms the post-V21 bottleneck moved from safety feasibility to task learning and metric semantics. Reward signs, nominal acceleration realization, BC data quality, and success measurement all mattered; none alone solved long-horizon reaching under the full curriculum.

## Decision

Do not continue trying to solve the task purely with V22 PPO reward surgery. Preserve the V21/V22 safety math and move toward staged navigation, waypoint/pursuit-teacher data, and later V23/V24/V25 BC.

## Evidence

- `v22_4_reward_pipeline_audit.md`
- `v22_5_signal_to_noise_sanity.md`
- `v22_6_bc_pipeline_report.md`
- `v22_7_residual_pivot_report.md`
- `v22_8_nominal_engine_audit_report.md`
- `v22_10_limit_cycle_patch_layer2_summary.md`
- `v22_10d_bc_anchor_report.md`
- `v22_10d_layer3_summary.md`

