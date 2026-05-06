# Experiment Record Index

This index is the fast map. Detailed notes live in `runs/`.

| Date | Record | Status | One-line Verdict |
| --- | --- | --- | --- |
| 2026-04-24 | [Early Hover Baselines](runs/2026-04-24_early_hover_baselines.md) | `SUPERSEDED` | Simple hover learned quickly, but the baseline did not exercise the later safety stack and ended with abnormal policy std. |
| 2026-04-26 | [Seed42 Pilot and A/B Controls](runs/2026-04-26_seed42_pilot_and_ab_controls.md) | `SUPERSEDED` | Early telemetry showed instrumentation and QP visibility mattered more than apparent clean-pilot reward. |
| 2026-04-26 | [V3-V11 Safety Training Sweep](runs/2026-04-26_v3_v11_safety_training_sweep.md) | `FAIL` | PPO variants could not escape 10%+ fallback/QP infeasibility and poor goal distance, forcing offline QP diagnosis. |
| 2026-04-26 | [V12-V20 Layer-1 QP Diagnostics](runs/2026-04-26_v12_v20_layer1_qp_diagnostics.md) | `FAIL` | Feasibility improved above 99%, but robust-margin and cap metrics stayed out of band until V21. |
| 2026-04-27 | [V22 Reward, BC, and Residual Pipeline](runs/2026-04-27_v22_reward_bc_residual_pipeline.md) | `FAIL` | After V21 safety stabilized, V22 exposed task-learning, teacher, BC, and metric-semantics bottlenecks. |
| 2026-04-28 to 2026-04-30 | [V21-V23 Planning and Dataset Prelude](runs/2026-04-30_v21_v23_planning_dataset_prelude.md) | `PARTIAL` | A* waypointing, pursuit teacher, and Layer-2 safety plumbing produced a usable but imperfect V23 dataset pipeline. |
| 2026-04-30 | [V24 GRU BC Underfit](runs/2026-04-30_v24_gru_bc_underfit.md) | `FAIL` | GRU BC reached `0.0594598` loss and underfit 4.1M timesteps, motivating the Transformer pivot. |
| 2026-05-01 | [V25 Causal Transformer BC Success](runs/2026-05-01_v25_transformer_bc_success.md) | `PASS` | Transformer BC reached best validation loss `0.0060398`, decisively below the GRU floor. |
| 2026-05-01 | [Stage-0 Phantom Collision and Layer-1 Fix](runs/2026-05-01_stage0_phantom_collision_layer1_fix.md) | `PASS` | Forensics showed Layer 2 was safe; Stage-0 failures were out-of-arena policy drift, not obstacle collisions. |
| 2026-05-02 | [DualBarrier PPO Fatal Funnel](runs/2026-05-02_v25_ppo_dualbarrier_fatal_funnel.md) | `FAIL` | DualBarrier allowed 100% boundary exits, starving the PPO problem of useful recovery dynamics. |
| 2026-05-02 | [TripleBarrier Attempt 2](runs/2026-05-02_triplebarrier_attempt2.md) | `HALTED` | Arena barrier eliminated exits/collisions and EV peaked at `0.7428`, but Gate V1 held actor frozen too long. |
| 2026-05-02 | [Gate V2 Resume](runs/2026-05-02_gate_v2_resume.md) | `HALTED` | Gate V2 opened full PPO at `1,507,328` transitions; later watchdog halted after very high initial PPO KL/clip saturation. |
| 2026-05-02 | [Experiment Record System Backfill](runs/2026-05-02_experiment_record_system_backfill.md) | `PASS` | Created the canonical structured lab notebook and backfilled the V21-V26 narrative from reports/logs. |
| 2026-05-03 | [PPO Handoff Slipper Resume Patch](runs/2026-05-03_ppo_handoff_slipper.md) | `HALTED` | Slipper fixed the KL handoff (`max_kl=0.01634`), but an over-sensitive value-loss-slope watchdog halted at `1,863,680`. |
| 2026-05-03 | [Phase-Aware Watchdog Amendment 5 Resume](runs/2026-05-03_phase_aware_watchdog_resume.md) | `COMPLETED` | Amendment 5 eliminated the F-015 false halt and completed 10M transitions with stable KL and clean safety; success-rate evaluation remains separate. |
| 2026-05-03 | [Final Evaluation R-7 Halt](runs/2026-05-03_final_eval_r7_halt.md) | `HALTED` | Final CPU eval halted at episode 0 step 16 on arena stopping-barrier R-7 autopsy before aggregate metrics were generated. |
| 2026-05-03 | [Dual-Track Eval Completion and Attempt-5 Ignition](runs/2026-05-03_dual_track_eval_attempt5.md) | `COMPLETED` | Final eval completed 600/600 with 0/0/0/600 outcomes; attempt-5 proved F-016 was fixed but halted on critic EV watchdog. |
| 2026-05-03 | [Attempt-6 PBRS Diagnostic Readback](runs/2026-05-03_attempt6_pbrs_readback.md) | `HALTED` | PBRS implementation launched cleanly, but the 1M diagnostic halted at `151,552` transitions with EV stuck near zero under high arena projection. |
| 2026-05-03 | [Attempt-7 KF-005 4x PBRS Final Burn](runs/2026-05-03_attempt7_kf005_4x_final_burn.md) | `PASS` | 4x shaping improved critic fit but did not break the wall-hugging attractor; EV peaked at `0.5927`, safety stayed clean, and success remained `0`. |

## Current Open Questions

- What is the true 500-step episode success rate of the 10M final checkpoint under an evaluation loop that does not reset every 64 rollout steps?
- Should R-7 treat tiny `h_arena_stopping_min` negatives near `1e-8` as hard Layer-2 violations, or use a numerical tolerance separate from physical collision/out-of-arena safety?
- Should `rollout/episode_return_std=nan` during no-termination windows be reported separately from zero variance to avoid misleading watchdog traces?

## Coverage Map

See [COVERAGE.md](COVERAGE.md) for the mapping from historical reports/logs to these structured records.
