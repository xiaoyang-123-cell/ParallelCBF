# Experiment Timeline

## Early Hover and Pilot Runs

The earliest runs proved the Isaac/rsl_rl loop could learn simple hover. A representative full hover run reached `Train/mean_reward=938.0741`, `Train/mean_episode_length=924.71`, and `hover/goal_dist_mean=0.1191`, but ended with abnormal `Policy/mean_std=15.9442`. The first Seed42 safety pilots then showed that apparent reward was not enough: once CBF/QP telemetry was exposed, runs showed non-trivial infeasibility and poor task progress.

Key evidence: `logs/rsl_rl/parallel_cbf_uav_hover/2026-04-24_23-16-06`, `logs/rsl_rl/uav_safe_3B1_pilot_seed42_*`, `experiment_records/runs/2026-04-24_early_hover_baselines.md`, `experiment_records/runs/2026-04-26_seed42_pilot_and_ab_controls.md`.

## V3-V20: QP Bottleneck Isolation

V3-V11 tried PPO/safety variants directly. V3 full ended with fallback/infeasibility `0.180806`, persistent terminations `0.020752`, and goal distance `2.04055`; V6 and V7 hit abort gates. This forced a methodological pivot: stop tuning PPO and diagnose Layer-1 QP math offline using the V11 checkpoint. V12-V20 then improved feasibility above `99%` in many settings but repeatedly missed `D_t`, hard-cap, or soft-clamp gates. V20 removed mass and alpha-rate phantoms, proving wind/attitude-lag modeling was the remaining core issue.

Key evidence: `experiment_records/runs/2026-04-26_v3_v11_safety_training_sweep.md`, `experiment_records/runs/2026-04-26_v12_v20_layer1_qp_diagnostics.md`, `qp_diagnostic_report_v20_summary.md`.

## V21-V22: Safety/QP Stabilization but Task Bottleneck

V21 solved the structural QP feasibility problem: the 1500-iteration run completed normally, collision rate stayed around `0.6205%`, persistent QP terminations were nearly zero, and hard-cap binding stayed below the alarm line. V22.10/V22.10d then showed a recurring separation: safety metrics remained clean, but task success collapsed under long-horizon curriculum. V22.10d ended with `0.0%` collision rate but only `1.4388%` episode success.

Key evidence: `v21_1500_training_summary.md`, `v22_10_recovery_and_layer2_summary.md`, `v22_10d_layer3_summary.md`.

## V23: Navigation and Dataset Formation

V23 moved the stack from local reactive behavior toward waypoint-conditioned navigation. A* planning, waypoint blending, and a pursuit-teacher pipeline were introduced, with Layer-2 CBF checks kept active. The V23 200-episode sanity run accepted 122/200 episodes and showed zero hard CBF violations, but teacher success stayed around `0.61`; the dataset was useful for BC but not a finished policy result.

Key evidence: `v23_stream1_eod1_report.md`, `v23_stream2_eod2_report.md`, `v23_oversampling_sanity_200_report.md`, `v23_oversampling_50k_shard_*.md`.

## V24: GRU Capacity Limit

V24 trained a GRU behavior-cloning anchor on the accepted V23 50K artifact. The dataset audit passed on `31,415` episodes and `4,109,264` timesteps, but the GRU final BC loss stopped at `0.059459819820711426`. This became the old floor and the reason to pivot from recurrent small models to a causal Transformer.

Key evidence: `v24_gru_bc_anchor_report.md`, `v24_phase_alpha_safety_gates_report.md`.

## V25 Day 1-5: Transformer BC Breakthrough

The pure causal Transformer BC pipeline replaced the underpowered GRU. Full training reached best validation loss `0.0060398161485550345`; by epoch 5, train loss was already `0.0279513070431`, below the GRU floor. This established that the main BC bottleneck was model capacity/architecture rather than dataset corruption.

Key evidence: `logs/v25_train_full.log`, `checkpoints/v25_transformer_bc_full/best.pt`.

## V25 Day 6: Evaluation Bug and True Stage-0 Failure Mode

The first Stage-0 evaluation appeared to show collisions, but forensic logging showed `h_hard` was extremely positive near the reported event. The environment was conflating arena exits/dummy sentinels with obstacle collisions. After separating termination reasons, the real Stage-0 result was `0` collisions, `200` out-of-arena events, `0` successes.

Key evidence: `logs/forensic_console.log`, `logs/v25_stage0_eval.json`.

## V25/V26 PPO: Fatal Funnel and TripleBarrier Pivot

KL-anchored PPO with DualBarrier preserved obstacle safety but did not prevent boundary exits. Diagnostic metrics showed the critic could briefly reach EV above `0.6`, but the rollout distribution was dominated by boundary termination. TripleBarrier promoted arena boundaries to Layer 2. The calibrated attempt eliminated collisions and out-of-arena events, with arena projection active around `0.53125` and EV peaking at `0.7428237199783325`.

Key evidence: `runs/v26_diagnostic_20260502T110047Z/metrics.jsonl`, `runs/v02_attempt2_20260502T114503Z/metrics.jsonl`.

## Gate V2: Correct Transition Signal, New PPO Shock

Gate V1 waited for value-loss plateau and incorrectly kept the actor frozen. Gate V2 used sustained rolling EV and opened at `global_step=1,507,328`. The resumed run entered full PPO and reached EV peak `0.9675523042678833`, but forensics captured huge immediate PPO KL (`approx_kl` about `1.87`) and `clip_fraction=1.0`, followed by a watchdog halt at `1,974,272` transitions.

Key evidence: `AMENDMENT_LOG.md`, `runs/v02_gate_v2_resume_20260502T121701Z/launch.log`, `runs/v02_gate_v2_resume_20260502T121701Z/checkpoints/forensics/forensics_20260502_122113.json`.
