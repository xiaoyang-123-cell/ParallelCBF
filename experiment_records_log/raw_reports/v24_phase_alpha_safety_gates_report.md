# V24 Phase Alpha Safety Gates

Generated: `2026-04-30T14:27:00+08:00`

## Status

All requested pre-trigger gates are implemented before the V23 50K dataset artifact exists. The old ungated autostart watcher was stopped, and the gated watcher is now running.

## Gate Map

| Gate | Implementation | Halt behavior | Telemetry |
| --- | --- | --- | --- |
| Step 1.0 Inline Dataset Audit | `ParallelCBF_UAV/tools/v24_train_gru_bc_anchor.py` | exits `10` before BC training if audit fails | `logs/v24_step1_dataset_audit`, `v24_step1_dataset_audit_report.md`, `data/v24_step1_dataset_audit.json` |
| Step 1.5 Layer 2 Sanity Rollout | `ParallelCBF_UAV/tools/v24_layer2_sanity_rollout.py` | exits `20` before PPO if Stage 0 < 85% or Stage 1 < 55% | `logs/v24_step1_5_layer2_sanity`, `v24_step1_5_layer2_sanity_report.md`, `data/v24_step1_5_layer2_sanity.json` |
| Step 3.0 Dynamic Critic Warmup | `ParallelCBF_UAV/training/entropy_runner.py` + `PPOWithClipFraction` | raises RuntimeError if 750K cap is hit before plateau exit | TensorBoard `v24/critic_warmup_*`, stdout `[V24 CRITIC WARMUP COMPLETE]` |
| Phase alpha Section 7 Watchdogs | `ParallelCBF_UAV/training/entropy_runner.py` | raises RuntimeError on h_hard > 2% or freeze suspicion | TensorBoard `safety/h_hard_violation_rate`, `v24/freeze_streak` |

## Clarifications Implemented

- KL clock starts after warmup exit via `kl_bc_reference_iteration`, not at launch.
- KL reference is captured from the local pre-trained BC checkpoint loaded through `--bc_warmstart_path`.
- BC training now uses true episode-sequential GRU BPTT instead of treating timesteps as parallel batch entries.
- Step 1.0 smoke passed on `data/v23_oversampling_sanity_200.pt`.

## Active Processes

- V23 50K collection remains active.
- Gated V24 autostart PID: `83536`.
- Gated V24 autostart log: `run_logs/v24_phase_alpha_autostart_20260430_142639.log`.
