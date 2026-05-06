# Experiment Record Coverage

This file tracks which historical artifacts have been folded into the structured experiment records. It is not a replacement for the raw logs; it is the map from raw evidence to paper-usable notes.

For a per-file root report inventory, see `ARTIFACT_INVENTORY.md`.

## Coverage Levels

- `Detailed`: dedicated run record exists.
- `Grouped`: covered inside a broader phase record.
- `Indexed`: listed as evidence but not summarized deeply because it is a shard, retry, or duplicate diagnostic.

## Pre-V21 Coverage

| Artifact group | Coverage | Record |
| --- | --- | --- |
| Early hover runs under `logs/rsl_rl/parallel_cbf_uav_hover/` | `Detailed` | `runs/2026-04-24_early_hover_baselines.md` |
| Seed42 clean/CBF pilot and A/B controls | `Detailed` | `runs/2026-04-26_seed42_pilot_and_ab_controls.md` |
| V3-V11 PPO safety sweep | `Detailed` | `runs/2026-04-26_v3_v11_safety_training_sweep.md` |
| `qp_diagnostic_report_v12.md` through `qp_diagnostic_report_v20_summary.md` | `Detailed` | `runs/2026-04-26_v12_v20_layer1_qp_diagnostics.md` |
| Per-wind QP reports V14-V20 | `Grouped` | `runs/2026-04-26_v12_v20_layer1_qp_diagnostics.md` |

## V21-V24 Coverage

| Artifact group | Coverage | Record |
| --- | --- | --- |
| `qp_diagnostic_report_v21_summary.md`, V21 wind reports, V21 full run | `Grouped` | `runs/2026-04-30_v21_v23_planning_dataset_prelude.md` |
| V22 reward, signal, BC, residual, nominal, attitude-lag, V22.10c/d reports | `Detailed` | `runs/2026-04-27_v22_reward_bc_residual_pipeline.md` |
| V23 planning and GRU plumbing reports | `Grouped` | `runs/2026-04-30_v21_v23_planning_dataset_prelude.md` |
| V23 phase-3 dry runs, controller diagnostics, pursuit-teacher validations, and shard reports | `Grouped` | `runs/2026-04-30_v21_v23_planning_dataset_prelude.md` |
| V24 GRU BC anchor and safety-gate reports | `Detailed` | `runs/2026-04-30_v24_gru_bc_underfit.md` |

## V25-V26 Coverage

| Artifact group | Coverage | Record |
| --- | --- | --- |
| V25 Transformer BC training | `Detailed` | `runs/2026-05-01_v25_transformer_bc_success.md` |
| Stage-0 phantom collision forensics | `Detailed` | `runs/2026-05-01_stage0_phantom_collision_layer1_fix.md` |
| DualBarrier PPO diagnostic burn | `Detailed` | `runs/2026-05-02_v25_ppo_dualbarrier_fatal_funnel.md` |
| TripleBarrier attempt 2 | `Detailed` | `runs/2026-05-02_triplebarrier_attempt2.md` |
| Gate V2 resume | `Detailed` | `runs/2026-05-02_gate_v2_resume.md` |

## Known Gaps

- Some early runs only have TensorBoard event files and checkpoints, not narrative reports. Their records therefore cite extracted final scalars rather than full hand-written summaries.
- V23 has many small dry-run reports. They are intentionally grouped in the V21-V23 prelude instead of one file per mini-run to keep the record system readable.
- `run_logs/` launcher logs are cited only when they add evidence beyond the markdown report or `metrics.jsonl`.
