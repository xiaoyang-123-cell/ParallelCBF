# V21-V23 Planning and Dataset Prelude

Status: `PARTIAL`

## Purpose

Build the safety, navigation, and demonstration substrate that later BC/PPO experiments rely on: QP/CBF stabilization, waypoint planning, recurrent-policy plumbing, pursuit-teacher data collection, and Layer-2 safety auditing.

## Setup

- Safety/QP reports: `v21_1500_training_summary.md`, `v22_10_recovery_and_layer2_summary.md`, `v22_10d_layer3_summary.md`
- Planner reports: `v23_stream1_eod1_report.md`, `v23_stream2_eod2_report.md`
- Dataset sanity report: `v23_oversampling_sanity_200_report.md`
- Full shard reports: `v23_oversampling_50k_shard_00_v1_report.md` through `v23_oversampling_50k_shard_04_v1_report.md`

## Result

- V21 completed 1500 iterations with final collision rate `0.6205%`, fallback activation `0.6856%`, and nearly zero persistent QP terminations.
- V22.10d completed 1500 iterations with `0.0%` collision rate but only `1.4388%` final episode success.
- Stream 1 implemented A* and waypoint blending; standalone tests passed.
- Stream 2 implemented the V23 GRU path and 26-dim waypoint observation schema; 7 GRU pitfall tests passed.
- V23 200-episode sanity collection accepted `122/200` episodes with teacher success rate `0.61`.
- Layer-2 hard barrier violation rate was `0.0` in the sanity run.
- The sanity run still failed its dry-run status because teacher success and A* no-path gates were not yet strong enough.

## Diagnosis

The early stack made safety increasingly reliable before it made task performance reliable. V21/V22 established that the bottleneck had moved away from raw QP feasibility and toward long-horizon navigation. V23 responded by adding planning and pursuit-teacher structure, producing a usable demonstration source rather than a deployable policy.

## Decision

Proceed to larger accepted-demo aggregation and BC anchoring, while adding stronger pre-trigger gates in V24.

## Evidence

- `v21_1500_training_summary.md`
- `v22_10_recovery_and_layer2_summary.md`
- `v22_10d_layer3_summary.md`
- `v23_stream1_eod1_report.md`
- `v23_stream2_eod2_report.md`
- `v23_oversampling_sanity_200_report.md`
- `v24_phase_alpha_safety_gates_report.md`
