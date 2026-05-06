# Experiment Record System Backfill

Status: `PASS`

## Purpose

Create a canonical experiment-record folder and backfill the historical V21-V26 record from chat context, reports, and run logs. This is infrastructure for paper writing and future directive handoffs.

## Setup

- New folder: `experiment_records/`
- Source reports: root `v21_*`, `v22_*`, `v23_*`, `v24_*` markdown reports
- Source logs: `logs/v25_train_full.log`, `logs/v25_stage0_eval.json`, `logs/forensic_console.log`
- Source run metrics: `runs/*/metrics.jsonl` for V25/V26 PPO diagnostics

## Result

- Added a stable documentation schema: `README.md`, `TEMPLATE.md`, `INDEX.md`, `timeline.md`, and `lessons_learned.md`.
- Added focused run records under `experiment_records/runs/` for the main V21-V26 narrative.
- Backfilled the core evidence chain: safety/QP stabilization, GRU underfit, Transformer BC breakthrough, phantom collision diagnosis, DualBarrier fatal funnel, TripleBarrier pivot, and Gate V2 resume halt.
- Established the rule that each future directive should end by updating a relevant record and the index.

## Diagnosis

Before this backfill, experiment knowledge was spread across chat history, root-level reports, logs, checkpoints, and amendments. That made the scientific story easy to lose. The new record system compresses evidence into paper-usable notes while preserving links to raw artifacts.

## Decision

Use `experiment_records/` as the canonical lab notebook from this point forward. Keep `AMENDMENT_LOG.md` for formal pre-registration amendments only; keep broader lessons and run narratives here.

## Evidence

- `experiment_records/README.md`
- `experiment_records/INDEX.md`
- `experiment_records/timeline.md`
- `experiment_records/lessons_learned.md`
