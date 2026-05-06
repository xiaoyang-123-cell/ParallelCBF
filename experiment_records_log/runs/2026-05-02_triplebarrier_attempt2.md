# TripleBarrier Attempt 2

Status: `HALTED`

## Purpose

Test the TripleBarrier pivot: add arena-boundary constraints to Layer 2 so the safety filter prevents boundary exits rather than merely reporting them.

## Setup

- Failed early attempt: `runs/v02_attempt2_20260502T112534Z`
- Calibrated attempt: `runs/v02_attempt2_20260502T114503Z`
- Metrics: `metrics.jsonl` in both run directories
- Safety filter: TripleBarrier with arena projection

## Result

- First attempt showed arena projection inactive: final `rollout/arena_projection_active_rate=0.0` with `rollout/out_of_arena=64`.
- Calibrated attempt eliminated exits and collisions at the halt point: `rollout/out_of_arena=0`, `rollout/collisions=0`.
- Calibrated arena projection active rate: `0.53125`.
- Calibrated EV peak: `0.7428237199783325`.
- Final EV before halt: `0.731096625328064`.
- Final value loss before halt: `319.74920654296875`.
- Halted at `1,503,232` transitions while still in critic warmup.

## Diagnosis

TripleBarrier solved the boundary-exit failure mode. The remaining problem was not safety but phase-gating: Gate V1 used value-loss plateau as a proxy for critic readiness, even though explained variance had already crossed a useful threshold.

## Decision

Replace Gate V1 with Gate V2: release actor based on sustained rolling explained variance and non-negative EV slope, not value-loss plateau.

## Evidence

- `runs/v02_attempt2_20260502T112534Z/metrics.jsonl`
- `runs/v02_attempt2_20260502T114503Z/metrics.jsonl`
- `AMENDMENT_LOG.md`

