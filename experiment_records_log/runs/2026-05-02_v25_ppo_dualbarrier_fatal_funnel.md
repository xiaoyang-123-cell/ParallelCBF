# V25 PPO DualBarrier Fatal Funnel

Status: `FAIL`

## Purpose

Run a diagnostic KL-anchored PPO control experiment with the existing DualBarrier safety filter before implementing arena-boundary CBF constraints.

## Setup

- Run directory: `runs/v26_diagnostic_20260502T110047Z`
- Metrics: `runs/v26_diagnostic_20260502T110047Z/metrics.jsonl`
- Safety filter: DualBarrier CBF, no arena barrier
- Total diagnostic budget: `1,000,000` transitions

## Result

- Final phase: `critic_warmup` (`phase=0`).
- EV peak: `0.6279959082603455`.
- Final EV: `-1.6167211532592773` on the terminal partial window.
- Final value loss: `354.1780700683594`; minimum recorded value loss also `354.1780700683594`.
- Representative near-final windows had `rollout/out_of_arena=64`, `rollout/collisions=0`, `rollout/successes=0`, `rollout/timeouts=0`.

## Diagnosis

DualBarrier protected obstacle safety but left arena exits as a Layer-1/reward issue. The rollout distribution became dominated by boundary exits, creating the “fatal funnel”: critic learning signal was narrow and actor improvement could not address the dominant failure safely.

## Decision

Promote arena boundary to Layer 2 via TripleBarrierCBF instead of relying on reward shaping or post-hoc termination penalties.

## Evidence

- `runs/v26_diagnostic_20260502T110047Z/metrics.jsonl`
- `runs/v26_diagnostic_20260502T110047Z/launch.log`

