# Gate V2 Resume

Status: `HALTED`

## Purpose

Resume from the saved TripleBarrier checkpoint and test the amended Gate V2 critic-to-PPO transition conditions.

## Setup

- Run directory: `runs/v02_gate_v2_resume_20260502T121701Z`
- Manifest amendment: `AMENDMENT_LOG.md`
- Metrics: `runs/v02_gate_v2_resume_20260502T121701Z/metrics.jsonl`
- Forensics: `runs/v02_gate_v2_resume_20260502T121701Z/checkpoints/forensics/forensics_20260502_122113.json`
- Gate V2 manifest SHA: `04239083e521976faa6fa109bd5624296af0c8a69c037da2f52703424f58d88d`

## Result

- Gate V2 transition occurred at `global_step=1507328` with `ev=0.647888` and `ev_slope_100k=2.7114e-07`.
- Final recorded phase: `ppo` (`phase=1`).
- EV peak during resume: `0.9675523042678833`.
- Final EV: `0.8907701373100281`.
- Value loss dropped to minimum `20.638092041015625`; final value loss `30.59918785095215`.
- Safety remained clean at the final metrics row: `rollout/collisions=0`, `rollout/out_of_arena=0`, `safety/h_hard_min=4.792019367218018`.
- Run halted by watchdog at `1,974,272` transitions; failed checkpoint exists at `runs/v02_gate_v2_resume_20260502T121701Z/checkpoints/failed_step_01974272.pt`.

## 2026-05-02 20:45 CST Status Check

- No active `train_ppo_v25` or V25 training Python process was running.
- `runs/CURRENT_RUN` still points at the older Attempt-2 halt directory, and `runs/CURRENT_RUN_V2_RESUME` points at this resume directory.
- The latest resume log still ends with `[HALT][V25] reason=watchdog halt`.
- Current interpretation: training is stopped, not progressing in the background.

## Diagnosis

Gate V2 opened the phase door correctly, but the first full-PPO updates were too abrupt. Forensics captured immediate PPO shock after transition: `approx_kl` around `1.87`, `clip_fraction=1.0`, and gradient norms above `1400` in the first PPO records. The later console rows show lower gradient norms, but the watchdog halt indicates the handoff envelope was still violated.

## Decision

Next work should focus on full-PPO handoff control: actor LR warmup magnitude, KL anchor activation schedule, policy ratio reset/normalization, and watchdog thresholds for the first post-transition window. Do not discard TripleBarrier; safety results remained clean.

## Evidence

- `runs/v02_gate_v2_resume_20260502T121701Z/launch.log`
- `runs/v02_gate_v2_resume_20260502T121701Z/metrics.jsonl`
- `runs/v02_gate_v2_resume_20260502T121701Z/checkpoints/forensics/forensics_20260502_122113.json`
- `AMENDMENT_LOG.md`
