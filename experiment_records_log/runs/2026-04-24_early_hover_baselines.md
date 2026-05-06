# Early Hover Baselines

Status: `SUPERSEDED`

## Purpose

Establish the initial Isaac/rsl_rl hover training baseline before the later CBF-heavy safety campaigns. These runs predate the V3-V21 naming sequence but are part of the project history.

## Setup

- Main run family: `logs/rsl_rl/parallel_cbf_uav_hover/`
- Representative full run: `logs/rsl_rl/parallel_cbf_uav_hover/2026-04-24_23-16-06`
- Evidence type: TensorBoard scalars, checkpoints, and `iter50_report.txt`

## Result

- Iter 50 report showed rapid reward and episode-length growth: reward from `44.9225` at iter 25 to `204.148` at iter 50, episode length from `187.58` to `510.65`.
- Representative full-run TensorBoard final values at step `1499`: `Train/mean_reward=938.0741`, `Train/mean_episode_length=924.7100`, `hover/goal_dist_mean=0.1191427`, `hover/collision_rate=0.0`.
- The same full run had `Policy/mean_std=15.9442`, which is an abnormal policy-scale warning even though hover metrics looked strong.

## Diagnosis

The earliest hover task was learnable and produced superficially strong reward/length metrics, but it did not yet exercise the later obstacle/QP safety stack. The very large final policy std means the baseline cannot be treated as a stable safety-capable controller.

## Decision

Use these runs only as evidence that the simulator/training loop could learn simple hover. They are superseded by the later seed42 pilot, QP instrumentation, and V3-V21 safety campaigns.

## Evidence

- `logs/rsl_rl/parallel_cbf_uav_hover/2026-04-24_23-16-06/iter50_report.txt`
- `logs/rsl_rl/parallel_cbf_uav_hover/2026-04-24_23-16-06/events.out.tfevents.1777043769.smartlab.52061.0`
- `logs/rsl_rl/parallel_cbf_uav_hover/2026-04-24_23-16-06/model_1499.pt`

