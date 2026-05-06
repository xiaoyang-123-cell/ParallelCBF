# Seed42 Pilot and A/B Controls

Status: `SUPERSEDED`

## Purpose

Instrument the first obstacle-aware Seed 42 training path, test whether CBF/QP telemetry exists, and compare early mitigations before the formal V3-V21 sweep.

## Setup

- Clean pilot: `logs/rsl_rl/uav_safe_3B1_pilot_seed42_clean/2026-04-26_11-57-35`
- CBF v2 pilot: `logs/rsl_rl/uav_safe_3B1_pilot_seed42_cbf_v2/2026-04-26_12-29-00`
- Early abort: `logs/rsl_rl/uav_safe_3B1_pilot_seed42_cbf_v2/2026-04-26_12-23-43/pilot_abort_report.txt`
- A/B controls: `logs/rsl_rl/expA_pterm100/`, `logs/rsl_rl/expB_reduced_dr/`, `logs/rsl_rl/expB_patched_sanity/`
- Anti-collapse run: `logs/rsl_rl/uav_safe_3B1_pilot_seed42_anticoll/2026-04-26_17-56-46`

## Result

- Clean pilot completed 1499 iterations with high reward `731.8403`, long episode length `803.44`, and `hover/collision_rate=0.0`, but it lacked the CBF/QP telemetry needed for safety diagnosis.
- First CBF v2 attempt aborted at iteration 5 because CBF metrics were missing or `N/A`.
- CBF v2 full run completed but was task-poor: final reward `3.6982`, episode length `15.92`, `qp/infeasibility_rate=0.061615`, and `safety/collision_rate=0.00280762`.
- Patched sanity control improved the mechanical metrics by iter 199: `qp/infeasibility_rate=0.0280558`, `safety/collision_rate=0.00120036`, but `hover/goal_dist_mean=2.01555` remained poor.
- Anti-collapse final run kept policy std high (`0.9945`) and collision around `0.001241`, but goal distance stayed `2.02162`.

## Diagnosis

This phase exposed the core split that shaped the next week: telemetry and safety instrumentation could be improved, but goal-reaching did not follow automatically. The clean pilot looked good only because it did not expose the QP/CBF failure surfaces; once the shield metrics were visible, infeasibility and fallback dominated the story.

## Decision

Move from ad-hoc pilot patches to named versioned safety experiments. The V3-V11 sequence inherits this lesson and tracks fallback, persistent QP terminations, collision rate, and goal distance explicitly.

## Evidence

- `logs/rsl_rl/uav_safe_3B1_pilot_seed42_clean/2026-04-26_11-57-35/milestone_1499_report.txt`
- `logs/rsl_rl/uav_safe_3B1_pilot_seed42_cbf_v2/2026-04-26_12-23-43/pilot_abort_report.txt`
- `logs/rsl_rl/uav_safe_3B1_pilot_seed42_cbf_v2/2026-04-26_12-29-00/milestone_1499_report.txt`
- `logs/rsl_rl/expB_patched_sanity/2026-04-26_16-11-46/milestone_200_report.txt`
- `logs/rsl_rl/uav_safe_3B1_pilot_seed42_anticoll/2026-04-26_17-56-46/final_anticoll_report.txt`

