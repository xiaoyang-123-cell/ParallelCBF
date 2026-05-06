# V3-V11 Safety Training Sweep

Status: `FAIL`

## Purpose

Run a sequence of PPO/safety variants to reduce QP fallback and persistent terminations while preserving task learning under wind and obstacle pressure.

## Setup

- V3 blackhole sanity/full: `logs/rsl_rl/uav_safe_3B1_v3_blackhole_*`
- V4 anti-overrel sanity: `logs/rsl_rl/uav_safe_3B1_v4_anti_overrel_sanity/2026-04-26_20-38-51`
- V5 threshold sanity: `logs/rsl_rl/uav_safe_3B1_v5_threshold_sanity/2026-04-26_20-46-31`
- V6 wide-threshold/full: `logs/rsl_rl/uav_safe_3B1_v6_*`
- V7 PPO tune: `logs/rsl_rl/uav_safe_3B1_v7_ppo_tune/2026-04-26_22-15-16`
- V8 anti-trap sanity, V9 goldilocks, V10 inverted sanity, V11 simplified sanity: `logs/rsl_rl/uav_safe_3B1_v8_*` through `logs/rsl_rl/uav_safe_3B1_v11_*`

## Result

- V3 full failed badly by iter 1499: reward `-23.9178`, episode length `40.7`, fallback/infeasibility `0.180806`, persistent terminations `0.020752`, goal distance `2.04055`.
- V4/V5 50-iteration sanity runs still had fallback around `0.11` and goal distance around `1.84`.
- V6 full triggered abort criteria at iter 500: reward `-24.9643`, episode length `63.62`, fallback `0.115285`, persistent terminations `0.0132345`.
- V7 PPO tune hit its abort gate at iter 750: reward `-61.3613`, episode length `59.46`, fallback `0.130727`, persistent terminations `0.015625`.
- V8/V9/V10/V11 sanity runs improved short-horizon reward relative to V6/V7 in some cases, but fallback stayed around `0.105-0.111` and goal distance stayed around `1.82-1.84`.
- V11 became the reference checkpoint for later offline QP diagnostics: `logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/model_99.pt`.

## Diagnosis

The sweep showed a persistent structural QP/safety bottleneck. PPO tuning and threshold changes could not reduce fallback enough, and task distance did not materially improve. This justified suspending RL hyperparameter tuning and switching to offline QP math for V12 onward.

## Decision

Freeze PPO iteration as a diagnostic path and use V11 states/checkpoint as a reproducible source for QP feasibility reconstruction.

## Evidence

- `logs/rsl_rl/uav_safe_3B1_v3_blackhole_full/2026-04-26_19-42-28/iter1500_final_report.txt`
- `logs/rsl_rl/uav_safe_3B1_v6_full/2026-04-26_21-05-39/iter500_report.txt`
- `logs/rsl_rl/uav_safe_3B1_v7_ppo_tune/2026-04-26_22-15-16/iter750_report.txt`
- `logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/events.out.tfevents.1777218974.smartlab.44187.0`
- `logs/rsl_rl/uav_safe_3B1_v11_simplified_sanity/2026-04-26_23-56-10/model_99.pt`

