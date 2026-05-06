# Phase-Aware Watchdog Amendment 5 Resume

Status: `COMPLETED`

## Purpose

Implement Amendment 5 after the Slipper resume halted on a false-positive
watchdog. The goal is to keep all safety and KL protections active while making
watchdogs explicitly regime-aware across `CRITIC_WARMUP` and `FULL_PPO`.

## Trigger

- Previous run: `runs/v02_slipper_resume_20260503T064452Z`
- Previous final transition: `1,863,680`
- Previous halt reason: `watchdog halt`
- Evidence against true failure: `max(train/approx_kl)=0.01633099466562271`,
  final EV `0.9573695063591003`, collisions `0`, out-of-arena `0`,
  `safety/h_hard_min=4.7846245765686035`.
- Likely false-positive rule: `value_loss_increasing`, because final
  `value_loss_slope=0.004083183484199719` crossed an immediate `> 0`
  threshold in `FULL_PPO`.

## Patch

- Active watchdog envelope: `configs/v25_watchdog_envelope.yaml`.
- Every watchdog rule now declares `active_phases`.
- `value_loss_increasing` was replaced by `value_loss_increasing_warmup`.
- `value_loss_increasing_warmup` is active only in `CRITIC_WARMUP`, uses
  `threshold_slope=1.0e-3`, and has severity `WARN`.
- Added `explained_variance_dropping_full_ppo`, active only in `FULL_PPO`,
  halting if `train/value_explained_variance < 0.5` is sustained for
  `200,000` steps.
- Launcher now asserts that every rule has `active_phases`, phase-gates every
  watchdog update, normalizes `WARN/HALT` severity aliases, forwards
  `phase_name` and `train/value_explained_variance` to watchdogs, and logs the
  alias `rollout/termination_success`.

## Validation

- Targeted phase-aware watchdog tests: `3 passed`.
- Full dev package tests: `91 passed, 1 skipped`.
- Strict type check on patched launcher and new tests: `Success: no issues
  found in 2 source files`.
- New pre-registration manifest SHA:
  `bec97d2e2672588a135b4df09b74386bf4596b44ff9c854c258765c2cf91b0e5`.

## Resume Plan

- Resume checkpoint:
  `runs/v02_slipper_resume_20260503T064452Z/checkpoints/last_safe_step.pt`.
- Expected resume state: clean `FULL_PPO` near the previous safe checkpoint,
  before the false-positive halt.
- Watch target: page on the first non-zero `rollout/termination_success`.

## 2026-05-03 16:34 CST Resume Launch

- Run directory: `runs/v02_phase_aware_resume_20260503T083420Z`.
- PID: `8931`.
- Detachment: `TTY=?`, session ID equals PID.
- Manifest SHA accepted by launcher:
  `bec97d2e2672588a135b4df09b74386bf4596b44ff9c854c258765c2cf91b0e5`.
- Resume state: `start_transitions=1863680`, `phase=ppo`,
  `global_step=1863680`.
- Optimizer restore: `restored_optimizer_state_dict=True`.
- Watchdog registry printed all nine phase-aware rules, including
  `value_loss_increasing_warmup` active only in `critic_warmup` and
  `explained_variance_dropping_full_ppo` active only in `ppo`.

## 2026-05-03 16:42 CST Watch Snapshot

- Latest transition: `2,736,128`.
- Process status: running, detached.
- `rollout/termination_success`: `0`.
- First success crossing: not yet observed.
- `rollout/collisions`: `0`.
- `rollout/out_of_arena`: `0`.
- `rollout/timeouts`: `0`.
- Minimum EV observed in this resume window: `0.8788855075836182`.
- Maximum KL observed in this resume window: `0.01391325518488884`.
- Minimum `safety/h_hard_min`: `4.541052341461182`.
- Latest metrics: EV `0.8825459480285645`, approx KL
  `0.011261692270636559`, value loss `28.97418212890625`,
  `safety/h_hard_min=4.8215250968933105`.

## Interim Interpretation

The phase-aware watchdog patch succeeded at its immediate engineering purpose:
the previous F-015 false-positive halt did not recur after roughly `872k`
additional transitions. Safety is still clean and the run is continuing in
`FULL_PPO`. The open science question is now policy progress: despite stable
KL, finite EV, and clean barriers, no episode termination or success has
occurred yet in the monitored window.

## 2026-05-03 17:40 CST Final Result

- Completion: natural `[DONE][V25] training complete`.
- Final transition: `10,000,000`.
- Resume transition delta: `8,136,320` from `1,863,680` to `10,000,000`.
- Logged metric delta: `8,132,224` from first logged row `1,867,776` to
  `10,000,000`.
- Elapsed wall-clock from metric stream: `3,962.956402725` seconds
  (`66.04927337875` minutes).
- Effective throughput: `2,052.059920318108` transitions/sec.
- Final checkpoint: `runs/v02_phase_aware_resume_20260503T083420Z/checkpoints/final.pt`.
- Final step checkpoint:
  `runs/v02_phase_aware_resume_20260503T083420Z/checkpoints/step_010000000.pt`.
- Final checkpoint SHA-256:
  `4405712474d8a9d46482cf4c71692e33c9d334aec876fb9e758287537c216b21`.

## Final Metrics

- `train/value_explained_variance`: first `0.9661282300949097`, final
  `0.8274534940719604`, min `0.8244217038154602`, max
  `0.9871420860290527`, mean `0.9212568651267975`.
- `train/approx_kl`: first `0.007833650335669518`, final
  `0.008865231648087502`, min `0.0037321471609175205`, max
  `0.01391325518488884`, mean `0.008478587451585437`.
- `train/value_loss`: first `22.187705993652344`, final
  `11.227445602416992`, min `2.479966640472412`, max
  `39.04972839355469`, mean `13.575838461435795`.
- `train/grad_norm`: first `0.650026261806488`, final
  `0.550621509552002`, min `0.0598815381526947`, max
  `19.288801193237305`, mean `1.060160982988261`.
- `safety/h_hard_min`: first `4.84821081161499`, final
  `2.775484561920166`, min `2.775484561920166`, max
  `4.998889446258545`, mean `4.825678850096197`.
- `rollout/cbf_active_rate`: first `0.53125`, final `0.0`, min `0.0`,
  max `0.754638671875`, mean `0.5280987797637771`.
- `rollout/arena_projection_active_rate`: first `0.53125`, final `0.0`,
  min `0.0`, max `0.546875`, mean `0.5277918530998364`.

## Termination Metrics

- `rollout/terminations`: `0`.
- `rollout/termination_success`: `0`.
- `rollout/collisions`: `0`.
- `rollout/out_of_arena`: `0`.
- `rollout/timeouts`: `0`.

These termination counters are not a valid final success-rate estimate for
this run. The launcher collects `64` steps per rollout while the environment
has `max_steps=500`, and `collect_rollout()` resets the vector environment at
the start of each call. Most episodes therefore never reach a natural terminal
condition during training telemetry. A post-training evaluation must run full
`500`-step episodes without this short-rollout reset artifact.

## Final Interpretation

Amendment 5 succeeded as an engineering fix: phase-aware watchdogs eliminated
the F-015 false halt, the run completed the 10M target, KL stayed controlled,
the critic stayed useful, and Layer-2 safety remained clean with no observed
collisions or out-of-arena events. The open scientific question remains policy
task success, because training-time termination telemetry is structurally
uninformative under the current rollout lifecycle.

## Evidence

- `configs/v25_watchdog_envelope.yaml`
- `scripts/train_ppo_v25.py`
- `tests/test_phase_aware_watchdogs.py`
- `AMENDMENT_LOG.md`
- `configs/v25_ppo_pre_registration.yaml`
