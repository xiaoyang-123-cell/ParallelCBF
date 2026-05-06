# PPO Handoff Slipper Resume Patch

Status: `HALTED`

## Purpose

Implement the Chief Architect's PPO Handoff Slipper after Gate V2 opened full
PPO but the handoff produced an abrupt KL/clip shock. The goal is to preserve
the good TripleBarrier and Gate V2 behavior while making the actor release
gradual enough for the first 50k post-transition steps.

## Setup

- Algorithm path: `/home/smartlab/parallelcbf_dev/parallelcbf/algorithms/causal_transformer_ppo.py`
- Launcher path: `scripts/train_ppo_v25.py`
- Manifest: `configs/v25_ppo_pre_registration.yaml`
- New manifest SHA: `2e5dffd6271c93a356c5eb27d8eee46221d4707378119319f499b0da923fb551`
- Rejected resume checkpoint: `runs/v02_gate_v2_resume_20260502T121701Z/checkpoints/last_safe_step.pt`
- Fallback checkpoint: `runs/v02_attempt2_20260502T114503Z/checkpoints/last_safe_step.pt`

## Result

- Added actor LR warmup: `0 -> cfg.lr_actor` over `50,000` post-transition environment steps.
- Added PPO clip warmup: `0.05 -> 0.2` over `50,000` post-transition environment steps.
- Added telemetry for `train/approx_kl`, `train/clip_fraction`, `train/actor_lr`, `train/clip_epsilon`, and `train/post_transition_steps`.
- Targeted tests passed: `26 passed, 1 skipped`.
- Local unchanged-policy smoke produced `approx_kl_smoke=0.0` with `clip_eps_step0=0.05`.

## 2026-05-03 14:45 CST Resume Launch

- Run directory: `runs/v02_slipper_resume_20260503T064452Z`
- PID: `7536`
- Detachment: `TTY=?`, session ID equals PID.
- Resume checkpoint: `runs/v02_attempt2_20260502T114503Z/checkpoints/last_safe_step.pt`
- Manifest SHA: `2e5dffd6271c93a356c5eb27d8eee46221d4707378119319f499b0da923fb551`
- Gate transition: `critic_warmup -> full_ppo` at `global_step=1507328`, `ev=0.647888`, `ev_slope_100k=2.7114e-07`.
- 50k Slipper watch: `max(train/approx_kl)=0.01633099466562271`, below the `0.05` halt threshold.
- Last in-window Slipper row: `post_transition_steps=49152`, `clip_epsilon=0.19745600000000002`, `actor_lr=0.000294912`.
- Latest sampled row after launch: `rollout/transitions=1683456`, `train/approx_kl=0.016057971864938736`, `EV=0.9611353278160095`, `collisions=0`, `out_of_arena=0`, `safety/h_hard_min=4.951394557952881`.

## 2026-05-03 14:48 CST Halt

- Final transition: `1,863,680`.
- Halt reason: watchdog halt.
- Failed checkpoint: `runs/v02_slipper_resume_20260503T064452Z/checkpoints/failed_step_01863680.pt`.
- Forensics: `runs/v02_slipper_resume_20260503T064452Z/checkpoints/forensics/forensics_20260503_064759.json`.
- Final `train/approx_kl`: `0.0063484348356723785`.
- Max observed `train/approx_kl`: `0.01633099466562271`.
- Final EV: `0.9573695063591003`.
- Final value loss: `28.743431091308594`.
- Safety: `collisions=0`, `out_of_arena=0`, `safety/h_hard_min=4.7846245765686035`.
- Likely triggering rule: `value_loss_increasing`, because final `value_loss_slope=0.004083183484199719` crossed the current `> 0` threshold.

## Diagnosis

The buffer audit confirmed PPO stores the nominal actor action, not the
SafetyWrapper-projected safe action. This is Option i. The environment executes
the safe action, while `RolloutStep.action`, `u_nominal`, and `u_safe` remain
available separately for PPO and diagnostics.

During the audit, we found an additional Layer-3 PPO accounting issue:
`old_log_prob` had been stored as a zero placeholder. That can inflate
`approx_kl` even when the policy has not moved. The patch now stores the true
log-probability of the nominal action under the rollout policy distribution.

Checkpoint triage showed the V2 resume `last_safe_step.pt` was already
`phase=ppo` at `global_step=1,900,544`, so it was saved after the handoff shock.
The next resume should fall back to the pre-transition attempt-2 checkpoint.

The Slipper fixed the KL handoff problem, but the current value-loss slope
watchdog is too sensitive for full PPO. It halted on a tiny positive rolling
slope despite healthy KL, high EV, and clean safety. This watchdog likely needs
phase-aware sustained semantics rather than an immediate `> 0` threshold.

## Decision

Do not treat this as a Slipper failure. The next amendment should refine
`value_loss_increasing` into a phase-aware sustained watchdog with a positive
tolerance, while preserving the KL watch and safety gates.

## Evidence

- `AMENDMENT_LOG.md`
- `configs/v25_ppo_pre_registration.yaml`
- `configs/v25_ppo_pre_registration.seal.json`
- `/home/smartlab/parallelcbf_dev/tests/test_actor_critic.py`
- `/home/smartlab/parallelcbf_dev/tests/test_rollout_buffer.py`
