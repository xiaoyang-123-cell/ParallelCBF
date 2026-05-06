# Amendment Log

## 2026-05-03T13:34:36Z - Amendment 7 / Attempt-6 Potential-Based Reward Shaping

Reason: `vs_ccd_variance_starvation_pbrs_diagnostic`.

R-10 was resolved as Scenario A: `eta_cbf * cbf_delta_norm` was already active
in attempt-5 PPO rollout collection as part of the training reward baseline
(`env_reward - eta_cbf * cbf_delta_norm`). Attempt-6 does not introduce or
change that penalty. The only controlled intervention is the
Ng-Harada-Russell potential-based shaping increment:
`gamma * Phi(next_pre_reset_state) - Phi(current_state)`.

Pinned constants: `K_d=1.0`, `K_v=0.5`, `V_target=1.0`, `eta_cbf=0.05`, and
`gamma=0.99`. The implementation asserts that shaping gamma matches PPO gamma.
Terminal transitions compute `Phi(next_state)` from the post-step, pre-reset
safety state. Zero-distance cases are explicitly tested to avoid NaNs.

Validation before ignition:
`pytest -q tests` from `/home/smartlab/parallelcbf_dev` reported
`98 passed, 1 skipped`; focused rollout/PBRS tests reported `15 passed`; strict
mypy on the touched PPO, rollout-test, and launcher files reported
`Success: no issues found in 3 source files`.

New pre-registration manifest SHA:
`80cc153947515063c529a07d7aa3e412094ea9218f7284b301568eb3ae513cc9`.

## 2026-05-03T11:44:59Z - Amendment 6 / F-016 Rollout Reset Fix

Reason: `f016_rollout_reset_fix`.

Post-final-eval review isolated a Layer-3 collection defect: `collect_rollout`
reset the vector environment at every rollout boundary and reset all lanes
whenever any lane finished. This artificially chopped episodes around the
Transformer context length and erased hidden-state continuity, explaining why
`rollout/termination_success` could remain at zero even after otherwise stable
training.

The fix persists `_last_observation`, `_last_hidden_state`, per-lane episode
ids, timesteps, returns, and lengths across rollout calls. Rollout collection no
longer calls `env.reset()` at the start unless no persisted state exists or the
vector batch shape changes. Natural done signals reset only their lane through
`reset_done(done_mask)`, and the Transformer hidden obs buffer is cleared only
for those done lanes. New checkpoints also persist the rollout state fields.

Focused validation:
`pytest tests/test_v25_evaluate_final_r7v2.py tests/test_rollout_buffer.py`
reported `12 passed`, and strict mypy on the touched eval/PPO/test files
reported `Success: no issues found in 3 source files`.

New pre-registration manifest SHA:
`7c57bfc80f4faaa2247cf354bad4d4b367b1f80355552a13378ba578a5b09e6f`.

## 2026-05-03T08:31:50Z - Amendment 5 / F-015 Phase-Aware Watchdog Rules

Reason: `phase_aware_watchdog_rules`.

The Slipper resume demonstrated that the PPO handoff itself was healthy:
`max(train/approx_kl)=0.01633099466562271`, final EV was
`0.9573695063591003`, and safety remained clean with zero collisions and zero
out-of-arena events. The halt was a regime-blind watchdog false alarm:
`value_loss_increasing` fired in FULL_PPO on a tiny positive rolling
`value_loss_slope=0.004083183484199719`.

The watchdog envelope now requires explicit `active_phases` on every rule.
`value_loss_increasing` is replaced by `value_loss_increasing_warmup`, active
only in `CRITIC_WARMUP`, relaxed to `threshold_slope=1.0e-3`, and downgraded to
`WARN`. A new `explained_variance_dropping_full_ppo` HALT rule is active only
in `FULL_PPO`, with `train/value_explained_variance < 0.5` sustained for
`200,000` steps.

The resume checkpoint is
`runs/v02_slipper_resume_20260503T064452Z/checkpoints/last_safe_step.pt`, which
preserves the clean Slipper-stabilized FULL_PPO state before the false-positive
halt.

New pre-registration manifest SHA:
`bec97d2e2672588a135b4df09b74386bf4596b44ff9c854c258765c2cf91b0e5`.

## 2026-05-03T06:42:20Z - PPO Handoff Slipper and Nominal Log-Prob Accounting

Gate V2 opened correctly, but the first full-PPO resume exposed a handoff
shock: the rollout entered `phase=ppo` immediately and early forensics showed
very large policy displacement symptoms (`approx_kl` previously reported near
the 1.8 range, clip saturation, and large first-PPO grad norms).

The amendment adds a PPO Handoff Slipper: actor LR linearly ramps from `0` to
`cfg.lr_actor` over the first `50,000` post-transition environment steps, and
PPO `clip_eps` linearly ramps from `0.05` to `0.2` over the same interval. We
did not change `max_grad_norm`, the KL-anchor lambda schedule, or advantage
normalization.

Rollout buffer audit: PPO stores the nominal actor action in `RolloutStep.action`
and separately stores `u_nominal`/`u_safe`; the environment receives the
SafetyWrapper-projected `safe_action`. Verdict: Option i, not Option ii. During
the audit we also found a Layer-3 PPO accounting bug: `old_log_prob` was a zero
placeholder. It now records the true log-probability of the nominal action
under the rollout policy distribution, so `approx_kl` starts from a meaningful
near-zero baseline rather than a bookkeeping artifact.

Checkpoint decision: `runs/v02_gate_v2_resume_20260502T121701Z/checkpoints/last_safe_step.pt`
was rejected because it is already `phase=ppo` at `global_step=1,900,544`,
after the handoff shock. Resume will fall back to the pre-transition critic
warmup checkpoint at
`runs/v02_attempt2_20260502T114503Z/checkpoints/last_safe_step.pt`.

New pre-registration manifest SHA:
`2e5dffd6271c93a356c5eb27d8eee46221d4707378119319f499b0da923fb551`.

## 2026-05-02T12:08:37Z - Gate V2 Correction After V0.2 Attempt-2 Halt

The V1 critic warmup gate used value-loss plateau as the release condition.
Attempt-2 proved that this was the wrong proxy: the critic reached strong
explained variance while value loss was still improving, so the plateau test
kept the actor frozen even though the critic was ready enough for full PPO.

Gate V2 replaces the plateau proxy with an explained-variance breakthrough:
after global step 500,000, rolling-50k EV must stay at or above 0.60 for
20,000 steps, and the rolling EV slope over the last 100k steps must be at
least -1.0e-6. This directly measures critic usefulness rather than waiting for
the loss curve to flatten.

Attempt-2 evidence: `runs/v02_attempt2_20260502T114503Z` halted at 1,503,232
transitions with EV peak 0.7428237199783325, final EV 0.731096625328064,
final value loss 319.74920654296875, and zero collisions/out-of-arena events.

Resume caveat: the legacy `last_safe_step.pt` created before this amendment
contains model weights, phase, critic loss history, and last update step, but
does not contain Adam optimizer state or a rollout buffer. The resume path will
restore every available legacy field and all new checkpoints will include
optimizer state and training global step.
