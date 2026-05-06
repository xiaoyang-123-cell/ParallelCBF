# Attempt-6 PBRS Diagnostic Readback and Implementation

Status: `HALTED`

## Purpose

Prepare v0.2-attempt-6 to test the VS-CCD hypothesis: the shielded policy is
safe but over-protected, creating reward/return variance starvation. The
controlled intervention is potential-based reward shaping only.

## R-10 Resolution

R-10 was resolved as Scenario A. The `eta_cbf * cbf_delta_norm` term was already
active in attempt-5 PPO rollout collection as part of the training reward
baseline:

`baseline_reward = env_reward - eta_cbf * cbf_delta_norm`

Attempt-6 keeps `eta_cbf=0.05` unchanged. It adds only:

`gamma * Phi(next_pre_reset_state) - Phi(current_state)`

## Implementation Pins

- `Phi(s) = -K_d * ||position - goal|| - K_v * max(0, V_target - dot(velocity, direction_to_goal))`.
- `K_d=1.0`, `K_v=0.5`, `V_target=1.0`.
- `gamma=0.99`, with assertion that shaping gamma matches PPO gamma.
- Terminal `Phi(next_state)` is computed from the post-step, pre-reset safety
  state.
- PPO hyperparameters remain frozen from attempt-5.
- Final evaluation continuity remains the same 600-episode configuration from
  attempt-5.

## Observability

Added rollout metrics:

- `rollout/potential_phi_std`
- `rollout/potential_delta_std`
- `rollout/shaping_reward_share`
- `rollout/shaping_reward_per_step`

Support metrics also log mean Phi, mean PBRS delta, baseline reward mean, CBF
penalty mean, and final shaped reward mean.

## Validation

- Focused rollout/PBRS tests: `15 passed`.
- Strict mypy on touched files:
  `Success: no issues found in 3 source files`.
- Full test suite from `/home/smartlab/parallelcbf_dev`: `98 passed, 1 skipped`.

## Amendment 7

Reason: `vs_ccd_variance_starvation_pbrs_diagnostic`.

Manifest SHA:
`80cc153947515063c529a07d7aa3e412094ea9218f7284b301568eb3ae513cc9`.

## Diagnostic Burn

Run directory:
`runs/v02_attempt6_pbrs_diag_20260503T133731Z`.

The 1M-step diagnostic burn launched successfully in detached headless mode,
but halted early by watchdog at `151,552` transitions:

- Phase at halt: `critic_warmup`.
- Halt reason: `explained_variance_stuck_negative`.
- `train/value_explained_variance` max: `0.014817774295806885`.
- Final `train/value_explained_variance`: `-1.1920928955078125e-07`.
- Final `train/value_loss`: `761.89697265625`.
- `rollout/termination_success` total: `0`.
- `rollout/collisions` total: `0`.
- `rollout/out_of_arena` total: `0`.
- `rollout/timeouts` observed before halt: `256`.
- Minimum `safety/h_hard_min`: `2.0206384658813477`.
- Mean `rollout/arena_projection_active_rate`: `0.9349662162162162`.
- Final `rollout/arena_projection_active_rate`: `1.0`.
- Mean `rollout/shaping_reward_share`: `0.013324399170565408`.
- Final `rollout/shaping_reward_share`: `0.012140181674443742`.
- Max `rollout/potential_delta_std`: `0.027390150323112067`.

Artifacts:

- `metrics.jsonl`
- `launch.log`
- `checkpoints/failed_step_00151552.pt`
- `checkpoints/last_safe_step.pt`
- `checkpoints/step_000102400.pt`
- `checkpoints/forensics/forensics_20260503_133846.json`

## Interpretation

The implementation and launch path are clean: the manifest guard passed,
the actor stayed frozen in critic warmup, `approx_kl=0`, and Layer 2 remained
safe with no collisions or out-of-arena events.

The first PBRS constants did not fix critic variance starvation. The shaping
signal was present but small, averaging about `1.33%` of final training reward,
and arena projection was active in most rollout windows. This suggests the
diagnostic hypothesis should move to the authorized one-shot scaling adjustment
for `K_d` and `K_v`, or to a deeper check of why arena projection collapses
state diversity after early rollout windows.

## Next Step

Use the R-5 authorization for exactly one `K_d`/`K_v` scale adjustment within
`[0.25, 4.0]`, or pause for Architect review if changing the shaping magnitude
would weaken the controlled negative-result narrative.
