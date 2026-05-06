# Attempt-7 KF-005 4x PBRS Final Burn

Status: `PASS`

## Purpose

Run the single authorized R-5 adjustment on top of the PBRS diagnostic setup to
test whether stronger potential shaping could break the KF-005 wall-hugging /
state-degeneracy attractor.

## Setup

- Base lineage: `v0.2-attempt-6` PBRS diagnostic.
- Only allowed tuning: `K_d: 1.0 -> 4.0`, `K_v: 0.5 -> 2.0`.
- `V_target` stayed `1.0`.
- PPO hyperparameters stayed frozen.
- Shape reward remained potential-based only; no extra reward terms were added.
- Run directory:
  `runs/v02_attempt7_kf005_4x_20260503T155101Z`.

## What Changed

- `Phi(s)` kept the same Ng-Harada-Russell form.
- The shaping magnitude was increased by 4x on distance and velocity terms.
- New KF-005 telemetry was logged:
  - `rollout/state_distribution_entropy`
  - `rollout/distance_to_goal_std`
  - `rollout/distance_to_nearest_wall_mean`
  - `rollout/distance_to_nearest_wall_std`
  - `rollout/cbf_arena_intervention_duration_mean`

## Validation

- Full test suite: `98 passed, 1 skipped`.
- Strict mypy on touched files: success.

## Result

- Final status: natural completion.
- Final step: `1,000,000`.
- Final `train/value_explained_variance`: `-0.008253335952758789`.
- Peak `train/value_explained_variance`: `0.592685341835022` at transition
  `897,024`.
- Final `train/value_loss`: `44.68305969238281`.
- Final `train/approx_kl`: `0.0`.
- Final `train/actor_lr`: `0.0`.
- Final `safety/h_hard_min`: `6.262282848358154`.
- Final `rollout/arena_projection_active_rate`: `1.0`.
- Final `rollout/cbf_active_rate`: `1.0`.
- Final `rollout/collisions`: `0`.
- Final `rollout/out_of_arena`: `0`.
- Final `rollout/timeouts`: `0`.
- Final `rollout/termination_success`: `0`.
- Final `rollout/episode_return_std`: `NaN`.
- Final `rollout/potential_delta_std`: `0.0`.
- Final `rollout/shaping_reward_share`: `0.01213502115131008`.

## Interpretation

The 4x shaping strength did not break the KF-005 attractor. The critic became
numerically healthier than in the earlier PBRS run, but the policy still failed
to produce any successful terminations. Safety stayed clean and the arena
projection remained saturated, which is consistent with the trap-state
hypothesis rather than a collision or obstacle-safety failure.

The final log pattern is particularly telling:

- Most of training stayed in `critic_warmup`.
- EV peaked below the 0.6 transition gate.
- Success remained at `0`.
- Collisions and out-of-arena stayed at `0`.
- The agent still spent most rollout windows under full arena projection.

## Consequence

This is a negative-result but paper-useful outcome: the stronger PBRS signal
improves value learning somewhat, yet does not escape the state-distribution
degeneracy. The KF-005 hypothesis remains supported.

## Evidence

- `runs/v02_attempt7_kf005_4x_20260503T155101Z/launch.log`
- `runs/v02_attempt7_kf005_4x_20260503T155101Z/metrics.jsonl`
- `runs/v02_attempt7_kf005_4x_20260503T155101Z/checkpoints/final.pt`
- `runs/v02_attempt7_kf005_4x_20260503T155101Z/checkpoints/last_safe_step.pt`

