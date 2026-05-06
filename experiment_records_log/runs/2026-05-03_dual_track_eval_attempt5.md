# Dual-Track Eval Completion and Attempt-5 Ignition

Status: `COMPLETED`

## Purpose

Execute the post-review dual track:

- Complete final CPU evaluation after filtering invalid `wide_*` scenes against
  the TripleBarrier Layer-2 arena precondition.
- Launch v0.2-attempt-5 after fixing F-016, the Layer-3 rollout reset defect.

## Track 1: Eval Artifact Completion

Implemented `_validate_scene_within_arena` in `scripts/v25_evaluate_final.py`
with `MARGIN=0.05`. The evaluator now clamps invalid `wide_*` start/goal
coordinates into the inscribed-square Layer-2 envelope before execution. For
the previously failing `wide_single_static` scene, the invalid `x=-2.2` start is
shrunk to `x=-2.0713203435596426`, matching
`arena_radius / sqrt(2) - 0.05`.

The final detached CPU evaluation completed all `600` episodes and produced
`aggregate.json`.

Final aggregate:

- `total_episodes=600`
- `successes=0`
- `success_rate=0.0`
- `termination_counts`: `timeout=600`, `success=0`, `collision=0`,
  `out_of_arena=0`, `layer2_violation=0`
- `layer2_residuals.total_numerical_residuals=22576`
- `layer2_residuals.violations_above_tolerance=0`

Interpretation: the policy never reached the goal, but the Layer-2 shield kept
the evaluation clean under the clamp-fixed scene envelope. The same blind
policy failed safely for all 600 episodes.

## Track 2: F-016 Fix

Root cause: `collect_rollout` reset the environment at the start of every
rollout and reset all vector lanes when any lane finished. This truncated
episodes near the context boundary and erased Transformer hidden-state
continuity.

Implemented Layer-3-only fix in
`parallelcbf/algorithms/causal_transformer_ppo.py`:

- Persist `_last_observation` and `_last_hidden_state` across rollout calls.
- Persist per-lane episode ids, timesteps, returns, and lengths.
- Do not call `env.reset()` at rollout start unless state is absent or batch
  shape changes.
- Reset only naturally done lanes via `reset_done(done_mask)`.
- Clear hidden-state obs buffers only for done lanes.
- Persist rollout state fields in future checkpoints.

Added launcher support in `scripts/train_ppo_v25.py` for
`VectorSafetyWrapper.reset_done(done_mask)`.

## Validation

- Focused pytest:
  `tests/test_v25_evaluate_final_r7v2.py` and
  `/home/smartlab/parallelcbf_dev/tests/test_rollout_buffer.py`:
  `12 passed`.
- Focused strict mypy:
  `scripts/v25_evaluate_final.py`,
  `parallelcbf/algorithms/causal_transformer_ppo.py`, and
  `tests/test_rollout_buffer.py`:
  `Success: no issues found in 3 source files`.

## Amendment 6

Reason: `f016_rollout_reset_fix`.

New pre-registration manifest SHA:
`7c57bfc80f4faaa2247cf354bad4d4b367b1f80355552a13378ba578a5b09e6f`.

## Track 2 Result

The detached v0.2-attempt-5 burn started from BC, not resume, and ran until a
watchdog halt at `global_step=1,007,616`.

Observed outcome:

- `phase=critic_warmup` throughout the run.
- `rollout/termination_success` remained `0`.
- `rollout/episode_length_mean` reached `500.0`, which confirms the rollout
  reset bug was fixed.
- `explained_variance_stuck_negative` triggered the halt.
- `approx_kl` stayed at `0.0` during warmup, so the stop was not a KL shock.
- `safety/h_hard_min` remained positive throughout.

Checkpoint evidence:

- `last_safe_step.pt` preserved.
- `failed_step_01007616.pt` written.
- Forensics captured under
  `runs/v02_attempt5_20260503T114723Z/checkpoints/forensics/`.

## Final Verdict

F-016 is fixed. The CPU evaluation is complete. The retrain did not reach
full PPO because the critic warmup EV watchdog still fired first, so the next
iteration should focus on critic observability rather than rollout plumbing.
