# Final Evaluation R-7 Halt

Status: `HALTED`

## Purpose

Execute the post-Architect final V25/V0.2 evaluation after the 10M-step PPO
run completed. The evaluation was required to pin refinements R-1 through R-7,
generate final paper artifacts, and run `500` deterministic plus `100`
stochastic CPU-only episodes.

## Pre-Execution Pins

- R-1: Hidden state reset to `model.initial_hidden_state(1, ...)` at every
  episode start.
- R-2: Success radius pinned to `Toy2DConfig().goal_radius = 0.15`.
- R-3: Stochastic action seed pinned to `env_seed + 10000`.
- R-4: Termination classes pinned to `layer2_violation`, `collision`,
  `out_of_arena`, `success`, `timeout`.
- R-5: Stochastic seeds `500..599` paired to deterministic scene specs by
  `seed % 500`.
- R-6: CPU-only single-thread eval with `torch.set_num_threads(1)` and
  `torch.use_deterministic_algorithms(True)`.
- R-7: Any Layer-2 violation writes JSON/NPZ autopsy and exits with code `2`.

## Execution

- Script: `scripts/v25_evaluate_final.py`.
- Checkpoint:
  `runs/v02_phase_aware_resume_20260503T083420Z/checkpoints/final.pt`.
- Output directory: `runs/v25_eval_final_20260503T100118Z`.
- Manifest: `runs/v25_eval_final_20260503T100118Z/manifest.json`.
- Exit code: `2`.

## Halt Autopsy

- Autopsy JSON:
  `runs/v25_eval_final_20260503T100118Z/autopsy/layer2_violation_ep0000_step0016.json`.
- Autopsy NPZ:
  `runs/v25_eval_final_20260503T100118Z/autopsy/layer2_violation_ep0000_step0016.npz`.
- Episode: `0`.
- Step: `16`.
- Scene: `open`.
- Start: `[-2.0, 0.0]`.
- Goal: `[2.0, 0.0]`.
- Obstacle: `[100.0, 100.0]`.
- Nominal action: `[[-0.6179484724998474, -0.22648227214813232]]`.
- Safe action: `[[0.6588190197944641, -0.22648227214813232]]`.
- `h_hard_min`: `20443.8671875`.
- `h_soft_min`: `142.2827606201172`.
- `h_arena_min`: `0.0005056858062744141`.
- `h_arena_stopping_min`: `-5.3783878684043884e-08`.
- `arena_projection_active`: `true`.
- `arena_projection_saturated`: `true`.

## Interpretation

The final evaluation did not reach aggregate metrics because R-7 halted the run
before the first episode completed. The autopsy shows obstacle safety was not
the issue: `h_hard_min` and `h_soft_min` were strongly positive with a dummy
far obstacle. The halt came from the arena stopping barrier, with
`h_arena_min` still positive but `h_arena_stopping_min` slightly negative
(`-5.38e-08`) and `arena_projection_saturated=true`.

This is a paper-grade diagnostic: the final policy pushes toward the arena
boundary in the open scene early in the episode, and the TripleBarrier arena
projection intervenes. The exact halt value is close to numerical tolerance,
so the next review should decide whether R-7 should distinguish physical
Layer-2 violation from tiny stopping-barrier saturation jitter, or whether the
policy/arena barrier should be further calibrated before claiming final
success.

## Missing Artifacts

- `aggregate.json`: not generated because R-7 halted by design.
- `stage0_comparison.json`: not generated because R-7 halted by design.
- `value_calibration.json`: not generated because R-7 halted by design.
- Full `per_episode.jsonl`: empty because the halt occurred before episode 0
  completed.

## R-7v2 Re-Ignition Result

Status: `HALTED`

After the Architect ruled the first halt a numerical false positive, R-7v2 was
implemented with `TOL_LAYER2_HALT = -1.0e-5`:

- `h_min >= 0`: pass.
- `-1.0e-5 <= h_min < 0`: log `severity="numerical"` to
  `numerical_residuals.jsonl` and continue.
- `h_min < -1.0e-5`: write autopsy and exit `2` with `severity="real"`.

Validation passed:

- R-8 synthetic unit tests: `2 passed`.
- `mypy --strict scripts/v25_evaluate_final.py tests/test_v25_evaluate_final_r7v2.py`:
  `Success: no issues found in 2 source files`.

The previous autopsy files were moved to:
`runs/v25_eval_final_20260503T100118Z/halt_autopsies/ep0000_step0016_NUMERICAL_FALSE_POSITIVE/`.

Re-run output directory:
`runs/v25_eval_final_20260503T101705Z`.

The R-7v2 run completed `450` episodes before halting at episode `450`, step
`1`:

- Completed episodes before halt: `450`.
- Partial termination breakdown before halt: `timeout=450`, `success=0`,
  `collision=0`, `out_of_arena=0`.
- Partial scenes completed: `open=125`, `single_static=125`,
  `offset_single_static=125`, `near_goal_open=75`.
- Numerical residuals logged before real halt: `8,727`.
- Numerical residual barrier distribution before real halt:
  `h_arena_stopping=8,726`, `h_arena=1`.

R-7v2 real halt autopsy:

- Autopsy JSON:
  `runs/v25_eval_final_20260503T101705Z/autopsy/layer2_violation_ep0450_step0001.json`.
- Autopsy NPZ:
  `runs/v25_eval_final_20260503T101705Z/autopsy/layer2_violation_ep0450_step0001.npz`.
- Episode: `450`.
- Step: `1`.
- Scene: `wide_single_static`.
- Start: `[-2.2, -0.5]`.
- Goal: `[2.1, 0.25]`.
- Obstacle: `[0.7, 0.2]`.
- `h_hard_min`: `8.680068969726562`.
- `h_soft_min`: `2.2803637981414795`.
- `h_arena_min`: `-0.07617974281311035`.
- `h_arena_stopping_min`: `-0.07617974281311035`.
- Tolerance threshold: `-1.0e-5`.
- Severity: `real`.

Interpretation: this second halt is not numerical noise. It is an evaluation
scene precondition violation. `wide_single_static` starts at `x=-2.2`, while
the TripleBarrier evaluator uses the same inscribed-square arena convention as
training (`arena_radius / sqrt(2) ~= 2.121`). Therefore the episode begins
outside Layer 2's square arena before the policy meaningfully acts. Obstacle
safety is clean, but the scene itself is invalid for this Layer-2 arena.

No final `aggregate.json` was produced because R-7v2 halted by design.
