# Changelog

## v0.1.1 - 2026-05-01

- Changed pre-registration metric and artifact parsing to raise `ParseError`
  on missing, non-numeric, or non-finite required fields instead of silently
  treating malformed data as a failed comparison.
- Added compatibility exports for `parallelcbf.ops.pre_registration`.
- Added V25 Stage 0 evaluation tooling for the Transformer BC checkpoint.

## v0.1.0 - 2026-04-30

- Added formal `SafeEnv`, `SafetyFilter`, `Algorithm`, and ops ABCs.
- Added `SafetyWrapper`, `Toy2DAvoidanceEnv`, and `Toy2DAvoidanceVecEnv`.
- Added NumPy `NaiveDistanceCBF` and CPU PyTorch `DualBarrierCBF`.
- Added watchdog registry, pre-registration artifacts, forensics buffer, atomic
  checkpointing, and pydantic telemetry schema.
- Added pytest, Hypothesis, mypy, and coverage-oriented CI scaffolding.

### Internal Tooling

- Added `scripts/verify_bc_artifact.py` after the first V23 rescue aggregator
  omitted `attempt_distribution`, causing the V24 dataset audit to halt before
  BC pre-training. The verifier asserts the 31,415-episode count, validates
  attempt-distribution metadata, checks the first 100 episodes for non-finite
  tensors, and prints a SHA-256 digest before V24 launch.
