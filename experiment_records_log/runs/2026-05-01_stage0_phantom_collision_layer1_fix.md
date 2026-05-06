# Stage-0 Phantom Collision and Layer-1 Fix

Status: `PASS`

## Purpose

Evaluate the V25 BC checkpoint in Stage 0 and determine why the initial evaluation appeared to show a 100% collision rate despite Layer-2 safety.

## Setup

- Evaluation report: `logs/v25_stage0_eval.json`
- Forensic console: `logs/forensic_console.log`
- Checkpoint: `checkpoints/v25_transformer_bc_full/best.pt`
- Episodes: `200` deterministic Stage-0 episodes (`100` open, `100` single)

## Result

- Final corrected Stage-0 collision rate: `0.0`.
- Overall success: `0.0`.
- Termination breakdown: `total_success=0`, `total_collision=0`, `total_out_of_arena=200`, `total_timeout=0`.
- Per-scene: open `100/100` out-of-arena; single `100/100` out-of-arena.
- Forensics near phantom collision showed `h_hard_before_min=20042.73828125` and `h_hard_next_min=20036.388671875`, so Layer 2 was not failing.

## Diagnosis

The environment conflated arena exit or dummy sentinel interactions with physical obstacle collision. The policy was drifting out of the arena; it was not colliding with real obstacles.

## Decision

Separate termination reasons in Layer 1 and treat arena exit as policy drift. Do not modify Layer 2 or the Transformer for this specific bug.

## Evidence

- `logs/forensic_console.log`
- `logs/v25_stage0_eval.json`

