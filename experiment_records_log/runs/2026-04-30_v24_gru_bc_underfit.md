# V24 GRU BC Underfit

Status: `FAIL`

## Purpose

Train a GRU behavior-cloning anchor on the accepted V23 50K dataset and determine whether recurrent BC is sufficient as the policy prior.

## Setup

- Dataset: `data/v23_oversampling_50k_bc.pt`
- Checkpoint: `checkpoints/v24_gru_bc_anchor.pt`
- Report: `v24_gru_bc_anchor_report.md`
- Architecture: `RNNModel(GRU hidden=128, mlp=[64], action_dim=3)`

## Result

- Dataset audit passed with `31,415` episodes and `4,109,264` timesteps.
- Final BC loss after 30 epochs: `0.059459819820711426`.
- Zero-action fraction: `0.0`; shape/Nan checks passed.
- Status: `FAIL` due severe underfit relative to dataset scale.

## Diagnosis

The dataset was not the immediate culprit: audits passed, BPTT continuity held, and action data were non-degenerate. The GRU had only about 65K parameters and did not have enough modeling capacity for the long-horizon, mixed-scene demonstration corpus.

## Decision

Pivot to V25 pure causal Transformer BC with substantially larger context-modeling capacity.

## Evidence

- `v24_gru_bc_anchor_report.md`
- `v24_phase_alpha_safety_gates_report.md`

