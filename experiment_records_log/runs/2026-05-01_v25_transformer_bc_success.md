# V25 Causal Transformer BC Success

Status: `PASS`

## Purpose

Replace the underfit GRU anchor with a causal Transformer behavior-cloning model and test whether capacity/context were the main imitation bottleneck.

## Setup

- Training log: `logs/v25_train_full.log`
- Best checkpoint: `checkpoints/v25_transformer_bc_full/best.pt`
- Dataset: `data/v23_oversampling_50k_bc.pt`
- Validation split: episode-level split, `val_frac=0.05`

## Result

- Epoch 5 train loss: `0.0279513070431`; validation loss: `0.026384708848942458`.
- Best validation loss: `0.0060398161485550345` at epoch 49.
- Final epoch 50 validation loss: `0.006040255930515584`.
- Best checkpoint recorded at `checkpoints/v25_transformer_bc_full/best.pt`.

## Diagnosis

The Transformer broke the GRU floor early and kept improving. This confirms the V24 failure was primarily an architecture/capacity bottleneck rather than corrupted demonstrations.

## Decision

Use the V25 Transformer BC checkpoint as the frozen reference and initialization source for KL-anchored PPO.

## Evidence

- `logs/v25_train_full.log`
- `checkpoints/v25_transformer_bc_full/best.pt`

