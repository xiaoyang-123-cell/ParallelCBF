#!/usr/bin/env python3
"""Run V25 causal Transformer behavior-cloning training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from parallelcbf.algorithms import CausalTransformerBC, CausalTransformerConfig
from parallelcbf.dataloaders import SlidingWindowDataset
from parallelcbf.ops import DefaultWatchdogRegistry, ThresholdWatchdog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints/v25_transformer_bc"))
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--obs_dim", type=int, default=0)
    parser.add_argument("--action_dim", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-2)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    return parser.parse_args()


def infer_obs_dim(dataset_path: Path) -> int:
    payload = torch.load(dataset_path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise TypeError("dataset artifact must contain an episodes list")
    episodes = payload["episodes"]
    if len(episodes) == 0:
        raise ValueError("dataset contains no episodes")
    first = episodes[0]
    if not isinstance(first, dict) or not isinstance(first.get("obs"), torch.Tensor):
        raise TypeError("episode 0 must contain an obs tensor")
    return int(first["obs"].shape[1])


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    obs_dim = args.obs_dim if args.obs_dim > 0 else infer_obs_dim(args.dataset)
    full_dataset = SlidingWindowDataset(
        args.dataset,
        context_length=args.context_length,
        stride=args.stride,
        obs_dim=obs_dim,
    )
    train_dataset, val_dataset = full_dataset.split_by_episode(val_frac=args.val_frac, seed=args.seed)
    config = CausalTransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        context_length=args.context_length,
        obs_dim=obs_dim,
        action_dim=args.action_dim,
    )
    algorithm = CausalTransformerBC(config, device=torch.device(args.device))
    print(f"dataset={args.dataset}")
    print(f"obs_dim={obs_dim}")
    print(f"train_windows={len(train_dataset)}")
    print(f"val_windows={len(val_dataset)}")
    print(f"parameters={sum(parameter.numel() for parameter in algorithm.model.parameters())}")
    watchdog = DefaultWatchdogRegistry()
    watchdog.register(ThresholdWatchdog("train_loss", 1.0e6, label="Train Loss Explosion"))
    watchdog.register(ThresholdWatchdog("grad_norm", 1.0e4, label="Grad Norm Blowup"))
    watchdog.register(ThresholdWatchdog("val_overfit_ratio", 5.0, label="Val Overfit"))
    print("=" * 60)
    print("WATCHDOG REGISTRY (V25 BC training):")
    for registered_watchdog in watchdog.list_registered():
        metric = getattr(registered_watchdog, "metric", "unknown")
        threshold = getattr(registered_watchdog, "threshold", "unknown")
        when = getattr(registered_watchdog, "when", "unknown")
        print(f"  - {registered_watchdog.name}: metric={metric}, threshold={threshold}, when={when}")
    print("=" * 60)
    report = algorithm.learn(
        train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        watchdogs=watchdog,
        early_stop_patience=args.early_stop_patience,
    )
    print(f"epochs_run={report.epochs_run}")
    print(f"steps_run={report.steps_run}")
    print(f"train_losses={list(report.train_losses)}")
    print(f"val_losses={list(report.val_losses)}")
    print(f"best_val_loss={report.best_val_loss}")
    print(f"checkpoint_paths={[str(path) for path in report.checkpoint_paths]}")
    print(f"best_checkpoint_path={report.best_checkpoint_path}")
    print(f"halted={report.halted}")
    print(f"halt_reason={report.halt_reason}")


if __name__ == "__main__":
    main()
