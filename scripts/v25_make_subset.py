#!/usr/bin/env python3
"""Create a small V23-compatible subset artifact for V25 dry-runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from random import Random
from typing import cast

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/v23_oversampling_50k_bc.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/v23_subset_1k.pt"))
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("first", "random"), default="first")
    return parser.parse_args()


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    payload = torch.load(args.source, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise TypeError("source artifact must contain an episodes list")
    episodes = cast(list[object], payload["episodes"])
    if args.episodes > len(episodes):
        raise ValueError(f"requested {args.episodes} episodes from artifact with {len(episodes)}")
    if args.mode == "random":
        indices = list(range(len(episodes)))
        Random(args.seed).shuffle(indices)
        selected_indices = sorted(indices[: args.episodes])
    else:
        selected_indices = list(range(args.episodes))
    subset_episodes = [episodes[index] for index in selected_indices]
    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {}
    metadata.update(
        {
            "source_artifact": str(args.source),
            "subset_episodes": args.episodes,
            "subset_mode": args.mode,
            "subset_seed": args.seed,
        }
    )
    atomic_torch_save({"episodes": subset_episodes, "metadata": metadata}, args.output)
    verified = torch.load(args.output, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(verified, dict) or not isinstance(verified.get("episodes"), list):
        raise RuntimeError("subset verification failed: malformed output")
    actual_count = len(verified["episodes"])
    if actual_count != args.episodes:
        raise RuntimeError(f"subset verification failed: expected {args.episodes}, got {actual_count}")
    print(f"wrote_subset={args.output}")
    print(f"episodes={actual_count}")


if __name__ == "__main__":
    main()
