#!/usr/bin/env python3
"""Aggregate internal V23 BC shards into one CPU torch artifact.

This script is intentionally excluded from the public v0.1 API. It exists to
rescue the internal V23 sharded data collection after the original manifest
builder ran in an environment without torch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: dict[str, Any], output_path: Path) -> None:
    """Write a torch artifact using `.tmp -> fsync -> rename`."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with tmp_path.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, output_path)
    directory_fd = os.open(output_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_shard(path: Path) -> dict[str, Any]:
    """Load and minimally validate one V23 shard."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a dictionary")
    if "episodes" not in payload or "metadata" not in payload:
        raise KeyError(f"{path} must contain episodes and metadata")
    episodes = payload["episodes"]
    metadata = payload["metadata"]
    if not isinstance(episodes, list):
        raise TypeError(f"{path} episodes must be a list")
    if not isinstance(metadata, dict):
        raise TypeError(f"{path} metadata must be a dict")
    expected = metadata.get("episodes_accepted")
    if isinstance(expected, int) and expected != len(episodes):
        raise ValueError(f"{path} accepted count mismatch: metadata={expected} actual={len(episodes)}")
    return payload


def aggregate(shard_paths: list[Path], output_path: Path) -> int:
    """Aggregate shard episodes and return the verified episode count."""

    all_episodes: list[Any] = []
    shard_metadata: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        shard = load_shard(shard_path)
        episodes = shard["episodes"]
        metadata = shard["metadata"]
        all_episodes.extend(episodes)
        shard_metadata.append(metadata)

    artifact = {
        "episodes": all_episodes,
        "metadata": {
            "source": "v23_sharded_bc_rescue",
            "num_shards": len(shard_paths),
            "episodes_accepted": len(all_episodes),
            "shard_paths": [str(path) for path in shard_paths],
            "shards": shard_metadata,
        },
    }
    atomic_torch_save(artifact, output_path)
    verified = torch.load(output_path, map_location="cpu", weights_only=False)
    if not isinstance(verified, dict) or "episodes" not in verified:
        raise TypeError("verified artifact did not contain episodes")
    verified_count = len(verified["episodes"])
    if verified_count != len(all_episodes):
        raise ValueError(f"verified count mismatch: expected={len(all_episodes)} actual={verified_count}")
    return verified_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=os.environ.get("PARALLELCBF_V23_SHARD_ROOT"),
        required=os.environ.get("PARALLELCBF_V23_SHARD_ROOT") is None,
        help="Directory containing shard_XX_vY.pt files. Can also be set with PARALLELCBF_V23_SHARD_ROOT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=os.environ.get("PARALLELCBF_V23_BC_OUTPUT"),
        required=os.environ.get("PARALLELCBF_V23_BC_OUTPUT") is None,
        help="Output .pt artifact. Can also be set with PARALLELCBF_V23_BC_OUTPUT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_paths = sorted(args.shard_root.glob("shard_*_v*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard artifacts found under {args.shard_root}")
    count = aggregate(shard_paths, args.output)
    print(f"[V23 AGGREGATE] wrote {args.output} episodes={count} shards={len(shard_paths)}")


if __name__ == "__main__":
    main()
