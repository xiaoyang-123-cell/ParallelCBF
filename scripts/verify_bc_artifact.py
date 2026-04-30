#!/usr/bin/env python3
"""Verify the rescued V23 BC artifact before V24 consumes it."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch


EXPECTED_EPISODES = 31_415


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_finite_tensor_tree(value: Any, *, prefix: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite tensor detected at {prefix}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tensor_tree(child, prefix=f"{prefix}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_tensor_tree(child, prefix=f"{prefix}[{index}]")


def verify(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("BC artifact must be a dictionary")
    episodes = payload.get("episodes")
    metadata = payload.get("metadata")
    if not isinstance(episodes, list):
        raise TypeError("BC artifact must contain an episodes list")
    if not isinstance(metadata, dict):
        raise TypeError("BC artifact must contain metadata")
    if len(episodes) != EXPECTED_EPISODES:
        raise ValueError(f"episode count mismatch: expected={EXPECTED_EPISODES} actual={len(episodes)}")

    attempt_distribution = metadata.get("attempt_distribution")
    if not isinstance(attempt_distribution, dict):
        raise TypeError("metadata.attempt_distribution must be present")
    total = sum(float(value) for value in attempt_distribution.values())
    if abs(total - 1.0) > 1.0e-9:
        raise ValueError(f"attempt_distribution must sum to 1.0, got {total}")

    for index, episode in enumerate(episodes[:100]):
        assert_finite_tensor_tree(episode, prefix=f"episodes[{index}]")

    digest = sha256_file(path)
    return {
        "path": str(path),
        "episodes": len(episodes),
        "attempt_distribution": attempt_distribution,
        "attempt_distribution_sum": total,
        "first_100_finite": True,
        "sha256": digest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = verify(args.path)
    print(f"[BC VERIFY] path={report['path']}")
    print(f"[BC VERIFY] episodes={report['episodes']}")
    print(f"[BC VERIFY] attempt_distribution={report['attempt_distribution']}")
    print(f"[BC VERIFY] attempt_distribution_sum={report['attempt_distribution_sum']:.12f}")
    print(f"[BC VERIFY] first_100_finite={report['first_100_finite']}")
    print(f"[BC VERIFY] sha256={report['sha256']}")


if __name__ == "__main__":
    main()
