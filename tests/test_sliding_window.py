from __future__ import annotations

from pathlib import Path
from os import PathLike
from typing import IO, Any

import torch

from parallelcbf.dataloaders import SlidingWindowDataset


def _write_episode_dataset(path: Path, lengths: list[int], *, obs_dim: int = 4, action_dim: int = 2) -> None:
    episodes: list[dict[str, object]] = []
    for episode_id, length in enumerate(lengths):
        obs = torch.zeros((length, obs_dim), dtype=torch.float64)
        obs[:, 0] = float(episode_id)
        obs[:, 1] = torch.arange(length, dtype=torch.float64)
        actions = torch.full((length, action_dim), float(episode_id), dtype=torch.float64)
        waypoint_index = torch.arange(length, dtype=torch.int64)
        episodes.append(
            {
                "obs": obs,
                "actions": actions,
                "waypoint_index": waypoint_index,
                "scene_type": episode_id % 4,
                "episode_id": 1000 + episode_id,
            }
        )
    torch.save({"episodes": episodes, "metadata": {"source": "test"}}, path)


def test_no_cross_episode_leakage(tmp_path: Path) -> None:
    artifact = tmp_path / "toy_bc.pt"
    _write_episode_dataset(artifact, [6, 7, 8])

    dataset = SlidingWindowDataset(artifact, context_length=4, stride=2)

    assert len(dataset) == 7
    for index in range(len(dataset)):
        sample = dataset[index]
        episode_id = sample["episode_id"] - 1000
        observations = sample["observations"]
        actions = sample["actions"]
        waypoints = sample["waypoint_index"]
        start_index = sample["start_index"]
        assert torch.all(observations[:, 0] == float(episode_id))
        assert torch.all(actions == float(episode_id))
        assert torch.equal(waypoints, torch.arange(start_index, start_index + 4, dtype=torch.int64))


def test_train_val_episode_disjoint(tmp_path: Path) -> None:
    artifact = tmp_path / "toy_bc.pt"
    _write_episode_dataset(artifact, [8] * 10)
    dataset = SlidingWindowDataset(artifact, context_length=4, stride=2)

    train_dataset, val_dataset = dataset.split_by_episode(val_frac=0.2, seed=42)

    train_ids = set(train_dataset.episode_ids)
    val_ids = set(val_dataset.episode_ids)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(dataset.episode_ids)
    assert train_dataset.window_count > 0
    assert val_dataset.window_count > 0


def test_short_episode_skipped(tmp_path: Path) -> None:
    artifact = tmp_path / "toy_bc.pt"
    _write_episode_dataset(artifact, [3, 4, 5, 8])

    dataset = SlidingWindowDataset(artifact, context_length=5, stride=1)

    sampled_episode_ids = {dataset[index]["episode_id"] for index in range(len(dataset))}
    assert sampled_episode_ids == {1002, 1003}
    assert dataset.window_count == 5


def test_torch_load_requests_mmap(tmp_path: Path, monkeypatch: Any) -> None:
    artifact = tmp_path / "toy_bc.pt"
    _write_episode_dataset(artifact, [6])
    original_load = torch.load
    mmap_values: list[object] = []

    def recording_load(
        f: str | PathLike[str] | IO[bytes],
        map_location: object = None,
        pickle_module: object = None,
        *,
        weights_only: bool | None = None,
        mmap: bool | None = None,
        **pickle_load_args: object,
    ) -> object:
        _ = pickle_module
        mmap_values.append(mmap)
        return original_load(
            f,
            map_location=map_location,  # type: ignore[arg-type]
            weights_only=weights_only,
            mmap=mmap,
            **pickle_load_args,
        )

    monkeypatch.setattr(torch, "load", recording_load)

    dataset = SlidingWindowDataset(artifact, context_length=4, stride=1)

    assert dataset.window_count == 3
    assert mmap_values == [True]
