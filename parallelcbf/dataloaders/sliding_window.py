"""Episode-safe sliding-window dataset for BC training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import SupportsInt, TypedDict, cast

import torch
from torch.utils.data import Dataset


class EpisodeRecord(TypedDict):
    """Accepted episode record stored in the V23 dataset artifacts."""

    obs: torch.Tensor
    actions: torch.Tensor
    waypoint_index: torch.Tensor
    scene_type: int | str
    episode_id: int


class SlidingWindowSample(TypedDict):
    """One fixed-length training window."""

    observations: torch.Tensor
    actions: torch.Tensor
    waypoint_index: torch.Tensor
    episode_id: int
    scene_type: int | str
    start_index: int


@dataclass(frozen=True, slots=True)
class _WindowIndex:
    episode_position: int
    start_index: int


class SlidingWindowDataset(Dataset[SlidingWindowSample]):
    """Memory-mapped sliding windows that never cross episode boundaries."""

    def __init__(
        self,
        path: str | Path,
        *,
        context_length: int = 64,
        stride: int = 16,
        obs_dim: int | None = None,
        min_episode_length: int | None = None,
        _payload: dict[str, object] | None = None,
        _selected_episode_positions: Sequence[int] | None = None,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self._source_path = Path(path)
        self.context_length = int(context_length)
        self.stride = int(stride)
        self.obs_dim = obs_dim
        self._min_episode_length = int(min_episode_length) if min_episode_length is not None else int(context_length)
        self._payload = _payload if _payload is not None else self._load_payload(self._source_path)
        self._episodes = self._coerce_episodes(self._payload)
        self._selected_episode_positions = self._validate_selected_positions(
            _selected_episode_positions if _selected_episode_positions is not None else range(len(self._episodes))
        )
        self._window_index = self._build_window_index(self._selected_episode_positions)

    @property
    def source_path(self) -> Path:
        """Return the backing artifact path."""

        return self._source_path

    @property
    def metadata(self) -> dict[str, object]:
        """Return artifact metadata if present."""

        raw = self._payload.get("metadata")
        return dict(raw) if isinstance(raw, dict) else {}

    @property
    def episode_ids(self) -> tuple[int, ...]:
        """Return selected episode IDs in deterministic order."""

        return tuple(self._episode_id_at(position) for position in self._selected_episode_positions)

    @property
    def selected_episode_positions(self) -> tuple[int, ...]:
        """Return the selected episode positions within the source artifact."""

        return tuple(self._selected_episode_positions)

    @property
    def selected_episode_count(self) -> int:
        """Return the number of selected episodes."""

        return len(self._selected_episode_positions)

    @property
    def window_count(self) -> int:
        """Return the number of sample windows."""

        return len(self._window_index)

    def __len__(self) -> int:
        return len(self._window_index)

    def __getitem__(self, index: int) -> SlidingWindowSample:
        if index < 0:
            index += len(self._window_index)
        window = self._window_index[index]
        episode = self._episodes[window.episode_position]
        obs = self._slice_observations(episode["obs"], window.start_index)
        actions = self._slice_actions(episode["actions"], window.start_index)
        waypoint_index = self._slice_waypoints(episode["waypoint_index"], window.start_index)
        return {
            "observations": obs,
            "actions": actions,
            "waypoint_index": waypoint_index,
            "episode_id": self._episode_id_at(window.episode_position),
            "scene_type": self._scene_type_at(window.episode_position),
            "start_index": window.start_index,
        }

    def split_by_episode(self, *, val_frac: float = 0.05, seed: int = 42) -> tuple[SlidingWindowDataset, SlidingWindowDataset]:
        """Split at episode granularity to avoid leakage across windows."""

        if not 0.0 <= val_frac < 1.0:
            raise ValueError("val_frac must be in [0, 1)")
        eligible = [position for position in self._selected_episode_positions if self._episode_length(position) >= self._min_episode_length]
        shuffled = list(eligible)
        Random(seed).shuffle(shuffled)
        if len(shuffled) <= 1:
            val_positions: list[int] = []
        else:
            val_count = int(round(len(shuffled) * val_frac))
            val_count = max(1, min(len(shuffled) - 1, val_count))
            val_positions = shuffled[:val_count]
        val_set = set(val_positions)
        train_positions = [position for position in eligible if position not in val_set]
        return self._spawn(train_positions), self._spawn(val_positions)

    def _spawn(self, selected_episode_positions: Sequence[int]) -> SlidingWindowDataset:
        return SlidingWindowDataset(
            self._source_path,
            context_length=self.context_length,
            stride=self.stride,
            obs_dim=self.obs_dim,
            min_episode_length=self._min_episode_length,
            _payload=self._payload,
            _selected_episode_positions=selected_episode_positions,
        )

    def _build_window_index(self, episode_positions: Sequence[int]) -> list[_WindowIndex]:
        windows: list[_WindowIndex] = []
        for position in episode_positions:
            episode_length = self._episode_length(position)
            if episode_length < self._min_episode_length or episode_length < self.context_length:
                continue
            last_start = episode_length - self.context_length
            for start_index in range(0, last_start + 1, self.stride):
                windows.append(_WindowIndex(episode_position=position, start_index=start_index))
        return windows

    def _validate_selected_positions(self, positions: Sequence[int]) -> list[int]:
        selected: list[int] = []
        episode_count = len(self._episodes)
        for position in positions:
            if position < 0 or position >= episode_count:
                raise IndexError(f"episode position out of range: {position}")
            selected.append(int(position))
        return selected

    def _coerce_episodes(self, payload: dict[str, object]) -> list[EpisodeRecord]:
        raw_episodes = payload.get("episodes")
        if not isinstance(raw_episodes, list):
            raise TypeError("dataset payload must contain an 'episodes' list")
        episodes: list[EpisodeRecord] = []
        for idx, raw_episode in enumerate(raw_episodes):
            if not isinstance(raw_episode, dict):
                raise TypeError(f"episode {idx} must be a dictionary")
            episode = cast(dict[str, object], raw_episode)
            obs = episode.get("obs")
            actions = episode.get("actions")
            waypoint_index = episode.get("waypoint_index")
            scene_type = episode.get("scene_type")
            episode_id = episode.get("episode_id")
            if not isinstance(obs, torch.Tensor):
                raise TypeError(f"episode {idx} obs must be a tensor")
            if not isinstance(actions, torch.Tensor):
                raise TypeError(f"episode {idx} actions must be a tensor")
            if not isinstance(waypoint_index, torch.Tensor):
                raise TypeError(f"episode {idx} waypoint_index must be a tensor")
            if obs.ndim != 2:
                raise ValueError(f"episode {idx} obs must be 2D, got {tuple(obs.shape)}")
            if actions.ndim != 2:
                raise ValueError(f"episode {idx} actions must be 2D, got {tuple(actions.shape)}")
            if waypoint_index.ndim != 1:
                raise ValueError(f"episode {idx} waypoint_index must be 1D, got {tuple(waypoint_index.shape)}")
            if obs.shape[0] != actions.shape[0] or obs.shape[0] != waypoint_index.shape[0]:
                raise ValueError(f"episode {idx} length mismatch across obs/actions/waypoint_index")
            if self.obs_dim is not None and obs.shape[1] < self.obs_dim:
                raise ValueError(
                    f"episode {idx} obs_dim {obs.shape[1]} is smaller than requested slice {self.obs_dim}"
                )
            if isinstance(scene_type, str):
                scene_id: int | str = scene_type
            elif isinstance(scene_type, SupportsInt):
                scene_id = int(scene_type)
            elif scene_type is None:
                scene_id = 0
            else:
                raise TypeError(f"episode {idx} scene_type must be an integer or string")
            try:
                ep_id = int(episode_id) if isinstance(episode_id, SupportsInt) else idx if episode_id is None else int(str(episode_id))
            except (TypeError, ValueError) as exc:
                raise TypeError(f"episode {idx} episode_id must be an integer") from exc
            episodes.append(
                EpisodeRecord(
                    obs=obs,
                    actions=actions,
                    waypoint_index=waypoint_index,
                    scene_type=scene_id,
                    episode_id=ep_id,
                )
            )
        return episodes

    def _slice_observations(self, obs: torch.Tensor, start_index: int) -> torch.Tensor:
        window = obs[start_index : start_index + self.context_length]
        if self.obs_dim is not None:
            window = window[:, : self.obs_dim]
        return window.contiguous()

    def _slice_actions(self, actions: torch.Tensor, start_index: int) -> torch.Tensor:
        return actions[start_index : start_index + self.context_length].contiguous()

    def _slice_waypoints(self, waypoint_index: torch.Tensor, start_index: int) -> torch.Tensor:
        return waypoint_index[start_index : start_index + self.context_length].contiguous()

    def _episode_length(self, position: int) -> int:
        return int(self._episodes[position]["obs"].shape[0])

    def _episode_id_at(self, position: int) -> int:
        return int(self._episodes[position]["episode_id"])

    def _scene_type_at(self, position: int) -> int | str:
        return self._episodes[position]["scene_type"]

    @staticmethod
    def _load_payload(path: Path) -> dict[str, object]:
        try:
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("SlidingWindowDataset expects a dict payload")
        return cast(dict[str, object], payload)
