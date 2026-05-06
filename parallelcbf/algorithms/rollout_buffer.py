"""Episode-aware PPO rollout storage for Transformer policies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import random
from typing import Literal, TypeAlias, cast

import torch


TerminationReason: TypeAlias = Literal[
    "success",
    "collision",
    "out_of_arena",
    "cbf_arena_projection_active",
    "timeout",
    "ongoing",
]
ZERO_BOOTSTRAP_REASONS = frozenset(("success", "collision", "out_of_arena", "cbf_arena_projection_active"))


@dataclass(slots=True)
class RolloutStep:
    """One transition stored for GAE and PPO minibatching."""

    env_id: int
    episode_id: int
    timestep: int
    context: torch.Tensor
    observation: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    value: float
    reward: float
    shaped_reward: float
    done: bool
    terminated: bool
    truncated: bool
    termination_reason: TerminationReason
    u_nominal: torch.Tensor
    u_safe: torch.Tensor
    cbf_delta_norm: float
    bootstrap_value: float | None = None
    advantage: float = 0.0
    return_: float = 0.0


class RolloutBuffer:
    """Append-only rollout store with episode-aware and shuffled iteration."""

    def __init__(self, *, context_length: int, obs_dim: int, action_dim: int) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if obs_dim <= 0:
            raise ValueError("obs_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.context_length = int(context_length)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self._steps: list[RolloutStep] = []

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator[RolloutStep]:
        return iter(self._steps)

    @property
    def steps(self) -> tuple[RolloutStep, ...]:
        """Return an immutable view of stored transitions."""

        return tuple(self._steps)

    def append(self, step: RolloutStep) -> None:
        """Append one transition after checking the core tensor ranks."""

        if step.context.ndim != 2 or step.context.shape[-1] != self.obs_dim:
            raise ValueError(f"context must have shape (T, {self.obs_dim}), got {tuple(step.context.shape)}")
        if step.context.shape[0] > self.context_length:
            raise ValueError(f"context length exceeds {self.context_length}: {step.context.shape[0]}")
        if step.observation.shape != (self.obs_dim,):
            raise ValueError(f"observation must have shape ({self.obs_dim},), got {tuple(step.observation.shape)}")
        if step.action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},), got {tuple(step.action.shape)}")
        if step.u_nominal.shape != step.action.shape:
            raise ValueError("u_nominal must match action shape")
        if step.u_safe.shape != step.action.shape:
            raise ValueError("u_safe must match action shape")
        self._steps.append(step)

    def clear(self) -> None:
        """Drop all stored transitions."""

        self._steps.clear()

    def iter_episodes(self) -> Iterator[tuple[tuple[int, int], tuple[RolloutStep, ...]]]:
        """Yield transitions grouped by `(env_id, episode_id)` in time order."""

        grouped: dict[tuple[int, int], list[RolloutStep]] = defaultdict(list)
        for step in self._steps:
            grouped[(step.env_id, step.episode_id)].append(step)
        for key in sorted(grouped):
            episode = sorted(grouped[key], key=lambda item: item.timestep)
            yield key, tuple(episode)

    def shuffled_batches(self, *, batch_size: int, seed: int | None = None) -> Iterator[dict[str, torch.Tensor]]:
        """Yield PPO-style shuffled minibatches covering every stored step once."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = list(range(len(self._steps)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield self._batch_from_indices(batch_indices)

    def _batch_from_indices(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        selected = [self._steps[index] for index in indices]
        contexts = torch.stack([_left_pad_context(step.context, self.context_length, self.obs_dim) for step in selected])
        observations = torch.stack([step.observation for step in selected])
        actions = torch.stack([step.action for step in selected])
        old_log_probs = torch.stack([step.log_prob.reshape(()) for step in selected])
        values = torch.as_tensor([step.value for step in selected], dtype=contexts.dtype)
        rewards = torch.as_tensor([step.shaped_reward for step in selected], dtype=contexts.dtype)
        advantages = torch.as_tensor([step.advantage for step in selected], dtype=contexts.dtype)
        returns = torch.as_tensor([step.return_ for step in selected], dtype=contexts.dtype)
        u_nominal = torch.stack([step.u_nominal for step in selected])
        u_safe = torch.stack([step.u_safe for step in selected])
        cbf_delta_norm = torch.as_tensor([step.cbf_delta_norm for step in selected], dtype=contexts.dtype)
        env_ids = torch.as_tensor([step.env_id for step in selected], dtype=torch.long)
        episode_ids = torch.as_tensor([step.episode_id for step in selected], dtype=torch.long)
        timesteps = torch.as_tensor([step.timestep for step in selected], dtype=torch.long)
        return {
            "context": contexts,
            "observation": observations,
            "action": actions,
            "old_log_prob": old_log_probs,
            "value": values,
            "reward": rewards,
            "advantage": advantages,
            "return": returns,
            "u_nominal": u_nominal,
            "u_safe": u_safe,
            "cbf_delta_norm": cbf_delta_norm,
            "env_id": env_ids,
            "episode_id": episode_ids,
            "timestep": timesteps,
        }


def compute_gae_in_place(
    buffer: RolloutBuffer,
    *,
    last_values: Mapping[int, float] | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> None:
    """Compute GAE using termination reasons to decide value bootstrapping."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    value_map = {} if last_values is None else dict(last_values)
    for _, episode in buffer.iter_episodes():
        next_advantage = 0.0
        for index in range(len(episode) - 1, -1, -1):
            step = episode[index]
            has_next_step = index + 1 < len(episode) and not step.done
            if has_next_step:
                next_value = episode[index + 1].value
                next_nonterminal = 1.0
            else:
                next_value = _terminal_next_value(step, value_map)
                next_nonterminal = _next_nonterminal(step)
                next_advantage = 0.0
            delta = step.shaped_reward + gamma * next_value * next_nonterminal - step.value
            step.advantage = delta + gamma * gae_lambda * next_nonterminal * next_advantage
            step.return_ = step.advantage + step.value
            next_advantage = step.advantage


def _left_pad_context(context: torch.Tensor, context_length: int, obs_dim: int) -> torch.Tensor:
    if context.shape == (context_length, obs_dim):
        return context
    padded = torch.zeros((context_length, obs_dim), dtype=context.dtype, device=context.device)
    length = min(context.shape[0], context_length)
    padded[-length:, :] = context[-length:, :]
    return padded


def _next_nonterminal(step: RolloutStep) -> float:
    if step.termination_reason in ZERO_BOOTSTRAP_REASONS:
        return 0.0
    return 1.0


def _terminal_next_value(step: RolloutStep, last_values: Mapping[int, float]) -> float:
    if step.termination_reason in ZERO_BOOTSTRAP_REASONS:
        return 0.0
    if step.bootstrap_value is not None:
        return float(step.bootstrap_value)
    if step.env_id in last_values:
        return float(last_values[step.env_id])
    if step.termination_reason in {"timeout", "ongoing"}:
        raise ValueError(f"missing bootstrap value for {step.termination_reason} step in env {step.env_id}")
    return 0.0


def normalize_termination_reason(value: object) -> TerminationReason:
    """Normalize env info values into the rollout termination vocabulary."""

    if value is None:
        return "ongoing"
    if isinstance(value, str) and value in {
        "success",
        "collision",
        "out_of_arena",
        "cbf_arena_projection_active",
        "timeout",
    }:
        return cast(TerminationReason, value)
    return "ongoing"


__all__ = [
    "RolloutBuffer",
    "RolloutStep",
    "TerminationReason",
    "compute_gae_in_place",
    "normalize_termination_reason",
]
