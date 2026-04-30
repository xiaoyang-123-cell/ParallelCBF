"""NumPy-batched Toy2D fixture for vectorization tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from parallelcbf.envs.toy2d import Toy2DConfig


ArrayF32 = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


class Toy2DAvoidanceVecEnv:
    """Small vectorized point-mass environment with shape-stable operations."""

    def __init__(self, *, num_envs: int, config: Toy2DConfig | None = None) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.config = config or Toy2DConfig()
        self._step_count = np.zeros((num_envs,), dtype=np.int32)
        self._pos = np.zeros((num_envs, 2), dtype=np.float32)
        self._vel = np.zeros((num_envs, 2), dtype=np.float32)
        self._goal = np.broadcast_to(np.array([2.0, 0.0], dtype=np.float32), (num_envs, 2)).copy()
        self._obstacle = np.broadcast_to(np.array([0.4, 0.25], dtype=np.float32), (num_envs, 2)).copy()
        self._last_hard_violation = np.zeros((num_envs,), dtype=np.bool_)

    def reset(self, *, seed: int | None = None) -> tuple[ArrayF32, dict[str, Any]]:
        """Reset all vectorized environments."""

        _ = seed
        self._step_count.fill(0)
        self._pos[:, :] = np.array([-2.0, 0.0], dtype=np.float32)
        self._vel.fill(0.0)
        self._goal[:, :] = np.array([2.0, 0.0], dtype=np.float32)
        self._obstacle[:, :] = np.array([0.4, 0.25], dtype=np.float32)
        self._last_hard_violation.fill(False)
        return self._observation(), {"safety_metrics": self.safety_metrics()}

    def step(self, actions: ArrayF32) -> tuple[ArrayF32, ArrayF32, BoolArray, BoolArray, dict[str, Any]]:
        """Advance every vector lane by one step."""

        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (self.num_envs, 2):
            raise ValueError(f"actions must have shape ({self.num_envs}, 2), got {action_array.shape}")
        applied = np.clip(action_array, -self.config.action_limit, self.config.action_limit).astype(np.float32)
        dt = np.float32(self.config.dt)
        self._vel = (self._vel + applied * dt).astype(np.float32)
        self._pos = (self._pos + self._vel * dt).astype(np.float32)
        self._step_count += 1

        obstacle_clearance = self._obstacle_clearance()
        arena_clearance = self.config.arena_radius - np.linalg.norm(self._pos, axis=1)
        self._last_hard_violation = np.logical_or(obstacle_clearance < 0.0, arena_clearance < 0.0)
        dist_to_goal = np.linalg.norm(self._goal - self._pos, axis=1)
        terminated = np.logical_or(dist_to_goal <= self.config.goal_radius, self._last_hard_violation)
        truncated = self._step_count >= self.config.max_steps
        rewards = (-dist_to_goal - np.where(self._last_hard_violation, 10.0, 0.0)).astype(np.float32)
        info = {
            "safety_metrics": self.safety_metrics(),
            "hard_constraint_violations": {"collision_or_oob": self._last_hard_violation.copy()},
        }
        return self._observation(), rewards, terminated.astype(np.bool_), truncated.astype(np.bool_), info

    def safety_metrics(self) -> dict[str, ArrayF32 | BoolArray]:
        """Return per-lane safety metrics."""

        arena_clearance = self.config.arena_radius - np.linalg.norm(self._pos, axis=1)
        return {
            "obstacle_clearance": self._obstacle_clearance().astype(np.float32),
            "arena_clearance": np.asarray(arena_clearance, dtype=np.float32),
            "h_hard_violation": self._last_hard_violation.copy(),
        }

    def _observation(self) -> ArrayF32:
        return np.concatenate([self._pos, self._vel, self._goal, self._obstacle], axis=1).astype(np.float32)

    def _obstacle_clearance(self) -> ArrayF32:
        combined_radius = np.float32(self.config.robot_radius + self.config.obstacle_radius)
        return np.asarray(np.linalg.norm(self._pos - self._obstacle, axis=1) - combined_radius, dtype=np.float32)
