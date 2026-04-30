"""A minimal non-Isaac 2D avoidance environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from parallelcbf.api import ConstraintViolations, MetricDict, SafeEnv, SafetyState


ArrayF32 = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Toy2DConfig:
    """Configuration for `Toy2DAvoidanceEnv`."""

    dt: float = 0.05
    max_steps: int = 200
    action_limit: float = 1.0
    arena_radius: float = 3.0
    obstacle_radius: float = 0.35
    robot_radius: float = 0.10
    goal_radius: float = 0.15


class Toy2DAvoidanceEnv(SafeEnv[ArrayF32, ArrayF32]):
    """Point-mass 2D navigation with one circular obstacle.

    Observation layout: `[px, py, vx, vy, gx, gy, ox, oy]`.
    Action layout: `[ax, ay]`.
    """

    metadata: dict[str, Any] = {"render_modes": []}
    reward_range: tuple[float, float] = (-float("inf"), float("inf"))
    spec: Any = None

    def __init__(self, config: Toy2DConfig | None = None) -> None:
        self.config = config or Toy2DConfig()
        high = np.full((8,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-self.config.action_limit,
            high=self.config.action_limit,
            shape=(2,),
            dtype=np.float32,
        )
        self._rng = np.random.default_rng()
        self._step_count = 0
        self._pos = np.zeros((2,), dtype=np.float32)
        self._vel = np.zeros((2,), dtype=np.float32)
        self._goal = np.array([2.0, 0.0], dtype=np.float32)
        self._obstacle = np.array([0.8, 0.0], dtype=np.float32)
        self._last_hard_violation = False
        self.last_observation = self._observation()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ArrayF32, dict[str, Any]]:
        """Reset the toy environment."""

        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._pos = np.array([-2.0, 0.0], dtype=np.float32)
        self._vel = np.zeros((2,), dtype=np.float32)
        self._goal = np.array([2.0, 0.0], dtype=np.float32)
        self._obstacle = np.array([0.4, 0.25], dtype=np.float32)
        self._last_hard_violation = False
        self.last_observation = self._observation()
        return self.last_observation, {"safety_metrics": dict(self.safety_metrics())}

    def step(self, action: ArrayF32) -> tuple[ArrayF32, float, bool, bool, dict[str, Any]]:
        """Advance one simulation step."""

        nominal = np.asarray(action, dtype=np.float32).reshape(2)
        applied = np.clip(nominal, self.action_space.low, self.action_space.high).astype(np.float32)

        self._vel = (self._vel + applied * np.float32(self.config.dt)).astype(np.float32)
        self._pos = (self._pos + self._vel * np.float32(self.config.dt)).astype(np.float32)
        self._step_count += 1

        obstacle_clearance = self._obstacle_clearance()
        arena_clearance = self.config.arena_radius - float(np.linalg.norm(self._pos))
        self._last_hard_violation = obstacle_clearance < 0.0 or arena_clearance < 0.0
        dist_to_goal = float(np.linalg.norm(self._goal - self._pos))
        terminated = dist_to_goal <= self.config.goal_radius or self._last_hard_violation
        truncated = self._step_count >= self.config.max_steps
        reward = -dist_to_goal - (10.0 if self._last_hard_violation else 0.0)
        info = {
            "safety_metrics": dict(self.safety_metrics()),
            "hard_constraint_violations": dict(self.hard_constraint_violations()),
        }
        self.last_observation = self._observation()
        return self.last_observation, float(reward), bool(terminated), bool(truncated), info

    def safety_state(self) -> SafetyState:
        """Return state needed by a safety filter."""

        return SafetyState(
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            goal=self._goal.copy(),
            obstacles=self._obstacle.reshape(1, 2).copy(),
            robot_radius=float(self.config.robot_radius),
            obstacle_radius=float(self.config.obstacle_radius),
            arena_bounds=np.array(
                [
                    -self.config.arena_radius,
                    self.config.arena_radius,
                    -self.config.arena_radius,
                    self.config.arena_radius,
                ],
                dtype=np.float32,
            ),
            metadata={},
        )

    def safety_metrics(self) -> MetricDict:
        """Return scalar safety metrics."""

        return {
            "obstacle_clearance": self._obstacle_clearance(),
            "arena_clearance": self.config.arena_radius - float(np.linalg.norm(self._pos)),
            "h_hard_violation": self._last_hard_violation,
        }

    def hard_constraint_violations(self) -> ConstraintViolations:
        """Return hard safety flags."""

        return {"collision_or_oob": self._last_hard_violation}

    def render(self) -> None:
        """Rendering is intentionally omitted for the minimal API proof."""

        return None

    def _observation(self) -> ArrayF32:
        return np.concatenate([self._pos, self._vel, self._goal, self._obstacle]).astype(np.float32)

    def _obstacle_clearance(self) -> float:
        combined_radius = self.config.robot_radius + self.config.obstacle_radius
        return float(np.linalg.norm(self._pos - self._obstacle) - combined_radius)
