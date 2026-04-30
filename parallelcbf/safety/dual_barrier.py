"""Reference analytical and PyTorch dual-barrier CBF safety filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch

from parallelcbf.api import BarrierState, MetricDict, SafetyFilter, SafetyFilterResult, SafetyState


ArrayF32 = NDArray[np.float32]
Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class NaiveDistanceCBFConfig:
    """Configuration for `NaiveDistanceCBF`."""

    dt: float = 0.05
    action_limit: float = 1.0
    clearance_epsilon: float = 1.0e-6


@dataclass(frozen=True, slots=True)
class DualBarrierCBFConfig:
    """Configuration for the CPU PyTorch dual-barrier filter."""

    dt: float = 0.05
    action_limit: float = 1.0
    hard_margin: float = 1.0e-8
    soft_margin: float = 0.25
    eps: float = 1.0e-12


class NaiveDistanceCBF(SafetyFilter[ArrayF32, ArrayF32]):
    """Analytical one-step distance filter for a 2D point mass.

    The filter projects the nominal acceleration so that the predicted next
    position remains outside the hard obstacle radius. It is deliberately small:
    no cvxpy, no simulator assumptions, no GPU.
    """

    def __init__(self, config: NaiveDistanceCBFConfig | None = None) -> None:
        self.config = config or NaiveDistanceCBFConfig()
        self._last_barriers: dict[str, BarrierState] = {}
        self._last_metrics: MetricDict = {"modified": False, "h_hard_min": 0.0}

    def reset(self, *, seed: int | None = None) -> None:
        """Reset filter diagnostics."""

        _ = seed
        self._last_barriers = {}
        self._last_metrics = {"modified": False, "h_hard_min": 0.0}

    def filter_action(
        self,
        observation: ArrayF32,
        nominal_action: ArrayF32,
        safety_state: SafetyState,
    ) -> SafetyFilterResult[ArrayF32]:
        """Project a 2D acceleration onto the one-step hard-safe set."""

        _ = observation
        action = np.asarray(nominal_action, dtype=np.float32).reshape(2)
        action = np.clip(action, -self.config.action_limit, self.config.action_limit).astype(np.float32)
        pos = np.asarray(safety_state.position, dtype=np.float32).reshape(2)
        vel = np.asarray(safety_state.velocity, dtype=np.float32).reshape(2)
        obstacles = np.asarray(safety_state.obstacles, dtype=np.float32).reshape(-1, 2)
        hard_radius = float(safety_state.robot_radius + safety_state.obstacle_radius)
        safe_action = action.copy()
        modified = False

        for obstacle in obstacles:
            safe_action, changed = self._project_for_obstacle(pos, vel, obstacle, hard_radius, safe_action)
            modified = modified or changed

        h_values = self._h_hard_after_action(pos, vel, obstacles, hard_radius, safe_action)
        active = h_values <= np.float32(1.0e-3)
        violation = h_values < np.float32(-1.0e-6)
        self._last_barriers = {
            "hard_distance": BarrierState(
                h=h_values.astype(np.float32),
                active=active,
                violation=violation,
                barrier_type="hard",
                metadata={"radius": hard_radius},
            )
        }
        self._last_metrics = {
            "modified": modified,
            "h_hard_min": min(float(value) for value in h_values) if h_values.size > 0 else 0.0,
        }
        return SafetyFilterResult(
            safe_action=safe_action.astype(np.float32),
            nominal_action=action,
            modified=modified,
            barrier_states=dict(self._last_barriers),
            metrics=dict(self._last_metrics),
        )

    def barrier_state(self) -> dict[str, BarrierState]:
        """Return the most recent hard-distance barrier."""

        return dict(self._last_barriers)

    def metrics(self) -> MetricDict:
        """Return scalar filter diagnostics."""

        return dict(self._last_metrics)

    def _project_for_obstacle(
        self,
        pos: ArrayF32,
        vel: ArrayF32,
        obstacle: ArrayF32,
        hard_radius: float,
        action: ArrayF32,
    ) -> tuple[ArrayF32, bool]:
        dt = np.float32(self.config.dt)
        rel = pos - obstacle
        dist = float(np.linalg.norm(rel))
        if dist <= 1.0e-9:
            normal = np.array([1.0, 0.0], dtype=np.float32)
        else:
            normal = (rel / np.float32(dist)).astype(np.float32)
        predicted_without_action = pos + vel * dt
        rhs = hard_radius + self.config.clearance_epsilon - float(np.dot(normal, predicted_without_action - obstacle))
        coeff = float(dt * dt)
        if coeff <= 0.0 or float(np.dot(normal, action)) >= rhs / coeff:
            return action, False
        corrected = self._project_halfspace_box(action, normal, rhs / coeff)
        return corrected, True

    def _project_halfspace_box(self, action: ArrayF32, normal: ArrayF32, target: float) -> ArrayF32:
        limit = float(self.config.action_limit)
        clipped = np.clip(action, -limit, limit).astype(np.float32)
        if float(np.dot(normal, clipped)) >= target:
            return clipped

        max_feasible = limit * (abs(float(normal[0])) + abs(float(normal[1])))
        if target > max_feasible:
            best_effort = limit * np.sign(normal)
            return np.asarray(best_effort, dtype=np.float32)

        candidates: list[ArrayF32] = []
        denom = float(np.dot(normal, normal))
        if denom > 0.0:
            projected = clipped + np.float32((target - float(np.dot(normal, clipped))) / denom) * normal
            if self._is_box_feasible(projected, limit) and float(np.dot(normal, projected)) >= target - 1.0e-6:
                candidates.append(projected.astype(np.float32))

        for fixed_index in range(2):
            free_index = 1 - fixed_index
            free_coeff = float(normal[free_index])
            if abs(free_coeff) <= 1.0e-9:
                continue
            for fixed_value in (-limit, limit):
                free_value = (target - float(normal[fixed_index]) * fixed_value) / free_coeff
                candidate = np.empty((2,), dtype=np.float32)
                candidate[fixed_index] = np.float32(fixed_value)
                candidate[free_index] = np.float32(free_value)
                if self._is_box_feasible(candidate, limit) and float(np.dot(normal, candidate)) >= target - 1.0e-6:
                    candidates.append(candidate)

        for x_value in (-limit, limit):
            for y_value in (-limit, limit):
                candidate = np.array([x_value, y_value], dtype=np.float32)
                if float(np.dot(normal, candidate)) >= target - 1.0e-6:
                    candidates.append(candidate)

        if not candidates:
            best_effort = limit * np.sign(normal)
            return np.asarray(best_effort, dtype=np.float32)

        best = min(candidates, key=lambda candidate: self._squared_distance(candidate, clipped))
        return np.asarray(best, dtype=np.float32)

    @staticmethod
    def _is_box_feasible(action: ArrayF32, limit: float) -> bool:
        return bool(
            -limit - 1.0e-6 <= float(action[0]) <= limit + 1.0e-6
            and -limit - 1.0e-6 <= float(action[1]) <= limit + 1.0e-6
        )

    @staticmethod
    def _squared_distance(left: ArrayF32, right: ArrayF32) -> float:
        dx = float(left[0] - right[0])
        dy = float(left[1] - right[1])
        return dx * dx + dy * dy

    def _h_hard_after_action(
        self,
        pos: ArrayF32,
        vel: ArrayF32,
        obstacles: ArrayF32,
        hard_radius: float,
        action: ArrayF32,
    ) -> ArrayF32:
        dt = np.float32(self.config.dt)
        next_pos = pos + vel * dt + action * dt * dt
        if obstacles.size == 0:
            return np.array([], dtype=np.float32)
        distances = np.linalg.norm(next_pos.reshape(1, 2) - obstacles, axis=1)
        h_values: ArrayF32 = np.asarray(distances - np.float32(hard_radius), dtype=np.float32)
        return h_values


class DualBarrierCBF(SafetyFilter[Tensor, Tensor]):
    """CPU-only PyTorch implementation of the V23-style dual barrier.

    The hard barrier is squared distance:
    `h_hard = ||rel_pos||^2 - R^2`.
    The soft barrier is linear distance margin:
    `h_soft = ||rel_pos|| - (R + soft_margin)`.
    """

    def __init__(self, config: DualBarrierCBFConfig | None = None) -> None:
        self.config = config or DualBarrierCBFConfig()
        self._last_barriers: dict[str, BarrierState] = {}
        self._last_metrics: MetricDict = {"modified": False, "h_hard_min": 0.0, "h_soft_min": 0.0}

    def reset(self, *, seed: int | None = None) -> None:
        """Reset filter diagnostics."""

        _ = seed
        self._last_barriers = {}
        self._last_metrics = {"modified": False, "h_hard_min": 0.0, "h_soft_min": 0.0}

    def filter_action(
        self,
        observation: Tensor,
        nominal_action: Tensor,
        safety_state: SafetyState,
    ) -> SafetyFilterResult[Tensor]:
        """Filter a batched action tensor using explicit broadcast discipline."""

        _ = observation
        action = self._as_cpu_tensor(nominal_action)
        self._assert_action_shape(action)
        position = self._as_cpu_tensor(safety_state.position, dtype=action.dtype)
        velocity = self._as_cpu_tensor(safety_state.velocity, dtype=action.dtype)
        obstacles = self._as_cpu_tensor(safety_state.obstacles, dtype=action.dtype)
        radii = self._combined_radius(safety_state, action)
        self._assert_state_shapes(position, velocity, obstacles, radii, action)

        safe_action = torch.clamp(action, -self.config.action_limit, self.config.action_limit)
        safe_action = self._project_batch(position, velocity, obstacles, radii, safe_action)
        h_hard = self.h_hard(position, obstacles, radii)
        h_soft = self.h_soft(position, obstacles, radii)
        next_position = self.next_position(position, velocity, safe_action)
        h_hard_next = self.h_hard(next_position, obstacles, radii)
        h_soft_next = self.h_soft(next_position, obstacles, radii)
        modified = bool(torch.any(torch.abs(safe_action - action) > 1.0e-8).detach().cpu().item())

        hard_active = h_hard_next <= self.config.hard_margin
        soft_active = h_soft <= 0.0
        hard_violation = h_hard_next < -self.config.hard_margin
        soft_violation = h_soft_next < -self.config.soft_margin
        self._last_barriers = {
            "hard_squared_distance": BarrierState(
                h=h_hard_next.detach(),
                active=hard_active.detach(),
                violation=hard_violation.detach(),
                barrier_type="hard",
                metadata={"formula": "||rel_pos||^2 - R^2"},
            ),
            "soft_linear_distance": BarrierState(
                h=h_soft_next.detach(),
                active=soft_active.detach(),
                violation=soft_violation.detach(),
                barrier_type="soft",
                metadata={"formula": "||rel_pos|| - (R + soft_margin)"},
            ),
        }
        self._last_metrics = {
            "modified": modified,
            "h_hard_min": float(torch.min(h_hard_next).detach().cpu().item()),
            "h_soft_min": float(torch.min(h_soft_next).detach().cpu().item()),
        }
        return SafetyFilterResult(
            safe_action=safe_action,
            nominal_action=action,
            modified=modified,
            barrier_states=dict(self._last_barriers),
            metrics=dict(self._last_metrics),
        )

    def barrier_state(self) -> dict[str, BarrierState]:
        """Return the latest hard/soft barrier diagnostics."""

        return dict(self._last_barriers)

    def metrics(self) -> MetricDict:
        """Return scalar filter diagnostics."""

        return dict(self._last_metrics)

    def h_hard(self, position: Tensor, obstacles: Tensor, radii: Tensor) -> Tensor:
        """Compute squared hard barrier values with explicit unsqueeze."""

        rel_pos = position.unsqueeze(1) - obstacles
        self._assert_rel_shape(rel_pos, position, obstacles)
        return torch.sum(rel_pos * rel_pos, dim=-1) - radii.unsqueeze(-1) * radii.unsqueeze(-1)

    def h_soft(self, position: Tensor, obstacles: Tensor, radii: Tensor) -> Tensor:
        """Compute linear soft barrier values with explicit unsqueeze."""

        rel_pos = position.unsqueeze(1) - obstacles
        self._assert_rel_shape(rel_pos, position, obstacles)
        distance = torch.sqrt(torch.sum(rel_pos * rel_pos, dim=-1) + self.config.eps)
        return distance - (radii.unsqueeze(-1) + self.config.soft_margin)

    def next_position(self, position: Tensor, velocity: Tensor, action: Tensor) -> Tensor:
        """One-step point-mass position update used by tests and filtering."""

        dt = torch.as_tensor(self.config.dt, dtype=action.dtype, device=action.device)
        return position + velocity * dt + action * dt * dt

    def _project_batch(
        self,
        position: Tensor,
        velocity: Tensor,
        obstacles: Tensor,
        radii: Tensor,
        action: Tensor,
    ) -> Tensor:
        safe_action = action
        for obstacle_index in range(obstacles.shape[1]):
            obstacle = obstacles[:, obstacle_index, :]
            radius = radii
            safe_action = self._project_one_obstacle(position, velocity, obstacle, radius, safe_action)
        return safe_action

    def _project_one_obstacle(
        self,
        position: Tensor,
        velocity: Tensor,
        obstacle: Tensor,
        radius: Tensor,
        action: Tensor,
    ) -> Tensor:
        dt = torch.as_tensor(self.config.dt, dtype=action.dtype, device=action.device)
        rel = position - obstacle
        dist = torch.sqrt(torch.sum(rel * rel, dim=-1) + self.config.eps)
        normal = rel / dist.unsqueeze(-1)
        predicted_without_action = position + velocity * dt
        rhs = radius + self.config.hard_margin - torch.sum(normal * (predicted_without_action - obstacle), dim=-1)
        coeff = dt * dt
        required_normal_action = rhs / coeff
        current_normal_action = torch.sum(normal * action, dim=-1)
        delta = torch.clamp(required_normal_action - current_normal_action, min=0.0)
        corrected = action + delta.unsqueeze(-1) * normal
        return torch.clamp(corrected, -self.config.action_limit, self.config.action_limit)

    def _combined_radius(self, safety_state: SafetyState, action: Tensor) -> Tensor:
        batch = action.shape[0]
        value = float(safety_state.robot_radius + safety_state.obstacle_radius)
        return torch.full((batch,), value, dtype=action.dtype, device=action.device)

    @staticmethod
    def _as_cpu_tensor(value: object, *, dtype: torch.dtype | None = None) -> Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.device.type != "cpu":
            raise RuntimeError("DualBarrierCBF is CPU-only; received non-CPU tensor.")
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    @staticmethod
    def _assert_action_shape(action: Tensor) -> None:
        if action.ndim != 2 or action.shape[-1] != 2:
            raise ValueError(f"action must have shape (N, 2), got {tuple(action.shape)}")

    @staticmethod
    def _assert_state_shapes(position: Tensor, velocity: Tensor, obstacles: Tensor, radii: Tensor, action: Tensor) -> None:
        if position.shape != action.shape:
            raise ValueError(f"position must have shape {tuple(action.shape)}, got {tuple(position.shape)}")
        if velocity.shape != action.shape:
            raise ValueError(f"velocity must have shape {tuple(action.shape)}, got {tuple(velocity.shape)}")
        if obstacles.ndim != 3 or obstacles.shape[0] != action.shape[0] or obstacles.shape[-1] != 2:
            raise ValueError(f"obstacles must have shape (N, M, 2), got {tuple(obstacles.shape)}")
        if radii.shape != (action.shape[0],):
            raise ValueError(f"radii must have shape ({action.shape[0]},), got {tuple(radii.shape)}")

    @staticmethod
    def _assert_rel_shape(rel_pos: Tensor, position: Tensor, obstacles: Tensor) -> None:
        expected = (position.shape[0], obstacles.shape[1], position.shape[1])
        if rel_pos.shape != expected:
            raise ValueError(f"broadcast rel_pos must have shape {expected}, got {tuple(rel_pos.shape)}")


class ChanceConstrainedDualBarrierCBF(DualBarrierCBF):
    """Skeleton for V25 delta-aware safety inflation.

    `sigma_d=0` is the degenerate placeholder, so this currently reduces to the
    deterministic dual-barrier radius until the probabilistic model lands.
    """

    def __init__(self, config: DualBarrierCBFConfig | None = None, *, delta: float = 0.05, sigma_d: float = 0.0) -> None:
        super().__init__(config=config)
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if sigma_d < 0.0:
            raise ValueError("sigma_d must be nonnegative")
        self.delta = delta
        self.sigma_d = sigma_d

    def _combined_radius(self, safety_state: SafetyState, action: Tensor) -> Tensor:
        base_radius = super()._combined_radius(safety_state, action)
        inflation = torch.as_tensor(self.sigma_d, dtype=action.dtype, device=action.device)
        return base_radius + inflation
