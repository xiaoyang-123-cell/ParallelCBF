"""Triple-barrier CBF with obstacle and square-arena halfspace constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from parallelcbf.api import BarrierState, MetricDict, SafetyFilterResult, SafetyState
from parallelcbf.safety.dual_barrier import DualBarrierCBF, DualBarrierCBFConfig, Tensor


@dataclass(frozen=True, slots=True)
class TripleBarrierCBFConfig(DualBarrierCBFConfig):
    """Configuration for the obstacle plus square-arena CBF filter."""

    alpha: float = 1.0
    arena_margin: float = 0.0


class TripleBarrierCBF(DualBarrierCBF):
    """Project controls through hard, soft, and 4 arena halfspace barriers.

    The arena uses Option alpha's 4-barrier decomposition:
    `L + p_x`, `L - p_x`, `L + p_y`, and `L - p_y`. Each halfspace is
    enforced with a bounded-acceleration stopping-distance guard built from
    the same linear halfspace. This keeps the public barrier decomposition
    linear while activating early enough for the Toy2D acceleration limit.
    """

    def __init__(self, config: TripleBarrierCBFConfig | None = None) -> None:
        triple_config = config or TripleBarrierCBFConfig()
        if not 0.0 < triple_config.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if triple_config.arena_margin < 0.0:
            raise ValueError("arena_margin must be nonnegative")
        super().__init__(config=triple_config)
        self.config: TripleBarrierCBFConfig = triple_config
        self._last_metrics = {
            "modified": False,
            "h_hard_min": 0.0,
            "h_soft_min": 0.0,
            "h_arena_min": 0.0,
            "h_arena_stopping_min": 0.0,
            "arena_projection_active": False,
            "arena_projection_active_rate": 0.0,
            "arena_projection_active_count": 0,
            "arena_projection_saturated": False,
        }

    def reset(self, *, seed: int | None = None) -> None:
        """Reset filter diagnostics."""

        _ = seed
        self._last_barriers = {}
        self._last_metrics = {
            "modified": False,
            "h_hard_min": 0.0,
            "h_soft_min": 0.0,
            "h_arena_min": 0.0,
            "h_arena_stopping_min": 0.0,
            "arena_projection_active": False,
            "arena_projection_active_rate": 0.0,
            "arena_projection_active_count": 0,
            "arena_projection_saturated": False,
        }

    def filter_action(
        self,
        observation: Tensor,
        nominal_action: Tensor,
        safety_state: SafetyState,
    ) -> SafetyFilterResult[Tensor]:
        """Filter a batched action tensor with obstacle and arena barriers."""

        _ = observation
        action = self._as_cpu_tensor(nominal_action)
        self._assert_action_shape(action)
        position = self._as_cpu_tensor(safety_state.position, dtype=action.dtype)
        velocity = self._as_cpu_tensor(safety_state.velocity, dtype=action.dtype)
        obstacles = self._as_cpu_tensor(safety_state.obstacles, dtype=action.dtype)
        arena_bounds = self._arena_bounds(safety_state, action)
        radii = self._combined_radius(safety_state, action)
        self._assert_state_shapes(position, velocity, obstacles, radii, action)

        safe_action = torch.clamp(action, -self.config.action_limit, self.config.action_limit)
        safe_action = self._project_obstacle_barriers(position, velocity, obstacles, radii, safe_action)
        before_arena = safe_action
        safe_action = self._project_arena_barriers(position, velocity, arena_bounds, safe_action)
        arena_active_by_env = torch.any(torch.abs(safe_action - before_arena) > 1.0e-8, dim=-1)

        next_position = self.next_position(position, velocity, safe_action)
        next_velocity = velocity + safe_action * torch.as_tensor(self.config.dt, dtype=safe_action.dtype)
        h_hard_next = self.h_hard(next_position, obstacles, radii)
        h_soft_next = self.h_soft(next_position, obstacles, radii)
        h_arena_next = self.h_arena(next_position, arena_bounds)
        h_arena_stopping_next = self.h_arena_stopping(next_position, next_velocity, arena_bounds)
        modified = bool(torch.any(torch.abs(safe_action - action) > 1.0e-8).detach().cpu().item())
        arena_active_count = int(torch.sum(arena_active_by_env.to(dtype=torch.int64)).detach().cpu().item())
        arena_rate = float(arena_active_count) / float(max(action.shape[0], 1))
        saturated = bool(torch.any(h_arena_stopping_next < -self.config.hard_margin).detach().cpu().item())

        hard_active = h_hard_next <= self.config.hard_margin
        soft_active = h_soft_next <= 0.0
        arena_active = torch.logical_or(
            h_arena_next <= self.config.arena_margin + self.config.hard_margin,
            h_arena_stopping_next <= self.config.hard_margin,
        )
        hard_violation = h_hard_next < -self.config.hard_margin
        soft_violation = h_soft_next < -self.config.soft_margin
        arena_violation = h_arena_next < -self.config.hard_margin

        self._last_barriers = {
            "hard_squared_distance": BarrierState(
                h=h_hard_next.detach(),
                active=hard_active.detach(),
                violation=hard_violation.detach(),
                barrier_type="hard",
                metadata={"formula": "||rel_pos||^2 - R^2", "alpha": self.config.alpha},
            ),
            "soft_linear_distance": BarrierState(
                h=h_soft_next.detach(),
                active=soft_active.detach(),
                violation=soft_violation.detach(),
                barrier_type="soft",
                metadata={"formula": "||rel_pos|| - (R + soft_margin)", "alpha": self.config.alpha},
            ),
            "arena_linear_halfspace": BarrierState(
                h=h_arena_next.detach(),
                active=arena_active.detach(),
                violation=arena_violation.detach(),
                barrier_type="arena",
                metadata={
                    "formula": "[x-xmin, xmax-x, y-ymin, ymax-y]",
                    "constraint_count": 4,
                    "alpha": self.config.alpha,
                },
            ),
            "arena_projection": BarrierState(
                h=torch.min(h_arena_stopping_next, dim=-1).values.detach(),
                active=arena_active_by_env.detach(),
                violation=torch.any(h_arena_stopping_next < -self.config.hard_margin, dim=-1).detach(),
                barrier_type="arena_projection",
                metadata={
                    "formula": "h - max(0, -h_dot)^2 / (2 * action_limit)",
                    "constraint_count": 4,
                    "alpha": self.config.alpha,
                },
            ),
        }
        self._last_metrics = {
            "modified": modified,
            "h_hard_min": float(torch.min(h_hard_next).detach().cpu().item()),
            "h_soft_min": float(torch.min(h_soft_next).detach().cpu().item()),
            "h_arena_min": float(torch.min(h_arena_next).detach().cpu().item()),
            "h_arena_stopping_min": float(torch.min(h_arena_stopping_next).detach().cpu().item()),
            "arena_projection_active": bool(arena_active_count > 0),
            "arena_projection_active_rate": arena_rate,
            "arena_projection_active_count": arena_active_count,
            "arena_projection_saturated": saturated,
            "constraint_count": 6,
        }
        return SafetyFilterResult(
            safe_action=safe_action,
            nominal_action=action,
            modified=modified,
            barrier_states=dict(self._last_barriers),
            metrics=dict(self._last_metrics),
        )

    def h_arena(self, position: Tensor, arena_bounds: Tensor) -> Tensor:
        """Return four square-arena halfspace barriers per batch row."""

        if arena_bounds.shape != (4,):
            raise ValueError(f"arena_bounds must have shape (4,), got {tuple(arena_bounds.shape)}")
        x_min, x_max, y_min, y_max = arena_bounds.unbind()
        x = position[:, 0]
        y = position[:, 1]
        return torch.stack((x - x_min, x_max - x, y - y_min, y_max - y), dim=-1)

    def h_arena_stopping(self, position: Tensor, velocity: Tensor, arena_bounds: Tensor) -> Tensor:
        """Return stopping-distance arena barriers for bounded acceleration."""

        h_values = self.h_arena(position, arena_bounds)
        normal_velocity = self._arena_normal_velocity(velocity, dtype=position.dtype)
        outward_speed = torch.clamp(-normal_velocity, min=0.0)
        denominator = 2.0 * torch.as_tensor(self.config.action_limit, dtype=position.dtype, device=position.device)
        return h_values - self.config.arena_margin - (outward_speed * outward_speed) / denominator

    def _project_obstacle_barriers(
        self,
        position: Tensor,
        velocity: Tensor,
        obstacles: Tensor,
        radii: Tensor,
        action: Tensor,
    ) -> Tensor:
        safe_action = action
        soft_radii = radii + self.config.soft_margin
        for obstacle_index in range(obstacles.shape[1]):
            obstacle = obstacles[:, obstacle_index, :]
            safe_action = self._project_one_radius_barrier(position, velocity, obstacle, radii, safe_action)
            safe_action = self._project_one_radius_barrier(position, velocity, obstacle, soft_radii, safe_action)
        return safe_action

    def _project_one_radius_barrier(
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
        h_current = dist - radius
        target_distance = radius + (1.0 - self.config.alpha) * h_current + self.config.hard_margin
        predicted_without_action = position + velocity * dt
        required_normal_action = (
            target_distance - torch.sum(normal * (predicted_without_action - obstacle), dim=-1)
        ) / (dt * dt)
        current_normal_action = torch.sum(normal * action, dim=-1)
        delta = torch.clamp(required_normal_action - current_normal_action, min=0.0)
        corrected = action + delta.unsqueeze(-1) * normal
        return torch.clamp(corrected, -self.config.action_limit, self.config.action_limit)

    def _project_arena_barriers(
        self,
        position: Tensor,
        velocity: Tensor,
        arena_bounds: Tensor,
        action: Tensor,
    ) -> Tensor:
        x_min, x_max, y_min, y_max = arena_bounds.unbind()
        normals = torch.tensor(
            ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
            dtype=action.dtype,
            device=action.device,
        )
        offsets = torch.stack((-x_min, x_max, -y_min, y_max))
        safe_action = action
        for _ in range(2):
            for index in range(4):
                normal = normals[index]
                offset = offsets[index]
                h_current = offset + torch.sum(position * normal, dim=-1)
                normal_velocity = torch.sum(velocity * normal, dim=-1)
                current_normal_action = torch.sum(safe_action * normal, dim=-1)
                required_normal_action = self._required_arena_normal_action(
                    h_current,
                    normal_velocity,
                    current_normal_action,
                    dtype=action.dtype,
                )
                current_normal_action = torch.sum(safe_action * normal, dim=-1)
                delta = torch.clamp(required_normal_action - current_normal_action, min=0.0)
                safe_action = safe_action + delta.unsqueeze(-1) * normal
                safe_action = torch.clamp(safe_action, -self.config.action_limit, self.config.action_limit)
        return safe_action

    def _required_arena_normal_action(
        self,
        h_current: Tensor,
        normal_velocity: Tensor,
        current_normal_action: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        """Return the least inward acceleration satisfying the stopping guard."""

        limit = torch.as_tensor(self.config.action_limit, dtype=dtype, device=h_current.device)
        lower = torch.clamp(current_normal_action, min=-limit, max=limit)
        if torch.all(self._arena_stopping_guard(h_current, normal_velocity, lower, dtype=dtype) >= 0.0):
            return lower
        upper = torch.full_like(lower, float(self.config.action_limit))
        unsafe = self._arena_stopping_guard(h_current, normal_velocity, lower, dtype=dtype) < 0.0
        for _ in range(32):
            midpoint = torch.where(unsafe, (lower + upper) * 0.5, lower)
            midpoint_safe = self._arena_stopping_guard(h_current, normal_velocity, midpoint, dtype=dtype) >= 0.0
            upper = torch.where(torch.logical_and(unsafe, midpoint_safe), midpoint, upper)
            lower = torch.where(torch.logical_and(unsafe, torch.logical_not(midpoint_safe)), midpoint, lower)
        return torch.where(unsafe, upper, lower)

    def _arena_stopping_guard(
        self,
        h_current: Tensor,
        normal_velocity: Tensor,
        normal_action: Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tensor:
        dt = torch.as_tensor(self.config.dt, dtype=dtype, device=h_current.device)
        limit = torch.as_tensor(self.config.action_limit, dtype=dtype, device=h_current.device)
        h_next = h_current + normal_velocity * dt + normal_action * dt * dt - self.config.arena_margin
        velocity_next = normal_velocity + normal_action * dt
        outward_speed_next = torch.clamp(-velocity_next, min=0.0)
        stopping_distance = (outward_speed_next * outward_speed_next) / (2.0 * limit)
        return h_next - stopping_distance - self.config.hard_margin

    @staticmethod
    def _arena_normal_velocity(velocity: Tensor, *, dtype: torch.dtype) -> Tensor:
        normals = torch.tensor(
            ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
            dtype=dtype,
            device=velocity.device,
        )
        return velocity @ normals.transpose(0, 1)

    @staticmethod
    def _arena_bounds(safety_state: SafetyState, action: Tensor) -> Tensor:
        bounds = safety_state.arena_bounds
        tensor = bounds if isinstance(bounds, torch.Tensor) else torch.as_tensor(bounds)
        if tensor.device.type != "cpu":
            raise RuntimeError("TripleBarrierCBF is CPU-only; received non-CPU arena bounds.")
        tensor = tensor.to(dtype=action.dtype)
        if tensor.shape != (4,):
            raise ValueError(f"arena_bounds must have shape (4,), got {tuple(tensor.shape)}")
        return tensor
