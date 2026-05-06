from __future__ import annotations

from typing import cast

import pytest
import torch

from parallelcbf.api import SafetyState
from parallelcbf.safety import TripleBarrierCBF, TripleBarrierCBFConfig


def _state(
    *,
    position: torch.Tensor,
    velocity: torch.Tensor,
    obstacles: torch.Tensor | None = None,
    arena: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
) -> SafetyState:
    batch = position.shape[0]
    obs = obstacles if obstacles is not None else torch.full((batch, 1, 2), 100.0, dtype=position.dtype)
    return SafetyState(
        position=position,
        velocity=velocity,
        goal=torch.zeros((batch, 2), dtype=position.dtype),
        obstacles=obs,
        robot_radius=0.1,
        obstacle_radius=0.2,
        arena_bounds=torch.tensor(arena, dtype=position.dtype),
        metadata={},
    )


def test_triple_barrier_adds_four_arena_constraints_to_dual_barriers() -> None:
    cbf = TripleBarrierCBF()
    position = torch.zeros((2, 2), dtype=torch.float64)
    velocity = torch.zeros_like(position)
    action = torch.zeros_like(position)

    result = cbf.filter_action(torch.zeros((2, 8), dtype=torch.float64), action, _state(position=position, velocity=velocity))

    assert result.safe_action.shape == (2, 2)
    assert result.barrier_states["hard_squared_distance"].h.shape == (2, 1)
    assert result.barrier_states["soft_linear_distance"].h.shape == (2, 1)
    assert result.barrier_states["arena_linear_halfspace"].h.shape == (2, 4)
    assert result.metrics["constraint_count"] == 6


def test_triple_barrier_preserves_positive_arena_invariance_at_boundary() -> None:
    cbf = TripleBarrierCBF(TripleBarrierCBFConfig(action_limit=20.0, alpha=1.0))
    position = torch.tensor([[2.99, 0.0]], dtype=torch.float64)
    velocity = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    nominal = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    result = cbf.filter_action(torch.zeros((1, 8), dtype=torch.float64), nominal, _state(position=position, velocity=velocity))
    next_position = cbf.next_position(position, velocity, result.safe_action)
    h_arena = cbf.h_arena(next_position, torch.tensor([-3.0, 3.0, -3.0, 3.0], dtype=torch.float64))

    assert bool(torch.all(h_arena >= -1.0e-7).item())
    assert bool(result.metrics["arena_projection_active"])


def test_triple_barrier_projection_pushes_away_from_right_wall() -> None:
    cbf = TripleBarrierCBF(TripleBarrierCBFConfig(action_limit=20.0, alpha=1.0))
    position = torch.tensor([[2.99, 0.0]], dtype=torch.float64)
    velocity = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
    nominal = torch.tensor([[20.0, 0.0]], dtype=torch.float64)

    result = cbf.filter_action(torch.zeros((1, 8), dtype=torch.float64), nominal, _state(position=position, velocity=velocity))

    assert result.safe_action[0, 0].item() < nominal[0, 0].item()
    assert result.safe_action[0, 0].item() <= -6.0
    assert result.metrics["arena_projection_active_rate"] == pytest.approx(1.0)


def test_triple_barrier_corner_case_enforces_both_axes() -> None:
    cbf = TripleBarrierCBF(TripleBarrierCBFConfig(action_limit=30.0, alpha=1.0))
    position = torch.tensor([[2.99, 2.99]], dtype=torch.float64)
    velocity = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    nominal = torch.tensor([[30.0, 30.0]], dtype=torch.float64)

    result = cbf.filter_action(torch.zeros((1, 8), dtype=torch.float64), nominal, _state(position=position, velocity=velocity))
    next_position = cbf.next_position(position, velocity, result.safe_action)
    h_arena = cbf.h_arena(next_position, torch.tensor([-3.0, 3.0, -3.0, 3.0], dtype=torch.float64))

    assert bool(torch.all(h_arena >= -1.0e-7).item())
    assert result.safe_action[0, 0].item() < 0.0
    assert result.safe_action[0, 1].item() < 0.0


def test_triple_barrier_keeps_obstacle_invariance_with_arena_constraints() -> None:
    cbf = TripleBarrierCBF(TripleBarrierCBFConfig(action_limit=10.0, alpha=1.0))
    position = torch.tensor([[0.31, 0.0]], dtype=torch.float64)
    velocity = torch.zeros((1, 2), dtype=torch.float64)
    obstacle = torch.zeros((1, 1, 2), dtype=torch.float64)
    nominal = torch.tensor([[-10.0, 0.0]], dtype=torch.float64)
    state = _state(position=position, velocity=velocity, obstacles=obstacle)

    result = cbf.filter_action(torch.zeros((1, 8), dtype=torch.float64), nominal, state)
    next_position = cbf.next_position(position, velocity, result.safe_action)
    radii = torch.full((1,), 0.3, dtype=torch.float64)
    h_hard = cbf.h_hard(next_position, obstacle, radii)

    assert bool(torch.all(h_hard >= -1.0e-7).item())


def test_triple_barrier_stopping_guard_prevents_late_arena_exit_at_action_limit() -> None:
    cbf = TripleBarrierCBF(TripleBarrierCBFConfig(action_limit=1.0, alpha=1.0))
    position = torch.tensor([[-2.5, 0.0]], dtype=torch.float64)
    velocity = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    nominal = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    state = _state(position=position, velocity=velocity)

    result = cbf.filter_action(torch.zeros((1, 8), dtype=torch.float64), nominal, state)
    next_position = cbf.next_position(position, velocity, result.safe_action)
    next_velocity = velocity + result.safe_action * cbf.config.dt
    h_stopping = cbf.h_arena_stopping(next_position, next_velocity, torch.tensor([-3.0, 3.0, -3.0, 3.0], dtype=torch.float64))

    assert result.safe_action[0, 0].item() > 0.9
    assert bool(torch.all(h_stopping >= -1.0e-7).item())
    active = cast(torch.Tensor, result.barrier_states["arena_projection"].active)
    assert bool(active[0].item())
