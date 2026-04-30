from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.autograd import gradcheck

from parallelcbf.api import SafetyState
from parallelcbf.envs import Toy2DAvoidanceEnv
from parallelcbf.safety import ChanceConstrainedDualBarrierCBF, DualBarrierCBF, DualBarrierCBFConfig


def _state(
    *,
    position: torch.Tensor,
    velocity: torch.Tensor,
    obstacles: torch.Tensor,
    robot_radius: float = 0.2,
    obstacle_radius: float = 0.3,
) -> SafetyState:
    batch = position.shape[0]
    return SafetyState(
        position=position,
        velocity=velocity,
        goal=torch.zeros((batch, 2), dtype=position.dtype),
        obstacles=obstacles,
        robot_radius=robot_radius,
        obstacle_radius=obstacle_radius,
        arena_bounds=torch.tensor([-5.0, 5.0, -5.0, 5.0], dtype=position.dtype),
        metadata={},
    )


@st.composite
def safe_torch_state(draw: st.DrawFn) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = draw(st.integers(min_value=1, max_value=8))
    angles = []
    clearances = []
    actions = []
    for _ in range(batch):
        angles.append(draw(st.floats(min_value=-3.14159, max_value=3.14159, allow_nan=False, allow_infinity=False)))
        clearances.append(draw(st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False)))
        actions.append(
            [
                draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
                draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
            ]
        )
    radius = 0.5
    dirs = torch.tensor([[np.cos(angle), np.sin(angle)] for angle in angles], dtype=torch.float64)
    position = dirs * torch.tensor([[radius + clearance] for clearance in clearances], dtype=torch.float64)
    velocity = torch.zeros((batch, 2), dtype=torch.float64)
    obstacles = torch.zeros((batch, 1, 2), dtype=torch.float64)
    nominal_action = torch.tensor(actions, dtype=torch.float64)
    return position, velocity, obstacles, nominal_action


@given(safe_torch_state())
@settings(max_examples=80, deadline=None)
def test_dual_barrier_positive_invariance(data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    position, velocity, obstacles, nominal_action = data
    cbf = DualBarrierCBF(DualBarrierCBFConfig(action_limit=1.0))
    state = _state(position=position, velocity=velocity, obstacles=obstacles)
    result = cbf.filter_action(torch.zeros((position.shape[0], 8), dtype=torch.float64), nominal_action, state)
    next_position = cbf.next_position(position, velocity, result.safe_action)
    h_hard = cbf.h_hard(next_position, obstacles, torch.full((position.shape[0],), 0.5, dtype=torch.float64))
    assert bool(torch.all(h_hard >= -1.0e-7).item())


@pytest.mark.parametrize("num_envs", [1, 2, 4, 16])
def test_dual_barrier_vectorization_shapes(num_envs: int) -> None:
    position = torch.zeros((num_envs, 2), dtype=torch.float64)
    position[:, 0] = 1.0
    velocity = torch.zeros((num_envs, 2), dtype=torch.float64)
    obstacles = torch.zeros((num_envs, 3, 2), dtype=torch.float64)
    action = torch.zeros((num_envs, 2), dtype=torch.float64)
    cbf = DualBarrierCBF()
    state = _state(position=position, velocity=velocity, obstacles=obstacles)
    result = cbf.filter_action(torch.zeros((num_envs, 8), dtype=torch.float64), action, state)
    assert result.safe_action.shape == (num_envs, 2)
    assert result.barrier_states["hard_squared_distance"].h.shape == (num_envs, 3)
    assert result.barrier_states["soft_linear_distance"].h.shape == (num_envs, 3)


def test_dual_barrier_rejects_non_batched_action() -> None:
    cbf = DualBarrierCBF()
    position = torch.zeros((1, 2), dtype=torch.float64)
    velocity = torch.zeros((1, 2), dtype=torch.float64)
    obstacles = torch.zeros((1, 1, 2), dtype=torch.float64)
    state = _state(position=position, velocity=velocity, obstacles=obstacles)
    with pytest.raises(ValueError, match="shape"):
        cbf.filter_action(torch.zeros((1, 8), dtype=torch.float64), torch.zeros((2,), dtype=torch.float64), state)


def test_dual_barrier_rejects_mismatched_position_shape() -> None:
    cbf = DualBarrierCBF()
    position = torch.zeros((2, 2), dtype=torch.float64)
    velocity = torch.zeros((2, 2), dtype=torch.float64)
    obstacles = torch.zeros((2, 1, 2), dtype=torch.float64)
    state = _state(position=position, velocity=velocity, obstacles=obstacles)
    with pytest.raises(ValueError, match="position"):
        cbf.filter_action(torch.zeros((1, 8), dtype=torch.float64), torch.zeros((1, 2), dtype=torch.float64), state)


def test_dual_barrier_rejects_invalid_obstacle_shape() -> None:
    cbf = DualBarrierCBF()
    position = torch.zeros((1, 2), dtype=torch.float64)
    velocity = torch.zeros((1, 2), dtype=torch.float64)
    bad_obstacles = torch.zeros((1, 2), dtype=torch.float64)
    state = _state(position=position, velocity=velocity, obstacles=bad_obstacles)
    with pytest.raises(ValueError, match="obstacles"):
        cbf.filter_action(torch.zeros((1, 8), dtype=torch.float64), torch.zeros((1, 2), dtype=torch.float64), state)


def test_dual_barrier_metrics_and_reset_are_stable() -> None:
    cbf = DualBarrierCBF()
    position = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    velocity = torch.zeros((1, 2), dtype=torch.float64)
    obstacles = torch.zeros((1, 1, 2), dtype=torch.float64)
    state = _state(position=position, velocity=velocity, obstacles=obstacles)
    cbf.filter_action(torch.zeros((1, 8), dtype=torch.float64), torch.zeros((1, 2), dtype=torch.float64), state)
    assert "h_hard_min" in cbf.metrics()
    assert "hard_squared_distance" in cbf.barrier_state()
    cbf.reset()
    assert cbf.barrier_state() == {}


def test_chance_constrained_dual_barrier_validates_parameters() -> None:
    with pytest.raises(ValueError, match="delta"):
        ChanceConstrainedDualBarrierCBF(delta=0.0)
    with pytest.raises(ValueError, match="sigma_d"):
        ChanceConstrainedDualBarrierCBF(sigma_d=-1.0)


def test_dual_barrier_gradcheck_filter_action() -> None:
    cbf = DualBarrierCBF(DualBarrierCBFConfig(action_limit=10.0))
    position = torch.tensor([[0.8, 0.0]], dtype=torch.float64, requires_grad=True)
    velocity = torch.zeros((1, 2), dtype=torch.float64)
    obstacles = torch.zeros((1, 1, 2), dtype=torch.float64)
    nominal_action = torch.tensor([[-0.2, 0.1]], dtype=torch.float64, requires_grad=True)

    def wrapped(pos: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        state = _state(position=pos, velocity=velocity, obstacles=obstacles)
        return cbf.filter_action(torch.zeros((1, 8), dtype=torch.float64), act, state).safe_action

    assert bool(gradcheck(wrapped, (position, nominal_action), eps=1.0e-6, atol=1.0e-4))


def test_chance_constrained_dual_barrier_degenerate_matches_base() -> None:
    position = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    velocity = torch.zeros((1, 2), dtype=torch.float64)
    obstacles = torch.zeros((1, 1, 2), dtype=torch.float64)
    action = torch.tensor([[-1.0, 0.0]], dtype=torch.float64)
    state = _state(position=position, velocity=velocity, obstacles=obstacles)
    base = DualBarrierCBF()
    chance = ChanceConstrainedDualBarrierCBF(sigma_d=0.0)
    base_action = base.filter_action(torch.zeros((1, 8), dtype=torch.float64), action, state).safe_action
    chance_action = chance.filter_action(torch.zeros((1, 8), dtype=torch.float64), action, state).safe_action
    assert torch.allclose(base_action, chance_action)


def test_dual_barrier_toy2d_combination_is_deterministic_over_100_steps() -> None:
    env_a = Toy2DAvoidanceEnv()
    env_b = Toy2DAvoidanceEnv()
    obs_a, _ = env_a.reset(seed=404)
    obs_b, _ = env_b.reset(seed=404)
    cbf_a = DualBarrierCBF()
    cbf_b = DualBarrierCBF()
    rng = np.random.default_rng(91)
    for _ in range(100):
        nominal_np = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float64)
        nominal = torch.tensor(nominal_np.reshape(1, 2), dtype=torch.float64)
        state_a = env_a.safety_state()
        state_b = env_b.safety_state()
        torch_state_a = _state(
            position=torch.as_tensor(state_a.position, dtype=torch.float64).reshape(1, 2),
            velocity=torch.as_tensor(state_a.velocity, dtype=torch.float64).reshape(1, 2),
            obstacles=torch.as_tensor(state_a.obstacles, dtype=torch.float64).reshape(1, 1, 2),
            robot_radius=state_a.robot_radius,
            obstacle_radius=state_a.obstacle_radius,
        )
        torch_state_b = _state(
            position=torch.as_tensor(state_b.position, dtype=torch.float64).reshape(1, 2),
            velocity=torch.as_tensor(state_b.velocity, dtype=torch.float64).reshape(1, 2),
            obstacles=torch.as_tensor(state_b.obstacles, dtype=torch.float64).reshape(1, 1, 2),
            robot_radius=state_b.robot_radius,
            obstacle_radius=state_b.obstacle_radius,
        )
        filtered_a = cbf_a.filter_action(torch.as_tensor(obs_a, dtype=torch.float64).reshape(1, 8), nominal, torch_state_a)
        filtered_b = cbf_b.filter_action(torch.as_tensor(obs_b, dtype=torch.float64).reshape(1, 8), nominal, torch_state_b)
        obs_a, reward_a, terminated_a, truncated_a, _ = env_a.step(
            filtered_a.safe_action.detach().numpy().reshape(2).astype(np.float32)
        )
        obs_b, reward_b, terminated_b, truncated_b, _ = env_b.step(
            filtered_b.safe_action.detach().numpy().reshape(2).astype(np.float32)
        )
        assert np.allclose(obs_a, obs_b)
        assert reward_a == pytest.approx(reward_b)
        assert terminated_a == terminated_b
        assert truncated_a == truncated_b
