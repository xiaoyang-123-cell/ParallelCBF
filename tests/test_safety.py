from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from parallelcbf.api import BarrierState, MetricDict, SafetyFilter, SafetyFilterResult, SafetyState, SafetyWrapper
from parallelcbf.envs import Toy2DAvoidanceEnv
from parallelcbf.safety import NaiveDistanceCBF


ArrayF32 = NDArray[np.float32]


class IdentitySafetyFilter(SafetyFilter[ArrayF32, ArrayF32]):
    def reset(self, *, seed: int | None = None) -> None:
        _ = seed

    def filter_action(
        self,
        observation: ArrayF32,
        nominal_action: ArrayF32,
        safety_state: SafetyState,
    ) -> SafetyFilterResult[ArrayF32]:
        _ = observation
        _ = safety_state
        return SafetyFilterResult(
            safe_action=nominal_action,
            nominal_action=nominal_action,
            modified=False,
            barrier_states={},
            metrics={"identity_filter": True},
        )

    def barrier_state(self) -> dict[str, BarrierState]:
        return {}

    def metrics(self) -> MetricDict:
        return {"identity_filter": True}


def test_toy2d_runs_100_steps_with_valid_safety_state() -> None:
    env = Toy2DAvoidanceEnv()
    observation, info = env.reset(seed=7)
    assert observation.shape == (8,)
    assert "safety_metrics" in info
    for _ in range(100):
        observation, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        state = env.safety_state()
        assert observation.shape == (8,)
        assert state.position.shape == (2,)
        assert state.velocity.shape == (2,)
        assert state.obstacles.shape == (1, 2)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "hard_constraint_violations" in info


def test_toy2d_is_deterministic_for_identical_seed_over_100_steps() -> None:
    env_a = Toy2DAvoidanceEnv()
    env_b = Toy2DAvoidanceEnv()
    obs_a, _ = env_a.reset(seed=2024)
    obs_b, _ = env_b.reset(seed=2024)
    assert np.allclose(obs_a, obs_b)
    rng = np.random.default_rng(77)
    for _ in range(100):
        action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        step_a = env_a.step(action)
        step_b = env_b.step(action)
        assert np.allclose(step_a[0], step_b[0])
        assert step_a[1] == pytest.approx(step_b[1])
        assert step_a[2] == step_b[2]
        assert step_a[3] == step_b[3]


def test_safety_wrapper_requires_reset_before_step() -> None:
    wrapped = SafetyWrapper(Toy2DAvoidanceEnv(), IdentitySafetyFilter())
    with pytest.raises(RuntimeError, match="before reset"):
        wrapped.step(np.zeros(2, dtype=np.float32))


def test_safety_wrapper_caches_reset_and_step_observations() -> None:
    wrapped = SafetyWrapper(Toy2DAvoidanceEnv(), IdentitySafetyFilter())
    observation, _ = wrapped.reset(seed=11)
    assert observation.shape == (8,)
    next_observation, _, _, _, info = wrapped.step(np.zeros(2, dtype=np.float32))
    assert next_observation.shape == (8,)
    assert info["safety_filter"].modified is False
    assert wrapped.safety_state().position.shape == (2,)
    assert "identity_filter" in wrapped.safety_metrics()
    assert "collision_or_oob" in wrapped.hard_constraint_violations()


@st.composite
def safe_state_and_action(draw: st.DrawFn) -> tuple[SafetyState, ArrayF32]:
    radius = draw(st.floats(min_value=0.2, max_value=1.0, allow_nan=False, allow_infinity=False))
    clearance = draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    angle = draw(st.floats(min_value=-3.14159, max_value=3.14159, allow_nan=False, allow_infinity=False))
    radial_speed = draw(st.floats(min_value=0.0, max_value=0.25, allow_nan=False, allow_infinity=False))
    tangential_speed = draw(st.floats(min_value=-0.10, max_value=0.10, allow_nan=False, allow_infinity=False))
    action_x = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    action_y = draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    obstacle = np.array([0.0, 0.0], dtype=np.float32)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    tangent = np.array([-direction[1], direction[0]], dtype=np.float32)
    position = direction * np.float32(radius + clearance)
    velocity = direction * np.float32(radial_speed) + tangent * np.float32(tangential_speed)
    state = SafetyState(
        position=position.astype(np.float32),
        velocity=velocity.astype(np.float32),
        goal=np.array([2.0, 0.0], dtype=np.float32),
        obstacles=obstacle.reshape(1, 2),
        robot_radius=float(radius * 0.5),
        obstacle_radius=float(radius * 0.5),
        arena_bounds=np.array([-5.0, 5.0, -5.0, 5.0], dtype=np.float32),
        metadata={},
    )
    action = np.array([action_x, action_y], dtype=np.float32)
    return state, action


@given(safe_state_and_action())
@settings(max_examples=200, deadline=None)
def test_naive_distance_cbf_preserves_nonnegative_h_hard(data: tuple[SafetyState, ArrayF32]) -> None:
    state, nominal_action = data
    cbf = NaiveDistanceCBF()
    result = cbf.filter_action(np.zeros(8, dtype=np.float32), nominal_action, state)
    next_position = (
        np.asarray(state.position, dtype=np.float32)
        + np.asarray(state.velocity, dtype=np.float32) * np.float32(cbf.config.dt)
        + result.safe_action * np.float32(cbf.config.dt * cbf.config.dt)
    )
    obstacle = np.asarray(state.obstacles, dtype=np.float32).reshape(1, 2)[0]
    h_hard = float(np.linalg.norm(next_position - obstacle) - (state.robot_radius + state.obstacle_radius))
    assert h_hard >= -1.0e-5
