from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
import torch

from parallelcbf.algorithms import (
    CausalTransformerActorCritic,
    CausalTransformerConfig,
    RolloutBuffer,
    RolloutStep,
    TerminationReason,
    compute_gae_in_place,
)
from parallelcbf.algorithms.causal_transformer_ppo import _arena_projection_flags, _potential_phi
from parallelcbf.api import BarrierState, MetricDict, SafeEnv, SafetyFilter, SafetyFilterResult, SafetyState, SafetyWrapper


def _step(
    *,
    env_id: int = 0,
    episode_id: int = 0,
    timestep: int = 0,
    value: float = 0.0,
    reward: float = 1.0,
    reason: str = "ongoing",
    bootstrap_value: float | None = None,
) -> RolloutStep:
    done = reason in {"success", "collision", "out_of_arena", "timeout"}
    return RolloutStep(
        env_id=env_id,
        episode_id=episode_id,
        timestep=timestep,
        context=torch.full((2, 4), float(timestep), dtype=torch.float64),
        observation=torch.full((4,), float(timestep), dtype=torch.float64),
        action=torch.full((3,), float(timestep), dtype=torch.float64),
        log_prob=torch.tensor(0.0, dtype=torch.float64),
        value=value,
        reward=reward,
        shaped_reward=reward,
        done=done,
        terminated=reason in {"success", "collision", "out_of_arena"},
        truncated=reason == "timeout",
        termination_reason=cast(TerminationReason, reason),
        u_nominal=torch.zeros((3,), dtype=torch.float64),
        u_safe=torch.ones((3,), dtype=torch.float64),
        cbf_delta_norm=1.0,
        bootstrap_value=bootstrap_value,
    )


def _tiny_model() -> CausalTransformerActorCritic:
    config = CausalTransformerConfig(d_model=32, n_heads=4, n_layers=1, context_length=4, obs_dim=6, action_dim=3)
    return CausalTransformerActorCritic(config)


def test_rollout_buffer_appends_and_shuffled_batches_cover_all_samples() -> None:
    buffer = RolloutBuffer(context_length=4, obs_dim=4, action_dim=3)
    for index in range(5):
        buffer.append(_step(timestep=index))

    batches = list(buffer.shuffled_batches(batch_size=2, seed=7))
    seen = torch.cat([batch["timestep"] for batch in batches]).tolist()

    assert len(buffer) == 5
    assert sorted(seen) == [0, 1, 2, 3, 4]
    assert batches[0]["context"].shape == (2, 4, 4)
    assert batches[0]["u_nominal"].shape == (2, 3)
    assert batches[0]["u_safe"].shape == (2, 3)


def test_rollout_buffer_iter_episodes_keeps_env_episode_boundaries() -> None:
    buffer = RolloutBuffer(context_length=4, obs_dim=4, action_dim=3)
    buffer.append(_step(env_id=1, episode_id=0, timestep=1))
    buffer.append(_step(env_id=0, episode_id=1, timestep=0))
    buffer.append(_step(env_id=1, episode_id=0, timestep=0))

    grouped = list(buffer.iter_episodes())

    assert [key for key, _ in grouped] == [(0, 1), (1, 0)]
    assert [step.timestep for step in grouped[1][1]] == [0, 1]


def test_gae_zero_bootstrap_on_success_terminal() -> None:
    buffer = RolloutBuffer(context_length=4, obs_dim=4, action_dim=3)
    buffer.append(_step(timestep=0, value=0.5, reward=1.0))
    buffer.append(_step(timestep=1, value=0.25, reward=1.0, reason="success", bootstrap_value=99.0))

    compute_gae_in_place(buffer, gamma=1.0, gae_lambda=1.0)

    assert buffer.steps[1].return_ == pytest.approx(1.0)
    assert buffer.steps[0].return_ == pytest.approx(2.0)


def test_gae_bootstraps_timeout_terminal() -> None:
    buffer = RolloutBuffer(context_length=4, obs_dim=4, action_dim=3)
    buffer.append(_step(timestep=0, value=0.0, reward=1.0, reason="timeout", bootstrap_value=3.0))

    compute_gae_in_place(buffer, gamma=0.5, gae_lambda=1.0)

    assert buffer.steps[0].return_ == pytest.approx(2.5)


def test_gae_bootstraps_ongoing_tail_from_value_head() -> None:
    buffer = RolloutBuffer(context_length=4, obs_dim=4, action_dim=3)
    buffer.append(_step(timestep=0, value=0.0, reward=2.0, reason="ongoing"))

    compute_gae_in_place(buffer, last_values={0: 4.0}, gamma=0.25, gae_lambda=1.0)

    assert buffer.steps[0].return_ == pytest.approx(3.0)


def test_reward_shaping_penalizes_cbf_interventions() -> None:
    model = _tiny_model()

    shaped = model._shape_reward(1.25, 3.0)

    assert shaped == pytest.approx(1.10)


class _FixedDeltaFilter(SafetyFilter[torch.Tensor, torch.Tensor]):
    def reset(self, *, seed: int | None = None) -> None:
        _ = seed

    def filter_action(
        self,
        observation: torch.Tensor,
        nominal_action: torch.Tensor,
        safety_state: SafetyState,
    ) -> SafetyFilterResult[torch.Tensor]:
        _ = observation
        _ = safety_state
        safe_action = nominal_action.clone()
        safe_action[:, 0] += 0.25
        return SafetyFilterResult(
            safe_action=safe_action,
            nominal_action=nominal_action,
            modified=True,
            barrier_states={},
            metrics={"modified": True},
        )

    def barrier_state(self) -> dict[str, BarrierState]:
        return {}

    def metrics(self) -> MetricDict:
        return {"modified": True}


class _TinyTensorEnv(SafeEnv[torch.Tensor, torch.Tensor]):
    metadata: dict[str, Any] = {"render_modes": []}
    reward_range: tuple[float, float] = (-float("inf"), float("inf"))
    spec: Any = None

    def __init__(self) -> None:
        self.observation_space = cast(
            Any,
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        )
        self.action_space = cast(
            Any,
            gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        )
        self.step_count = 0
        self._position = torch.zeros((1, 2), dtype=torch.float32)
        self._velocity = torch.zeros((1, 2), dtype=torch.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        _ = seed
        _ = options
        self.step_count = 0
        self._position.zero_()
        self._velocity.zero_()
        return torch.zeros((1, 6), dtype=torch.float32), {}

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        self._velocity = action[:, :2].detach().to(dtype=torch.float32)
        self._position = self._position + self._velocity * 0.05
        observation = torch.zeros((1, 6), dtype=torch.float32)
        observation[:, :2] = self._position
        reason = "timeout" if self.step_count % 5 == 0 else None
        return observation, 1.0, False, reason == "timeout", {"termination_reason": [reason]}

    def safety_state(self) -> SafetyState:
        return SafetyState(
            position=self._position,
            velocity=self._velocity,
            goal=torch.ones((1, 2), dtype=torch.float32),
            obstacles=torch.zeros((1, 1, 2), dtype=torch.float32),
            robot_radius=0.1,
            obstacle_radius=0.2,
            arena_bounds=torch.tensor([-3.0, 3.0, -3.0, 3.0], dtype=torch.float32),
            metadata={},
        )

    def safety_metrics(self) -> MetricDict:
        return {"h_hard_violation": False}

    def hard_constraint_violations(self) -> dict[str, bool]:
        return {"collision_or_oob": False}

    def render(self) -> None:
        return None


class _LongEpisodeTensorEnv(SafeEnv[torch.Tensor, torch.Tensor]):
    metadata: dict[str, Any] = {"render_modes": []}
    reward_range: tuple[float, float] = (-float("inf"), float("inf"))
    spec: Any = None

    def __init__(self, *, timeout_step: int = 130) -> None:
        self.observation_space = cast(
            Any,
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        )
        self.action_space = cast(
            Any,
            gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        )
        self.timeout_step = timeout_step
        self.reset_calls = 0
        self.reset_done_calls = 0
        self.step_count = 0
        self._position = torch.zeros((1, 2), dtype=torch.float32)
        self._velocity = torch.zeros((1, 2), dtype=torch.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        _ = seed
        _ = options
        self.reset_calls += 1
        self.step_count = 0
        self._position.zero_()
        self._velocity.zero_()
        return self._observation(), {}

    def reset_done(self, done_mask: torch.Tensor) -> torch.Tensor:
        mask = done_mask.detach().cpu().reshape(-1).to(dtype=torch.bool)
        self.reset_done_calls += int(torch.any(mask).item())
        if bool(mask[0].item()):
            self.step_count = 0
            self._position.zero_()
            self._velocity.zero_()
        return self._observation()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        self._velocity = action[:, :2].detach().to(dtype=torch.float32)
        self._position = self._position + self._velocity * 0.05
        done = self.step_count >= self.timeout_step
        reason = "timeout" if done else None
        return self._observation(), 1.0, False, done, {"termination_reason": [reason]}

    def safety_state(self) -> SafetyState:
        return SafetyState(
            position=self._position,
            velocity=self._velocity,
            goal=torch.ones((1, 2), dtype=torch.float32),
            obstacles=torch.zeros((1, 1, 2), dtype=torch.float32),
            robot_radius=0.1,
            obstacle_radius=0.2,
            arena_bounds=torch.tensor([-3.0, 3.0, -3.0, 3.0], dtype=torch.float32),
            metadata={},
        )

    def safety_metrics(self) -> MetricDict:
        return {"h_hard_violation": False}

    def hard_constraint_violations(self) -> dict[str, bool]:
        return {"collision_or_oob": False}

    def render(self) -> None:
        return None

    def _observation(self) -> torch.Tensor:
        observation = torch.zeros((1, 6), dtype=torch.float32)
        observation[:, 0] = float(self.step_count)
        observation[:, 1:3] = self._position
        return observation


class _TerminalPreResetPhiEnv(SafeEnv[torch.Tensor, torch.Tensor]):
    metadata: dict[str, Any] = {"render_modes": []}
    reward_range: tuple[float, float] = (-float("inf"), float("inf"))
    spec: Any = None

    def __init__(self) -> None:
        self.observation_space = cast(
            Any,
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        )
        self.action_space = cast(
            Any,
            gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        )
        self.reset_done_calls = 0
        self._position = torch.zeros((1, 2), dtype=torch.float32)
        self._velocity = torch.zeros((1, 2), dtype=torch.float32)
        self._goal = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        _ = seed
        _ = options
        self._position = torch.zeros((1, 2), dtype=torch.float32)
        self._velocity = torch.zeros((1, 2), dtype=torch.float32)
        return self._observation(), {}

    def reset_done(self, done_mask: torch.Tensor) -> torch.Tensor:
        _ = done_mask
        self.reset_done_calls += 1
        self._position = torch.tensor([[99.0, 99.0]], dtype=torch.float32)
        self._velocity = torch.zeros((1, 2), dtype=torch.float32)
        return self._observation()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        _ = action
        self._position = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
        self._velocity = torch.tensor([[0.5, 0.0]], dtype=torch.float32)
        return self._observation(), 0.0, False, True, {"termination_reason": ["timeout"]}

    def safety_state(self) -> SafetyState:
        return SafetyState(
            position=self._position,
            velocity=self._velocity,
            goal=self._goal,
            obstacles=torch.zeros((1, 1, 2), dtype=torch.float32),
            robot_radius=0.1,
            obstacle_radius=0.2,
            arena_bounds=torch.tensor([-3.0, 3.0, -3.0, 3.0], dtype=torch.float32),
            metadata={},
        )

    def safety_metrics(self) -> MetricDict:
        return {"h_hard_violation": False}

    def hard_constraint_violations(self) -> dict[str, bool]:
        return {"collision_or_oob": False}

    def render(self) -> None:
        return None

    def _observation(self) -> torch.Tensor:
        observation = torch.zeros((1, 6), dtype=torch.float32)
        observation[:, :2] = self._position
        observation[:, 2:4] = self._velocity
        return observation


def test_collect_rollout_fills_nominal_safe_and_cbf_delta() -> None:
    torch.manual_seed(5)
    model = _tiny_model()
    wrapped_env = SafetyWrapper(_TinyTensorEnv(), _FixedDeltaFilter())

    buffer, stats = model.collect_rollout(wrapped_env, num_steps=6, seed=1)

    assert len(buffer) == 6
    assert stats.steps == 6
    assert stats.cbf_active_steps == 6
    assert stats.timeouts == 1
    assert buffer.steps[0].context.shape[-1] == model.config.obs_dim
    assert buffer.steps[0].action.shape == (model.config.action_dim,)
    assert buffer.steps[0].u_nominal.shape == (model.config.action_dim,)
    assert buffer.steps[0].u_safe.shape == (model.config.action_dim,)
    assert buffer.steps[0].cbf_delta_norm == pytest.approx(0.25)
    assert torch.allclose(buffer.steps[0].action, buffer.steps[0].u_nominal)
    assert not torch.allclose(buffer.steps[0].action, buffer.steps[0].u_safe)
    assert buffer.steps[0].log_prob.item() != pytest.approx(0.0)


def test_arena_projection_flags_preserve_per_env_active_bits() -> None:
    barrier = BarrierState(
        h=torch.ones((4,), dtype=torch.float32),
        active=torch.tensor([True, False, True, False]),
        violation=torch.zeros((4,), dtype=torch.bool),
        barrier_type="arena_projection",
        metadata={},
    )
    result = SafetyFilterResult(
        safe_action=torch.zeros((4, 2), dtype=torch.float32),
        nominal_action=torch.zeros((4, 2), dtype=torch.float32),
        modified=True,
        barrier_states={"arena_projection": barrier},
        metrics={"arena_projection_active_count": 2},
    )

    assert _arena_projection_flags({"safety_filter": result}, 4) == [True, False, True, False]


def test_collect_rollout_episode_length_can_exceed_context_boundary() -> None:
    torch.manual_seed(13)
    model = _tiny_model()
    env = _LongEpisodeTensorEnv(timeout_step=130)

    first_buffer, first_stats = model.collect_rollout(env, num_steps=64, seed=4)
    second_buffer, second_stats = model.collect_rollout(env, num_steps=64, seed=999)
    third_buffer, third_stats = model.collect_rollout(env, num_steps=2, seed=1000)

    assert len(first_buffer) == 64
    assert len(second_buffer) == 64
    assert len(third_buffer) == 2
    assert first_stats.timeouts == 0
    assert second_stats.timeouts == 0
    assert third_stats.timeouts == 1
    assert third_stats.episode_length_max == pytest.approx(130.0)
    assert env.reset_calls == 1
    assert env.reset_done_calls == 1
    assert third_buffer.steps[-1].timestep == 129


def test_collect_rollout_hidden_persists_until_done_then_resets_lane() -> None:
    torch.manual_seed(17)
    model = _tiny_model()
    env = _LongEpisodeTensorEnv(timeout_step=6)

    first_buffer, _ = model.collect_rollout(env, num_steps=3, seed=8)
    assert model._last_hidden_state is not None
    first_hidden_len = model._last_hidden_state.obs_buffer.shape[1]

    second_buffer, second_stats = model.collect_rollout(env, num_steps=3, seed=9)
    assert second_stats.timeouts == 1
    assert model._last_hidden_state is not None
    assert first_hidden_len == 3
    assert model._last_hidden_state.obs_buffer.shape[1] == model.config.context_length
    assert second_buffer.steps[0].timestep == 3
    assert second_buffer.steps[-1].timestep == 5
    assert env.reset_calls == 1
    assert env.reset_done_calls == 1
    assert torch.allclose(model._last_hidden_state.obs_buffer, torch.zeros_like(model._last_hidden_state.obs_buffer))


def test_potential_phi_handles_exact_zero_distance() -> None:
    phi = _potential_phi(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
    )

    assert torch.isfinite(phi).all()
    assert phi.item() == pytest.approx(-0.5)


def test_potential_phi_handles_epsilon_distance_without_nan() -> None:
    phi = _potential_phi(
        torch.tensor([[1.0e-9, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[0.0, 0.0]]),
    )

    assert torch.isfinite(phi).all()
    assert phi.item() == pytest.approx(-0.500000001)


def test_potential_phi_goal_with_zero_velocity_is_finite() -> None:
    phi = _potential_phi(
        torch.tensor([[2.0, 3.0]]),
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([[2.0, 3.0]]),
    )

    assert torch.isfinite(phi).all()
    assert phi.item() == pytest.approx(-0.5)


def test_potential_phi_goal_with_nonzero_velocity_is_finite() -> None:
    phi = _potential_phi(
        torch.tensor([[2.0, 3.0]]),
        torch.tensor([[4.0, -7.0]]),
        torch.tensor([[2.0, 3.0]]),
    )

    assert torch.isfinite(phi).all()
    assert phi.item() == pytest.approx(-0.5)


def test_collect_rollout_terminal_phi_next_uses_pre_reset_state() -> None:
    model = _tiny_model()
    env = _TerminalPreResetPhiEnv()

    buffer, stats = model.collect_rollout(env, num_steps=1, seed=21)

    expected_phi_current = -1.5
    expected_phi_next_pre_reset = -0.75
    assert env.reset_done_calls == 1
    assert stats.timeouts == 1
    assert buffer.steps[0].termination_reason == "timeout"
    assert buffer.steps[0].shaped_reward == pytest.approx(
        model.gamma * expected_phi_next_pre_reset - expected_phi_current
    )
