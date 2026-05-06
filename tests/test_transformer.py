from __future__ import annotations

import pytest
import torch

from parallelcbf.algorithms import CausalTransformer, CausalTransformerBC, CausalTransformerConfig


def _small_config() -> CausalTransformerConfig:
    return CausalTransformerConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        context_length=8,
        obs_dim=16,
        action_dim=3,
    )


@pytest.mark.parametrize("num_envs", [1, 2, 4])
def test_causal_transformer_shape_correctness(num_envs: int) -> None:
    config = _small_config()
    model = CausalTransformer(config)
    observations = torch.randn((num_envs, config.context_length, config.obs_dim), dtype=torch.float64)
    actions = model(observations)
    assert actions.shape == (num_envs, config.context_length, config.action_dim)
    assert model.forward_last(observations).shape == (num_envs, config.action_dim)


def test_causal_mask_prevents_future_token_influence() -> None:
    torch.manual_seed(17)
    config = _small_config()
    model = CausalTransformer(config)
    model.eval()
    prefix_len = 4
    base = torch.randn((2, config.context_length, config.obs_dim), dtype=torch.float64)
    changed_future = base.clone()
    changed_future[:, prefix_len:, :] = torch.randn_like(changed_future[:, prefix_len:, :]) * 100.0
    with torch.no_grad():
        base_actions = model(base)
        changed_actions = model(changed_future)
    assert torch.allclose(base_actions[:, :prefix_len, :], changed_actions[:, :prefix_len, :], atol=1.0e-10)


def test_predict_hidden_state_round_trip() -> None:
    torch.manual_seed(31)
    config = _small_config()
    algo = CausalTransformerBC(config)
    hidden = algo.initial_hidden_state(batch_size=2, dtype=torch.float64)
    obs_1 = torch.randn((2, config.obs_dim), dtype=torch.float64)
    pred_1 = algo.predict(obs_1, hidden, deterministic=True)
    assert pred_1.action.shape == (2, config.action_dim)
    assert pred_1.hidden_state.obs_buffer.shape == (2, 1, config.obs_dim)
    obs_2 = torch.randn((2, config.obs_dim), dtype=torch.float64)
    pred_2 = algo.predict(obs_2, pred_1.hidden_state, deterministic=True)
    assert pred_2.action.shape == (2, config.action_dim)
    assert pred_2.hidden_state.obs_buffer.shape == (2, 2, config.obs_dim)
    assert torch.allclose(pred_2.hidden_state.obs_buffer[:, 0, :], obs_1)
    assert torch.allclose(pred_2.hidden_state.obs_buffer[:, 1, :], obs_2)


def test_predict_hidden_state_respects_context_length() -> None:
    config = _small_config()
    algo = CausalTransformerBC(config)
    hidden = algo.initial_hidden_state(batch_size=1, dtype=torch.float64)
    for step in range(config.context_length + 3):
        obs = torch.full((1, config.obs_dim), float(step), dtype=torch.float64)
        prediction = algo.predict(obs, hidden, deterministic=True)
        hidden = prediction.hidden_state
    assert hidden.obs_buffer.shape == (1, config.context_length, config.obs_dim)
    assert torch.all(hidden.obs_buffer[:, 0, :] == 3.0)
    assert torch.all(hidden.obs_buffer[:, -1, :] == float(config.context_length + 2))
