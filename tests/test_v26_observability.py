from __future__ import annotations

import pytest
import torch

from parallelcbf.algorithms import CausalTransformerActorCritic, CausalTransformerConfig
from parallelcbf.algorithms.causal_transformer_ppo import (
    _explained_variance,
    _is_zero_variance,
    _rolling_slope,
)
from parallelcbf.ops import DefaultWatchdogRegistry, SustainedPhaseWatchdog, ThresholdWatchdog


def _tiny_model() -> CausalTransformerActorCritic:
    config = CausalTransformerConfig(d_model=32, n_heads=4, n_layers=1, context_length=4, obs_dim=6, action_dim=3)
    return CausalTransformerActorCritic(config)


def test_explained_variance_matches_closed_form_and_handles_zero_variance() -> None:
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    predictions = torch.tensor([1.0, 2.0, 2.0, 2.0], dtype=torch.float64)

    expected = 1.0 - torch.var(targets - predictions, unbiased=False) / torch.var(targets, unbiased=False)

    assert _explained_variance(predictions, targets).item() == pytest.approx(float(expected.item()))
    assert _explained_variance(torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)).item() == 0.0


def test_rolling_slope_detects_increasing_and_flat_value_loss() -> None:
    assert _rolling_slope([1.0, 2.0, 3.0, 4.0]) > 0.0
    assert _rolling_slope([4.0, 3.0, 2.0, 1.0]) < 0.0
    assert _rolling_slope([2.0, 2.0, 2.0, 2.0]) == pytest.approx(0.0)


def test_zero_variance_detector_separates_flat_and_noisy_returns() -> None:
    assert _is_zero_variance([0.0, 0.0, 0.0, 0.0])
    assert not _is_zero_variance([0.0, 0.0, 1.0, 0.0])


def test_v26_watchdogs_trip_new_diagnostic_rules() -> None:
    registry = DefaultWatchdogRegistry()
    registry.register(SustainedPhaseWatchdog("phase", 0, 1_500_000, "critic_warmup_did_not_converge"))
    registry.register(ThresholdWatchdog("value_loss_slope", 0.0, label="value_loss_increasing"))
    registry.register(ThresholdWatchdog("episode_return_zero_variance", 0.5, label="episode_return_zero_variance"))
    registry.register(ThresholdWatchdog("explained_variance_stuck_negative", 0.5, label="explained_variance_stuck_negative"))

    events = registry.update(
        {
            "phase": 0,
            "value_loss_slope": 0.1,
            "episode_return_zero_variance": 1.0,
            "explained_variance_stuck_negative": 1.0,
        },
        step=1_500_000,
    )

    assert [event.name for event in events] == [
        "critic_warmup_did_not_converge",
        "value_loss_increasing",
        "episode_return_zero_variance",
        "explained_variance_stuck_negative",
    ]
    assert registry.should_halt()


def test_critic_lr_cosine_decay_updates_only_critic_param_group() -> None:
    model = _tiny_model()
    assert model.optimizer is not None
    actor_lr_before = float(model.optimizer.param_groups[model._actor_param_group_index]["lr"])

    critic_lr_mid = model.step_critic_lr_cosine(step=50, total_steps=100)
    critic_lr_end = model.step_critic_lr_cosine(step=100, total_steps=100)

    assert float(model.optimizer.param_groups[model._actor_param_group_index]["lr"]) == pytest.approx(actor_lr_before)
    assert critic_lr_mid < float(model.optimizer.param_groups[model._critic_param_group_index]["base_lr"])
    assert critic_lr_end == pytest.approx(float(model.optimizer.param_groups[model._critic_param_group_index]["base_lr"]) * 0.1)
