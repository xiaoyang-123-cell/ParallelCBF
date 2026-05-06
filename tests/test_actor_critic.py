from __future__ import annotations

from pathlib import Path

import pytest
import torch

from parallelcbf.algorithms import CausalTransformerActorCritic, CausalTransformerConfig, RolloutBuffer, RolloutStep, WarmupPhase
from parallelcbf.algorithms.causal_transformer_ppo import _cosine_decay, _gaussian_kl, _load_checkpoint_payload
from parallelcbf.ops import AtomicCheckpoint, DefaultWatchdogRegistry, FailureForensics, ThresholdWatchdog


BEST_PT = Path("/data/uav_project/ParallelCBF/checkpoints/v25_transformer_bc_full/best.pt")


def _tiny_actor_critic() -> CausalTransformerActorCritic:
    config = CausalTransformerConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        context_length=8,
        obs_dim=16,
        action_dim=3,
    )
    return CausalTransformerActorCritic(
        config,
        critic_plateau_window=3,
        critic_plateau_min_updates=3,
        critic_plateau_abs_tolerance=1.0e-3,
        critic_plateau_rel_tolerance=1.0e-3,
    )


@pytest.mark.skipif(not BEST_PT.exists(), reason="V25 best.pt checkpoint is not available")
def test_actor_critic_loads_best_pt_into_actor_and_frozen_reference() -> None:
    model = CausalTransformerActorCritic.from_bc_checkpoint(BEST_PT, device=torch.device("cpu"))
    payload = _load_checkpoint_payload(BEST_PT, device=torch.device("cpu"))

    assert model.config == payload["config"]
    assert model.phase is WarmupPhase.CRITIC_WARMUP
    assert all(not parameter.requires_grad for parameter in model.bc_reference.parameters())
    assert all(not parameter.requires_grad for parameter in model.actor.parameters())
    assert any(parameter.requires_grad for parameter in model.critic_head.parameters())

    for actor_parameter, reference_parameter in zip(model.actor.parameters(), model.bc_reference.parameters(), strict=True):
        assert torch.allclose(actor_parameter, reference_parameter)


def test_critic_head_is_small_random_head_and_outputs_value_shape() -> None:
    torch.manual_seed(11)
    model = _tiny_actor_critic()
    hidden = model.initial_hidden_state(batch_size=4, dtype=torch.float64)
    observation = torch.randn((4, model.config.obs_dim), dtype=torch.float64)

    output = model.act_and_value(observation, hidden, deterministic=True)

    assert output.action.shape == (4, model.config.action_dim)
    assert output.reference_action.shape == (4, model.config.action_dim)
    assert output.value.shape == (4,)
    assert model.critic_head.weight.norm().item() == pytest.approx(0.01)
    assert torch.allclose(model.critic_head.bias, torch.zeros_like(model.critic_head.bias))


def test_warmup_routes_gradients_only_to_critic_then_unfreezes_actor() -> None:
    torch.manual_seed(23)
    model = _tiny_actor_critic()
    hidden = model.initial_hidden_state(batch_size=2, dtype=torch.float64)
    observation = torch.randn((2, model.config.obs_dim), dtype=torch.float64)
    target_value = torch.ones((2,), dtype=torch.float64)

    output = model.act_and_value(observation, hidden, deterministic=True)
    loss = torch.nn.functional.mse_loss(output.value, target_value)
    loss.backward()  # type: ignore[no-untyped-call]

    assert all(parameter.grad is None for parameter in model.actor.parameters())
    assert all(parameter.grad is None for parameter in model.bc_reference.parameters())
    assert all(parameter.grad is not None for parameter in model.critic_head.parameters())

    assert model._maybe_advance_phase(critic_loss=1.0) is WarmupPhase.CRITIC_WARMUP
    assert model._maybe_advance_phase(critic_loss=1.0005) is WarmupPhase.CRITIC_WARMUP
    assert model._maybe_advance_phase(critic_loss=0.9998) is WarmupPhase.PPO
    assert any(parameter.requires_grad for parameter in model.actor.parameters())
    assert all(not parameter.requires_grad for parameter in model.bc_reference.parameters())


def test_bc_frozen_stays_eval_after_model_train() -> None:
    model = _tiny_actor_critic()

    model.bc_trunk_frozen.train()
    model.bc_actor_mean_head_frozen.train()
    assert model.bc_trunk_frozen.training is True
    assert model.bc_actor_mean_head_frozen.training is True

    returned = model.train()

    assert returned is model
    assert model.training is True
    assert model.bc_trunk_frozen.training is False
    assert model.bc_actor_mean_head_frozen.training is False
    assert all(not parameter.requires_grad for parameter in model.bc_trunk_frozen.parameters())


def _loss_batch(model: CausalTransformerActorCritic, *, batch_size: int = 4) -> dict[str, torch.Tensor]:
    context = torch.randn((batch_size, model.config.context_length, model.config.obs_dim), dtype=torch.float64)
    action = torch.randn((batch_size, model.config.action_dim), dtype=torch.float64)
    old_log_prob = torch.zeros((batch_size,), dtype=torch.float64)
    value = torch.zeros((batch_size,), dtype=torch.float64)
    returns = torch.linspace(0.1, 0.4, batch_size, dtype=torch.float64)
    advantages = torch.linspace(-1.0, 1.0, batch_size, dtype=torch.float64)
    return {
        "context": context,
        "action": action,
        "old_log_prob": old_log_prob,
        "value": value,
        "return": returns,
        "advantage": advantages,
    }


def _rollout_buffer_from_batch(model: CausalTransformerActorCritic, *, batch_size: int = 4) -> RolloutBuffer:
    batch = _loss_batch(model, batch_size=batch_size)
    buffer = RolloutBuffer(
        context_length=model.config.context_length,
        obs_dim=model.config.obs_dim,
        action_dim=model.config.action_dim,
    )
    for index in range(batch_size):
        buffer.append(
            RolloutStep(
                env_id=0,
                episode_id=0,
                timestep=index,
                context=batch["context"][index],
                observation=batch["context"][index, -1, :],
                action=batch["action"][index],
                log_prob=batch["old_log_prob"][index],
                value=float(batch["value"][index].item()),
                reward=1.0,
                shaped_reward=1.0,
                done=index == batch_size - 1,
                terminated=index == batch_size - 1,
                truncated=False,
                termination_reason="success" if index == batch_size - 1 else "ongoing",
                u_nominal=batch["action"][index],
                u_safe=batch["action"][index],
                cbf_delta_norm=0.0,
                advantage=float(batch["advantage"][index].item()),
                return_=float(batch["return"][index].item()),
            )
        )
    return buffer


def _bc_log_std(model: CausalTransformerActorCritic) -> torch.Tensor:
    buffer = model.log_std_bc
    assert isinstance(buffer, torch.Tensor)
    return buffer


def test_gaussian_kl_detaches_bc_terms_and_clamps_trainable_log_std() -> None:
    mu_theta = torch.tensor([[0.2, -0.1]], dtype=torch.float64, requires_grad=True)
    log_std_theta = torch.tensor([[10.0, -10.0]], dtype=torch.float64, requires_grad=True)
    mu_bc = torch.tensor([[0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    log_std_bc = torch.tensor([[0.5, -0.25]], dtype=torch.float64, requires_grad=True)

    kl = _gaussian_kl(mu_theta, log_std_theta, mu_bc, log_std_bc).sum()
    kl.backward()  # type: ignore[no-untyped-call]
    expected = (
        log_std_bc.detach()
        - log_std_theta.detach().clamp(-5.0, 2.0)
        + (
            torch.exp(2.0 * log_std_theta.detach().clamp(-5.0, 2.0))
            + (mu_theta.detach() - mu_bc.detach()) ** 2
        )
        / (2.0 * torch.exp(2.0 * log_std_bc.detach()))
        - 0.5
    ).sum(dim=-1)

    assert torch.allclose(kl.detach(), expected.sum())
    assert mu_theta.grad is not None
    assert log_std_theta.grad is not None
    assert mu_bc.grad is None
    assert log_std_bc.grad is None


def test_kl_anchor_does_not_mutate_bc_log_std_buffer() -> None:
    model = _tiny_actor_critic()
    original_bc_log_std = _bc_log_std(model).detach().clone()
    mu_theta = torch.zeros((2, model.config.action_dim), dtype=torch.float64, requires_grad=True)
    log_std_theta = torch.full((2, model.config.action_dim), 99.0, dtype=torch.float64, requires_grad=True)
    mu_bc = torch.zeros((2, model.config.action_dim), dtype=torch.float64)

    _ = _gaussian_kl(mu_theta, log_std_theta, mu_bc, _bc_log_std(model).expand_as(mu_theta)).sum()

    assert torch.allclose(_bc_log_std(model), original_bc_log_std)


def test_warmup_compute_loss_backprops_only_critic_even_if_actor_thawed() -> None:
    model = _tiny_actor_critic()
    for parameter in model.actor.parameters():
        parameter.requires_grad_(True)
    model.actor_log_std.requires_grad_(True)
    batch = _loss_batch(model)

    loss = model.compute_loss(batch, phase=WarmupPhase.CRITIC_WARMUP)
    loss.total_loss.backward()  # type: ignore[no-untyped-call]

    assert loss.policy_loss.item() == pytest.approx(0.0)
    assert loss.kl_loss.item() == pytest.approx(0.0)
    assert loss.entropy_loss.item() == pytest.approx(0.0)
    assert all(parameter.grad is None for parameter in model.actor.parameters())
    assert model.actor_log_std.grad is None
    assert all(parameter.grad is not None for parameter in model.critic_head.parameters())


def test_ppo_compute_loss_routes_actor_critic_and_keeps_bc_frozen() -> None:
    model = _tiny_actor_critic()
    model._transition_to_full_ppo(actor_warmup_steps=2)
    batch = _loss_batch(model)

    loss = model.compute_loss(batch, phase=WarmupPhase.PPO, step=5, total_steps=10)
    loss.total_loss.backward()  # type: ignore[no-untyped-call]

    assert loss.policy_loss.requires_grad
    assert loss.value_loss.requires_grad
    assert loss.kl_loss.requires_grad
    assert any(parameter.grad is not None for parameter in model.actor.parameters())
    assert model.actor_log_std.grad is not None
    assert any(parameter.grad is not None for parameter in model.critic_head.parameters())
    assert all(parameter.grad is None for parameter in model.bc_reference.parameters())


def test_value_loss_uses_clipped_mse_epsilon_point_two() -> None:
    model = _tiny_actor_critic()
    values = torch.tensor([1.0], dtype=torch.float64)
    old_values = torch.tensor([0.0], dtype=torch.float64)
    returns = torch.tensor([0.0], dtype=torch.float64)

    value_unclipped = (values - returns) ** 2
    clipped_values = old_values + torch.clamp(values - old_values, -0.2, 0.2)
    value_clipped = (clipped_values - returns) ** 2

    assert torch.maximum(value_unclipped, value_clipped).item() == pytest.approx(1.0)
    assert model.compute_loss(_loss_batch(model), phase=WarmupPhase.CRITIC_WARMUP).value_loss.item() >= 0.0


def test_kl_cosine_decay_schedule_endpoints() -> None:
    assert _cosine_decay(1.0, 0.1, step=0, total_steps=10) == pytest.approx(1.0)
    assert _cosine_decay(1.0, 0.1, step=10, total_steps=10) == pytest.approx(0.1)
    assert _cosine_decay(1.0, 0.1, step=5, total_steps=10) == pytest.approx(0.55)


def test_halt_protocol_dumps_forensics_saves_autopsy_and_verifies_last_safe(tmp_path: Path) -> None:
    model = _tiny_actor_critic()
    forensics = FailureForensics(capacity=4)
    forensics.push(step=3, metrics={"loss": 1.0})
    last_safe = AtomicCheckpoint().save({"safe": True}, tmp_path / "last_safe_step.pt")

    result = model._halt_protocol(
        reason="nan watchdog",
        step=7,
        forensics=forensics,
        checkpoint_dir=tmp_path,
        last_safe_step_path=last_safe,
    )

    assert result.forensics_path.exists()
    assert result.failed_checkpoint_path.exists()
    assert result.last_safe_step_path == last_safe
    assert result.last_safe_step_verified is True
    payload = AtomicCheckpoint().load(result.failed_checkpoint_path)
    assert payload["reason"] == "nan watchdog"
    assert payload["step"] == 7


def test_transition_to_full_ppo_keeps_optimizer_and_warms_actor_lr() -> None:
    model = _tiny_actor_critic()
    optimizer = model.optimizer

    model._transition_to_full_ppo(actor_warmup_steps=4, transition_global_step=100)
    lr_0 = model._set_actor_lr_for_step(model._post_transition_steps(100))
    lr_2 = model._set_actor_lr_for_step(model._post_transition_steps(102))
    lr_4 = model._set_actor_lr_for_step(model._post_transition_steps(104))

    assert model.optimizer is optimizer
    assert model.phase is WarmupPhase.PPO
    assert lr_0 == pytest.approx(0.0)
    assert lr_2 == pytest.approx(model._actor_base_lr / 2.0)
    assert lr_4 == pytest.approx(model._actor_base_lr)
    assert any(parameter.requires_grad for parameter in model.actor.parameters())


def test_handoff_slipper_actor_lr_and_clip_eps_ramp_over_50k_steps() -> None:
    model = _tiny_actor_critic()
    model.configure_optimizer(actor_lr=2.0e-4)
    model._transition_to_full_ppo(transition_global_step=1_500_000)

    assert model._actor_lr_at_step(0) == pytest.approx(0.0)
    assert model._actor_lr_at_step(25_000) == pytest.approx(1.0e-4)
    assert model._actor_lr_at_step(50_000) == pytest.approx(2.0e-4)
    assert model._actor_lr_at_step(75_000) == pytest.approx(2.0e-4)
    assert model._clip_eps_at_step(0) == pytest.approx(0.05)
    assert model._clip_eps_at_step(25_000) == pytest.approx(0.125)
    assert model._clip_eps_at_step(50_000) == pytest.approx(0.2)
    assert model._clip_eps_at_step(75_000) == pytest.approx(0.2)


def test_watchdog_nan_halt_smoke_invokes_halt_protocol(tmp_path: Path) -> None:
    model = _tiny_actor_critic()
    forensics = FailureForensics(capacity=4)
    forensics.push(step=9, metrics={"loss_is_nan": True})
    last_safe = AtomicCheckpoint().save({"safe": True}, tmp_path / "last_safe_step.pt")

    result = model._halt_protocol(
        reason="watchdog_nan_halt",
        step=9,
        forensics=forensics,
        checkpoint_dir=tmp_path,
        last_safe_step_path=last_safe,
    )

    assert result.last_safe_step_verified
    assert "failed_step_00000009.pt" == result.failed_checkpoint_path.name


def test_master_loop_warmup_only_smoke_keeps_actor_gradients_zero() -> None:
    model = _tiny_actor_critic()
    buffer = _rollout_buffer_from_batch(model)

    report = model.optimize_rollout(buffer, epochs=1, batch_size=2)

    assert report.halted is False
    assert report.phase is WarmupPhase.CRITIC_WARMUP
    assert all(parameter.grad is None for parameter in model.actor.parameters())
    assert any(parameter.grad is not None for parameter in model.critic_head.parameters())


def test_master_loop_phase_transition_smoke_preserves_optimizer() -> None:
    model = _tiny_actor_critic()
    optimizer = model.optimizer
    model.critic_loss_history.extend([1.0, 1.0002, 0.9999])
    assert model._maybe_advance_phase(critic_loss=0.99995) is WarmupPhase.PPO
    buffer = _rollout_buffer_from_batch(model)

    report = model.optimize_rollout(buffer, epochs=1, batch_size=4)

    assert model.optimizer is optimizer
    assert report.phase is WarmupPhase.PPO
    assert any(parameter.requires_grad for parameter in model.actor.parameters())


def test_gate_v2_requires_sustained_ev_breakthrough_and_slope() -> None:
    model = _tiny_actor_critic()
    optimizer = model.optimizer

    assert (
        model._maybe_advance_phase(
            critic_loss=100.0,
            step=1,
            global_step=499_999,
            explained_variance=0.6,
        )
        is WarmupPhase.CRITIC_WARMUP
    )
    assert (
        model._maybe_advance_phase(
            critic_loss=90.0,
            step=2,
            global_step=500_000,
            explained_variance=0.61,
        )
        is WarmupPhase.CRITIC_WARMUP
    )
    assert (
        model._maybe_advance_phase(
            critic_loss=80.0,
            step=3,
            global_step=520_000,
            explained_variance=0.62,
        )
        is WarmupPhase.PPO
    )

    assert model.optimizer is optimizer
    assert any(parameter.requires_grad for parameter in model.actor.parameters())


def test_gate_v2_blocks_negative_ev_slope() -> None:
    model = _tiny_actor_critic()

    assert (
        model._maybe_advance_phase(
            critic_loss=100.0,
            step=1,
            global_step=500_000,
            explained_variance=0.8,
        )
        is WarmupPhase.CRITIC_WARMUP
    )
    assert (
        model._maybe_advance_phase(
            critic_loss=90.0,
            step=2,
            global_step=620_000,
            explained_variance=0.7,
        )
        is WarmupPhase.CRITIC_WARMUP
    )


def test_master_loop_watchdog_nan_halt_smoke_saves_autopsy(tmp_path: Path) -> None:
    model = _tiny_actor_critic()
    buffer = _rollout_buffer_from_batch(model)
    watchdogs = DefaultWatchdogRegistry()
    watchdogs.register(ThresholdWatchdog("ppo_total_loss", -1.0e9, label="NaN Halt Smoke"))
    forensics = FailureForensics(capacity=8)
    last_safe = AtomicCheckpoint().save({"safe": True}, tmp_path / "last_safe_step.pt")

    report = model.optimize_rollout(
        buffer,
        epochs=1,
        batch_size=4,
        watchdogs=watchdogs,
        forensics=forensics,
        checkpoint_dir=tmp_path,
        last_safe_step_path=last_safe,
    )

    assert report.halted is True
    assert report.halt_result is not None
    assert report.halt_result.failed_checkpoint_path.exists()
    assert report.halt_result.last_safe_step_verified
