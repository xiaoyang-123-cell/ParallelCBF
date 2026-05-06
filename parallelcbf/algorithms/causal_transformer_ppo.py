"""KL-anchored PPO sidecar for the causal Transformer backbone."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import io
import math
from pathlib import Path
import pickle
from typing import Any, cast

import torch
from torch import nn

from parallelcbf.api import Algorithm, MetricDict, Prediction, SafeEnv, WatchdogRegistry
from parallelcbf.algorithms.causal_transformer import (
    CausalTransformer,
    CausalTransformerConfig,
    CausalTransformerHiddenState,
)
from parallelcbf.algorithms.rollout_buffer import (
    RolloutBuffer,
    RolloutStep,
    TerminationReason,
    compute_gae_in_place,
    normalize_termination_reason,
)
from parallelcbf.ops import AtomicCheckpoint, FailureForensics


SHAPING_K_D = 1.0
SHAPING_K_V = 0.5
SHAPING_V_TARGET = 1.0
SHAPING_DISTANCE_EPS = 1.0e-8
STATE_DISTRIBUTION_GRID_BINS = 16


class WarmupPhase(Enum):
    """High-level phase state for the critic-warmup PPO pipeline."""

    CRITIC_WARMUP = "critic_warmup"
    PPO = "ppo"


TrainingPhase = WarmupPhase


@dataclass(frozen=True, slots=True)
class ActorCriticOutput:
    """Policy/value output for PPO rollouts and diagnostics."""

    action: torch.Tensor
    value: torch.Tensor
    reference_action: torch.Tensor
    hidden_state: CausalTransformerHiddenState


@dataclass(frozen=True, slots=True)
class PPOLossOutput:
    """Fully decomposed PPO objective used for audit gates."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor
    kl_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    kl_weight: float
    diagnostics: MetricDict


@dataclass(frozen=True, slots=True)
class HaltProtocolResult:
    """Artifacts produced by the PPO halt/autopsy protocol."""

    forensics_path: Path
    failed_checkpoint_path: Path
    last_safe_step_path: Path
    last_safe_step_verified: bool


@dataclass(frozen=True, slots=True)
class PPOUpdateReport:
    """Summary emitted by the Day 4 PPO master update loop."""

    updates: int
    phase: WarmupPhase
    halted: bool
    halt_reason: str | None
    last_metrics: MetricDict
    halt_result: HaltProtocolResult | None


@dataclass(frozen=True, slots=True)
class GateV2Config:
    """Explained-variance transition gate for critic warmup release."""

    global_step_min: int = 500_000
    ev_threshold: float = 0.60
    ev_sustain_steps: int = 20_000
    ev_slope_window_steps: int = 100_000
    ev_slope_min: float = -1.0e-6


@dataclass(frozen=True, slots=True)
class RolloutCollectionStats:
    """Aggregate diagnostics emitted by PPO rollout collection."""

    steps: int
    episodes_started: int
    terminations: int
    successes: int
    collisions: int
    out_of_arena: int
    arena_projection_terminations: int
    timeouts: int
    cbf_active_steps: int
    cbf_active_rate: float
    mean_cbf_delta_norm: float
    arena_projection_active_steps: int
    arena_projection_active_rate: float
    episode_return_mean: float
    episode_return_std: float
    episode_return_min: float
    episode_return_max: float
    episode_length_mean: float
    episode_length_std: float
    episode_length_min: float
    episode_length_max: float
    h_hard_p01: float
    h_hard_p05: float
    h_hard_p50: float
    h_hard_p95: float
    h_hard_p99: float
    potential_phi_mean: float
    potential_phi_std: float
    potential_delta_mean: float
    potential_delta_std: float
    shaping_reward_share: float
    shaping_reward_per_step: float
    base_reward_mean: float
    cbf_penalty_mean: float
    shaped_reward_mean: float
    state_distribution_entropy: float
    distance_to_goal_std: float
    distance_to_nearest_wall_mean: float
    distance_to_nearest_wall_std: float
    cbf_arena_intervention_duration_mean: float


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def _gaussian_kl(
    mu_theta: torch.Tensor,
    log_std_theta: torch.Tensor,
    mu_bc: torch.Tensor,
    log_std_bc: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL `D_KL(N_theta || N_BC)` for diagonal Gaussians."""

    mu_bc_detached = mu_bc.detach()
    log_std_bc_detached = log_std_bc.detach()
    log_std_theta_clamped = log_std_theta.clamp(-5.0, 2.0)
    var_theta = torch.exp(2.0 * log_std_theta_clamped)
    var_bc = torch.exp(2.0 * log_std_bc_detached)
    mean_delta_sq = (mu_theta - mu_bc_detached) * (mu_theta - mu_bc_detached)
    per_dim = log_std_bc_detached - log_std_theta_clamped + (var_theta + mean_delta_sq) / (2.0 * var_bc) - 0.5
    return torch.sum(per_dim, dim=-1)


def _gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Return summed log probability under a diagonal Gaussian."""

    clamped_log_std = log_std.clamp(-5.0, 2.0)
    variance = torch.exp(2.0 * clamped_log_std)
    log_two_pi = math.log(2.0 * math.pi)
    per_dim = -0.5 * (((action - mean) * (action - mean)) / variance + 2.0 * clamped_log_std + log_two_pi)
    return torch.sum(per_dim, dim=-1)


def _gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """Return summed entropy for a diagonal Gaussian."""

    clamped_log_std = log_std.clamp(-5.0, 2.0)
    per_dim = clamped_log_std + 0.5 * (1.0 + math.log(2.0 * math.pi))
    return torch.sum(per_dim, dim=-1)


def _cosine_decay(start_value: float, end_value: float, *, step: int, total_steps: int) -> float:
    """Cosine decay from `start_value` to `end_value` over `total_steps`."""

    if total_steps <= 0:
        return float(end_value)
    progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
    return float(end_value + 0.5 * (start_value - end_value) * (1.0 + math.cos(math.pi * progress)))


def _explained_variance(predictions: torch.Tensor, targets: torch.Tensor, *, epsilon: float = 1.0e-8) -> torch.Tensor:
    """Return `1 - Var[target - pred] / Var[target]` with finite zero-variance handling."""

    pred = predictions.detach().reshape(-1)
    target = targets.detach().reshape(-1)
    if pred.shape != target.shape:
        raise ValueError("predictions and targets must have the same flattened shape")
    target_variance = torch.var(target, unbiased=False)
    if bool((target_variance <= epsilon).detach().cpu().item()):
        return target.new_zeros(())
    residual_variance = torch.var(target - pred, unbiased=False)
    return 1.0 - residual_variance / target_variance


def _rolling_slope(values: Sequence[float]) -> float:
    """Return the least-squares slope over equally spaced samples."""

    count = len(values)
    if count < 2:
        return 0.0
    mean_x = float(count - 1) / 2.0
    mean_y = sum(float(value) for value in values) / float(count)
    numerator = 0.0
    denominator = 0.0
    for index, raw_value in enumerate(values):
        dx = float(index) - mean_x
        numerator += dx * (float(raw_value) - mean_y)
        denominator += dx * dx
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _is_zero_variance(values: Sequence[float], *, tolerance: float = 1.0e-8) -> bool:
    """Return whether a scalar series has effectively no population variance."""

    if len(values) < 2:
        return True
    mean = sum(float(value) for value in values) / float(len(values))
    variance = sum((float(value) - mean) * (float(value) - mean) for value in values) / float(len(values))
    return variance <= tolerance * tolerance


def _value_diagnostics(predictions: torch.Tensor, targets: torch.Tensor) -> MetricDict:
    """Build train-side critic observability metrics for one optimization minibatch."""

    pred = predictions.detach().reshape(-1)
    target = targets.detach().reshape(-1)
    per_sample_loss = (pred - target) * (pred - target)
    sorted_loss, _ = torch.sort(per_sample_loss)
    count = max(int(sorted_loss.numel()), 1)
    p50_index = min(int(0.50 * float(count - 1)), count - 1)
    p99_index = min(int(math.ceil(0.99 * float(count - 1))), count - 1)
    pred_std = torch.std(pred, unbiased=False) if pred.numel() > 1 else pred.new_zeros(())
    target_std = torch.std(target, unbiased=False) if target.numel() > 1 else target.new_zeros(())
    return {
        "value_explained_variance": float(_explained_variance(pred, target).cpu().item()),
        "value_target_mean": float(torch.mean(target).cpu().item()),
        "value_target_std": float(target_std.cpu().item()),
        "value_pred_mean": float(torch.mean(pred).cpu().item()),
        "value_pred_std": float(pred_std.cpu().item()),
        "value_loss_p50": float(sorted_loss[p50_index].cpu().item()),
        "value_loss_p99": float(sorted_loss[p99_index].cpu().item()),
    }


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Normalize advantages only when a minibatch has meaningful variance."""

    if advantages.numel() <= 1:
        return advantages
    std = torch.std(advantages, unbiased=False)
    if bool((std <= 1.0e-8).detach().cpu().item()):
        return advantages - torch.mean(advantages)
    return (advantages - torch.mean(advantages)) / (std + 1.0e-8)


def _potential_phi(
    position: torch.Tensor,
    velocity: torch.Tensor,
    goal: torch.Tensor,
    *,
    k_d: float = SHAPING_K_D,
    k_v: float = SHAPING_K_V,
    v_target: float = SHAPING_V_TARGET,
) -> torch.Tensor:
    """Ng-Harada-Russell potential for goal distance and velocity alignment."""

    if k_d < 0.0 or k_v < 0.0 or v_target < 0.0:
        raise ValueError("potential shaping constants must be nonnegative")
    pos = position if isinstance(position, torch.Tensor) else torch.as_tensor(position)
    vel = velocity if isinstance(velocity, torch.Tensor) else torch.as_tensor(velocity)
    target = goal if isinstance(goal, torch.Tensor) else torch.as_tensor(goal)
    pos = pos.to(dtype=torch.float32)
    vel = vel.to(dtype=torch.float32, device=pos.device)
    target = target.to(dtype=torch.float32, device=pos.device)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    if vel.ndim == 1:
        vel = vel.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if pos.shape != vel.shape or pos.shape != target.shape or pos.shape[-1] != 2:
        raise ValueError("position, velocity, and goal must share shape (B, 2)")
    delta_to_goal = target - pos
    distance = torch.linalg.vector_norm(delta_to_goal, dim=-1)
    safe_distance = torch.clamp(distance, min=SHAPING_DISTANCE_EPS)
    direction = torch.where(
        (distance > SHAPING_DISTANCE_EPS).unsqueeze(-1),
        delta_to_goal / safe_distance.unsqueeze(-1),
        torch.zeros_like(delta_to_goal),
    )
    velocity_to_goal = torch.sum(vel * direction, dim=-1)
    velocity_deficit = torch.clamp(
        torch.as_tensor(float(v_target), dtype=pos.dtype, device=pos.device) - velocity_to_goal,
        min=0.0,
    )
    phi = cast(torch.Tensor, -float(k_d) * distance - float(k_v) * velocity_deficit)
    if not bool(torch.all(torch.isfinite(phi)).detach().cpu().item()):
        raise ValueError("potential shaping produced a non-finite Phi")
    return phi


def _potential_phi_from_safety_state(safety_state: object) -> torch.Tensor:
    """Compute potential from the physical SafetyState fields."""

    position = torch.as_tensor(getattr(safety_state, "position"), dtype=torch.float32)
    velocity = torch.as_tensor(getattr(safety_state, "velocity"), dtype=torch.float32)
    goal = torch.as_tensor(getattr(safety_state, "goal"), dtype=torch.float32)
    return _potential_phi(position, velocity, goal)


def _distance_to_nearest_wall(position: torch.Tensor, arena_bounds: torch.Tensor) -> torch.Tensor:
    """Return the signed distance to the nearest square-arena wall."""

    pos = torch.as_tensor(position, dtype=torch.float32)
    bounds = torch.as_tensor(arena_bounds, dtype=torch.float32)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    if bounds.ndim == 1:
        if bounds.numel() != 4:
            raise ValueError("arena_bounds must have 4 elements")
        bounds = bounds.unsqueeze(0).expand(pos.shape[0], -1)
    elif bounds.ndim == 2 and bounds.shape[0] == 1:
        bounds = bounds.expand(pos.shape[0], -1)
    if bounds.ndim != 2 or bounds.shape[0] != pos.shape[0] or bounds.shape[1] != 4:
        raise ValueError("arena_bounds must broadcast to position batch size")
    distances = torch.stack(
        (
            pos[:, 0] - bounds[:, 0],
            bounds[:, 1] - pos[:, 0],
            pos[:, 1] - bounds[:, 2],
            bounds[:, 3] - pos[:, 1],
        ),
        dim=-1,
    )
    return torch.min(distances, dim=-1).values


def _state_distribution_entropy(position: torch.Tensor, arena_bounds: torch.Tensor, *, bins: int) -> float:
    """Compute Shannon entropy over a discretized position grid."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    pos = torch.as_tensor(position, dtype=torch.float64)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    if pos.shape[-1] != 2 or pos.numel() == 0:
        return 0.0
    bounds = torch.as_tensor(arena_bounds, dtype=torch.float64).reshape(-1)
    if bounds.numel() != 4:
        raise ValueError("arena_bounds must contain exactly 4 values")
    x_min, x_max, y_min, y_max = [float(value) for value in bounds.tolist()]
    x_width = max(x_max - x_min, 1.0e-12)
    y_width = max(y_max - y_min, 1.0e-12)
    x_index = torch.clamp(((pos[:, 0] - x_min) / x_width * float(bins)).floor().to(dtype=torch.long), 0, bins - 1)
    y_index = torch.clamp(((pos[:, 1] - y_min) / y_width * float(bins)).floor().to(dtype=torch.long), 0, bins - 1)
    flat_index = x_index * bins + y_index
    histogram = torch.bincount(flat_index, minlength=bins * bins).to(dtype=torch.float64)
    total = torch.sum(histogram)
    if bool((total <= 0.0).detach().cpu().item()):
        return 0.0
    probabilities = histogram[histogram > 0.0] / total
    entropy = -torch.sum(probabilities * torch.log(probabilities))
    return float(entropy.detach().cpu().item())


def _coerce_config(raw_config: object) -> CausalTransformerConfig:
    if not isinstance(raw_config, CausalTransformerConfig):
        raise TypeError("checkpoint config must be CausalTransformerConfig")
    return raw_config


def _coerce_state_dict(raw_state_dict: object) -> dict[str, torch.Tensor]:
    if not isinstance(raw_state_dict, Mapping):
        raise TypeError("checkpoint state_dict must be a mapping")
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        if not isinstance(key, str):
            raise TypeError("checkpoint state_dict keys must be strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint tensor for {key!r} must be a torch.Tensor")
        state_dict[key] = value
    return state_dict


def _load_checkpoint_payload(path: str | Path, *, device: torch.device) -> dict[str, object]:
    class _TorchLoadUnpickler(pickle.Unpickler):
        def __init__(self, file: io.BufferedReader) -> None:
            super().__init__(file)

        def find_class(self, module: str, name: str) -> object:
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda data: torch.load(io.BytesIO(data), map_location=device, weights_only=False)
            return super().find_class(module, name)

    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        with checkpoint_path.open("rb") as handle:
            payload = _TorchLoadUnpickler(handle).load()
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    return cast(dict[str, object], payload)


def _infer_config_from_state_dict(state_dict: Mapping[str, torch.Tensor]) -> CausalTransformerConfig:
    obs_proj_weight = state_dict.get("obs_proj.weight")
    action_head_weight = state_dict.get("action_head.weight")
    if obs_proj_weight is None or action_head_weight is None:
        raise TypeError("pure BC state_dict is missing obs_proj.weight or action_head.weight")
    layer_indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("blocks.") and key.split(".")[1].isdigit()
    }
    return CausalTransformerConfig(
        d_model=int(obs_proj_weight.shape[0]),
        n_heads=8,
        n_layers=len(layer_indices),
        context_length=64,
        obs_dim=int(obs_proj_weight.shape[1]),
        action_dim=int(action_head_weight.shape[0]),
    )


def _extract_bc_state_dict(payload: Mapping[str, object]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in payload:
        return _coerce_state_dict(payload["model_state_dict"])
    if "state_dict" in payload:
        return _coerce_state_dict(payload["state_dict"])
    if all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in payload.items()):
        return _coerce_state_dict(payload)
    raise KeyError("checkpoint is missing model_state_dict/state_dict")


def _encode_backbone(model: CausalTransformer, observations: torch.Tensor) -> torch.Tensor:
    if observations.ndim != 3:
        raise ValueError(f"observations must have shape (B, T, D), got {tuple(observations.shape)}")
    if observations.shape[-1] != model.config.obs_dim:
        raise ValueError(f"obs_dim must be {model.config.obs_dim}, got {observations.shape[-1]}")
    if observations.shape[1] > model.config.context_length:
        raise ValueError(
            f"sequence length {observations.shape[1]} exceeds context_length {model.config.context_length}"
        )
    x = cast(torch.Tensor, model.obs_proj(observations))
    for block in model.blocks:
        x = block(x)
    return cast(torch.Tensor, model.final_norm(x))


class CausalTransformerActorCritic(nn.Module, Algorithm[torch.Tensor, torch.Tensor, CausalTransformerHiddenState]):
    """Actor-critic wrapper with a frozen BC anchor and a critic warmup phase."""

    def __init__(
        self,
        config: CausalTransformerConfig | None = None,
        *,
        device: torch.device | str | None = None,
        checkpoint_path: str | Path | None = None,
        bc_checkpoint_path: str | Path | None = None,
        critic_plateau_window: int = 8,
        critic_plateau_min_updates: int = 8,
        critic_plateau_abs_tolerance: float = 1.0e-4,
        critic_plateau_rel_tolerance: float = 2.0e-2,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or "cpu")
        checkpoint_source = checkpoint_path if checkpoint_path is not None else bc_checkpoint_path
        if checkpoint_path is not None and bc_checkpoint_path is not None and Path(checkpoint_path) != Path(bc_checkpoint_path):
            raise ValueError("checkpoint_path and bc_checkpoint_path must match when both are provided")

        payload: dict[str, object] | None = None
        loaded_config = config or CausalTransformerConfig()
        if checkpoint_source is not None:
            payload = _load_checkpoint_payload(checkpoint_source, device=self.device)
            if "config" in payload:
                loaded_config = _coerce_config(payload.get("config"))
            else:
                loaded_config = _infer_config_from_state_dict(_coerce_state_dict(payload))

        self.config = loaded_config
        self.actor = CausalTransformer(self.config).to(self.device)
        self.bc_reference = CausalTransformer(self.config).to(self.device)
        self.critic_head = nn.Linear(self.config.d_model, 1).to(self.device)
        self.actor_log_std = nn.Parameter(torch.zeros((self.config.action_dim,), device=self.device))
        self.register_buffer("_bc_log_std", torch.zeros((self.config.action_dim,), device=self.device), persistent=True)
        nn.init.orthogonal_(self.critic_head.weight, gain=0.01)
        nn.init.zeros_(self.critic_head.bias)

        self.phase = WarmupPhase.CRITIC_WARMUP
        self.optimizer: torch.optim.AdamW | None = None
        self._actor_param_group_index = 0
        self._critic_param_group_index = 1
        self._actor_base_lr = 3.0e-4
        self._actor_warmup_steps = 50_000
        self._actor_warmup_step = 0
        self._clip_eps_warmup_steps = 50_000
        self._clip_eps_warmup_initial = 0.05
        self._clip_eps_base = 0.2
        self._ppo_transition_global_step: int | None = None
        self.critic_loss_history: list[float] = []
        self._critic_plateau_window = int(critic_plateau_window)
        self._critic_plateau_min_updates = int(critic_plateau_min_updates)
        self._critic_plateau_abs_tolerance = float(critic_plateau_abs_tolerance)
        self._critic_plateau_rel_tolerance = float(critic_plateau_rel_tolerance)
        self._last_update_step = 0
        self.global_step = 0
        self.gamma = 0.99
        self.gate_v2_config = GateV2Config()
        self.ev_history: list[tuple[int, float]] = []
        self._ev_sustain_start_step: int | None = None
        self._last_observation: torch.Tensor | None = None
        self._last_hidden_state: CausalTransformerHiddenState | None = None
        self._rollout_episode_ids: list[int] | None = None
        self._rollout_timesteps: list[int] | None = None
        self._rollout_episode_returns: list[float] | None = None
        self._rollout_episode_lengths: list[int] | None = None

        if payload is not None:
            state_dict = _extract_bc_state_dict(payload)
            self.actor.load_state_dict(state_dict)
            self.bc_reference.load_state_dict(state_dict)

        self.bc_reference.requires_grad_(False)
        self.bc_reference.eval()
        self._freeze_actor_initial()
        self.configure_optimizer()

    @property
    def trunk(self) -> CausalTransformer:
        """Alias for the trainable actor backbone."""

        return self.actor

    @property
    def actor_head(self) -> nn.Linear:
        """Return the trainable policy head."""

        return self.actor.action_head

    @property
    def log_std_theta(self) -> nn.Parameter:
        """Return the trainable policy log standard deviation."""

        return self.actor_log_std

    @property
    def log_std_bc(self) -> torch.Tensor:
        """Return the frozen BC-reference log standard deviation buffer."""

        return cast(torch.Tensor, self._bc_log_std)

    @property
    def bc_trunk_frozen(self) -> CausalTransformer:
        """Compatibility alias for the frozen BC Transformer trunk."""

        return self.bc_reference

    @property
    def bc_actor_mean_head_frozen(self) -> nn.Linear:
        """Compatibility alias for the frozen BC action-mean head."""

        return self.bc_reference.action_head

    @property
    def reference(self) -> CausalTransformer:
        """Alias for the frozen BC anchor."""

        return self.bc_reference

    def train(self, mode: bool = True) -> "CausalTransformerActorCritic":
        """Set training mode without thawing or train-mode toggling the BC anchor."""

        super().train(mode)
        self.bc_reference.requires_grad_(False)
        self.bc_reference.eval()
        if self.phase is WarmupPhase.CRITIC_WARMUP:
            self.actor.eval()
        return self

    @classmethod
    def from_bc_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
        critic_plateau_window: int = 8,
        critic_plateau_min_updates: int = 8,
        critic_plateau_abs_tolerance: float = 1.0e-4,
        critic_plateau_rel_tolerance: float = 2.0e-2,
    ) -> "CausalTransformerActorCritic":
        """Build an actor-critic model directly from a BC checkpoint."""

        return cls(
            device=device,
            checkpoint_path=checkpoint_path,
            critic_plateau_window=critic_plateau_window,
            critic_plateau_min_updates=critic_plateau_min_updates,
            critic_plateau_abs_tolerance=critic_plateau_abs_tolerance,
            critic_plateau_rel_tolerance=critic_plateau_rel_tolerance,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> "CausalTransformerActorCritic":
        """Restore a saved actor-critic checkpoint."""

        instance = cls(device=device)
        instance.load(checkpoint_path)
        return instance

    def initial_hidden_state(
        self,
        batch_size: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> CausalTransformerHiddenState:
        """Return an empty rolling observation buffer."""

        state_device = device or self.device
        obs_buffer = torch.empty((batch_size, 0, self.config.obs_dim), dtype=dtype, device=state_device)
        return CausalTransformerHiddenState(obs_buffer=obs_buffer, kv_cache=None)

    def forward(
        self,
        observation: torch.Tensor,
        hidden_state: CausalTransformerHiddenState,
        *,
        deterministic: bool = False,
    ) -> ActorCriticOutput:
        """Return policy and value outputs for one rollout step."""

        _ = deterministic
        obs = observation.to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2 or obs.shape[-1] != self.config.obs_dim:
            raise ValueError(f"observation must have shape (B, {self.config.obs_dim}) or ({self.config.obs_dim},)")
        buffer = hidden_state.obs_buffer.to(device=self.device, dtype=obs.dtype)
        if buffer.ndim != 3 or buffer.shape[0] != obs.shape[0] or buffer.shape[-1] != self.config.obs_dim:
            raise ValueError("hidden_state.obs_buffer must have shape (B, T, obs_dim)")
        next_buffer = torch.cat((buffer, obs.unsqueeze(1)), dim=1)
        next_buffer = next_buffer[:, -self.config.context_length :, :]
        actor_latent = _encode_backbone(self.actor, next_buffer)
        reference_latent = _encode_backbone(self.bc_reference, next_buffer)
        action = self.actor.action_head(actor_latent[:, -1, :])
        reference_action = self.bc_reference.action_head(reference_latent[:, -1, :])
        value = self.critic_head(actor_latent[:, -1, :]).squeeze(-1)
        next_state = CausalTransformerHiddenState(obs_buffer=next_buffer.detach(), kv_cache=hidden_state.kv_cache)
        return ActorCriticOutput(
            action=action,
            value=value,
            reference_action=reference_action,
            hidden_state=next_state,
        )

    def act_and_value(
        self,
        observation: torch.Tensor,
        hidden_state: CausalTransformerHiddenState,
        *,
        deterministic: bool = False,
    ) -> ActorCriticOutput:
        """Explicit alias for the combined policy/value forward pass."""

        return self.forward(observation, hidden_state, deterministic=deterministic)

    def value(
        self,
        observation: torch.Tensor,
        hidden_state: CausalTransformerHiddenState,
    ) -> torch.Tensor:
        """Return only the critic value estimate."""

        return self.forward(observation, hidden_state).value

    def predict(
        self,
        observation: torch.Tensor,
        hidden_state: CausalTransformerHiddenState,
        *,
        deterministic: bool = False,
    ) -> Prediction[torch.Tensor, CausalTransformerHiddenState]:
        """Predict an action while preserving the rolling hidden state."""

        with torch.no_grad():
            output = self.forward(observation, hidden_state, deterministic=deterministic)
        return Prediction(action=output.action, hidden_state=output.hidden_state)

    def _freeze_actor_initial(self) -> None:
        """Freeze the actor backbone and head for critic warmup."""

        _set_requires_grad(self.actor, False)
        self.actor_log_std.requires_grad_(False)
        self.actor.eval()
        self.phase = WarmupPhase.CRITIC_WARMUP

    def _unfreeze_actor(self) -> None:
        """Unfreeze the actor backbone and head for PPO optimization."""

        _set_requires_grad(self.actor, True)
        self.actor_log_std.requires_grad_(True)
        self.actor.train()

    def configure_optimizer(
        self,
        *,
        actor_lr: float = 3.0e-4,
        critic_lr: float = 3.0e-4,
        weight_decay: float = 1.0e-2,
        actor_warmup_steps: int = 50_000,
    ) -> torch.optim.AdamW:
        """Create one AdamW up front and preserve it across phase transitions."""

        if actor_lr < 0.0 or critic_lr < 0.0:
            raise ValueError("learning rates must be nonnegative")
        if actor_warmup_steps < 0:
            raise ValueError("actor_warmup_steps must be nonnegative")
        self._actor_base_lr = float(actor_lr)
        self._actor_warmup_steps = int(actor_warmup_steps)
        actor_params = list(self.actor.parameters()) + [self.actor_log_std]
        critic_params = list(self.critic_head.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": actor_params, "lr": 0.0, "base_lr": float(actor_lr), "name": "actor"},
                {"params": critic_params, "lr": float(critic_lr), "base_lr": float(critic_lr), "name": "critic"},
            ],
            weight_decay=weight_decay,
        )
        self._actor_param_group_index = 0
        self._critic_param_group_index = 1
        return self.optimizer

    def _transition_to_full_ppo(
        self,
        *,
        actor_warmup_steps: int | None = None,
        transition_global_step: int | None = None,
    ) -> None:
        """Release actor gradients without rebuilding AdamW."""

        optimizer_id = id(self.optimizer) if self.optimizer is not None else None
        self._unfreeze_actor()
        self.phase = WarmupPhase.PPO
        if actor_warmup_steps is not None:
            if actor_warmup_steps < 0:
                raise ValueError("actor_warmup_steps must be nonnegative")
            self._actor_warmup_steps = int(actor_warmup_steps)
        self._actor_warmup_step = 0
        self._ppo_transition_global_step = int(
            self.global_step if transition_global_step is None else transition_global_step
        )
        if self.optimizer is not None:
            actor_group = self.optimizer.param_groups[self._actor_param_group_index]
            actor_group["lr"] = self._actor_lr_at_step(0)
            if optimizer_id is not None and id(self.optimizer) != optimizer_id:
                raise RuntimeError("optimizer was unexpectedly rebuilt during PPO transition")

    def _actor_lr_at_step(self, post_transition_step: int) -> float:
        """Return actor LR after `post_transition_step` environment transitions."""

        if post_transition_step < 0:
            raise ValueError("post_transition_step must be nonnegative")
        if self._actor_warmup_steps <= 0:
            return float(self._actor_base_lr)
        progress = min(float(post_transition_step) / float(self._actor_warmup_steps), 1.0)
        return float(self._actor_base_lr * progress)

    def _clip_eps_at_step(self, post_transition_step: int) -> float:
        """Return PPO clip epsilon after `post_transition_step` transitions."""

        if post_transition_step < 0:
            raise ValueError("post_transition_step must be nonnegative")
        if self._clip_eps_warmup_steps <= 0:
            return float(self._clip_eps_base)
        progress = min(float(post_transition_step) / float(self._clip_eps_warmup_steps), 1.0)
        return float(self._clip_eps_warmup_initial + (self._clip_eps_base - self._clip_eps_warmup_initial) * progress)

    def _post_transition_steps(self, current_global_step: int | None = None) -> int:
        """Return transitions elapsed since the actor was released."""

        if self.phase is not WarmupPhase.PPO:
            return 0
        current_step = int(self.global_step if current_global_step is None else current_global_step)
        transition_step = self._ppo_transition_global_step
        if transition_step is None:
            transition_step = current_step
            self._ppo_transition_global_step = transition_step
        return max(0, current_step - int(transition_step))

    def _set_actor_lr_for_step(self, post_transition_step: int) -> float:
        """Apply the Slipper actor LR schedule and return the active LR."""

        if self.optimizer is None:
            raise RuntimeError("optimizer is not configured")
        if self.phase is not WarmupPhase.PPO:
            return float(self.optimizer.param_groups[self._actor_param_group_index]["lr"])
        lr = self._actor_lr_at_step(post_transition_step)
        self.optimizer.param_groups[self._actor_param_group_index]["lr"] = lr
        return float(lr)

    def _step_actor_lr_warmup(self) -> float:
        """Compatibility wrapper for tests and older callers."""

        return self._set_actor_lr_for_step(self._post_transition_steps())

    def _is_critic_plateaued(self, critic_losses: Sequence[float] | None = None) -> bool:
        """Return whether the critic has converged enough to release the actor."""

        history = tuple(self.critic_loss_history if critic_losses is None else critic_losses)
        if len(history) < self._critic_plateau_min_updates:
            return False
        window = min(self._critic_plateau_window, len(history))
        recent = history[-window:]
        if len(recent) < self._critic_plateau_window:
            return False
        baseline = sum(recent) / float(len(recent))
        amplitude = max(recent) - min(recent)
        threshold = max(
            self._critic_plateau_abs_tolerance,
            self._critic_plateau_rel_tolerance * max(abs(baseline), 1.0e-6),
        )
        return amplitude <= threshold

    def _maybe_advance_phase(
        self,
        critic_loss: float | None = None,
        *,
        step: int | None = None,
        global_step: int | None = None,
        explained_variance: float | None = None,
    ) -> WarmupPhase:
        """Update critic history and release the actor through Gate V2 when available."""

        if step is not None:
            self._last_update_step = int(step)
        if global_step is not None:
            self.global_step = int(global_step)
        if critic_loss is not None:
            self.critic_loss_history.append(float(critic_loss))
        if explained_variance is not None and math.isfinite(float(explained_variance)):
            ev_step = int(global_step if global_step is not None else self.global_step)
            self._record_explained_variance(ev_step, float(explained_variance))
        if self.phase is WarmupPhase.CRITIC_WARMUP:
            if explained_variance is not None:
                if self._gate_v2_ready():
                    self._transition_to_full_ppo(transition_global_step=self.global_step)
            elif self._is_critic_plateaued():
                self._transition_to_full_ppo(transition_global_step=self.global_step)
        return self.phase

    def _record_explained_variance(self, step: int, value: float) -> None:
        """Record finite EV samples and keep the Gate V2 horizon bounded."""

        if self.ev_history and step < self.ev_history[-1][0]:
            self.ev_history.clear()
            self._ev_sustain_start_step = None
        self.ev_history.append((int(step), float(value)))
        min_step = int(step) - max(self.gate_v2_config.ev_slope_window_steps * 2, self.gate_v2_config.ev_sustain_steps * 4)
        self.ev_history = [(sample_step, sample) for sample_step, sample in self.ev_history if sample_step >= min_step]
        if value >= self.gate_v2_config.ev_threshold:
            if self._ev_sustain_start_step is None:
                self._ev_sustain_start_step = int(step)
        else:
            self._ev_sustain_start_step = None

    def _gate_v2_ready(self) -> bool:
        """Return whether the pre-registered EV breakthrough gate is open."""

        if self.phase is not WarmupPhase.CRITIC_WARMUP or not self.ev_history:
            return False
        current_step = self.global_step
        if current_step < self.gate_v2_config.global_step_min:
            return False
        if self._ev_sustain_start_step is None:
            return False
        if current_step - self._ev_sustain_start_step < self.gate_v2_config.ev_sustain_steps:
            return False
        slope = self._ev_slope_over_window(self.gate_v2_config.ev_slope_window_steps)
        return slope >= self.gate_v2_config.ev_slope_min

    def _ev_slope_over_window(self, window_steps: int) -> float:
        """Return EV slope per transition over the latest requested horizon."""

        if len(self.ev_history) < 2:
            return float("-inf")
        current_step = self.ev_history[-1][0]
        window = [(step, value) for step, value in self.ev_history if step >= current_step - window_steps]
        if len(window) < 2:
            return float("-inf")
        x0 = float(window[0][0])
        xs = [float(step) - x0 for step, _ in window]
        ys = [float(value) for _, value in window]
        mean_x = sum(xs) / float(len(xs))
        mean_y = sum(ys) / float(len(ys))
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        denominator = sum((x - mean_x) * (x - mean_x) for x in xs)
        if denominator <= 0.0:
            return float("-inf")
        return numerator / denominator

    def _shape_reward(self, reward: float, cbf_delta_norm: float, *, eta_cbf: float = 0.05) -> float:
        """Apply the V25 intervention penalty to the environment reward."""

        if eta_cbf < 0.0:
            raise ValueError("eta_cbf must be nonnegative")
        return float(reward) - float(eta_cbf) * float(cbf_delta_norm)

    def _potential_shaping_delta(self, phi_current: float, phi_next: float, *, gamma: float | None = None) -> float:
        """Return the Ng-Harada-Russell PBRS increment using the PPO discount."""

        active_gamma = self.gamma if gamma is None else float(gamma)
        if not math.isclose(active_gamma, self.gamma, rel_tol=0.0, abs_tol=1.0e-12):
            raise AssertionError("SHAPING_GAMMA must match the PPO gamma")
        return float(active_gamma) * float(phi_next) - float(phi_current)

    def compute_loss(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        phase: TrainingPhase | None = None,
        step: int = 0,
        total_steps: int = 1,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        kl_coef_start: float = 1.0,
        kl_coef_end: float = 0.05,
    ) -> PPOLossOutput:
        """Compute the KL-anchored PPO objective for one minibatch."""

        active_phase = self.phase if phase is None else phase
        context = batch["context"].to(device=self.device, dtype=next(self.actor.parameters()).dtype)
        action = batch["action"].to(device=self.device, dtype=context.dtype)
        old_log_prob = batch["old_log_prob"].to(device=self.device, dtype=context.dtype)
        old_value = batch["value"].to(device=self.device, dtype=context.dtype)
        returns = batch["return"].to(device=self.device, dtype=context.dtype)
        advantages = batch["advantage"].to(device=self.device, dtype=context.dtype)
        actor_latent = _encode_backbone(self.actor, context)
        if active_phase is WarmupPhase.CRITIC_WARMUP:
            critic_latent = actor_latent.detach()
            values = self.critic_head(critic_latent[:, -1, :]).squeeze(-1)
            per_sample_value_loss = (values - returns) * (values - returns)
            value_loss = torch.mean(per_sample_value_loss)
            zero = value_loss.new_zeros(())
            return PPOLossOutput(
                total_loss=value_loss,
                policy_loss=zero,
                value_loss=value_loss,
                entropy_loss=zero,
                kl_loss=zero,
                entropy=zero,
                approx_kl=zero,
                clip_fraction=zero,
                kl_weight=0.0,
                diagnostics=_value_diagnostics(values, returns),
            )

        mean = self.actor.action_head(actor_latent[:, -1, :])
        values = self.critic_head(actor_latent[:, -1, :]).squeeze(-1)
        with torch.no_grad():
            reference_latent = _encode_backbone(self.bc_reference, context)
            reference_mean = self.bc_reference.action_head(reference_latent[:, -1, :])
        log_std_theta = self.actor_log_std.expand_as(mean)
        log_std_bc = self.log_std_bc.expand_as(reference_mean)
        new_log_prob = _gaussian_log_prob(action, mean, log_std_theta)
        ratio = torch.exp(new_log_prob - old_log_prob)
        normalized_advantages = _normalize_advantages(advantages)
        unclipped_policy = ratio * normalized_advantages
        clipped_policy = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * normalized_advantages
        policy_loss = -torch.mean(torch.minimum(unclipped_policy, clipped_policy))
        value_unclipped = (values - returns) * (values - returns)
        clipped_values = old_value + torch.clamp(values - old_value, -clip_epsilon, clip_epsilon)
        value_clipped = (clipped_values - returns) * (clipped_values - returns)
        per_sample_value_loss = torch.maximum(value_unclipped, value_clipped)
        value_loss = 0.5 * torch.mean(per_sample_value_loss)
        entropy = torch.mean(_gaussian_entropy(log_std_theta))
        entropy_loss = -entropy_coef * entropy
        kl_values = _gaussian_kl(mean, log_std_theta, reference_mean, log_std_bc)
        kl_weight = _cosine_decay(kl_coef_start, kl_coef_end, step=step, total_steps=total_steps)
        kl_loss = kl_weight * torch.mean(kl_values)
        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1.0) - torch.log(ratio))
            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_epsilon).to(dtype=context.dtype))
        total_loss = policy_loss + value_coef * value_loss + entropy_loss + kl_loss
        return PPOLossOutput(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy_loss=entropy_loss,
            kl_loss=kl_loss,
            entropy=entropy,
            approx_kl=approx_kl,
            clip_fraction=clip_fraction,
            kl_weight=kl_weight,
            diagnostics=_value_diagnostics(values, returns),
        )

    def collect_rollout(
        self,
        env: SafeEnv[torch.Tensor, torch.Tensor],
        *,
        num_steps: int,
        seed: int | None = None,
        deterministic: bool = False,
        action_noise_std: float = 0.0,
        eta_cbf: float = 0.05,
        shaping_gamma: float | None = None,
    ) -> tuple[RolloutBuffer, RolloutCollectionStats]:
        """Collect a PPO rollout using the Transformer's sliding observation context."""

        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if action_noise_std < 0.0:
            raise ValueError("action_noise_std must be nonnegative")
        active_shaping_gamma = self.gamma if shaping_gamma is None else float(shaping_gamma)
        if not math.isclose(active_shaping_gamma, self.gamma, rel_tol=0.0, abs_tol=1.0e-12):
            raise AssertionError("SHAPING_GAMMA must match the PPO gamma")
        model_dtype = next(self.actor.parameters()).dtype
        episodes_started = 0
        if self._last_observation is None or self._last_hidden_state is None:
            observation_raw, _ = env.reset(seed=seed)
            observation = _coerce_policy_observation(
                observation_raw,
                self.config.obs_dim,
                device=self.device,
                dtype=model_dtype,
            )
            hidden_state = self.initial_hidden_state(observation.shape[0], dtype=observation.dtype, device=self.device)
            self._reset_rollout_lane_state(observation.shape[0])
            episodes_started = observation.shape[0]
        else:
            observation = self._last_observation.to(device=self.device, dtype=model_dtype)
            hidden_state = self._last_hidden_state
            if hidden_state.obs_buffer.shape[0] != observation.shape[0]:
                observation_raw, _ = env.reset(seed=seed)
                observation = _coerce_policy_observation(
                    observation_raw,
                    self.config.obs_dim,
                    device=self.device,
                    dtype=model_dtype,
                )
                hidden_state = self.initial_hidden_state(observation.shape[0], dtype=observation.dtype, device=self.device)
                self._reset_rollout_lane_state(observation.shape[0])
                episodes_started = observation.shape[0]
        if self._rollout_episode_ids is None or len(self._rollout_episode_ids) != observation.shape[0]:
            self._reset_rollout_lane_state(observation.shape[0])
            episodes_started = observation.shape[0]
        buffer = RolloutBuffer(
            context_length=self.config.context_length,
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
        )
        episode_ids = self._rollout_episode_ids
        timesteps = self._rollout_timesteps
        episode_returns = self._rollout_episode_returns
        episode_lengths = self._rollout_episode_lengths
        if episode_ids is None or timesteps is None or episode_returns is None or episode_lengths is None:
            raise RuntimeError("rollout lane state was not initialized")
        terminations = 0
        successes = 0
        collisions = 0
        out_of_arena = 0
        arena_projection_terminations = 0
        timeouts = 0
        cbf_active_steps = 0
        cbf_delta_total = 0.0
        arena_projection_active_steps = 0
        phi_samples: list[float] = []
        potential_delta_samples: list[float] = []
        state_positions: list[torch.Tensor] = []
        distance_to_goal_samples: list[float] = []
        distance_to_wall_samples: list[float] = []
        arena_projection_duration_steps: list[int] = []
        current_arena_duration = [0 for _ in range(observation.shape[0])]
        baseline_reward_total = 0.0
        cbf_penalty_total = 0.0
        final_shaped_reward_total = 0.0
        abs_potential_delta_total = 0.0
        abs_final_shaped_reward_total = 0.0
        completed_episode_returns: list[float] = []
        completed_episode_lengths: list[float] = []
        h_hard_samples: list[float] = []

        for rollout_step in range(num_steps):
            phi_current = _potential_phi_from_safety_state(env.safety_state()).detach().cpu().reshape(-1)
            output = self.act_and_value(observation, hidden_state, deterministic=deterministic)
            action_mean = output.action.detach()
            action = output.action.detach()
            if action_noise_std > 0.0:
                action = action + torch.randn_like(action) * action_noise_std
            old_log_prob = _gaussian_log_prob(
                action,
                action_mean,
                self.actor_log_std.detach().expand_as(action_mean),
            ).detach().cpu()
            env_action_dim = _env_action_dim(env, default=self.config.action_dim)
            env_action = action[:, :env_action_dim].detach().cpu()
            next_observation_raw, reward_raw, terminated_raw, truncated_raw, info = env.step(env_action)
            phi_next = _potential_phi_from_safety_state(env.safety_state()).detach().cpu().reshape(-1)
            h_hard_samples.extend(_h_hard_samples(info, env))
            rollout_safety_state = env.safety_state()
            rollout_position = torch.as_tensor(rollout_safety_state.position, dtype=torch.float32)
            if rollout_position.ndim == 1:
                rollout_position = rollout_position.unsqueeze(0)
            rollout_goal = torch.as_tensor(rollout_safety_state.goal, dtype=torch.float32)
            if rollout_goal.ndim == 1:
                rollout_goal = rollout_goal.unsqueeze(0)
            state_positions.extend([row.detach().cpu() for row in rollout_position])
            distance_to_goal_samples.extend(
                float(torch.linalg.vector_norm(rollout_goal[row_index] - rollout_position[row_index]).item())
                for row_index in range(min(rollout_position.shape[0], observation.shape[0]))
            )
            wall_distance_batch = _distance_to_nearest_wall(
                rollout_position,
                torch.as_tensor(rollout_safety_state.arena_bounds, dtype=torch.float32),
            ).detach().cpu().reshape(-1)
            distance_to_wall_samples.extend(float(value.item()) for value in wall_distance_batch[: observation.shape[0]])
            next_observation = _coerce_policy_observation(
                next_observation_raw,
                self.config.obs_dim,
                device=self.device,
                dtype=model_dtype,
            )
            with torch.no_grad():
                next_output = self.act_and_value(next_observation, output.hidden_state, deterministic=True)
            rewards = _float_vector(reward_raw, observation.shape[0])
            reasons = _termination_reasons(info, observation.shape[0], terminated_raw, truncated_raw)
            u_nominal_env, u_safe_env = _safety_actions(info, fallback=env_action, action_dim=env_action_dim)
            arena_projection_flags = _arena_projection_flags(info, observation.shape[0])
            u_nominal = _pad_action(u_nominal_env, action_dim=self.config.action_dim, reference=action.detach().cpu())
            u_safe = _pad_action(u_safe_env, action_dim=self.config.action_dim, reference=action.detach().cpu())
            next_values = next_output.value.detach().cpu()
            done_flags: list[bool] = []

            for env_id in range(observation.shape[0]):
                if phi_current.numel() not in {1, observation.shape[0]}:
                    raise ValueError(f"Phi current batch must have 1 or {observation.shape[0]} entries")
                if phi_next.numel() not in {1, observation.shape[0]}:
                    raise ValueError(f"Phi next batch must have 1 or {observation.shape[0]} entries")
                current_phi_value = float(phi_current[min(env_id, phi_current.numel() - 1)].item())
                next_phi_value = float(phi_next[min(env_id, phi_next.numel() - 1)].item())
                reason = reasons[env_id]
                if reason == "out_of_arena" and arena_projection_flags[env_id]:
                    reason = "cbf_arena_projection_active"
                done = reason in {"success", "collision", "out_of_arena", "cbf_arena_projection_active", "timeout"}
                done_flags.append(done)
                terminated = reason in {"success", "collision", "out_of_arena", "cbf_arena_projection_active"}
                truncated = reason == "timeout"
                delta = torch.linalg.vector_norm(u_safe[env_id] - u_nominal[env_id]).item()
                baseline_reward = self._shape_reward(rewards[env_id], delta, eta_cbf=eta_cbf)
                potential_delta = self._potential_shaping_delta(
                    current_phi_value,
                    next_phi_value,
                    gamma=active_shaping_gamma,
                )
                shaped_reward = baseline_reward + potential_delta
                if delta > 1.0e-8:
                    cbf_active_steps += 1
                if arena_projection_flags[env_id]:
                    arena_projection_active_steps += 1
                    current_arena_duration[env_id] += 1
                else:
                    if current_arena_duration[env_id] > 0:
                        arena_projection_duration_steps.append(current_arena_duration[env_id])
                    current_arena_duration[env_id] = 0
                cbf_delta_total += float(delta)
                phi_samples.append(current_phi_value)
                potential_delta_samples.append(float(potential_delta))
                baseline_reward_total += float(baseline_reward)
                cbf_penalty_total += float(eta_cbf) * float(delta)
                final_shaped_reward_total += float(shaped_reward)
                abs_potential_delta_total += abs(float(potential_delta))
                abs_final_shaped_reward_total += abs(float(shaped_reward))
                episode_returns[env_id] += shaped_reward
                episode_lengths[env_id] += 1
                if done:
                    terminations += 1
                    successes += int(reason == "success")
                    collisions += int(reason == "collision")
                    out_of_arena += int(reason == "out_of_arena")
                    arena_projection_terminations += int(reason == "cbf_arena_projection_active")
                    timeouts += int(reason == "timeout")
                    completed_episode_returns.append(float(episode_returns[env_id]))
                    completed_episode_lengths.append(float(episode_lengths[env_id]))
                    if current_arena_duration[env_id] > 0:
                        arena_projection_duration_steps.append(current_arena_duration[env_id])
                        current_arena_duration[env_id] = 0
                bootstrap_value = float(next_values[env_id].item()) if reason in {"timeout", "ongoing"} else None
                buffer.append(
                    RolloutStep(
                        env_id=env_id,
                        episode_id=episode_ids[env_id],
                        timestep=timesteps[env_id],
                        context=output.hidden_state.obs_buffer.detach().cpu()[env_id].clone(),
                        observation=observation.detach().cpu()[env_id].clone(),
                        action=action.detach().cpu()[env_id].clone(),
                        log_prob=old_log_prob[env_id].clone(),
                        value=float(output.value.detach().cpu()[env_id].item()),
                        reward=rewards[env_id],
                        shaped_reward=shaped_reward,
                        done=done,
                        terminated=terminated,
                        truncated=truncated,
                        termination_reason=reason,
                        u_nominal=u_nominal[env_id].clone(),
                        u_safe=u_safe[env_id].clone(),
                        cbf_delta_norm=float(delta),
                        bootstrap_value=bootstrap_value,
                    )
                )
                timesteps[env_id] += 1
                if done:
                    episode_ids[env_id] += 1
                    timesteps[env_id] = 0
                    episode_returns[env_id] = 0.0
                    episode_lengths[env_id] = 0

            hidden_state = output.hidden_state
            observation = next_observation
            if any(done_flags):
                reset_seed = None if seed is None else seed + rollout_step + 1
                reset_observation_raw = _reset_done_env_lanes(env, done_flags, seed=reset_seed)
                observation = _coerce_policy_observation(
                    reset_observation_raw,
                    self.config.obs_dim,
                    device=self.device,
                    dtype=model_dtype,
                )
                hidden_state = _reset_done_hidden_lanes(hidden_state, done_flags)
                episodes_started += sum(int(done) for done in done_flags)

        self._last_observation = observation.detach()
        self._last_hidden_state = CausalTransformerHiddenState(
            obs_buffer=hidden_state.obs_buffer.detach(),
            kv_cache=hidden_state.kv_cache,
        )
        self._rollout_episode_ids = list(episode_ids)
        self._rollout_timesteps = list(timesteps)
        self._rollout_episode_returns = list(episode_returns)
        self._rollout_episode_lengths = list(episode_lengths)
        for duration in current_arena_duration:
            if duration > 0:
                arena_projection_duration_steps.append(duration)
        total_lane_steps = max(len(buffer), 1)
        return_summary = _series_summary(completed_episode_returns)
        length_summary = _series_summary(completed_episode_lengths)
        h_hard_summary = _percentile_summary(h_hard_samples, prefix="h_hard")
        phi_summary = _series_summary(phi_samples)
        potential_delta_summary = _series_summary(potential_delta_samples)
        state_entropy = _state_distribution_entropy(
            torch.stack(state_positions) if state_positions else torch.zeros((0, 2), dtype=torch.float32),
            torch.as_tensor(env.safety_state().arena_bounds, dtype=torch.float32),
            bins=STATE_DISTRIBUTION_GRID_BINS,
        )
        distance_to_goal_summary = _series_summary(distance_to_goal_samples)
        wall_distance_summary = _series_summary(distance_to_wall_samples)
        arena_duration_summary = _series_summary([float(value) for value in arena_projection_duration_steps]) if arena_projection_duration_steps else {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        stats = RolloutCollectionStats(
            steps=len(buffer),
            episodes_started=episodes_started,
            terminations=terminations,
            successes=successes,
            collisions=collisions,
            out_of_arena=out_of_arena,
            arena_projection_terminations=arena_projection_terminations,
            timeouts=timeouts,
            cbf_active_steps=cbf_active_steps,
            cbf_active_rate=float(cbf_active_steps) / float(total_lane_steps),
            mean_cbf_delta_norm=cbf_delta_total / float(total_lane_steps),
            arena_projection_active_steps=arena_projection_active_steps,
            arena_projection_active_rate=float(arena_projection_active_steps) / float(total_lane_steps),
            episode_return_mean=return_summary["mean"],
            episode_return_std=return_summary["std"],
            episode_return_min=return_summary["min"],
            episode_return_max=return_summary["max"],
            episode_length_mean=length_summary["mean"],
            episode_length_std=length_summary["std"],
            episode_length_min=length_summary["min"],
            episode_length_max=length_summary["max"],
            h_hard_p01=h_hard_summary["h_hard_p01"],
            h_hard_p05=h_hard_summary["h_hard_p05"],
            h_hard_p50=h_hard_summary["h_hard_p50"],
            h_hard_p95=h_hard_summary["h_hard_p95"],
            h_hard_p99=h_hard_summary["h_hard_p99"],
            potential_phi_mean=phi_summary["mean"],
            potential_phi_std=phi_summary["std"],
            potential_delta_mean=potential_delta_summary["mean"],
            potential_delta_std=potential_delta_summary["std"],
            shaping_reward_share=(
                abs_potential_delta_total / abs_final_shaped_reward_total
                if abs_final_shaped_reward_total > 1.0e-12
                else 0.0
            ),
            shaping_reward_per_step=potential_delta_summary["mean"],
            base_reward_mean=baseline_reward_total / float(total_lane_steps),
            cbf_penalty_mean=cbf_penalty_total / float(total_lane_steps),
            shaped_reward_mean=final_shaped_reward_total / float(total_lane_steps),
            state_distribution_entropy=state_entropy,
            distance_to_goal_std=distance_to_goal_summary["std"],
            distance_to_nearest_wall_mean=wall_distance_summary["mean"],
            distance_to_nearest_wall_std=wall_distance_summary["std"],
            cbf_arena_intervention_duration_mean=arena_duration_summary["mean"],
        )
        return buffer, stats

    def _reset_rollout_lane_state(self, batch_size: int) -> None:
        """Initialize persistent per-lane rollout accounting."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._rollout_episode_ids = [0 for _ in range(batch_size)]
        self._rollout_timesteps = [0 for _ in range(batch_size)]
        self._rollout_episode_returns = [0.0 for _ in range(batch_size)]
        self._rollout_episode_lengths = [0 for _ in range(batch_size)]

    def step_critic_lr_cosine(self, *, step: int, total_steps: int, end_fraction: float = 0.1) -> float:
        """Apply cosine decay to the critic LR param group and return the new LR."""

        if self.optimizer is None:
            raise RuntimeError("optimizer is not configured")
        if not 0.0 <= end_fraction <= 1.0:
            raise ValueError("end_fraction must be in [0, 1]")
        critic_group = self.optimizer.param_groups[self._critic_param_group_index]
        base_lr = float(critic_group.get("base_lr", critic_group["lr"]))
        lr = _cosine_decay(base_lr, base_lr * end_fraction, step=step, total_steps=total_steps)
        critic_group["lr"] = lr
        return float(lr)

    def learn(
        self,
        env: SafeEnv[torch.Tensor, torch.Tensor],
        *,
        total_timesteps: int,
        callback: WatchdogRegistry | None = None,
    ) -> None:
        """Run a minimal collect-GAE-optimize PPO loop."""

        if total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        remaining = int(total_timesteps)
        global_step = 0
        checkpoint_root = Path("checkpoints/v25_ppo")
        last_safe = AtomicCheckpoint().save(self._checkpoint_payload(step=0), checkpoint_root / "last_safe_step.pt")
        forensics = FailureForensics(capacity=4096)
        while remaining > 0:
            rollout_steps = min(remaining, self.config.context_length)
            buffer, _ = self.collect_rollout(env, num_steps=rollout_steps, deterministic=False)
            compute_gae_in_place(buffer, gamma=self.gamma)
            report = self.optimize_rollout(
                buffer,
                epochs=1,
                batch_size=min(64, max(len(buffer), 1)),
                watchdogs=callback,
                forensics=forensics,
                checkpoint_dir=checkpoint_root,
                last_safe_step_path=last_safe,
                start_step=global_step,
                total_steps=total_timesteps,
            )
            global_step += report.updates
            remaining -= rollout_steps
            if report.halted:
                break
            last_safe = AtomicCheckpoint().save(self._checkpoint_payload(step=global_step), checkpoint_root / "last_safe_step.pt")

    def _checkpoint_payload(self, *, step: int | None = None, reason: str | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "config": self.config,
            "state_dict": self.state_dict(),
            "phase": self.phase.value,
            "critic_loss_history": tuple(self.critic_loss_history),
            "last_update_step": self._last_update_step,
            "global_step": self.global_step,
            "ev_history": tuple(self.ev_history),
            "ev_sustain_start_step": self._ev_sustain_start_step,
            "ppo_transition_global_step": self._ppo_transition_global_step,
            "actor_lr_warmup_steps": self._actor_warmup_steps,
            "clip_eps_warmup_steps": self._clip_eps_warmup_steps,
            "clip_eps_warmup_initial": self._clip_eps_warmup_initial,
            "clip_eps_base": self._clip_eps_base,
            "rollout_last_observation": (
                self._last_observation.detach().cpu() if self._last_observation is not None else None
            ),
            "rollout_last_hidden_obs_buffer": (
                self._last_hidden_state.obs_buffer.detach().cpu() if self._last_hidden_state is not None else None
            ),
            "rollout_episode_ids": tuple(self._rollout_episode_ids or ()),
            "rollout_timesteps": tuple(self._rollout_timesteps or ()),
            "rollout_episode_returns": tuple(self._rollout_episode_returns or ()),
            "rollout_episode_lengths": tuple(self._rollout_episode_lengths or ()),
        }
        if self.optimizer is not None:
            payload["optimizer_state_dict"] = self.optimizer.state_dict()
        if step is not None:
            payload["step"] = int(step)
        if reason is not None:
            payload["reason"] = reason
        return payload

    def _halt_protocol(
        self,
        *,
        reason: str,
        step: int,
        forensics: FailureForensics,
        checkpoint_dir: str | Path,
        last_safe_step_path: str | Path,
    ) -> HaltProtocolResult:
        """Atomically dump black-box forensics, autopsy checkpoint, then verify safe checkpoint."""

        root = Path(checkpoint_dir)
        forensics_path = forensics.dump_to_disk(reason=reason, path=root / "forensics")
        failed_checkpoint_path = AtomicCheckpoint().save(
            self._checkpoint_payload(step=step, reason=reason),
            root / f"failed_step_{step:08d}.pt",
        )
        last_safe = Path(last_safe_step_path)
        try:
            _ = AtomicCheckpoint().load(last_safe)
        except Exception as exc:
            raise RuntimeError(f"last_safe_step checkpoint verification failed: {last_safe}") from exc
        return HaltProtocolResult(
            forensics_path=forensics_path,
            failed_checkpoint_path=failed_checkpoint_path,
            last_safe_step_path=last_safe,
            last_safe_step_verified=True,
        )

    def optimize_rollout(
        self,
        buffer: RolloutBuffer,
        *,
        epochs: int = 1,
        batch_size: int = 64,
        max_grad_norm: float = 1.0,
        watchdogs: WatchdogRegistry | None = None,
        forensics: FailureForensics | None = None,
        checkpoint_dir: str | Path | None = None,
        last_safe_step_path: str | Path | None = None,
        start_step: int = 0,
        global_step: int | None = None,
        total_steps: int = 1,
    ) -> PPOUpdateReport:
        """Run the Day 4 PPO update loop over an already-filled rollout buffer."""

        if self.optimizer is None:
            self.configure_optimizer()
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        active_forensics = forensics or FailureForensics(capacity=4096)
        updates = 0
        last_metrics: MetricDict = {}
        halt_result: HaltProtocolResult | None = None
        halt_reason: str | None = None
        for epoch in range(epochs):
            for batch in buffer.shuffled_batches(batch_size=batch_size, seed=start_step + epoch):
                step = start_step + updates
                schedule_global_step = int(global_step if global_step is not None else step)
                post_transition_steps = self._post_transition_steps(schedule_global_step)
                clip_epsilon = self._clip_eps_at_step(post_transition_steps)
                actor_lr = self._set_actor_lr_for_step(post_transition_steps)
                loss = self.compute_loss(batch, step=step, total_steps=total_steps, clip_epsilon=clip_epsilon)
                if not bool(torch.isfinite(loss.total_loss).detach().cpu().item()):
                    halt_reason = "nonfinite PPO loss"
                    active_forensics.push(step=step, metrics={"loss_is_finite": False})
                    if checkpoint_dir is not None and last_safe_step_path is not None:
                        halt_result = self._halt_protocol(
                            reason=halt_reason,
                            step=step,
                            forensics=active_forensics,
                            checkpoint_dir=checkpoint_dir,
                            last_safe_step_path=last_safe_step_path,
                        )
                    return PPOUpdateReport(updates=updates, phase=self.phase, halted=True, halt_reason=halt_reason, last_metrics=last_metrics, halt_result=halt_result)
                optimizer = self.optimizer
                if optimizer is None:
                    raise RuntimeError("optimizer is not configured")
                optimizer.zero_grad(set_to_none=True)
                loss.total_loss.backward()  # type: ignore[no-untyped-call]
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()
                critic_lr = self.step_critic_lr_cosine(step=step, total_steps=total_steps)
                updates += 1
                value_loss_float = float(loss.value_loss.detach().cpu().item())
                last_metrics = {
                    "ppo_total_loss": float(loss.total_loss.detach().cpu().item()),
                    "policy_loss": float(loss.policy_loss.detach().cpu().item()),
                    "value_loss": value_loss_float,
                    "entropy": float(loss.entropy.detach().cpu().item()),
                    "kl_loss": float(loss.kl_loss.detach().cpu().item()),
                    "approx_kl": float(loss.approx_kl.detach().cpu().item()),
                    "clip_fraction": float(loss.clip_fraction.detach().cpu().item()),
                    "grad_norm": float(grad_norm.detach().cpu().item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                    "actor_lr": actor_lr,
                    "clip_epsilon": clip_epsilon,
                    "post_transition_steps": post_transition_steps,
                    "critic_lr": critic_lr,
                    "phase": self.phase.value,
                }
                last_metrics.update(loss.diagnostics)
                active_forensics.push(step=step, metrics=last_metrics)
                if watchdogs is not None:
                    watchdogs.update(last_metrics, step=step)
                    if watchdogs.should_halt():
                        halt_reason = "watchdog halt"
                        if checkpoint_dir is not None and last_safe_step_path is not None:
                            halt_result = self._halt_protocol(
                                reason=halt_reason,
                                step=step,
                                forensics=active_forensics,
                                checkpoint_dir=checkpoint_dir,
                                last_safe_step_path=last_safe_step_path,
                        )
                        return PPOUpdateReport(updates=updates, phase=self.phase, halted=True, halt_reason=halt_reason, last_metrics=last_metrics, halt_result=halt_result)
                previous_phase = self.phase
                ev_value = last_metrics.get("value_explained_variance")
                transition_step = int(global_step if global_step is not None else step)
                self._maybe_advance_phase(
                    critic_loss=value_loss_float,
                    step=step,
                    global_step=transition_step,
                    explained_variance=float(ev_value) if isinstance(ev_value, int | float) else None,
                )
                if previous_phase is WarmupPhase.CRITIC_WARMUP and self.phase is WarmupPhase.PPO:
                    print(
                        "[V25] phase transition: critic_warmup -> full_ppo "
                        f"global_step={transition_step} "
                        f"ev={float(ev_value) if isinstance(ev_value, int | float) else float('nan'):.6g} "
                        f"ev_slope_100k={self._ev_slope_over_window(self.gate_v2_config.ev_slope_window_steps):.6g}",
                        flush=True,
                    )
                last_metrics["phase"] = self.phase.value
        return PPOUpdateReport(
            updates=updates,
            phase=self.phase,
            halted=False,
            halt_reason=None,
            last_metrics=last_metrics,
            halt_result=None,
        )

    def save(self, path: str | Path) -> None:
        """Persist the actor-critic state for later PPO stages."""

        AtomicCheckpoint().save(self._checkpoint_payload(), path)

    def load(self, path: str | Path) -> None:
        """Restore a saved actor-critic checkpoint."""

        payload = _load_checkpoint_payload(path, device=self.device)
        config = _coerce_config(payload.get("config"))
        if config != self.config:
            self.config = config
            self.actor = CausalTransformer(self.config).to(self.device)
            self.bc_reference = CausalTransformer(self.config).to(self.device)
            self.critic_head = nn.Linear(self.config.d_model, 1).to(self.device)
            self.actor_log_std = nn.Parameter(torch.zeros((self.config.action_dim,), device=self.device))
            self.register_buffer("_bc_log_std", torch.zeros((self.config.action_dim,), device=self.device), persistent=True)
            nn.init.orthogonal_(self.critic_head.weight, gain=0.01)
            nn.init.zeros_(self.critic_head.bias)
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise TypeError("actor-critic checkpoint must contain state_dict")
        self.load_state_dict(_coerce_state_dict(state_dict))
        phase_raw = payload.get("phase", WarmupPhase.CRITIC_WARMUP.value)
        self.phase = WarmupPhase(phase_raw) if isinstance(phase_raw, str) else WarmupPhase.CRITIC_WARMUP
        raw_history = payload.get("critic_loss_history", ())
        if isinstance(raw_history, Sequence):
            self.critic_loss_history = [float(value) for value in raw_history]
        else:
            self.critic_loss_history = []
        last_update_step = payload.get("last_update_step", 0)
        self._last_update_step = int(last_update_step) if isinstance(last_update_step, int) else 0
        global_step = payload.get("global_step", payload.get("step", 0))
        self.global_step = int(global_step) if isinstance(global_step, int) else 0
        raw_ev_history = payload.get("ev_history", ())
        if isinstance(raw_ev_history, Sequence):
            parsed_history: list[tuple[int, float]] = []
            for item in raw_ev_history:
                if isinstance(item, Sequence) and not isinstance(item, str) and len(item) == 2:
                    parsed_history.append((int(item[0]), float(item[1])))
            self.ev_history = parsed_history
        else:
            self.ev_history = []
        raw_sustain = payload.get("ev_sustain_start_step")
        self._ev_sustain_start_step = int(raw_sustain) if isinstance(raw_sustain, int) else None
        raw_transition_step = payload.get("ppo_transition_global_step")
        self._ppo_transition_global_step = int(raw_transition_step) if isinstance(raw_transition_step, int) else None
        self.bc_reference.requires_grad_(False)
        self.bc_reference.eval()
        if self.phase is WarmupPhase.CRITIC_WARMUP:
            self._freeze_actor_initial()
        else:
            self._unfreeze_actor()
        self.configure_optimizer()
        raw_actor_warmup = payload.get("actor_lr_warmup_steps")
        if isinstance(raw_actor_warmup, int):
            self._actor_warmup_steps = int(raw_actor_warmup)
        raw_clip_warmup = payload.get("clip_eps_warmup_steps")
        if isinstance(raw_clip_warmup, int):
            self._clip_eps_warmup_steps = int(raw_clip_warmup)
        raw_clip_initial = payload.get("clip_eps_warmup_initial")
        if isinstance(raw_clip_initial, int | float):
            self._clip_eps_warmup_initial = float(raw_clip_initial)
        raw_clip_base = payload.get("clip_eps_base")
        if isinstance(raw_clip_base, int | float):
            self._clip_eps_base = float(raw_clip_base)
        optimizer_state = payload.get("optimizer_state_dict")
        if isinstance(optimizer_state, Mapping):
            optimizer = self.optimizer
            if optimizer is None:
                raise RuntimeError("optimizer is not configured")
            optimizer.load_state_dict(dict(optimizer_state))
        raw_last_observation = payload.get("rollout_last_observation")
        self._last_observation = (
            raw_last_observation.to(device=self.device) if isinstance(raw_last_observation, torch.Tensor) else None
        )
        raw_hidden_obs = payload.get("rollout_last_hidden_obs_buffer")
        self._last_hidden_state = (
            CausalTransformerHiddenState(obs_buffer=raw_hidden_obs.to(device=self.device), kv_cache=None)
            if isinstance(raw_hidden_obs, torch.Tensor)
            else None
        )
        self._rollout_episode_ids = _optional_int_list(payload.get("rollout_episode_ids"))
        self._rollout_timesteps = _optional_int_list(payload.get("rollout_timesteps"))
        self._rollout_episode_returns = _optional_float_list(payload.get("rollout_episode_returns"))
        self._rollout_episode_lengths = _optional_int_list(payload.get("rollout_episode_lengths"))


def _coerce_policy_observation(
    observation: object,
    obs_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = observation if isinstance(observation, torch.Tensor) else torch.as_tensor(observation)
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"observation must have rank 1 or 2, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] == obs_dim:
        return tensor
    coerced = torch.zeros((tensor.shape[0], obs_dim), dtype=tensor.dtype, device=tensor.device)
    copy_dim = min(tensor.shape[-1], obs_dim)
    coerced[:, :copy_dim] = tensor[:, :copy_dim]
    return coerced


def _optional_int_list(value: object) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str):
        return None
    return [int(item) for item in value]


def _optional_float_list(value: object) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str):
        return None
    return [float(item) for item in value]


def _reset_done_hidden_lanes(
    hidden_state: CausalTransformerHiddenState,
    done_flags: Sequence[bool],
) -> CausalTransformerHiddenState:
    """Clear Transformer history only for lanes whose episode naturally ended."""

    mask = torch.as_tensor([bool(flag) for flag in done_flags], dtype=torch.bool, device=hidden_state.obs_buffer.device)
    if mask.numel() != hidden_state.obs_buffer.shape[0]:
        raise ValueError(f"done_flags must have {hidden_state.obs_buffer.shape[0]} entries, got {mask.numel()}")
    if bool(torch.any(mask).detach().cpu().item()):
        obs_buffer = hidden_state.obs_buffer.clone()
        obs_buffer[mask] = 0.0
    else:
        obs_buffer = hidden_state.obs_buffer
    return CausalTransformerHiddenState(obs_buffer=obs_buffer.detach(), kv_cache=None)


def _reset_done_env_lanes(
    env: SafeEnv[torch.Tensor, torch.Tensor],
    done_flags: Sequence[bool],
    *,
    seed: int | None,
) -> object:
    """Reset completed lanes when available, falling back to whole-env reset."""

    reset_done = getattr(env, "reset_done", None)
    if callable(reset_done):
        return reset_done(torch.as_tensor([bool(flag) for flag in done_flags], dtype=torch.bool))
    if not all(bool(flag) for flag in done_flags):
        raise RuntimeError("partial vector-lane resets require env.reset_done(done_mask)")
    observation_raw, _ = env.reset(seed=seed)
    return observation_raw


def _float_vector(value: object, count: int) -> list[float]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().reshape(-1)
        if tensor.numel() == 1:
            return [float(tensor.item()) for _ in range(count)]
        if tensor.numel() != count:
            raise ValueError(f"reward tensor must have {count} elements, got {tensor.numel()}")
        return [float(item.item()) for item in tensor]
    if isinstance(value, float | int):
        return [float(value) for _ in range(count)]
    tensor = torch.as_tensor(value).reshape(-1)
    if tensor.numel() == 1:
        return [float(tensor.item()) for _ in range(count)]
    if tensor.numel() != count:
        raise ValueError(f"reward value must have {count} elements, got {tensor.numel()}")
    return [float(item.item()) for item in tensor]


def _series_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan}
    tensor = torch.as_tensor([float(value) for value in values], dtype=torch.float64)
    std = torch.std(tensor, unbiased=False) if tensor.numel() > 1 else tensor.new_zeros(())
    return {
        "mean": float(torch.mean(tensor).item()),
        "std": float(std.item()),
        "min": float(torch.min(tensor).item()),
        "max": float(torch.max(tensor).item()),
    }


def _percentile_summary(values: Sequence[float], *, prefix: str) -> dict[str, float]:
    if not values:
        nan = float("nan")
        return {
            f"{prefix}_p01": nan,
            f"{prefix}_p05": nan,
            f"{prefix}_p50": nan,
            f"{prefix}_p95": nan,
            f"{prefix}_p99": nan,
        }
    tensor = torch.as_tensor([float(value) for value in values], dtype=torch.float64)
    return {
        f"{prefix}_p01": float(torch.quantile(tensor, 0.01).item()),
        f"{prefix}_p05": float(torch.quantile(tensor, 0.05).item()),
        f"{prefix}_p50": float(torch.quantile(tensor, 0.50).item()),
        f"{prefix}_p95": float(torch.quantile(tensor, 0.95).item()),
        f"{prefix}_p99": float(torch.quantile(tensor, 0.99).item()),
    }


def _h_hard_samples(info: Mapping[str, Any], env: SafeEnv[torch.Tensor, torch.Tensor]) -> list[float]:
    raw_values: object | None = None
    safety_metrics = info.get("safety_metrics")
    if isinstance(safety_metrics, Mapping):
        raw_values = safety_metrics.get("h_hard")
        if raw_values is None:
            raw_values = safety_metrics.get("h_hard_min")
    if raw_values is None:
        direct = info.get("h_hard")
        raw_values = direct if direct is not None else info.get("h_hard_min")
    if raw_values is None:
        try:
            env_metrics = env.safety_metrics()
        except Exception:
            env_metrics = {}
        raw_values = env_metrics.get("h_hard")
        if raw_values is None:
            raw_values = env_metrics.get("h_hard_min")
    if raw_values is None:
        return []
    try:
        tensor = raw_values if isinstance(raw_values, torch.Tensor) else torch.as_tensor(raw_values)
    except Exception:
        if isinstance(raw_values, int | float):
            return [float(raw_values)]
        return []
    tensor = tensor.detach().cpu().reshape(-1).to(dtype=torch.float64)
    return [float(value.item()) for value in tensor if math.isfinite(float(value.item()))]


def _termination_reasons(
    info: Mapping[str, Any],
    count: int,
    terminated: object,
    truncated: object,
) -> list[TerminationReason]:
    raw_reason = info.get("termination_reason")
    if isinstance(raw_reason, Sequence) and not isinstance(raw_reason, str):
        listed_reasons = [normalize_termination_reason(item) for item in raw_reason]
        if len(listed_reasons) != count:
            raise ValueError(f"termination_reason must have {count} entries, got {len(listed_reasons)}")
        return listed_reasons
    if raw_reason is not None:
        return [normalize_termination_reason(raw_reason) for _ in range(count)]
    terminated_flags = _bool_vector(terminated, count)
    truncated_flags = _bool_vector(truncated, count)
    reasons: list[TerminationReason] = []
    for is_terminated, is_truncated in zip(terminated_flags, truncated_flags, strict=True):
        if is_truncated:
            reasons.append("timeout")
        elif is_terminated:
            reasons.append("success")
        else:
            reasons.append("ongoing")
    return reasons


def _bool_vector(value: object, count: int) -> list[bool]:
    if isinstance(value, bool):
        return [value for _ in range(count)]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().reshape(-1)
    else:
        tensor = torch.as_tensor(value).reshape(-1)
    if tensor.numel() == 1:
        return [bool(tensor.item()) for _ in range(count)]
    if tensor.numel() != count:
        raise ValueError(f"boolean value must have {count} entries, got {tensor.numel()}")
    return [bool(item.item()) for item in tensor]


def _safety_actions(
    info: Mapping[str, Any],
    *,
    fallback: torch.Tensor,
    action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    filter_result = info.get("safety_filter")
    nominal_raw = getattr(filter_result, "nominal_action", fallback)
    safe_raw = getattr(filter_result, "safe_action", fallback)
    nominal = _coerce_action_tensor(nominal_raw, action_dim)
    safe = _coerce_action_tensor(safe_raw, action_dim)
    return nominal, safe


def _arena_projection_flags(info: Mapping[str, Any], count: int) -> list[bool]:
    filter_result = info.get("safety_filter")
    barrier_states = getattr(filter_result, "barrier_states", {})
    if isinstance(barrier_states, Mapping):
        arena_projection = barrier_states.get("arena_projection")
        raw_active = getattr(arena_projection, "active", None)
        if raw_active is not None:
            flags = _bool_vector(raw_active, count)
            if any(flags):
                return flags
    metrics = getattr(filter_result, "metrics", {})
    raw_count = metrics.get("arena_projection_active_count") if isinstance(metrics, Mapping) else None
    raw_rate = metrics.get("arena_projection_active_rate") if isinstance(metrics, Mapping) else None
    raw_active = metrics.get("arena_projection_active") if isinstance(metrics, Mapping) else None
    if isinstance(raw_count, int | float) and float(raw_count) >= float(count):
        return [True for _ in range(count)]
    if isinstance(raw_rate, int | float) and float(raw_rate) >= 1.0 - 1.0e-9:
        return [True for _ in range(count)]
    if isinstance(raw_active, bool) and raw_active:
        return [True for _ in range(count)]
    return [False for _ in range(count)]


def _env_action_dim(env: SafeEnv[torch.Tensor, torch.Tensor], *, default: int) -> int:
    action_space = getattr(env, "action_space", None)
    shape = getattr(action_space, "shape", None)
    if isinstance(shape, tuple) and len(shape) > 0 and isinstance(shape[-1], int):
        return int(shape[-1])
    return default


def _pad_action(action: torch.Tensor, *, action_dim: int, reference: torch.Tensor) -> torch.Tensor:
    if action.shape[-1] == action_dim:
        return action
    if action.shape[-1] > action_dim:
        return action[:, :action_dim]
    padded = reference.clone()
    padded[:, : action.shape[-1]] = action
    return padded


def _coerce_action_tensor(value: object, action_dim: int) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.detach().cpu().to(dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"action must have rank 1 or 2, got shape {tuple(tensor.shape)}")
    if tensor.shape[-1] != action_dim:
        raise ValueError(f"action_dim must be {action_dim}, got {tensor.shape[-1]}")
    return tensor


__all__ = [
    "ActorCriticOutput",
    "CausalTransformerActorCritic",
    "RolloutCollectionStats",
    "WarmupPhase",
]
