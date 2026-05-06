"""CPU-testable causal Transformer policy backbone for V25 BC-to-RL work."""

from __future__ import annotations

from collections.abc import Mapping, Sized
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, overload, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from parallelcbf.api import Algorithm, MetricDict, Prediction, SafeEnv, WatchdogRegistry
from parallelcbf.ops import AtomicCheckpoint, DefaultWatchdogRegistry, FailureForensics, ThresholdWatchdog


PosEncoding = Literal["rope"]
NormType = Literal["pre"]
ActivationName = Literal["gelu"]


@dataclass(frozen=True, slots=True)
class CausalTransformerConfig:
    """Configuration for the pure causal Transformer policy."""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    context_length: int = 64
    obs_dim: int = 16
    action_dim: int = 3
    pos_encoding: PosEncoding = "rope"
    norm_type: NormType = "pre"
    activation: ActivationName = "gelu"
    dropout: float = 0.0


@dataclass(frozen=True, slots=True)
class CausalTransformerHiddenState:
    """Inference state carried through the `Algorithm.predict` API."""

    obs_buffer: torch.Tensor
    kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None


@dataclass(frozen=True, slots=True)
class CausalTransformerTrainingReport:
    """Summary returned by offline BC training."""

    epochs_run: int
    steps_run: int
    train_losses: tuple[float, ...]
    val_losses: tuple[float, ...]
    best_val_loss: float | None
    checkpoint_paths: tuple[Path, ...]
    best_checkpoint_path: Path | None
    halted: bool
    halt_reason: str | None


def _validate_config(config: CausalTransformerConfig) -> None:
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if (config.d_model // config.n_heads) % 2 != 0:
        raise ValueError("RoPE requires an even per-head dimension")
    if config.context_length <= 0:
        raise ValueError("context_length must be positive")
    if config.pos_encoding != "rope":
        raise ValueError("only RoPE positional encoding is supported")
    if config.norm_type != "pre":
        raise ValueError("only Pre-LN Transformer blocks are supported")
    if config.activation != "gelu":
        raise ValueError("only GELU activation is supported")


def _causal_mask(seq_len: int, *, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)


def _rope_frequencies(head_dim: int, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    dims = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq = torch.pow(torch.full_like(dims, 10000.0), -dims / float(head_dim))
    return positions[:, None] * inv_freq[None, :]


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to a `(B, H, T, Dh)` tensor."""

    batch, heads, seq_len, head_dim = x.shape
    _ = batch, heads
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")
    freqs = _rope_frequencies(head_dim, seq_len, device=x.device, dtype=x.dtype)
    cos = torch.cos(freqs)[None, None, :, :]
    sin = torch.sin(freqs)[None, None, :, :]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with strict causal masking and RoPE."""

    def __init__(self, config: CausalTransformerConfig) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return causally attended token features."""

        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, T, D), got {tuple(x.shape)}")
        batch, seq_len, d_model = x.shape
        if d_model != self.config.d_model:
            raise ValueError(f"last dim must be {self.config.d_model}, got {d_model}")
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.config.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        q = apply_rope(q)
        k = apply_rope(k)
        scale = self.head_dim**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(_causal_mask(seq_len, device=x.device)[None, None, :, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(self.dropout(weights), v)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, seq_len, d_model)
        return cast(torch.Tensor, self.out_proj(attended))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with GELU feed-forward network."""

    def __init__(self, config: CausalTransformerConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_ff = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one residual causal-attention and feed-forward block."""

        x = x + self.attn(cast(torch.Tensor, self.ln_attn(x)))
        return x + cast(torch.Tensor, self.ff(cast(torch.Tensor, self.ln_ff(x))))


class CausalTransformer(nn.Module):
    """Causal Transformer that maps observation sequences to actions."""

    def __init__(self, config: CausalTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalTransformerConfig()
        _validate_config(self.config)
        self.obs_proj = nn.Linear(self.config.obs_dim, self.config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.n_layers)])
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.action_head = nn.Linear(self.config.d_model, self.config.action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Predict actions for every timestep in `(B, T, obs_dim)` observations."""

        if observations.ndim != 3:
            raise ValueError(f"observations must have shape (B, T, obs_dim), got {tuple(observations.shape)}")
        if observations.shape[-1] != self.config.obs_dim:
            raise ValueError(f"obs_dim must be {self.config.obs_dim}, got {observations.shape[-1]}")
        if observations.shape[1] > self.config.context_length:
            raise ValueError(f"sequence length {observations.shape[1]} exceeds context_length {self.config.context_length}")
        x = cast(torch.Tensor, self.obs_proj(observations))
        for block in self.blocks:
            x = block(x)
        return cast(torch.Tensor, self.action_head(cast(torch.Tensor, self.final_norm(x))))

    def forward_last(self, observations: torch.Tensor) -> torch.Tensor:
        """Return only the final action for each environment."""

        return self.forward(observations)[:, -1, :]


class CausalTransformerBC(Algorithm[torch.Tensor, torch.Tensor, CausalTransformerHiddenState]):
    """BC policy wrapper exposing the ParallelCBF `Algorithm` contract."""

    def __init__(self, config: CausalTransformerConfig | None = None, *, device: torch.device | None = None) -> None:
        self.config = config or CausalTransformerConfig()
        self.device = device or torch.device("cpu")
        self.model = CausalTransformer(self.config).to(self.device)
        self.timesteps_seen = 0

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

    @overload
    def learn(
        self,
        env: SafeEnv[torch.Tensor, torch.Tensor],
        *,
        total_timesteps: int,
        callback: WatchdogRegistry | None = None,
    ) -> None: ...

    @overload
    def learn(
        self,
        env: Dataset[object],
        *,
        val_dataset: Dataset[object] | None = None,
        epochs: int = 1,
        batch_size: int = 64,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 1.0e-2,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str | Path = "checkpoints/v25_transformer_bc",
        forensics_dir: str | Path | None = None,
        watchdogs: WatchdogRegistry | None = None,
        early_stop_patience: int | None = None,
    ) -> CausalTransformerTrainingReport: ...

    def learn(
        self,
        env: SafeEnv[torch.Tensor, torch.Tensor] | Dataset[object],
        *,
        total_timesteps: int | None = None,
        callback: WatchdogRegistry | None = None,
        val_dataset: Dataset[object] | None = None,
        epochs: int = 1,
        batch_size: int = 64,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 1.0e-2,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str | Path = "checkpoints/v25_transformer_bc",
        forensics_dir: str | Path | None = None,
        watchdogs: WatchdogRegistry | None = None,
        early_stop_patience: int | None = None,
    ) -> CausalTransformerTrainingReport | None:
        """Train online with an env or offline with BC sequence windows."""

        if isinstance(env, Dataset):
            return self.learn_bc(
                env,
                val_dataset=val_dataset,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                max_grad_norm=max_grad_norm,
                checkpoint_dir=checkpoint_dir,
                forensics_dir=forensics_dir,
                watchdogs=watchdogs,
                early_stop_patience=early_stop_patience,
            )
        if total_timesteps is None:
            raise ValueError("total_timesteps is required for online environment learning")
        self._learn_online(env, total_timesteps=total_timesteps, callback=callback)
        return None

    def _learn_online(
        self,
        env: SafeEnv[torch.Tensor, torch.Tensor],
        *,
        total_timesteps: int,
        callback: WatchdogRegistry | None = None,
    ) -> None:
        """Run a minimal online loop against a SafeEnv."""

        observation, _ = env.reset()
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        hidden_state = self.initial_hidden_state(observation.shape[0], dtype=observation.dtype, device=observation.device)
        for step in range(total_timesteps):
            prediction = self.predict(observation, hidden_state, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(prediction.action)
            if observation.ndim == 1:
                observation = observation.unsqueeze(0)
            hidden_state = prediction.hidden_state
            self.timesteps_seen += 1
            if callback is not None:
                callback.update(env.safety_metrics(), step=step)
                if callback.should_halt():
                    break
            if terminated or truncated:
                observation, _ = env.reset()
                if observation.ndim == 1:
                    observation = observation.unsqueeze(0)
                hidden_state = self.initial_hidden_state(observation.shape[0], dtype=observation.dtype, device=observation.device)

    def learn_bc(
        self,
        train_dataset: Dataset[object],
        *,
        val_dataset: Dataset[object] | None = None,
        epochs: int = 1,
        batch_size: int = 64,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 1.0e-2,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str | Path = "checkpoints/v25_transformer_bc",
        forensics_dir: str | Path | None = None,
        watchdogs: WatchdogRegistry | None = None,
        early_stop_patience: int | None = None,
    ) -> CausalTransformerTrainingReport:
        """Train the causal Transformer with behavior-cloning sequence windows."""

        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        train_size = _dataset_len(train_dataset)
        val_size = _dataset_len(val_dataset) if val_dataset is not None else 0
        if train_size == 0:
            raise ValueError("train_dataset must contain at least one window")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            if val_dataset is not None and val_size > 0
            else None
        )
        total_steps = max(1, epochs * len(train_loader))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _warmup_cosine_multiplier(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            ),
        )
        registry = watchdogs if watchdogs is not None else _default_bc_watchdogs()
        forensics = FailureForensics(capacity=4096)
        checkpoint_root = Path(checkpoint_dir)
        diagnostics_root = Path(forensics_dir) if forensics_dir is not None else checkpoint_root / "forensics"
        atomic = AtomicCheckpoint()
        parameter_dtype = next(self.model.parameters()).dtype

        train_losses: list[float] = []
        val_losses: list[float] = []
        checkpoint_paths: list[Path] = []
        best_val_loss: float | None = None
        best_checkpoint_path: Path | None = None
        best_epoch = 0
        global_step = 0
        halted = False
        halt_reason: str | None = None

        self.model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            batch_count = 0
            for raw_batch in train_loader:
                batch = _as_batch_mapping(raw_batch)
                observations = _batch_tensor(batch, "observations").to(
                    device=self.device,
                    dtype=parameter_dtype,
                )
                actions = _batch_tensor(batch, "actions").to(device=self.device, dtype=parameter_dtype)
                optimizer.zero_grad(set_to_none=True)
                predictions = self.model(observations)
                loss = torch.nn.functional.mse_loss(predictions, actions)
                loss.backward()  # type: ignore[no-untyped-call]
                grad_norm_raw = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                loss_value = float(loss.detach().cpu().item())
                grad_norm = float(grad_norm_raw.detach().cpu().item()) if isinstance(grad_norm_raw, torch.Tensor) else float(grad_norm_raw)
                running_loss += loss_value
                batch_count += 1
                global_step += 1
                self.timesteps_seen += int(observations.shape[0] * observations.shape[1])

                metrics: MetricDict = {
                    "phase": "train_step",
                    "epoch": epoch,
                    "train_loss": loss_value,
                    "grad_norm": grad_norm,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                forensics.push(step=global_step, metrics=metrics)
                registry.update(metrics, step=global_step)
                if registry.should_halt():
                    halted = True
                    halt_reason = _latest_halt_reason(registry)
                    forensics.dump_to_disk(reason=halt_reason, path=diagnostics_root)
                    break

            if batch_count == 0:
                raise RuntimeError("train_loader produced no batches")

            epoch_train_loss = running_loss / float(batch_count)
            train_losses.append(epoch_train_loss)
            epoch_val_loss = self._evaluate_bc_loader(val_loader, dtype=parameter_dtype)
            if epoch_val_loss is not None:
                val_losses.append(epoch_val_loss)

            overfit_ratio = (
                epoch_val_loss / max(epoch_train_loss, 1.0e-12)
                if epoch_val_loss is not None
                else 1.0
            )
            epoch_metrics: MetricDict = {
                "phase": "epoch",
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "grad_norm": 0.0,
                "val_loss": epoch_val_loss if epoch_val_loss is not None else -1.0,
                "val_overfit_ratio": overfit_ratio,
            }
            forensics.push(step=global_step, metrics=epoch_metrics)
            registry.update(epoch_metrics, step=global_step)

            checkpoint_path = checkpoint_root / f"epoch_{epoch:04d}.pt"
            checkpoint_paths.append(
                atomic.save(
                    self._checkpoint_payload(
                        epoch=epoch,
                        global_step=global_step,
                        train_losses=train_losses,
                        val_losses=val_losses,
                        best_val_loss=best_val_loss,
                    ),
                    checkpoint_path,
                )
            )
            comparison_loss = epoch_val_loss if epoch_val_loss is not None else epoch_train_loss
            if best_val_loss is None or comparison_loss < best_val_loss:
                best_val_loss = comparison_loss
                best_epoch = epoch
                best_checkpoint_path = atomic.save(
                    self._checkpoint_payload(
                        epoch=epoch,
                        global_step=global_step,
                        train_losses=train_losses,
                        val_losses=val_losses,
                        best_val_loss=best_val_loss,
                    ),
                    checkpoint_root / "best.pt",
                )

            print(
                "epoch="
                f"{epoch} train_loss={epoch_train_loss:.12g} "
                f"val_loss={epoch_val_loss if epoch_val_loss is not None else 'NA'} "
                f"best_val_loss={best_val_loss if best_val_loss is not None else 'NA'} "
                f"checkpoint={checkpoint_path}",
                flush=True,
            )

            if registry.should_halt() and not halted:
                halted = True
                halt_reason = _latest_halt_reason(registry)
                forensics.dump_to_disk(reason=halt_reason, path=diagnostics_root)
            if halted:
                break
            if early_stop_patience is not None and epoch - best_epoch >= early_stop_patience:
                halted = True
                halt_reason = f"early_stop_patience={early_stop_patience}"
                forensics.dump_to_disk(reason=halt_reason, path=diagnostics_root)
                break

        return CausalTransformerTrainingReport(
            epochs_run=len(train_losses),
            steps_run=global_step,
            train_losses=tuple(train_losses),
            val_losses=tuple(val_losses),
            best_val_loss=best_val_loss,
            checkpoint_paths=tuple(checkpoint_paths),
            best_checkpoint_path=best_checkpoint_path,
            halted=halted,
            halt_reason=halt_reason,
        )

    def _evaluate_bc_loader(
        self,
        val_loader: DataLoader[object] | None,
        *,
        dtype: torch.dtype,
    ) -> float | None:
        if val_loader is None:
            return None
        self.model.eval()
        loss_sum = 0.0
        batch_count = 0
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = _as_batch_mapping(raw_batch)
                observations = _batch_tensor(batch, "observations").to(device=self.device, dtype=dtype)
                actions = _batch_tensor(batch, "actions").to(device=self.device, dtype=dtype)
                predictions = self.model(observations)
                loss = torch.nn.functional.mse_loss(predictions, actions)
                loss_sum += float(loss.detach().cpu().item())
                batch_count += 1
        self.model.train()
        if batch_count == 0:
            return None
        return loss_sum / float(batch_count)

    def _checkpoint_payload(
        self,
        *,
        epoch: int,
        global_step: int,
        train_losses: list[float],
        val_losses: list[float],
        best_val_loss: float | None,
    ) -> dict[str, object]:
        return {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "timesteps_seen": self.timesteps_seen,
            "train_losses": tuple(train_losses),
            "val_losses": tuple(val_losses),
            "best_val_loss": best_val_loss,
        }

    def predict(
        self,
        observation: torch.Tensor,
        hidden_state: CausalTransformerHiddenState,
        *,
        deterministic: bool = False,
    ) -> Prediction[torch.Tensor, CausalTransformerHiddenState]:
        """Predict one action and carry a rolling observation context."""

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
        with torch.no_grad():
            action = self.model.forward_last(next_buffer)
        next_state = CausalTransformerHiddenState(obs_buffer=next_buffer.detach(), kv_cache=hidden_state.kv_cache)
        return Prediction(action=action, hidden_state=next_state)

    def save(self, path: str | Path) -> None:
        """Persist the model, config, and timestep counter."""

        payload = {
            "config": self.config,
            "state_dict": self.model.state_dict(),
            "timesteps_seen": self.timesteps_seen,
        }
        torch.save(payload, Path(path))

    def load(self, path: str | Path) -> None:
        """Load the model, config, and timestep counter."""

        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("CausalTransformerBC checkpoint must be a dictionary")
        config = payload["config"]
        if not isinstance(config, CausalTransformerConfig):
            raise ValueError("checkpoint config must be CausalTransformerConfig")
        self.config = config
        self.model = CausalTransformer(self.config).to(self.device)
        state_dict = cast(dict[str, torch.Tensor], payload["state_dict"])
        self.model.load_state_dict(state_dict)
        self.timesteps_seen = int(payload["timesteps_seen"])


def _warmup_cosine_multiplier(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    cosine_steps = max(1, total_steps - max(0, warmup_steps))
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(cosine_steps)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _dataset_len(dataset: Dataset[object] | None) -> int:
    if dataset is None:
        return 0
    if not isinstance(dataset, Sized):
        raise TypeError("dataset must be sized")
    return len(dataset)


def _default_bc_watchdogs() -> DefaultWatchdogRegistry:
    registry = DefaultWatchdogRegistry()
    registry.register(ThresholdWatchdog("train_loss", 1.0e6, label="Train Loss Explosion"))
    registry.register(ThresholdWatchdog("grad_norm", 1.0e4, label="Grad Norm Blowup"))
    registry.register(ThresholdWatchdog("val_overfit_ratio", 5.0, label="Val Overfit"))
    return registry


def _as_batch_mapping(raw_batch: object) -> Mapping[str, object]:
    if not isinstance(raw_batch, Mapping):
        raise TypeError("BC DataLoader batch must be a mapping")
    return cast(Mapping[str, object], raw_batch)


def _batch_tensor(batch: Mapping[str, object], key: str) -> torch.Tensor:
    raw_value = batch.get(key)
    if not isinstance(raw_value, torch.Tensor):
        raise TypeError(f"batch[{key!r}] must be a torch.Tensor")
    return raw_value


def _latest_halt_reason(registry: WatchdogRegistry) -> str:
    events = getattr(registry, "events", ())
    if isinstance(events, tuple) and len(events) > 0:
        latest = events[-1]
        reason = getattr(latest, "reason", None)
        if isinstance(reason, str):
            return reason
    return "watchdog_halt"
