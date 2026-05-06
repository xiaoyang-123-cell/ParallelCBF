#!/usr/bin/env python3
"""Manifest-driven V25 PPO launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self, SupportsFloat, SupportsIndex, cast


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEV_ROOT = Path("/home/smartlab/parallelcbf_dev")
CONDA_EXE = Path("/home/smartlab/miniforge3/bin/conda")
EXPECTED_ACTOR_PARAMETERS = 3_167_235
EXPECTED_MANIFEST_SHA256 = "80cc153947515063c529a07d7aa3e412094ea9218f7284b301568eb3ae513cc9"
PHASE_CRITIC_WARMUP = "critic_warmup"
PHASE_FULL_PPO = "ppo"


def _prefer_dev_package() -> None:
    dev = str(DEV_ROOT)
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != DEV_ROOT]
    sys.path.insert(0, dev)


def _running_in_parallel_env() -> bool:
    executable = Path(sys.executable).as_posix()
    return "envs/parallel_uav/bin/python" in executable or os.environ.get("CONDA_DEFAULT_ENV") == "parallel_uav"


def _bootstrap_parallel_env() -> None:
    if os.environ.get("V25_BOOTSTRAPPED") == "1" or _running_in_parallel_env():
        return
    if not CONDA_EXE.exists():
        raise FileNotFoundError(f"conda executable not found: {CONDA_EXE}")
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [str(DEV_ROOT)]
    if current_pythonpath:
        pythonpath_parts.append(current_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    os.environ["V25_BOOTSTRAPPED"] = "1"
    os.execv(
        str(CONDA_EXE),
        [
            str(CONDA_EXE),
            "run",
            "--no-capture-output",
            "-n",
            "parallel_uav",
            "python",
            "-u",
            "-m",
            "scripts.train_ppo_v25",
            *sys.argv[1:],
        ],
    )


_bootstrap_parallel_env()
_prefer_dev_package()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import torch  # noqa: E402

from parallelcbf.algorithms import CausalTransformerActorCritic, compute_gae_in_place  # noqa: E402
from parallelcbf.api import MetricDict, SafetyFilterResult, SafetyState, Watchdog, WatchdogEvent  # noqa: E402
from parallelcbf.envs.toy2d import Toy2DConfig  # noqa: E402
from parallelcbf.envs.toy2d_vec import Toy2DAvoidanceVecEnv  # noqa: E402
from parallelcbf.ops import (  # noqa: E402
    AtomicCheckpoint,
    DefaultWatchdogRegistry,
    FailureForensics,
    SustainedPhaseWatchdog,
    SustainedThresholdWatchdog,
    ThresholdWatchdog,
)
from parallelcbf.safety import TripleBarrierCBF  # noqa: E402


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    pre_registration: Path
    watchdog_config: Path
    bc_checkpoint: Path
    bc_dataset_sha256: str
    total_timesteps: int
    num_envs: int
    device: str
    seed: int
    output_dir: Path
    tb_logdir: Path
    wandb_project: str
    wandb_run_name: str
    checkpoint_every: int
    log_every: int
    strict_pre_registration: bool
    resume_from: Path | None


class JsonlLogger:
    """Tiny local telemetry sink used even when optional services are absent."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        _ = exc_type
        _ = exc
        _ = tb
        self._handle.close()

    def write(self, payload: dict[str, object]) -> None:
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()


class OptionalWandbRun:
    """Best-effort WandB integration that never prompts in headless mode."""

    def __init__(self, *, project: str, name: str, output_dir: Path, config: dict[str, object]) -> None:
        self._run: Any | None = None
        self.url: str | None = None
        try:
            import wandb  # type: ignore[import-not-found]
        except Exception as exc:
            print(f"wandb_status=unavailable reason={type(exc).__name__}:{exc}", flush=True)
            return
        mode = os.environ.get("WANDB_MODE")
        if not os.environ.get("WANDB_API_KEY") and not mode:
            os.environ["WANDB_MODE"] = "offline"
            mode = "offline"
        wandb_mode = cast(Any, mode)
        try:
            self._run = wandb.init(
                project=project,
                name=name,
                dir=str(output_dir / "wandb"),
                config=config,
                anonymous="allow",
                mode=wandb_mode,
            )
            raw_url = getattr(self._run, "url", None)
            self.url = str(raw_url) if raw_url else None
            print(f"wandb_status=initialized mode={mode or 'online'} url={self.url or 'unavailable'}", flush=True)
        except Exception as exc:
            print(f"wandb_status=failed reason={type(exc).__name__}:{exc}", flush=True)
            self._run = None

    def log(self, metrics: dict[str, object], *, step: int) -> None:
        if self._run is None:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception as exc:
            print(f"wandb_log_status=failed reason={type(exc).__name__}:{exc}", flush=True)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception:
            return


class OptionalTensorBoardRun:
    """Best-effort TensorBoard scalar writer for local diagnostic burns."""

    def __init__(self, logdir: Path) -> None:
        self._writer: Any | None = None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as exc:
            print(f"tensorboard_status=unavailable reason={type(exc).__name__}:{exc}", flush=True)
            return
        try:
            logdir.mkdir(parents=True, exist_ok=True)
            self._writer = cast(Any, SummaryWriter)(log_dir=str(logdir))
            print(f"tensorboard_status=initialized logdir={logdir}", flush=True)
        except Exception as exc:
            print(f"tensorboard_status=failed reason={type(exc).__name__}:{exc}", flush=True)
            self._writer = None

    def log(self, metrics: dict[str, object], *, step: int) -> None:
        if self._writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, bool):
                self._writer.add_scalar(key, float(value), step)
            elif isinstance(value, int | float) and math.isfinite(float(value)):
                self._writer.add_scalar(key, float(value), step)
        self._writer.flush()

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.close()


class VectorSafetyWrapper:
    """Layer-3-only adapter composing the vector fixture with the CBF filter."""

    metadata: dict[str, Any] = {"render_modes": []}
    reward_range: tuple[float, float] = (-float("inf"), float("inf"))
    spec: Any = None

    def __init__(self, env: Toy2DAvoidanceVecEnv, safety_filter: TripleBarrierCBF) -> None:
        self.env = env
        self.safety_filter = safety_filter
        obs_high = np.full((8,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._last_observation: npt.NDArray[np.float32] | None = None
        self._last_filter_result: SafetyFilterResult[torch.Tensor] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        _ = options
        self.safety_filter.reset(seed=seed)
        self._last_filter_result = None
        observation, info = self.env.reset(seed=seed)
        self._last_observation = observation
        return observation, dict(info)

    def reset_done(self, done_mask: torch.Tensor | npt.NDArray[np.bool_]) -> npt.NDArray[np.float32]:
        """Reset only completed vector lanes while preserving active episodes."""

        mask = np.asarray(done_mask, dtype=np.bool_).reshape(-1)
        if mask.shape != (self.env.num_envs,):
            raise ValueError(f"done_mask must have shape ({self.env.num_envs},), got {mask.shape}")
        if self._last_observation is None:
            observation, _ = self.reset()
            return observation
        if not np.any(mask):
            return self._last_observation
        self.env._step_count[mask] = 0
        self.env._pos[mask, :] = np.array([-2.0, 0.0], dtype=np.float32)
        self.env._vel[mask, :] = 0.0
        self.env._goal[mask, :] = np.array([2.0, 0.0], dtype=np.float32)
        self.env._obstacle[mask, :] = np.array([0.4, 0.25], dtype=np.float32)
        self.env._last_hard_violation[mask] = False
        for env_id in np.nonzero(mask)[0].tolist():
            self.env._last_termination_reason[env_id] = None
        self._last_observation = self.env._observation()
        return self._last_observation

    def step(
        self,
        action: torch.Tensor | npt.NDArray[np.float32],
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
        dict[str, Any],
    ]:
        if self._last_observation is None:
            raise RuntimeError("VectorSafetyWrapper.step() called before reset().")
        nominal = torch.as_tensor(action, dtype=torch.float32, device="cpu")
        filter_result = self.safety_filter.filter_action(
            torch.as_tensor(self._last_observation, dtype=torch.float32),
            nominal,
            self.safety_state(),
        )
        self._last_filter_result = filter_result
        safe_action = filter_result.safe_action.detach().cpu().numpy().astype(np.float32)
        observation, reward, terminated, truncated, info = self.env.step(safe_action)
        self._last_observation = observation
        wrapped_info = dict(info)
        wrapped_info["safety_filter"] = filter_result
        return observation, reward, terminated, truncated, wrapped_info

    def safety_state(self) -> SafetyState:
        pos = np.asarray(getattr(self.env, "_pos"), dtype=np.float32)
        vel = np.asarray(getattr(self.env, "_vel"), dtype=np.float32)
        goal = np.asarray(getattr(self.env, "_goal"), dtype=np.float32)
        obstacle = np.asarray(getattr(self.env, "_obstacle"), dtype=np.float32).reshape(self.env.num_envs, 1, 2)
        radius = float(self.env.config.arena_radius)
        half_extent = radius / math.sqrt(2.0)
        return SafetyState(
            position=pos,
            velocity=vel,
            goal=goal,
            obstacles=obstacle,
            robot_radius=float(self.env.config.robot_radius),
            obstacle_radius=float(self.env.config.obstacle_radius),
            arena_bounds=np.array([-half_extent, half_extent, -half_extent, half_extent], dtype=np.float32),
            metadata={"arena_bounds_source": "inscribed_square", "arena_radius": radius},
        )

    def safety_metrics(self) -> MetricDict:
        metrics: MetricDict = {}
        env_metrics = self.env.safety_metrics()
        for key, value in env_metrics.items():
            array = np.asarray(value)
            if array.dtype == np.bool_:
                metrics[key] = bool(np.any(array))
            else:
                metrics[key] = float(cast(SupportsFloat, np.min(array)))
        metrics.update(self.safety_filter.metrics())
        if self._last_filter_result is not None:
            metrics.update(self._last_filter_result.metrics)
        return metrics

    def hard_constraint_violations(self) -> dict[str, bool]:
        metrics = self.env.step  # keeps static analyzers from mistaking this adapter for the source env
        _ = metrics
        hard = self.env.safety_metrics()["h_hard_violation"]
        return {"collision_or_oob": bool(np.any(hard))}

    def close(self) -> None:
        return None


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-registration", type=Path, required=True)
    parser.add_argument("--watchdog-config", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--bc-dataset-sha256", type=str, required=True)
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tb-logdir", type=Path, required=True)
    parser.add_argument("--wandb-project", type=str, required=True)
    parser.add_argument("--wandb-run-name", type=str, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--strict-pre-registration", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()
    return RuntimeConfig(
        pre_registration=_resolve(args.pre_registration),
        watchdog_config=_resolve(args.watchdog_config),
        bc_checkpoint=_resolve(args.bc_checkpoint),
        bc_dataset_sha256=args.bc_dataset_sha256,
        total_timesteps=int(args.total_timesteps),
        num_envs=int(args.num_envs),
        device=str(args.device),
        seed=int(args.seed),
        output_dir=_resolve(args.output_dir),
        tb_logdir=_resolve(args.tb_logdir),
        wandb_project=str(args.wandb_project),
        wandb_run_name=str(args.wandb_run_name),
        checkpoint_every=int(args.checkpoint_every),
        log_every=int(args.log_every),
        strict_pre_registration=bool(args.strict_pre_registration),
        resume_from=_resolve(args.resume_from) if args.resume_from is not None else None,
    )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pre_registration(payload: dict[str, Any], config: RuntimeConfig, *, bc_sha256: str) -> None:
    required = {
        "bc_checkpoint",
        "bc_checkpoint_sha256",
        "bc_dataset_sha256",
        "total_timesteps",
        "num_envs",
        "seed",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"pre-registration missing required fields: {missing}")
    expected_checkpoint = _resolve(Path(str(payload["bc_checkpoint"])))
    if expected_checkpoint != config.bc_checkpoint:
        raise ValueError(f"BC checkpoint mismatch: manifest={expected_checkpoint} cli={config.bc_checkpoint}")
    if str(payload["bc_checkpoint_sha256"]) != bc_sha256:
        raise ValueError("BC checkpoint SHA-256 mismatch")
    if str(payload["bc_dataset_sha256"]) != config.bc_dataset_sha256:
        raise ValueError("BC dataset SHA-256 mismatch")
    if int(payload["num_envs"]) != config.num_envs:
        raise ValueError("num_envs mismatch")
    if int(payload["seed"]) != config.seed:
        raise ValueError("seed mismatch")
    allowed_timesteps = {int(payload["total_timesteps"])}
    diagnostic_timesteps = payload.get("diagnostic_burn_timesteps")
    if diagnostic_timesteps is not None:
        allowed_timesteps.add(int(diagnostic_timesteps))
    if config.total_timesteps != 0 and config.total_timesteps not in allowed_timesteps:
        raise ValueError(f"total_timesteps mismatch: cli={config.total_timesteps} allowed={sorted(allowed_timesteps)}")
    manifest_resume = payload.get("resume_allowed", True)
    if config.resume_from is not None and manifest_resume is False:
        raise ValueError("resume_from was provided but manifest forbids resume")


def _watchdog_direction(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"gt", ">", "greater_than"}:
        return True
    if normalized in {"lt", "<", "less_than"}:
        return False
    raise ValueError(f"unsupported watchdog comparison: {value}")


def _normalize_watchdog_severity(value: object) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "info": "info",
        "warn": "warning",
        "warning": "warning",
        "halt": "critical",
        "critical": "critical",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported watchdog severity: {value}")
    return aliases[normalized]


def _normalize_watchdog_phase(value: object) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "0": PHASE_CRITIC_WARMUP,
        "critic_warmup": PHASE_CRITIC_WARMUP,
        "warmup": PHASE_CRITIC_WARMUP,
        "phase_0": PHASE_CRITIC_WARMUP,
        "1": PHASE_FULL_PPO,
        "full_ppo": PHASE_FULL_PPO,
        "ppo": PHASE_FULL_PPO,
        "phase_1": PHASE_FULL_PPO,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported watchdog phase: {value}")
    return aliases[normalized]


def _current_phase(metrics: MetricDict) -> str | None:
    phase_name = metrics.get("phase_name")
    if isinstance(phase_name, str):
        return _normalize_watchdog_phase(phase_name)
    phase_index = metrics.get("phase")
    if isinstance(phase_index, str):
        return _normalize_watchdog_phase(phase_index)
    if isinstance(phase_index, int | float):
        return _normalize_watchdog_phase(str(int(phase_index)))
    return None


@dataclass(slots=True)
class PhaseGatedWatchdog(Watchdog):
    """Run an inner watchdog only in the pre-registered training phases."""

    inner: Watchdog
    active_phases: frozenset[str]

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def metric(self) -> str:
        return str(getattr(self.inner, "metric", "unknown"))

    @property
    def threshold(self) -> object:
        return getattr(self.inner, "threshold", "unknown")

    @property
    def when(self) -> str:
        return str(getattr(self.inner, "when", "unknown"))

    def update(self, metrics: MetricDict, *, step: int) -> WatchdogEvent | None:
        phase = _current_phase(metrics)
        if phase is None or phase not in self.active_phases:
            return None
        return self.inner.update(metrics, step=step)

    def reset(self) -> None:
        self.inner.reset()


def _build_watchdogs(path: Path) -> DefaultWatchdogRegistry:
    payload = _load_json_object(path)
    raw_watchdogs = payload.get("watchdogs")
    if not isinstance(raw_watchdogs, list):
        raise ValueError("watchdog config must contain a watchdogs list")
    assert all(isinstance(rule, dict) and "active_phases" in rule for rule in raw_watchdogs), (
        "all watchdog rules must declare active_phases"
    )
    registry = DefaultWatchdogRegistry()
    required_names = {"Train Loss Explosion", "Grad Norm Blowup", "Val Overfit"}
    seen_names: set[str] = set()
    for raw in raw_watchdogs:
        if not isinstance(raw, dict):
            raise TypeError("watchdog entries must be objects")
        active_phases_raw = raw["active_phases"]
        if not isinstance(active_phases_raw, list) or not active_phases_raw:
            raise ValueError("watchdog active_phases must be a non-empty list")
        active_phases = frozenset(_normalize_watchdog_phase(phase) for phase in active_phases_raw)
        kind = str(raw.get("type", "threshold"))
        if kind == "sustained_phase":
            for field in ("name", "metric", "value", "max_steps"):
                if field not in raw:
                    raise ValueError(f"watchdog entry missing field: {field}")
            name = str(raw["name"])
            seen_names.add(name)
            severity = _normalize_watchdog_severity(raw.get("severity", "critical"))
            registry.register(
                PhaseGatedWatchdog(
                    SustainedPhaseWatchdog(
                        phase_metric=str(raw["metric"]),
                        phase_value=raw["value"] if isinstance(raw["value"], int | float | str) else str(raw["value"]),
                        max_steps=int(raw["max_steps"]),
                        label=name,
                        severity=severity,
                    ),
                    active_phases,
                )
            )
            continue
        if kind != "threshold":
            raise ValueError(f"unsupported watchdog type: {kind}")
        for field in ("name", "metric", "threshold", "when"):
            if field not in raw and not (field == "threshold" and "threshold_slope" in raw):
                raise ValueError(f"watchdog entry missing field: {field}")
        name = str(raw["name"])
        seen_names.add(name)
        severity = _normalize_watchdog_severity(raw.get("severity", "critical"))
        raw_threshold = raw["threshold"] if "threshold" in raw else raw["threshold_slope"]
        threshold = float(raw_threshold)
        sustained_steps = raw.get("sustained_steps", raw.get("sustained_for_steps"))
        if sustained_steps is not None:
            registry.register(
                PhaseGatedWatchdog(
                    SustainedThresholdWatchdog(
                        str(raw["metric"]),
                        threshold,
                        sustained_steps=int(sustained_steps),
                        greater_than=_watchdog_direction(raw["when"]),
                        severity=severity,
                        label=name,
                    ),
                    active_phases,
                )
            )
            continue
        registry.register(
            PhaseGatedWatchdog(
                ThresholdWatchdog(
                    str(raw["metric"]),
                    threshold,
                    greater_than=_watchdog_direction(raw["when"]),
                    severity=severity,
                    label=name,
                ),
                active_phases,
            )
        )
    missing = sorted(required_names - seen_names)
    if missing:
        raise ValueError(f"watchdog config missing required entries: {missing}")
    return registry


def _print_watchdogs(registry: DefaultWatchdogRegistry) -> None:
    print("=" * 60)
    print("WATCHDOG REGISTRY (V25 PPO training):")
    for watchdog in registry.list_registered():
        metric = getattr(watchdog, "metric", "unknown")
        threshold = getattr(watchdog, "threshold", "unknown")
        when = getattr(watchdog, "when", "unknown")
        active_phases = ",".join(getattr(watchdog, "active_phases", ("unknown",)))
        print(f"  - {watchdog.name}: metric={metric}, threshold={threshold}, when={when}, phases={active_phases}")
    print("=" * 60)


def _make_env(num_envs: int) -> VectorSafetyWrapper:
    toy_config = Toy2DConfig(max_steps=500)
    env = Toy2DAvoidanceVecEnv(num_envs=num_envs, config=toy_config)
    return VectorSafetyWrapper(env, TripleBarrierCBF())


def _metric_float(metrics: MetricDict, key: str, default: float = float("nan")) -> float:
    value = metrics.get(key, default)
    if isinstance(value, bool | str):
        return default
    return float(value)


def _finite_float(value: object, default: float = float("nan")) -> float:
    if isinstance(value, bool):
        return default
    if value is None:
        return default
    if not isinstance(value, str | bytes | bytearray | SupportsFloat | SupportsIndex):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rolling_slope(values: list[float]) -> float:
    count = len(values)
    if count < 2:
        return 0.0
    mean_x = float(count - 1) / 2.0
    mean_y = sum(values) / float(count)
    numerator = 0.0
    denominator = 0.0
    for index, value in enumerate(values):
        dx = float(index) - mean_x
        numerator += dx * (value - mean_y)
        denominator += dx * dx
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _zero_variance(values: list[float], *, tolerance: float = 1.0e-8) -> bool:
    if len(values) < 2:
        return False
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) * (value - mean) for value in values) / float(len(values))
    return variance <= tolerance * tolerance


def _save_checkpoint(algo: CausalTransformerActorCritic, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    algo.save(path)
    return path


def _checkpoint_step_from_path(path: Path) -> int | None:
    stem = path.stem
    if stem.startswith("step_"):
        try:
            return int(stem.removeprefix("step_"))
        except ValueError:
            return None
    return None


def _infer_resume_transitions(path: Path, algo: CausalTransformerActorCritic) -> int:
    """Infer transition count from checkpoint payload or adjacent step files."""

    checkpoint = AtomicCheckpoint().load(path)
    if isinstance(checkpoint, dict):
        for key in ("global_step", "step"):
            value = checkpoint.get(key)
            if isinstance(value, int):
                return int(value)
    from_name = _checkpoint_step_from_path(path)
    if from_name is not None:
        return from_name
    siblings = sorted(path.parent.glob("step_*.pt"))
    if siblings:
        latest_step = _checkpoint_step_from_path(siblings[-1])
        if latest_step is not None:
            return latest_step
    if algo.global_step > 0:
        return int(algo.global_step)
    return 0


def _checkpoint_has_optimizer_state(path: Path) -> bool:
    checkpoint = AtomicCheckpoint().load(path)
    return isinstance(checkpoint, dict) and isinstance(checkpoint.get("optimizer_state_dict"), dict)


def _warm_start_gate_v2_from_metrics(algo: CausalTransformerActorCritic, resume_from: Path) -> None:
    """Seed Gate V2 EV history from a previous run's JSONL telemetry when available."""

    metrics_path = resume_from.parent.parent / "metrics.jsonl"
    if not metrics_path.exists():
        return
    rows: list[tuple[int, float]] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = payload.get("rollout/transitions")
        ev = payload.get("train/value_explained_variance")
        if isinstance(step, int) and isinstance(ev, int | float) and math.isfinite(float(ev)):
            rows.append((int(step), float(ev)))
    if not rows:
        return
    latest_step = rows[-1][0]
    min_step = latest_step - max(algo.gate_v2_config.ev_slope_window_steps * 2, algo.gate_v2_config.ev_sustain_steps * 4)
    algo.ev_history = [(step, ev) for step, ev in rows if step >= min_step]
    algo.global_step = max(algo.global_step, latest_step)
    sustain_start: int | None = None
    for step, ev in algo.ev_history:
        if ev >= algo.gate_v2_config.ev_threshold:
            if sustain_start is None:
                sustain_start = step
        else:
            sustain_start = None
    algo._ev_sustain_start_step = sustain_start
    print(
        "[RESUME][V25] seeded_gate_v2_history=True "
        f"samples={len(algo.ev_history)} latest_step={latest_step} "
        f"ev_sustain_start_step={sustain_start}",
        flush=True,
    )


def _run_training(
    config: RuntimeConfig,
    algo: CausalTransformerActorCritic,
    env: VectorSafetyWrapper,
    registry: DefaultWatchdogRegistry,
    *,
    start_transitions: int = 0,
) -> None:
    print("[INFO][V25] algo.learn entry reached", flush=True)
    if config.total_timesteps <= 0:
        print("[SMOKE][V25] total_timesteps=0; exiting cleanly after instantiation", flush=True)
        return

    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = config.output_dir / "metrics.jsonl"
    forensics = FailureForensics(capacity=4096)
    algo.global_step = int(start_transitions)
    last_safe_step = config.resume_from if config.resume_from is not None else _save_checkpoint(algo, checkpoint_dir / "last_safe_step.pt")
    telemetry_config: dict[str, object] = {
        "total_timesteps": config.total_timesteps,
        "num_envs": config.num_envs,
        "seed": config.seed,
        "bc_checkpoint": str(config.bc_checkpoint),
    }
    wandb_run = OptionalWandbRun(
        project=config.wandb_project,
        name=config.wandb_run_name,
        output_dir=config.output_dir,
        config=telemetry_config,
    )
    tb_run = OptionalTensorBoardRun(config.tb_logdir)
    transitions = int(start_transitions)
    updates = int(getattr(algo, "_last_update_step", 0))
    next_checkpoint_at = ((transitions // max(config.checkpoint_every, 1)) + 1) * max(config.checkpoint_every, 1)
    start_time = time.monotonic()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32 - 1))
    value_loss_window: list[float] = []
    episode_return_std_window: list[float] = []
    explained_variance_window: list[float] = []
    max_window = 64
    try:
        with JsonlLogger(jsonl_path) as jsonl:
            while transitions < config.total_timesteps:
                remaining = config.total_timesteps - transitions
                rollout_steps = max(1, min(algo.config.context_length, math.ceil(remaining / config.num_envs)))
                buffer, stats = algo.collect_rollout(
                    cast(Any, env),
                    num_steps=rollout_steps,
                    seed=config.seed + transitions,
                    deterministic=False,
                    action_noise_std=0.01,
                    shaping_gamma=algo.gamma,
                )
                compute_gae_in_place(buffer, gamma=algo.gamma)
                projected_transitions = transitions + stats.steps
                report = algo.optimize_rollout(
                    buffer,
                    epochs=1,
                    batch_size=min(256, max(len(buffer), 1)),
                    watchdogs=registry,
                    forensics=forensics,
                    checkpoint_dir=checkpoint_dir,
                    last_safe_step_path=last_safe_step,
                    start_step=updates,
                    global_step=projected_transitions,
                    total_steps=config.total_timesteps,
                )
                transitions += stats.steps
                algo.global_step = transitions
                updates += report.updates
                safety_metrics = env.safety_metrics()
                h_hard_min = _metric_float(safety_metrics, "h_hard_min")
                value_loss = float(report.last_metrics.get("value_loss", float("nan")))
                total_loss = float(report.last_metrics.get("ppo_total_loss", value_loss))
                grad_norm = float(report.last_metrics.get("grad_norm", float("nan")))
                explained_variance = _finite_float(report.last_metrics.get("value_explained_variance"))
                value_target_mean = _finite_float(report.last_metrics.get("value_target_mean"))
                value_target_std = _finite_float(report.last_metrics.get("value_target_std"))
                value_pred_mean = _finite_float(report.last_metrics.get("value_pred_mean"))
                value_pred_std = _finite_float(report.last_metrics.get("value_pred_std"))
                value_loss_p50 = _finite_float(report.last_metrics.get("value_loss_p50"))
                value_loss_p99 = _finite_float(report.last_metrics.get("value_loss_p99"))
                approx_kl = _finite_float(report.last_metrics.get("approx_kl"))
                clip_fraction = _finite_float(report.last_metrics.get("clip_fraction"))
                actor_lr = _finite_float(report.last_metrics.get("actor_lr"))
                clip_epsilon = _finite_float(report.last_metrics.get("clip_epsilon"))
                post_transition_steps = _finite_float(report.last_metrics.get("post_transition_steps"))
                for window, value in (
                    (value_loss_window, value_loss),
                    (episode_return_std_window, stats.episode_return_std),
                    (explained_variance_window, explained_variance),
                ):
                    if math.isfinite(value):
                        window.append(value)
                        del window[:-max_window]
                value_loss_slope = _rolling_slope(value_loss_window) if len(value_loss_window) >= max_window else 0.0
                episode_return_zero_variance = float(
                    len(episode_return_std_window) >= 8 and _zero_variance(episode_return_std_window)
                )
                explained_variance_stuck_negative = float(
                    len(explained_variance_window) >= 8 and max(explained_variance_window[-8:]) < 0.0
                )
                phase_index = 0 if algo.phase.value == "critic_warmup" else 1
                critic_lr = _finite_float(report.last_metrics.get("critic_lr"))
                metrics: dict[str, object] = {
                    "phase": phase_index,
                    "phase_name": algo.phase.value,
                    "rollout/transitions": transitions,
                    "rollout/terminations": stats.terminations,
                    "rollout/successes": stats.successes,
                    "rollout/termination_success": stats.successes,
                    "rollout/collisions": stats.collisions,
                    "rollout/out_of_arena": stats.out_of_arena,
                    "rollout/cbf_arena_projection_active": stats.arena_projection_terminations,
                    "rollout/timeouts": stats.timeouts,
                    "rollout/cbf_active_rate": stats.cbf_active_rate,
                    "rollout/arena_projection_active_rate": stats.arena_projection_active_rate,
                    "rollout/episode_return_mean": stats.episode_return_mean,
                    "rollout/episode_return_std": stats.episode_return_std,
                    "rollout/episode_return_min": stats.episode_return_min,
                    "rollout/episode_return_max": stats.episode_return_max,
                    "rollout/episode_length_mean": stats.episode_length_mean,
                    "rollout/episode_length_std": stats.episode_length_std,
                    "rollout/episode_length_min": stats.episode_length_min,
                    "rollout/episode_length_max": stats.episode_length_max,
                    "rollout/h_hard_p01": stats.h_hard_p01,
                    "rollout/h_hard_p05": stats.h_hard_p05,
                    "rollout/h_hard_p50": stats.h_hard_p50,
                    "rollout/h_hard_p95": stats.h_hard_p95,
                    "rollout/h_hard_p99": stats.h_hard_p99,
                    "rollout/potential_phi_mean": stats.potential_phi_mean,
                    "rollout/potential_phi_std": stats.potential_phi_std,
                    "rollout/potential_delta_mean": stats.potential_delta_mean,
                    "rollout/potential_delta_std": stats.potential_delta_std,
                    "rollout/shaping_reward_share": stats.shaping_reward_share,
                    "rollout/shaping_reward_per_step": stats.shaping_reward_per_step,
                    "rollout/base_reward_mean": stats.base_reward_mean,
                    "rollout/cbf_penalty_mean": stats.cbf_penalty_mean,
                    "rollout/shaped_reward_mean": stats.shaped_reward_mean,
                    "rollout/state_distribution_entropy": stats.state_distribution_entropy,
                    "rollout/distance_to_goal_std": stats.distance_to_goal_std,
                    "rollout/distance_to_nearest_wall_mean": stats.distance_to_nearest_wall_mean,
                    "rollout/distance_to_nearest_wall_std": stats.distance_to_nearest_wall_std,
                    "rollout/cbf_arena_intervention_duration_mean": stats.cbf_arena_intervention_duration_mean,
                    "safety/h_hard_min": h_hard_min,
                    "train/value_loss": value_loss,
                    "train/value_explained_variance": explained_variance,
                    "train/value_target_mean": value_target_mean,
                    "train/value_target_std": value_target_std,
                    "train/value_pred_mean": value_pred_mean,
                    "train/value_pred_std": value_pred_std,
                    "train/value_loss_p50": value_loss_p50,
                    "train/value_loss_p99": value_loss_p99,
                    "train/value_loss_slope": value_loss_slope,
                    "train/total_loss": total_loss,
                    "train/grad_norm": grad_norm,
                    "train/updates": updates,
                    "train/critic_lr": critic_lr,
                    "train/lr_critic": critic_lr,
                    "train/approx_kl": approx_kl,
                    "train/clip_fraction": clip_fraction,
                    "train/actor_lr": actor_lr,
                    "train/lr_actor": actor_lr,
                    "train/clip_epsilon": clip_epsilon,
                    "train/post_transition_steps": post_transition_steps,
                    "train_loss": total_loss,
                    "grad_norm": grad_norm,
                    "val_overfit_ratio": 1.0,
                    "critic_warmup_steps": transitions if phase_index == 0 else 0,
                    "critic_warmup_active": phase_index,
                    "value_loss_slope": value_loss_slope,
                    "episode_return_zero_variance": episode_return_zero_variance,
                    "explained_variance_stuck_negative": explained_variance_stuck_negative,
                    "elapsed_sec": time.monotonic() - start_time,
                }
                forensics.push(step=transitions, metrics={k: v for k, v in metrics.items() if isinstance(v, int | float | bool | str)})
                registry.update(
                    {
                        "train_loss": total_loss,
                        "grad_norm": grad_norm,
                        "val_overfit_ratio": 1.0,
                        "phase": phase_index,
                        "phase_name": algo.phase.value,
                        "critic_warmup_steps": transitions if phase_index == 0 else 0,
                        "value_loss_slope": value_loss_slope,
                        "train/value_explained_variance": explained_variance,
                        "episode_return_zero_variance": episode_return_zero_variance,
                        "explained_variance_stuck_negative": explained_variance_stuck_negative,
                        "arena_projection_active_rate": stats.arena_projection_active_rate,
                        "state_distribution_entropy": stats.state_distribution_entropy,
                        "distance_to_goal_std": stats.distance_to_goal_std,
                        "distance_to_nearest_wall_mean": stats.distance_to_nearest_wall_mean,
                        "distance_to_nearest_wall_std": stats.distance_to_nearest_wall_std,
                        "cbf_arena_intervention_duration_mean": stats.cbf_arena_intervention_duration_mean,
                    },
                    step=transitions,
                )
                jsonl.write(metrics)
                wandb_run.log(metrics, step=transitions)
                tb_run.log(metrics, step=transitions)
                print(
                    "[V25 PPO] "
                    f"phase={phase_index} "
                    f"rollout/transitions={transitions} "
                    f"safety/h_hard_min={h_hard_min:.6g} "
                    f"train/value_loss={value_loss:.6g} "
                    f"train/value_explained_variance={explained_variance:.6g} "
                    f"train/approx_kl={approx_kl:.6g} "
                    f"train/clip_epsilon={clip_epsilon:.6g} "
                    f"train/actor_lr={actor_lr:.6g} "
                    f"train/post_transition_steps={post_transition_steps:.0f} "
                    f"rollout/episode_return_std={stats.episode_return_std:.6g} "
                    f"rollout/potential_delta_std={stats.potential_delta_std:.6g} "
                    f"rollout/shaping_reward_share={stats.shaping_reward_share:.6g} "
                    f"train/grad_norm={grad_norm:.6g}",
                    flush=True,
                )
                if report.halted or registry.should_halt():
                    reason = report.halt_reason or "watchdog halt"
                    print(f"[HALT][V25] reason={reason}", flush=True)
                    if report.halt_result is None:
                        algo._halt_protocol(
                            reason=reason,
                            step=transitions,
                            forensics=forensics,
                            checkpoint_dir=checkpoint_dir,
                            last_safe_step_path=last_safe_step,
                        )
                    return
                if transitions >= next_checkpoint_at:
                    _save_checkpoint(algo, checkpoint_dir / f"step_{transitions:09d}.pt")
                    last_safe_step = _save_checkpoint(algo, checkpoint_dir / "last_safe_step.pt")
                    next_checkpoint_at += max(config.checkpoint_every, 1)
    finally:
        wandb_run.finish()
        tb_run.close()
    _save_checkpoint(algo, checkpoint_dir / "final.pt")
    print("[DONE][V25] training complete", flush=True)


def main() -> None:
    _prefer_dev_package()
    config = parse_args()
    if config.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if config.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.tb_logdir.mkdir(parents=True, exist_ok=True)
    pre_registration = _load_json_object(config.pre_registration)
    bc_sha256 = _sha256_file(config.bc_checkpoint)
    manifest_sha256 = _sha256_file(config.pre_registration)
    if config.strict_pre_registration:
        _validate_pre_registration(pre_registration, config, bc_sha256=bc_sha256)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"unexpected pre-registration SHA-256: {manifest_sha256}")
    registry = _build_watchdogs(config.watchdog_config)

    print(f"output_dir={config.output_dir}", flush=True)
    print(f"tb_logdir={config.tb_logdir}", flush=True)
    print(f"pre_registration_manifest_sha256={manifest_sha256}", flush=True)
    print(f"bc_checkpoint_sha256={bc_sha256}", flush=True)
    print(f"bc_dataset_sha256={config.bc_dataset_sha256}", flush=True)
    if config.resume_from is not None:
        print(f"resume_from={config.resume_from}", flush=True)
    _print_watchdogs(registry)
    print("[INFO][V25] phase=0 (CRITIC_WARMUP) entered cleanly", flush=True)

    if config.resume_from is None:
        algo = CausalTransformerActorCritic.from_bc_checkpoint(config.bc_checkpoint, device=config.device)
        start_transitions = 0
    else:
        algo = CausalTransformerActorCritic.from_checkpoint(config.resume_from, device=config.device)
        _warm_start_gate_v2_from_metrics(algo, config.resume_from)
        start_transitions = _infer_resume_transitions(config.resume_from, algo)
        start_transitions = max(start_transitions, int(algo.global_step))
        algo.global_step = start_transitions
        if _checkpoint_has_optimizer_state(config.resume_from):
            print("[RESUME][V25] restored optimizer_state_dict=True", flush=True)
        else:
            print(
                "[RESUME][V25] restored optimizer_state_dict=False "
                "reason=legacy_checkpoint_missing_optimizer_state",
                flush=True,
            )
        print(
            f"[RESUME][V25] start_transitions={start_transitions} "
            f"phase={algo.phase.value} global_step={algo.global_step}",
            flush=True,
        )
    actor_parameters = sum(parameter.numel() for parameter in algo.actor.parameters())
    print(f"loaded_actor_parameters={actor_parameters}", flush=True)
    if actor_parameters != EXPECTED_ACTOR_PARAMETERS:
        raise ValueError(f"actor parameter count mismatch: {actor_parameters} != {EXPECTED_ACTOR_PARAMETERS}")
    env = _make_env(config.num_envs)
    _run_training(config, algo, env, registry, start_transitions=start_transitions)


if __name__ == "__main__":
    main()
