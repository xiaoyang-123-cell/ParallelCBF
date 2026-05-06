#!/usr/bin/env python3
"""Post-burn critic calibration probe for V26 diagnostic PPO runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEV_ROOT = Path("/home/smartlab/parallelcbf_dev")
CONDA_EXE = Path("/home/smartlab/miniforge3/bin/conda")


def _prefer_dev_package() -> None:
    dev = str(DEV_ROOT)
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != DEV_ROOT]
    sys.path.insert(0, dev)


def _running_in_parallel_env() -> bool:
    executable = Path(sys.executable).as_posix()
    return "envs/parallel_uav/bin/python" in executable or os.environ.get("CONDA_DEFAULT_ENV") == "parallel_uav"


def _bootstrap_parallel_env() -> None:
    if os.environ.get("V26_CRITIC_EVAL_BOOTSTRAPPED") == "1" or _running_in_parallel_env():
        return
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [str(DEV_ROOT)]
    if current_pythonpath:
        pythonpath_parts.append(current_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    os.environ["V26_CRITIC_EVAL_BOOTSTRAPPED"] = "1"
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
            "scripts.v26_critic_return_eval",
            *sys.argv[1:],
        ],
    )


_bootstrap_parallel_env()
_prefer_dev_package()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from parallelcbf.algorithms import CausalTransformerActorCritic, RolloutBuffer, RolloutStep  # noqa: E402
from parallelcbf.api import MetricDict, SafeEnv, SafetyFilterResult, SafetyState  # noqa: E402
from parallelcbf.envs import Toy2DAvoidanceVecEnv, Toy2DConfig  # noqa: E402
from parallelcbf.safety import DualBarrierCBF  # noqa: E402


@dataclass(frozen=True, slots=True)
class CriticEvalConfig:
    checkpoint: Path
    output_json: Path
    output_scatter_json: Path
    output_scatter_csv: Path
    num_steps: int
    num_envs: int
    device: str
    seed: int
    gamma: float


class VectorSafetyWrapper:
    """Standalone Layer-3 adapter matching the diagnostic burn environment."""

    metadata: dict[str, Any] = {"render_modes": []}
    reward_range: tuple[float, float] = (-float("inf"), float("inf"))
    spec: Any = None

    def __init__(self, env: Toy2DAvoidanceVecEnv, safety_filter: DualBarrierCBF) -> None:
        self.env = env
        self.safety_filter = safety_filter
        obs_high = np.full((8,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._last_observation: np.ndarray | None = None
        self._last_filter_result: SafetyFilterResult[torch.Tensor] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        _ = options
        self.safety_filter.reset(seed=seed)
        self._last_filter_result = None
        observation, info = self.env.reset(seed=seed)
        self._last_observation = observation
        return observation, dict(info)

    def step(
        self,
        action: torch.Tensor | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
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
        return SafetyState(
            position=pos,
            velocity=vel,
            goal=goal,
            obstacles=obstacle,
            robot_radius=float(self.env.config.robot_radius),
            obstacle_radius=float(self.env.config.obstacle_radius),
            arena_bounds=np.array([-radius, radius, -radius, radius], dtype=np.float32),
            metadata={},
        )

    def safety_metrics(self) -> MetricDict:
        metrics: MetricDict = {}
        env_metrics = self.env.safety_metrics()
        for key, value in env_metrics.items():
            array = np.asarray(value)
            if array.dtype == np.bool_:
                metrics[key] = bool(np.any(array))
            else:
                metrics[key] = float(np.min(array))
        metrics.update(self.safety_filter.metrics())
        if self._last_filter_result is not None:
            metrics.update(self._last_filter_result.metrics)
        return metrics

    def hard_constraint_violations(self) -> dict[str, bool]:
        hard = self.env.safety_metrics()["h_hard_violation"]
        return {"collision_or_oob": bool(np.any(hard))}

    def close(self) -> None:
        return None


def _parse_args() -> CriticEvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-scatter-json", type=Path, required=True)
    parser.add_argument("--output-scatter-csv", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    return CriticEvalConfig(
        checkpoint=_resolve(args.checkpoint),
        output_json=_resolve(args.output_json),
        output_scatter_json=_resolve(args.output_scatter_json),
        output_scatter_csv=_resolve(args.output_scatter_csv),
        num_steps=int(args.num_steps),
        num_envs=int(args.num_envs),
        device=str(args.device),
        seed=int(args.seed),
        gamma=float(args.gamma),
    )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _make_env(num_envs: int) -> VectorSafetyWrapper:
    return VectorSafetyWrapper(Toy2DAvoidanceVecEnv(num_envs=num_envs, config=Toy2DConfig(max_steps=500)), DualBarrierCBF())


def _actual_returns(buffer: RolloutBuffer, *, gamma: float) -> dict[int, float]:
    returns_by_index: dict[int, float] = {}
    step_to_index = {id(step): index for index, step in enumerate(buffer.steps)}
    for _, episode in buffer.iter_episodes():
        running = 0.0
        for step in reversed(episode):
            running = step.shaped_reward + gamma * running
            returns_by_index[step_to_index[id(step)]] = running
    return returns_by_index


def _safe_corrcoef(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if float(np.std(x_array)) <= 1.0e-12 or float(np.std(y_array)) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _termination_histogram(steps: Sequence[RolloutStep]) -> dict[str, int]:
    reasons = {
        "success": 0,
        "collision": 0,
        "out_of_arena": 0,
        "timeout": 0,
        "ongoing": 0,
    }
    for step in steps:
        reasons[step.termination_reason] += 1
    return reasons


def _write_scatter_csv(path: Path, predicted: Sequence[float], actual: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("predicted_return,actual_return\n")
        for pred, ret in zip(predicted, actual, strict=True):
            handle.write(f"{pred:.12g},{ret:.12g}\n")


def main() -> None:
    config = _parse_args()
    if config.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if config.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed % (2**32 - 1))
    model = CausalTransformerActorCritic.from_checkpoint(config.checkpoint, device=config.device)
    model.eval()
    env = _make_env(config.num_envs)
    buffer, stats = model.collect_rollout(
        cast(SafeEnv[torch.Tensor, torch.Tensor], env),
        num_steps=config.num_steps,
        seed=config.seed,
        deterministic=True,
    )
    realized = _actual_returns(buffer, gamma=config.gamma)
    predicted_returns = [float(step.value) for step in buffer.steps]
    actual_returns = [realized[index] for index in range(len(buffer.steps))]
    errors = np.asarray(predicted_returns, dtype=np.float64) - np.asarray(actual_returns, dtype=np.float64)

    scatter = [
        {"predicted_return": pred, "actual_return": ret}
        for pred, ret in zip(predicted_returns, actual_returns, strict=True)
    ]
    summary: dict[str, object] = {
        "checkpoint": str(config.checkpoint),
        "seed": config.seed,
        "num_envs": config.num_envs,
        "num_steps": config.num_steps,
        "transitions": len(buffer),
        "gamma": config.gamma,
        "predicted_mean": float(np.mean(predicted_returns)),
        "predicted_std": float(np.std(predicted_returns)),
        "actual_mean": float(np.mean(actual_returns)),
        "actual_std": float(np.std(actual_returns)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "correlation": _safe_corrcoef(predicted_returns, actual_returns),
        "termination_histogram": _termination_histogram(buffer.steps),
        "rollout_stats": {
            "terminations": stats.terminations,
            "successes": stats.successes,
            "collisions": stats.collisions,
            "out_of_arena": stats.out_of_arena,
            "timeouts": stats.timeouts,
            "episode_return_mean": stats.episode_return_mean,
            "episode_return_std": stats.episode_return_std,
            "episode_length_mean": stats.episode_length_mean,
            "h_hard_p01": stats.h_hard_p01,
            "h_hard_p50": stats.h_hard_p50,
            "h_hard_p99": stats.h_hard_p99,
        },
        "scatter_json": str(config.output_scatter_json),
        "scatter_csv": str(config.output_scatter_csv),
    }

    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config.output_scatter_json.parent.mkdir(parents=True, exist_ok=True)
    config.output_scatter_json.write_text(json.dumps(scatter, separators=(",", ":")) + "\n", encoding="utf-8")
    _write_scatter_csv(config.output_scatter_csv, predicted_returns, actual_returns)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
