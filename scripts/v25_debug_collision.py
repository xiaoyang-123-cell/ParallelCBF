#!/usr/bin/env python3
"""Forensic probe for the V25 Stage 0 collision episode."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
import os
import pickle
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch

from parallelcbf.algorithms import CausalTransformerBC, CausalTransformerConfig
from parallelcbf.api import MetricDict, SafeEnv, SafetyState, SafetyWrapper
from parallelcbf.api.types import ConstraintViolations
from parallelcbf.envs import Toy2DAvoidanceVecEnv
from parallelcbf.envs.toy2d import Toy2DConfig
from parallelcbf.safety import DualBarrierCBF


ROOT = Path("/data/uav_project/ParallelCBF")
CHECKPOINT = ROOT / "checkpoints/v25_transformer_bc_full/best.pt"
OUTPUT = ROOT / "logs/v25_debug_collision.json"


class _TorchLoadUnpickler(pickle.Unpickler):
    def __init__(self, file: io.BufferedReader, *, map_location: torch.device) -> None:
        super().__init__(file)
        self._map_location = map_location

    def find_class(self, module: str, name: str) -> object:
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(io.BytesIO(data), map_location=self._map_location, weights_only=False)
        return super().find_class(module, name)


class Stage0VecEnv(SafeEnv[torch.Tensor, torch.Tensor]):
    metadata: dict[str, Any] = {"render_modes": []}
    reward_range: tuple[float, float] = (-float("inf"), float("inf"))
    spec: Any = None

    def __init__(self, *, num_envs: int, config: Toy2DConfig) -> None:
        self.base_env = Toy2DAvoidanceVecEnv(num_envs=num_envs, config=config)
        self.config = self.base_env.config
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-self.config.action_limit, high=self.config.action_limit, shape=(2,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        _ = options
        observation, info = self.base_env.reset(seed=seed)
        return torch.as_tensor(observation, dtype=torch.float32), info

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        observation, rewards, terminated, truncated, info = self.base_env.step(action.detach().cpu().numpy().astype(np.float32))
        return (
            torch.as_tensor(observation, dtype=torch.float32),
            float(rewards[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            info,
        )

    def safety_state(self) -> SafetyState:
        return SafetyState(
            position=torch.as_tensor(self.base_env._pos, dtype=torch.float32),
            velocity=torch.as_tensor(self.base_env._vel, dtype=torch.float32),
            goal=torch.as_tensor(self.base_env._goal, dtype=torch.float32),
            obstacles=torch.as_tensor(self.base_env._obstacle[:, None, :], dtype=torch.float32),
            robot_radius=float(self.config.robot_radius),
            obstacle_radius=float(self.config.obstacle_radius),
            arena_bounds=torch.as_tensor(
                [
                    -self.config.arena_radius,
                    self.config.arena_radius,
                    -self.config.arena_radius,
                    self.config.arena_radius,
                ],
                dtype=torch.float32,
            ),
            metadata={},
        )

    def hard_constraint_violations(self) -> ConstraintViolations:
        return {"collision_or_oob": bool(self.base_env._last_hard_violation[0])}

    def safety_metrics(self) -> MetricDict:
        raw_metrics = self.base_env.safety_metrics()
        return {
            "obstacle_clearance": float(raw_metrics["obstacle_clearance"][0]),
            "arena_clearance": float(raw_metrics["arena_clearance"][0]),
            "h_hard_violation": bool(raw_metrics["h_hard_violation"][0]),
        }

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class StepRecord:
    step: int
    position: list[float]
    velocity: list[float]
    obstacle: list[float]
    nominal_action_shape: list[int]
    safe_action_shape: list[int]
    rel_pos_shape: list[int]
    nominal_action: list[float]
    safe_action: list[float]
    nominal_all_finite: bool
    safe_all_finite: bool
    rel_pos_all_finite: bool
    h_hard_before_min: float
    h_hard_next_min: float
    h_soft_next_min: float
    modified: bool
    filter_skipped: bool
    collision: bool
    terminated: bool
    truncated: bool
    final_distance: float


class ForensicSafetyWrapper(SafetyWrapper[torch.Tensor, torch.Tensor]):
    """SafetyWrapper variant that records CBF inputs/outputs every step."""

    def __init__(self, env: Stage0VecEnv, safety_filter: DualBarrierCBF) -> None:
        super().__init__(env, safety_filter)
        self.records: list[StepRecord] = []

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        if self._last_observation is None:
            raise RuntimeError("ForensicSafetyWrapper.step() called before reset().")
        env = cast(Stage0VecEnv, self.env)
        safety_filter = cast(DualBarrierCBF, self.safety_filter)
        safety_state = env.safety_state()
        nominal_action = action.detach().cpu()
        position = torch.as_tensor(safety_state.position, dtype=nominal_action.dtype)
        velocity = torch.as_tensor(safety_state.velocity, dtype=nominal_action.dtype)
        obstacles = torch.as_tensor(safety_state.obstacles, dtype=nominal_action.dtype)
        radii = torch.full(
            (nominal_action.shape[0],),
            float(safety_state.robot_radius + safety_state.obstacle_radius),
            dtype=nominal_action.dtype,
        )
        rel_pos = position.unsqueeze(1) - obstacles
        h_hard_before = safety_filter.h_hard(position, obstacles, radii)
        filter_skipped = False
        try:
            filter_result = safety_filter.filter_action(self._last_observation, nominal_action, safety_state)
        except Exception:
            filter_skipped = True
            raise
        safe_action = filter_result.safe_action.detach().cpu()
        next_position = safety_filter.next_position(position, velocity, safe_action)
        h_hard_next = safety_filter.h_hard(next_position, obstacles, radii)
        h_soft_next = safety_filter.h_soft(next_position, obstacles, radii)
        next_observation, reward, terminated, truncated, info = env.step(safe_action)
        self._last_filter_result = filter_result
        self._last_observation = next_observation
        hard = info["hard_constraint_violations"]["collision_or_oob"]
        collision = bool(np.asarray(hard).reshape(-1)[0])
        final_distance = float(np.linalg.norm(env.base_env._goal[0] - env.base_env._pos[0]))
        self.records.append(
            StepRecord(
                step=len(self.records) + 1,
                position=[float(value) for value in position.reshape(-1).tolist()],
                velocity=[float(value) for value in velocity.reshape(-1).tolist()],
                obstacle=[float(value) for value in obstacles.reshape(-1).tolist()],
                nominal_action_shape=list(nominal_action.shape),
                safe_action_shape=list(safe_action.shape),
                rel_pos_shape=list(rel_pos.shape),
                nominal_action=[float(value) for value in nominal_action.reshape(-1).tolist()],
                safe_action=[float(value) for value in safe_action.reshape(-1).tolist()],
                nominal_all_finite=bool(torch.isfinite(nominal_action).all().item()),
                safe_all_finite=bool(torch.isfinite(safe_action).all().item()),
                rel_pos_all_finite=bool(torch.isfinite(rel_pos).all().item()),
                h_hard_before_min=float(torch.min(h_hard_before).item()),
                h_hard_next_min=float(torch.min(h_hard_next).item()),
                h_soft_next_min=float(torch.min(h_soft_next).item()),
                modified=bool(filter_result.modified),
                filter_skipped=filter_skipped,
                collision=collision,
                terminated=bool(terminated),
                truncated=bool(truncated),
                final_distance=final_distance,
            )
        )
        wrapped_info = dict(info)
        wrapped_info["safety_filter"] = filter_result
        return next_observation, float(reward), bool(terminated), bool(truncated), wrapped_info


def load_algorithm(device: torch.device) -> CausalTransformerBC:
    with CHECKPOINT.open("rb") as handle:
        payload = _TorchLoadUnpickler(handle, map_location=device).load()
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    config = payload.get("config")
    if not isinstance(config, CausalTransformerConfig):
        raise TypeError("checkpoint config must be CausalTransformerConfig")
    algorithm = CausalTransformerBC(config, device=device)
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint missing model_state_dict")
    algorithm.model.load_state_dict(cast(dict[str, torch.Tensor], state_dict))
    algorithm.model.eval()
    return algorithm


def policy_observation(toy_observation: np.ndarray[Any, np.dtype[np.float32]], obs_dim: int) -> torch.Tensor:
    padded = np.zeros((toy_observation.shape[0], obs_dim), dtype=np.float32)
    copy_dim = min(toy_observation.shape[1], obs_dim)
    padded[:, :copy_dim] = toy_observation[:, :copy_dim]
    return torch.as_tensor(padded, dtype=torch.float32)


def first_step(records: list[StepRecord], predicate_name: str) -> int | None:
    for record in records:
        if predicate_name == "filter_skip" and record.filter_skipped:
            return record.step
        if predicate_name == "nan_nominal" and not record.nominal_all_finite:
            return record.step
        if predicate_name == "nan_safe" and not record.safe_all_finite:
            return record.step
    return None


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algorithm = load_algorithm(device)
    env = Stage0VecEnv(num_envs=1, config=Toy2DConfig(max_steps=500))
    wrapper = ForensicSafetyWrapper(env, DualBarrierCBF())
    observation, _ = wrapper.reset(seed=42)
    env.base_env._obstacle[:, :] = np.array([100.0, 100.0], dtype=np.float32)
    toy_observation = np.asarray(observation, dtype=np.float32)
    hidden = algorithm.initial_hidden_state(batch_size=1, dtype=torch.float32, device=algorithm.device)
    crash_record: StepRecord | None = None
    for _ in range(500):
        model_obs = policy_observation(toy_observation, algorithm.config.obs_dim).to(algorithm.device)
        with torch.no_grad():
            prediction = algorithm.predict(model_obs, hidden, deterministic=True)
        hidden = prediction.hidden_state
        nominal_action = prediction.action[:, :2].detach().cpu()
        next_observation, _, terminated, truncated, _ = wrapper.step(nominal_action)
        toy_observation = np.asarray(next_observation, dtype=np.float32)
        latest = wrapper.records[-1]
        if latest.collision:
            crash_record = latest
            break
        if terminated or truncated:
            break
    records = wrapper.records
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CHECKPOINT),
        "seed": 42,
        "scene": "open",
        "steps_recorded": len(records),
        "collision_step": crash_record.step if crash_record is not None else None,
        "first_filter_skip_step": first_step(records, "filter_skip"),
        "first_nan_in_nominal_step": first_step(records, "nan_nominal"),
        "first_nan_in_safe_step": first_step(records, "nan_safe"),
        "crash_record": asdict(crash_record) if crash_record is not None else None,
        "last_record": asdict(records[-1]) if records else None,
        "records": [asdict(record) for record in records],
    }
    atomic_json_dump(summary, OUTPUT)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
