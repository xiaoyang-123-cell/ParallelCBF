#!/usr/bin/env python3
"""Final CPU-only V25/V0.2 evaluation with explicit audit artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from collections.abc import Mapping
from typing import Any, Literal, Sequence, cast


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEV_ROOT = Path("/home/smartlab/parallelcbf_dev")
for path in (str(PROJECT_ROOT), str(DEV_ROOT)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from parallelcbf.algorithms import CausalTransformerActorCritic  # noqa: E402
from parallelcbf.api import SafetyFilterResult, SafetyState  # noqa: E402
from parallelcbf.envs.toy2d import Toy2DConfig  # noqa: E402
from parallelcbf.safety import TripleBarrierCBF  # noqa: E402


AMENDMENT_5_SHA = "bec97d2e2672588a135b4df09b74386bf4596b44ff9c854c258765c2cf91b0e5"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs/v02_phase_aware_resume_20260503T083420Z/checkpoints/final.pt"
DETERMINISTIC_EPISODES = 500
STOCHASTIC_EPISODES = 100
MAX_STEPS = 500
SCENE_NAMES = ("open", "single_static", "offset_single_static", "near_goal_open", "wide_single_static")
TERMINATION_CLASSES = ("layer2_violation", "collision", "out_of_arena", "success", "timeout")
STAGE0_V01_SINGLE_STATIC = {
    "episodes": 125,
    "total_success": 0,
    "total_collision": 0,
    "total_out_of_arena": 125,
    "total_timeout": 0,
}
SUCCESS_RADIUS = float(Toy2DConfig().goal_radius)
FAR_OBSTACLE_THRESHOLD_M = np.float32(50.0)
TOL_LAYER2_HALT = -1.0e-5
SCENE_ARENA_MARGIN = 0.05
BARRIER_METRIC_KEYS = {
    "h_hard": "h_hard_min",
    "h_soft": "h_soft_min",
    "h_arena": "h_arena_min",
    "h_arena_stopping": "h_arena_stopping_min",
}


TerminationReason = Literal[
    "layer2_violation",
    "collision",
    "out_of_arena",
    "success",
    "timeout",
]


@dataclass(frozen=True, slots=True)
class SceneSpec:
    scene: str
    start: tuple[float, float]
    goal: tuple[float, float]
    obstacle: tuple[float, float]


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode_id: int
    paired_id: int
    mode: str
    scene: str
    env_seed: int
    action_seed: int | None
    success: bool
    termination_reason: str
    steps: int
    final_distance: float
    min_h_hard: float
    min_h_soft: float
    min_h_arena: float
    min_h_arena_stopping: float
    cbf_active_steps: int
    arena_projection_active_steps: int
    total_reward: float
    trajectory_file: str | None


@dataclass(frozen=True, slots=True)
class Layer2ResidualEvent:
    episode_id: int
    step: int
    barrier: str
    metric: str
    value: float
    severity: Literal["numerical", "real"]
    tolerance_threshold: float
    scene: str
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--deterministic-episodes", type=int, default=DETERMINISTIC_EPISODES)
    parser.add_argument("--stochastic-episodes", type=int, default=STOCHASTIC_EPISODES)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    return parser.parse_args()


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    np.random.seed(0)
    torch.manual_seed(0)
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or PROJECT_ROOT / f"runs/v25_eval_final_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    autopsy_dir = output_dir / "autopsy"
    autopsy_dir.mkdir(parents=True, exist_ok=True)
    residuals_path = output_dir / "numerical_residuals.jsonl"

    config = Toy2DConfig(max_steps=int(args.max_steps))
    manifest = build_manifest(args=args, output_dir=output_dir, config=config)
    write_json(output_dir / "manifest.json", manifest)
    model = CausalTransformerActorCritic.from_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    model.to("cpu")

    records: list[EpisodeRecord] = []
    success_saved = 0
    fail_saved = 0
    value_samples: list[dict[str, Any]] = []
    residual_events: list[Layer2ResidualEvent] = []
    per_episode_path = output_dir / "per_episode.jsonl"
    with per_episode_path.open("w", encoding="utf-8") as handle, residuals_path.open("w", encoding="utf-8") as residual_handle:
        for episode_id, mode, paired_id, spec, env_seed, stochastic in episode_schedule(
            int(args.deterministic_episodes),
            int(args.stochastic_episodes),
            config=config,
        ):
            save_trajectory = False
            record, trajectory, samples, episode_residuals = run_episode(
                model=model,
                spec=spec,
                config=config,
                episode_id=episode_id,
                mode=mode,
                paired_id=paired_id,
                env_seed=env_seed,
                stochastic=stochastic,
                autopsy_dir=autopsy_dir,
                residual_handle=residual_handle,
            )
            if record.success and success_saved < 10:
                save_trajectory = True
                success_saved += 1
            if not record.success and fail_saved < 10:
                save_trajectory = True
                fail_saved += 1
            if save_trajectory:
                traj_name = f"{mode}_{episode_id:04d}_{record.termination_reason}.npz"
                np.savez_compressed(trajectories_dir / traj_name, **cast(Any, trajectory))
                record = EpisodeRecord(**{**asdict(record), "trajectory_file": f"trajectories/{traj_name}"})
            records.append(record)
            residual_events.extend(episode_residuals)
            value_samples.extend(samples)
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            handle.flush()

    aggregate = summarize(records, residual_events=residual_events)
    aggregate["artifacts"] = {
        "manifest": "manifest.json",
        "per_episode": "per_episode.jsonl",
        "trajectories_dir": "trajectories",
        "value_calibration": "value_calibration.json",
        "stage0_comparison": "stage0_comparison.json",
    }
    write_json(output_dir / "aggregate.json", aggregate)
    write_json(output_dir / "stage0_comparison.json", stage0_comparison(records))
    write_json(output_dir / "value_calibration.json", value_calibration(value_samples, records))
    print(json.dumps({"output_dir": str(output_dir), "aggregate": aggregate}, indent=2, sort_keys=True))


def build_manifest(*, args: argparse.Namespace, output_dir: Path, config: Toy2DConfig) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "checkpoint": str(args.checkpoint),
        "amendment_5_sha": AMENDMENT_5_SHA,
        "cpu_only": True,
        "torch_num_threads": 1,
        "torch_deterministic_algorithms": True,
        "episodes": {
            "deterministic": int(args.deterministic_episodes),
            "stochastic": int(args.stochastic_episodes),
            "total": int(args.deterministic_episodes) + int(args.stochastic_episodes),
        },
        "r_pins": {
            "R-1": "hidden state reset to model.initial_hidden_state(batch=1) at each episode start",
            "R-2": {"success_radius": SUCCESS_RADIUS, "source": "Toy2DConfig.goal_radius"},
            "R-3": "stochastic action seed = env_seed + 10000",
            "R-4": list(TERMINATION_CLASSES),
            "R-5": "stochastic seeds 500-599 pair to deterministic scene specs by seed % 500",
            "R-6": "torch.set_num_threads(1), CPU-only, torch.use_deterministic_algorithms(True)",
            "R-7v2": {
                "pass": "h_min >= 0",
                "numerical": f"{TOL_LAYER2_HALT} <= h_min < 0, log numerical_residuals.jsonl without halt",
                "real": f"h_min < {TOL_LAYER2_HALT}, write autopsy and exit 2",
            },
        },
        "optional_artifacts": {
            "O-1": "125 single-static episodes compared against v0.1 Stage-0 0/0/125/0 metrics",
            "O-2": "value calibration samples every 10th transition",
        },
        "env": asdict(config),
        "scene_names": list(SCENE_NAMES),
        "layer2_tolerance_threshold": TOL_LAYER2_HALT,
        "scene_arena_precondition": {
            "validator": "_validate_scene_within_arena",
            "margin": SCENE_ARENA_MARGIN,
            "arena_bounds_source": "inscribed_square",
            "wide_scene_policy": "shrink start/goal coordinates into the Layer-2 envelope before evaluation",
        },
        "barrier_metric_keys": dict(BARRIER_METRIC_KEYS),
    }


def episode_schedule(
    deterministic_episodes: int,
    stochastic_episodes: int,
    *,
    config: Toy2DConfig | None = None,
) -> list[tuple[int, str, int, SceneSpec, int, bool]]:
    if deterministic_episodes != 500:
        raise ValueError("final evaluation pins deterministic episodes to 500")
    if stochastic_episodes != 100:
        raise ValueError("final evaluation pins stochastic episodes to 100")
    schedule: list[tuple[int, str, int, SceneSpec, int, bool]] = []
    active_config = config or Toy2DConfig()
    for index in range(deterministic_episodes):
        spec = _validate_scene_within_arena(deterministic_scene(index), active_config, margin=SCENE_ARENA_MARGIN)
        schedule.append((index, "deterministic", index, spec, index, False))
    for offset, env_seed in enumerate(range(500, 600)):
        paired_id = env_seed % deterministic_episodes
        spec = _validate_scene_within_arena(
            deterministic_scene(paired_id),
            active_config,
            margin=SCENE_ARENA_MARGIN,
        )
        schedule.append((deterministic_episodes + offset, "stochastic", paired_id, spec, env_seed, True))
    return schedule


def deterministic_scene(index: int) -> SceneSpec:
    if index < 125:
        return SceneSpec("open", (-2.0, 0.0), (2.0, 0.0), (100.0, 100.0))
    if index < 250:
        return SceneSpec("single_static", (-2.0, 0.0), (2.0, 0.0), (0.4, 0.25))
    if index < 375:
        lane = (index - 250) % 5
        y = -0.4 + 0.2 * lane
        return SceneSpec("offset_single_static", (-2.0, y), (2.0, -y), (0.35, 0.35))
    if index < 450:
        return SceneSpec("near_goal_open", (-1.2, 0.0), (1.6, 0.0), (100.0, 100.0))
    lane = (index - 450) % 5
    y = -0.5 + 0.25 * lane
    return SceneSpec("wide_single_static", (-2.2, y), (2.1, -0.5 * y), (0.7, 0.2))


def _validate_scene_within_arena(spec: SceneSpec, config: Toy2DConfig, *, margin: float = SCENE_ARENA_MARGIN) -> SceneSpec:
    """Ensure eval starts/goals satisfy the TripleBarrier square-arena precondition."""

    if margin < 0.0:
        raise ValueError("scene arena margin must be nonnegative")
    half_extent = float(config.arena_radius) / math.sqrt(2.0)
    allowed = half_extent - float(margin)
    if allowed <= 0.0:
        raise ValueError("scene arena margin leaves no valid arena interior")
    start = _point_within_arena(spec.start, allowed)
    goal = _point_within_arena(spec.goal, allowed)
    if start == spec.start and goal == spec.goal:
        return spec
    if spec.scene.startswith("wide_"):
        return SceneSpec(spec.scene, start, goal, spec.obstacle)
    raise ValueError(
        f"scene {spec.scene!r} violates Layer-2 arena precondition with margin={margin}: "
        f"start={spec.start}, goal={spec.goal}, allowed_abs={allowed:.6g}"
    )


def _point_within_arena(point: tuple[float, float], allowed_abs: float) -> tuple[float, float]:
    x, y = point
    clamped_x = min(max(float(x), -allowed_abs), allowed_abs)
    clamped_y = min(max(float(y), -allowed_abs), allowed_abs)
    return (clamped_x, clamped_y)


def run_episode(
    *,
    model: CausalTransformerActorCritic,
    spec: SceneSpec,
    config: Toy2DConfig,
    episode_id: int,
    mode: str,
    paired_id: int,
    env_seed: int,
    stochastic: bool,
    autopsy_dir: Path,
    residual_handle: Any,
) -> tuple[EpisodeRecord, dict[str, np.ndarray[Any, Any]], list[dict[str, Any]], list[Layer2ResidualEvent]]:
    env = FinalToy2D(config=config, spec=spec)
    safety_filter = TripleBarrierCBF()
    observation = env.reset(seed=env_seed)
    hidden = model.initial_hidden_state(1, dtype=torch.float32, device=torch.device("cpu"))
    generator = None
    action_seed = None
    if stochastic:
        action_seed = env_seed + 10000
        generator = torch.Generator(device="cpu").manual_seed(action_seed)
    positions: list[np.ndarray[Any, Any]] = [env.pos.copy()]
    velocities: list[np.ndarray[Any, Any]] = [env.vel.copy()]
    nominal_actions: list[np.ndarray[Any, Any]] = []
    safe_actions: list[np.ndarray[Any, Any]] = []
    rewards: list[float] = []
    h_hard_values: list[float] = []
    h_soft_values: list[float] = []
    h_arena_values: list[float] = []
    h_arena_stopping_values: list[float] = []
    value_samples: list[dict[str, Any]] = []
    residual_events: list[Layer2ResidualEvent] = []
    total_reward = 0.0
    cbf_active_steps = 0
    arena_active_steps = 0
    termination_reason: TerminationReason = "timeout"
    final_distance = float(np.linalg.norm(env.goal - env.pos))

    for step in range(1, config.max_steps + 1):
        obs_tensor = policy_observation(observation, model.config.obs_dim)
        with torch.no_grad():
            output = model.act_and_value(obs_tensor, hidden, deterministic=not stochastic)
        hidden = output.hidden_state
        nominal = output.action[:, :2].detach().cpu()
        if stochastic:
            log_std = model.actor_log_std.detach().cpu()[:2].clamp(-5.0, 2.0)
            noise = torch.randn((1, 2), generator=generator) * torch.exp(log_std).unsqueeze(0)
            nominal = nominal + noise
        filter_result = safety_filter.filter_action(
            torch.as_tensor(observation.reshape(1, -1), dtype=torch.float32),
            nominal,
            safety_state(env),
        )
        safe = filter_result.safe_action.detach().cpu()
        metrics = filter_result.metrics
        h_hard = float(metrics.get("h_hard_min", math.nan))
        h_soft = float(metrics.get("h_soft_min", math.nan))
        h_arena = float(metrics.get("h_arena_min", math.nan))
        h_arena_stopping = float(metrics.get("h_arena_stopping_min", math.nan))
        step_residuals = classify_layer2_residuals(
            metrics,
            episode_id=episode_id,
            step=step,
            scene=spec.scene,
            mode=mode,
        )
        for event in step_residuals:
            residual_handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
            residual_handle.flush()
        residual_events.extend(step_residuals)
        if any(event.severity == "real" for event in step_residuals):
            layer2_violation_halt(
                autopsy_dir=autopsy_dir,
                episode_id=episode_id,
                step=step,
                spec=spec,
                observation=observation,
                nominal=nominal.numpy(),
                safe=safe.numpy(),
                metrics=cast(dict[str, object], dict(metrics)),
                positions=positions,
            )
        next_observation, reward, reason, done = env.step(safe.squeeze(0).numpy())
        modified = bool(metrics.get("modified", False))
        arena_active = bool(metrics.get("arena_projection_active", False))
        cbf_active_steps += int(modified)
        arena_active_steps += int(arena_active)
        nominal_actions.append(nominal.squeeze(0).numpy().astype(np.float32))
        safe_actions.append(safe.squeeze(0).numpy().astype(np.float32))
        rewards.append(float(reward))
        h_hard_values.append(h_hard)
        h_soft_values.append(h_soft)
        h_arena_values.append(h_arena)
        h_arena_stopping_values.append(h_arena_stopping)
        total_reward += float(reward)
        positions.append(env.pos.copy())
        velocities.append(env.vel.copy())
        final_distance = float(np.linalg.norm(env.goal - env.pos))
        if step % 10 == 0:
            value_samples.append(
                {
                    "episode_id": episode_id,
                    "mode": mode,
                    "paired_id": paired_id,
                    "scene": spec.scene,
                    "env_seed": env_seed,
                    "step": step,
                    "predicted_value": float(output.value.detach().cpu().item()),
                    "prefix_return": total_reward,
                    "distance_to_goal": final_distance,
                    "termination_reason_so_far": reason if done else "ongoing",
                }
            )
        observation = next_observation
        if done:
            termination_reason = reason
            break
    else:
        step = config.max_steps
        termination_reason = "timeout"

    success = termination_reason == "success"
    record = EpisodeRecord(
        episode_id=episode_id,
        paired_id=paired_id,
        mode=mode,
        scene=spec.scene,
        env_seed=env_seed,
        action_seed=action_seed,
        success=success,
        termination_reason=termination_reason,
        steps=step,
        final_distance=final_distance,
        min_h_hard=float(np.nanmin(h_hard_values)) if h_hard_values else math.nan,
        min_h_soft=float(np.nanmin(h_soft_values)) if h_soft_values else math.nan,
        min_h_arena=float(np.nanmin(h_arena_values)) if h_arena_values else math.nan,
        min_h_arena_stopping=float(np.nanmin(h_arena_stopping_values)) if h_arena_stopping_values else math.nan,
        cbf_active_steps=cbf_active_steps,
        arena_projection_active_steps=arena_active_steps,
        total_reward=total_reward,
        trajectory_file=None,
    )
    trajectory = {
        "positions": np.asarray(positions, dtype=np.float32),
        "velocities": np.asarray(velocities, dtype=np.float32),
        "nominal_actions": np.asarray(nominal_actions, dtype=np.float32),
        "safe_actions": np.asarray(safe_actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "h_hard": np.asarray(h_hard_values, dtype=np.float32),
        "h_soft": np.asarray(h_soft_values, dtype=np.float32),
        "h_arena": np.asarray(h_arena_values, dtype=np.float32),
        "h_arena_stopping": np.asarray(h_arena_stopping_values, dtype=np.float32),
    }
    return record, trajectory, value_samples, residual_events


class FinalToy2D:
    def __init__(self, *, config: Toy2DConfig, spec: SceneSpec) -> None:
        self.config = config
        self.spec = spec
        self.rng = np.random.default_rng()
        self.step_count = 0
        self.pos = np.zeros((2,), dtype=np.float32)
        self.vel = np.zeros((2,), dtype=np.float32)
        self.goal = np.zeros((2,), dtype=np.float32)
        self.obstacle = np.zeros((2,), dtype=np.float32)

    def reset(self, *, seed: int) -> np.ndarray[Any, Any]:
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.pos = np.asarray(self.spec.start, dtype=np.float32).copy()
        self.vel = np.zeros((2,), dtype=np.float32)
        self.goal = np.asarray(self.spec.goal, dtype=np.float32).copy()
        self.obstacle = np.asarray(self.spec.obstacle, dtype=np.float32).copy()
        return self.observation()

    def observation(self) -> np.ndarray[Any, Any]:
        return np.concatenate((self.pos, self.vel, self.goal, self.obstacle)).astype(np.float32)

    def step(self, action: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], float, TerminationReason, bool]:
        applied = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -self.config.action_limit, self.config.action_limit)
        self.vel = (self.vel + applied * np.float32(self.config.dt)).astype(np.float32)
        self.pos = (self.pos + self.vel * np.float32(self.config.dt)).astype(np.float32)
        self.step_count += 1
        reason = classify_termination(self)
        done = reason != "timeout" or self.step_count >= self.config.max_steps
        if self.step_count >= self.config.max_steps and reason not in {"collision", "out_of_arena", "success"}:
            reason = "timeout"
            done = True
        distance = float(np.linalg.norm(self.goal - self.pos))
        reward = -distance - (10.0 if reason in {"collision", "out_of_arena"} else 0.0)
        return self.observation(), reward, reason, done

    def obstacle_clearance(self) -> float:
        if float(np.linalg.norm(self.obstacle)) > float(FAR_OBSTACLE_THRESHOLD_M):
            return math.inf
        return float(np.linalg.norm(self.pos - self.obstacle) - (self.config.robot_radius + self.config.obstacle_radius))


def classify_termination(env: FinalToy2D) -> TerminationReason:
    collision = env.obstacle_clearance() < 0.0
    out_of_arena = float(np.linalg.norm(env.pos)) > float(env.config.arena_radius)
    success = float(np.linalg.norm(env.goal - env.pos)) <= SUCCESS_RADIUS
    if collision:
        return "collision"
    if out_of_arena:
        return "out_of_arena"
    if success:
        return "success"
    return "timeout"


def safety_state(env: FinalToy2D) -> SafetyState:
    radius = float(env.config.arena_radius)
    half_extent = radius / math.sqrt(2.0)
    return SafetyState(
        position=torch.as_tensor(env.pos.reshape(1, 2), dtype=torch.float32),
        velocity=torch.as_tensor(env.vel.reshape(1, 2), dtype=torch.float32),
        goal=torch.as_tensor(env.goal.reshape(1, 2), dtype=torch.float32),
        obstacles=torch.as_tensor(env.obstacle.reshape(1, 1, 2), dtype=torch.float32),
        robot_radius=float(env.config.robot_radius),
        obstacle_radius=float(env.config.obstacle_radius),
        arena_bounds=torch.as_tensor([-half_extent, half_extent, -half_extent, half_extent], dtype=torch.float32),
        metadata={"arena_bounds_source": "inscribed_square", "arena_radius": radius},
    )


def policy_observation(observation: np.ndarray[Any, Any], obs_dim: int) -> torch.Tensor:
    padded = np.zeros((1, obs_dim), dtype=np.float32)
    copy_dim = min(int(observation.shape[-1]), obs_dim)
    padded[0, :copy_dim] = observation[:copy_dim]
    return torch.as_tensor(padded, dtype=torch.float32)


def classify_layer2_residuals(
    metrics: Mapping[str, object],
    *,
    episode_id: int,
    step: int,
    scene: str,
    mode: str,
) -> list[Layer2ResidualEvent]:
    events: list[Layer2ResidualEvent] = []
    for barrier, metric in BARRIER_METRIC_KEYS.items():
        raw_value = metrics.get(metric)
        if not isinstance(raw_value, int | float):
            continue
        value = float(raw_value)
        if value >= 0.0:
            continue
        severity: Literal["numerical", "real"] = "numerical" if value >= TOL_LAYER2_HALT else "real"
        events.append(
            Layer2ResidualEvent(
                episode_id=episode_id,
                step=step,
                barrier=barrier,
                metric=metric,
                value=value,
                severity=severity,
                tolerance_threshold=TOL_LAYER2_HALT,
                scene=scene,
                mode=mode,
            )
        )
    return events


def is_layer2_violation(filter_result: SafetyFilterResult[torch.Tensor]) -> bool:
    """Compatibility helper for tests and older callers."""

    events = classify_layer2_residuals(
        filter_result.metrics,
        episode_id=-1,
        step=-1,
        scene="unknown",
        mode="unknown",
    )
    return any(event.severity == "real" for event in events)


def layer2_violation_halt(
    *,
    autopsy_dir: Path,
    episode_id: int,
    step: int,
    spec: SceneSpec,
    observation: np.ndarray[Any, Any],
    nominal: np.ndarray[Any, Any],
    safe: np.ndarray[Any, Any],
    metrics: dict[str, object],
    positions: Sequence[np.ndarray[Any, Any]],
) -> None:
    payload = {
        "reason": "layer2_violation",
        "severity": "real",
        "tolerance_threshold": TOL_LAYER2_HALT,
        "episode_id": episode_id,
        "step": step,
        "scene": asdict(spec),
        "observation": observation.tolist(),
        "nominal_action": nominal.tolist(),
        "safe_action": safe.tolist(),
        "metrics": metrics,
    }
    write_json(autopsy_dir / f"layer2_violation_ep{episode_id:04d}_step{step:04d}.json", payload)
    np.savez_compressed(
        autopsy_dir / f"layer2_violation_ep{episode_id:04d}_step{step:04d}.npz",
        positions=np.asarray(positions, dtype=np.float32),
        observation=observation.astype(np.float32),
        nominal_action=nominal.astype(np.float32),
        safe_action=safe.astype(np.float32),
    )
    sys.exit(2)


def summarize(records: Sequence[EpisodeRecord], *, residual_events: Sequence[Layer2ResidualEvent]) -> dict[str, Any]:
    total = len(records)
    successes = sum(record.success for record in records)
    aggregate = {
        "total_episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "success_rate_wilson_95": wilson(successes, total),
        "termination_counts": {reason: sum(record.termination_reason == reason for record in records) for reason in TERMINATION_CLASSES},
        "mean_steps": mean(record.steps for record in records),
        "mean_final_distance": mean(record.final_distance for record in records),
        "min_h_hard": min(record.min_h_hard for record in records),
        "min_h_soft": min(record.min_h_soft for record in records),
        "min_h_arena": min(record.min_h_arena for record in records),
        "min_h_arena_stopping": min(record.min_h_arena_stopping for record in records),
        "layer2_residuals": layer2_residual_summary(records, residual_events=residual_events),
        "paired_det_stoch": paired_summary(records),
        "by_mode": group_summary(records, key="mode"),
        "by_scene": group_summary(records, key="scene"),
    }
    return aggregate


def layer2_residual_summary(
    records: Sequence[EpisodeRecord],
    *,
    residual_events: Sequence[Layer2ResidualEvent],
) -> dict[str, Any]:
    minima = {
        "h_hard": [record.min_h_hard for record in records],
        "h_soft": [record.min_h_soft for record in records],
        "h_arena": [record.min_h_arena for record in records],
        "h_arena_stopping": [record.min_h_arena_stopping for record in records],
    }
    by_barrier: dict[str, dict[str, Any]] = {}
    for barrier, values in minima.items():
        finite_values = [float(value) for value in values if math.isfinite(float(value))]
        minimum = min(finite_values) if finite_values else math.nan
        barrier_events = [event for event in residual_events if event.barrier == barrier]
        by_barrier[barrier] = {
            "min_value": minimum,
            "maximum_negative_excursion": min(0.0, minimum) if math.isfinite(minimum) else math.nan,
            "numerical_residual_count": sum(event.severity == "numerical" for event in barrier_events),
            "violations_above_tolerance": sum(event.severity == "real" for event in barrier_events),
        }
    return {
        "tolerance_threshold": TOL_LAYER2_HALT,
        "total_numerical_residuals": sum(event.severity == "numerical" for event in residual_events),
        "violations_above_tolerance": sum(event.severity == "real" for event in residual_events),
        "by_barrier": by_barrier,
    }


def group_summary(records: Sequence[EpisodeRecord], *, key: Literal["mode", "scene"]) -> dict[str, Any]:
    groups: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        groups.setdefault(getattr(record, key), []).append(record)
    return {
        name: {
            "episodes": len(items),
            "successes": sum(item.success for item in items),
            "success_rate": sum(item.success for item in items) / len(items),
            "success_rate_wilson_95": wilson(sum(item.success for item in items), len(items)),
            "termination_counts": {reason: sum(item.termination_reason == reason for item in items) for reason in TERMINATION_CLASSES},
            "mean_steps": mean(item.steps for item in items),
            "mean_final_distance": mean(item.final_distance for item in items),
        }
        for name, items in sorted(groups.items())
    }


def paired_summary(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    deterministic = {record.paired_id: record for record in records if record.mode == "deterministic"}
    stochastic = [record for record in records if record.mode == "stochastic"]
    pairs = [
        {
            "paired_id": record.paired_id,
            "scene": record.scene,
            "deterministic_success": deterministic[record.paired_id].success,
            "stochastic_success": record.success,
            "deterministic_reason": deterministic[record.paired_id].termination_reason,
            "stochastic_reason": record.termination_reason,
        }
        for record in stochastic
        if record.paired_id in deterministic
    ]
    return {
        "pairs": len(pairs),
        "both_success": sum(pair["deterministic_success"] and pair["stochastic_success"] for pair in pairs),
        "det_only_success": sum(pair["deterministic_success"] and not pair["stochastic_success"] for pair in pairs),
        "stoch_only_success": sum((not pair["deterministic_success"]) and pair["stochastic_success"] for pair in pairs),
        "both_fail": sum((not pair["deterministic_success"]) and (not pair["stochastic_success"]) for pair in pairs),
        "examples": pairs[:20],
    }


def stage0_comparison(records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    single_static = [record for record in records if record.mode == "deterministic" and record.scene == "single_static"]
    counts = {reason: sum(record.termination_reason == reason for record in single_static) for reason in TERMINATION_CLASSES}
    return {
        "scope": "first 125 deterministic single_static episodes",
        "v0_1_stage0_reference": STAGE0_V01_SINGLE_STATIC,
        "v0_2_observed": {
            "episodes": len(single_static),
            "total_success": counts["success"],
            "total_collision": counts["collision"],
            "total_out_of_arena": counts["out_of_arena"],
            "total_timeout": counts["timeout"],
            "total_layer2_violation": counts["layer2_violation"],
        },
        "delta_success": counts["success"] - STAGE0_V01_SINGLE_STATIC["total_success"],
        "delta_collision": counts["collision"] - STAGE0_V01_SINGLE_STATIC["total_collision"],
        "delta_out_of_arena": counts["out_of_arena"] - STAGE0_V01_SINGLE_STATIC["total_out_of_arena"],
        "delta_timeout": counts["timeout"] - STAGE0_V01_SINGLE_STATIC["total_timeout"],
    }


def value_calibration(samples: Sequence[dict[str, Any]], records: Sequence[EpisodeRecord]) -> dict[str, Any]:
    returns_by_episode = {record.episode_id: record.total_reward for record in records}
    enriched = []
    for sample in samples:
        total_return = float(returns_by_episode[int(sample["episode_id"])])
        enriched.append({**sample, "episode_total_return": total_return, "return_remaining_proxy": total_return - float(sample["prefix_return"])})
    return {
        "sampling_schema": "every 10th transition per episode",
        "num_samples": len(enriched),
        "fields": ["predicted_value", "prefix_return", "episode_total_return", "return_remaining_proxy"],
        "samples": enriched,
    }


def wilson(successes: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    z = 1.959963984540054
    phat = successes / total
    denominator = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denominator
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def mean(values: Sequence[float] | Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else math.nan


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
