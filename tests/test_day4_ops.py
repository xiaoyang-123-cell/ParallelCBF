from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from parallelcbf.algorithms import RandomActionAlgorithm
from parallelcbf.api import PreRegistrationSpec
from parallelcbf.envs import Toy2DAvoidanceEnv, Toy2DAvoidanceVecEnv
from parallelcbf.ops import (
    AtomicCheckpoint,
    DefaultWatchdogRegistry,
    FailureForensics,
    JsonPreRegistration,
    ThresholdWatchdog,
    V24Telemetry,
)


def test_watchdog_registry_triggers() -> None:
    registry = DefaultWatchdogRegistry()
    registry.register(ThresholdWatchdog("h_hard_violation_rate", 0.02))
    events = registry.update({"h_hard_violation_rate": 0.03}, step=5)
    assert len(events) == 1
    assert registry.should_halt()


def test_watchdog_registry_no_trigger() -> None:
    registry = DefaultWatchdogRegistry()
    registry.register(ThresholdWatchdog("h_hard_violation_rate", 0.02))
    events = registry.update({"h_hard_violation_rate": 0.0}, step=5)
    assert events == ()
    assert not registry.should_halt()


def test_watchdog_registry_mixed_watchdogs() -> None:
    registry = DefaultWatchdogRegistry()
    registry.register(ThresholdWatchdog("h_hard_violation_rate", 0.02))
    registry.register(ThresholdWatchdog("rolling_success", 0.60, greater_than=False))
    events = registry.update({"h_hard_violation_rate": 0.0, "rolling_success": 0.55}, step=8)
    assert [event.name for event in events] == ["rolling_success_lt_0.6"]
    assert registry.should_halt()


def test_watchdog_registry_reset() -> None:
    registry = DefaultWatchdogRegistry()
    registry.register(ThresholdWatchdog("h_hard_violation_rate", 0.02))
    registry.update({"h_hard_violation_rate": 0.03}, step=5)
    registry.reset()
    assert not registry.should_halt()
    assert registry.events == ()


def test_random_action_algorithm_learns_100_steps() -> None:
    env = Toy2DAvoidanceEnv()
    algorithm = RandomActionAlgorithm(seed=123)
    algorithm.learn(env, total_timesteps=100)
    assert algorithm.timesteps_seen == 100


def test_random_action_algorithm_predict_carries_hidden_state_contract() -> None:
    env = Toy2DAvoidanceEnv()
    observation, _ = env.reset(seed=1)
    algorithm = RandomActionAlgorithm(seed=123)
    prediction = algorithm.predict(observation, None, deterministic=True)
    assert prediction.action.shape == (2,)
    assert np.all(prediction.action == 0.0)
    assert prediction.hidden_state is None


def test_random_action_algorithm_serialization_replays_rng(tmp_path: Path) -> None:
    env = Toy2DAvoidanceEnv()
    observation, _ = env.reset(seed=3)
    algorithm = RandomActionAlgorithm(seed=99)
    _ = algorithm.predict(observation, None)
    checkpoint_path = tmp_path / "random.pt"
    algorithm.save(checkpoint_path)
    expected = algorithm.predict(observation, None).action
    restored = RandomActionAlgorithm(seed=0)
    restored.load(checkpoint_path)
    actual = restored.predict(observation, None).action
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("num_envs", [1, 2, 4])
def test_toy2d_vec_env_shape_stability(num_envs: int) -> None:
    env = Toy2DAvoidanceVecEnv(num_envs=num_envs)
    observations, info = env.reset(seed=17)
    assert observations.shape == (num_envs, 8)
    assert info["safety_metrics"]["h_hard_violation"].shape == (num_envs,)
    actions = np.zeros((num_envs, 2), dtype=np.float32)
    observations, rewards, terminated, truncated, info = env.step(actions)
    assert observations.shape == (num_envs, 8)
    assert rewards.shape == (num_envs,)
    assert terminated.shape == (num_envs,)
    assert truncated.shape == (num_envs,)
    assert info["hard_constraint_violations"]["collision_or_oob"].shape == (num_envs,)


def test_preregistration_atomic_round_trip(tmp_path: Path) -> None:
    prereg = JsonPreRegistration()
    prereg.add_spec(
        PreRegistrationSpec(
            name="zero_collision",
            hypothesis="Hard violations remain zero.",
            metric_name="h_hard_violation_rate",
            threshold=0.0,
            comparison="eq",
            sample_size=200,
        )
    )
    artifact_path = tmp_path / "prereg.json"
    commit = prereg.commit_to_artifact(artifact_path)
    content = artifact_path.read_bytes()
    loaded = json.loads(content.decode("utf-8"))
    assert loaded["specs"][0]["name"] == "zero_collision"
    assert hashlib.sha256(content).hexdigest() == commit.sha256
    assert not artifact_path.with_name("prereg.json.tmp").exists()


def test_preregistration_evaluates_registered_specs(tmp_path: Path) -> None:
    prereg = JsonPreRegistration()
    prereg.add_spec(
        PreRegistrationSpec(
            name="success_floor",
            hypothesis="Success rate clears the configured floor.",
            metric_name="success_rate",
            threshold=0.70,
            comparison="ge",
            sample_size=1000,
        )
    )
    prereg.commit_to_artifact(tmp_path / "prereg.json")
    report = prereg.evaluate({"success_rate": 0.72})
    assert report.status == "PASS"
    assert report.results == {"success_floor": True}


def test_failure_forensics_keeps_rolling_window_and_dumps(tmp_path: Path) -> None:
    forensics = FailureForensics(capacity=128)
    for step in range(1000):
        forensics.push(step=step, metrics={"rolling_success": step / 1000.0})
    assert len(forensics.records) == 128
    assert forensics.records[0].step == 872
    dump_path = forensics.dump_to_disk(reason="watchdog trip", path=tmp_path / "halt.json")
    assert dump_path.name.startswith("halt_")
    assert dump_path.suffix == ".json"
    loaded = json.loads(dump_path.read_text(encoding="utf-8"))
    assert loaded["reason"] == "watchdog trip"
    assert len(loaded["records"]) == 128
    assert loaded["records"][-1]["step"] == 999


def test_failure_forensics_directory_path_creates_timestamped_json(tmp_path: Path) -> None:
    forensics = FailureForensics(capacity=4)
    forensics.push(step=1, metrics={"h_hard_violation_rate": 0.0})
    dump_path = forensics.dump_to_disk(reason="manual audit", path=tmp_path)
    assert dump_path.parent == tmp_path
    assert dump_path.name.startswith("forensics_")
    assert dump_path.suffix == ".json"


def test_atomic_checkpoint_round_trip(tmp_path: Path) -> None:
    checkpoint = AtomicCheckpoint()
    path = checkpoint.save({"episode": 1000, "ok": True}, tmp_path / "checkpoint.pt")
    assert checkpoint.load(path) == {"episode": 1000, "ok": True}
    assert not path.with_name("checkpoint.pt.tmp").exists()


def test_atomic_checkpoint_mid_write_crash_leaves_previous_file(tmp_path: Path) -> None:
    checkpoint = AtomicCheckpoint()
    target = checkpoint.save({"version": 1}, tmp_path / "checkpoint.pt")
    tmp_path_obj = target.with_name("checkpoint.pt.tmp")
    with tmp_path_obj.open("wb") as handle:
        pickle.dump({"version": 2, "incomplete": True}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    assert checkpoint.load(target) == {"version": 1}
    assert tmp_path_obj.exists()


def test_atomic_checkpoint_overwrite_is_atomic(tmp_path: Path) -> None:
    checkpoint = AtomicCheckpoint()
    target = tmp_path / "checkpoint.pt"
    checkpoint.save({"version": 1}, target)
    checkpoint.save({"version": 2}, target)
    assert checkpoint.load(target) == {"version": 2}


def test_v24_telemetry_accepts_valid_metrics() -> None:
    telemetry = V24Telemetry(
        step=500000,
        stage=3,
        episode_success_rate=0.51,
        h_hard_violation_rate=0.0,
        mean_lateral_overshoot=0.12,
        mean_speed=0.87,
        policy_kl=0.03,
        critic_loss=0.42,
        actor_frozen=True,
        watchdog_halt=False,
    )
    assert telemetry.stage == 3


def test_v24_telemetry_rejects_invalid_metrics() -> None:
    payload = {
        "step": -1,
        "stage": 4,
        "episode_success_rate": 1.2,
        "h_hard_violation_rate": -0.1,
        "mean_lateral_overshoot": -0.01,
        "mean_speed": 0.0,
        "policy_kl": 0.0,
        "critic_loss": 0.0,
        "actor_frozen": True,
        "watchdog_halt": False,
        "extra_metric": 1.0,
    }
    with pytest.raises(ValidationError):
        V24Telemetry.model_validate(payload)
