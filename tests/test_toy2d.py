from __future__ import annotations

import numpy as np

from parallelcbf.envs import Toy2DAvoidanceEnv


def test_open_scene_far_obstacle_no_collision() -> None:
    env = Toy2DAvoidanceEnv()
    env.reset(seed=42)
    env._pos = np.array([0.0, 0.0], dtype=np.float32)
    env._vel = np.zeros((2,), dtype=np.float32)
    env._obstacle = np.array([100.0, 100.0], dtype=np.float32)

    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert info["collision"] is False
    assert info["termination_reason"] != "collision"


def test_arena_exit_not_collision() -> None:
    env = Toy2DAvoidanceEnv()
    env.reset(seed=7)
    env._pos = np.array([3.0, 0.0], dtype=np.float32)
    env._vel = np.zeros((2,), dtype=np.float32)
    env._obstacle = np.array([100.0, 100.0], dtype=np.float32)

    _, _, _, _, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

    assert info["termination_reason"] == "out_of_arena"
    assert info["collision"] is False
