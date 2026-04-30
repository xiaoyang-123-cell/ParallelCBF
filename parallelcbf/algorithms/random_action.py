"""Minimal random-action algorithm used to prove the Algorithm ABC."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from parallelcbf.api import Algorithm, Prediction, SafeEnv, WatchdogRegistry
from parallelcbf.ops import AtomicCheckpoint


ArrayF32 = NDArray[np.float32]


class RandomActionAlgorithm(Algorithm[ArrayF32, ArrayF32, None]):
    """Smoke-test algorithm that samples uniformly from a Box action space."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self.timesteps_seen = 0

    def learn(
        self,
        env: SafeEnv[ArrayF32, ArrayF32],
        *,
        total_timesteps: int,
        callback: WatchdogRegistry | None = None,
    ) -> None:
        """Run random actions for a fixed number of timesteps."""

        observation, _ = env.reset()
        for step in range(total_timesteps):
            prediction = self.predict(observation, None, deterministic=False)
            observation, _, terminated, truncated, _ = env.step(prediction.action)
            self.timesteps_seen += 1
            if callback is not None:
                callback.update(env.safety_metrics(), step=step)
                if callback.should_halt():
                    break
            if terminated or truncated:
                observation, _ = env.reset()

    def predict(
        self,
        observation: ArrayF32,
        hidden_state: None,
        *,
        deterministic: bool = False,
    ) -> Prediction[ArrayF32, None]:
        """Return a zero action in deterministic mode or a random action."""

        _ = observation
        _ = hidden_state
        if deterministic:
            action: ArrayF32 = np.zeros((2,), dtype=np.float32)
        else:
            action = cast(ArrayF32, self._rng.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32))
        return Prediction(action=action, hidden_state=None)

    def save(self, path: str | Path) -> None:
        """Persist minimal algorithm state."""

        AtomicCheckpoint().save({"timesteps_seen": self.timesteps_seen, "rng_state": self._rng.bit_generator.state}, path)

    def load(self, path: str | Path) -> None:
        """Load minimal algorithm state."""

        payload = AtomicCheckpoint().load(path)
        if not isinstance(payload, dict):
            raise ValueError("RandomActionAlgorithm checkpoint must be a dictionary")
        self.timesteps_seen = int(payload["timesteps_seen"])
        rng_state = payload["rng_state"]
        self._rng.bit_generator.state = rng_state
