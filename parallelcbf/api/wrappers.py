"""Composition wrappers for safety-filtered environments."""

from __future__ import annotations

from typing import Any, Generic

from parallelcbf.api.core import SafeEnv, SafetyFilter
from parallelcbf.api.types import ActT, ConstraintViolations, MetricDict, ObsT, SafetyFilterResult, SafetyState


class SafetyWrapper(SafeEnv[ObsT, ActT], Generic[ObsT, ActT]):
    """Compose a `SafeEnv` with a `SafetyFilter`.

    The wrapper intercepts `step(action)`, filters the nominal action, and then
    forwards only the filtered action to the underlying environment.
    """

    def __init__(self, env: SafeEnv[ObsT, ActT], safety_filter: SafetyFilter[ObsT, ActT]) -> None:
        self.env = env
        self.safety_filter = safety_filter
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = env.metadata
        self.reward_range = env.reward_range
        self.spec = env.spec
        self._last_filter_result: SafetyFilterResult[ActT] | None = None
        self._last_observation: ObsT | None = None

    @property
    def last_filter_result(self) -> SafetyFilterResult[ActT] | None:
        """Return diagnostics from the most recent filtered step."""

        return self._last_filter_result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsT, dict[str, Any]]:
        """Reset the environment and safety filter."""

        self.safety_filter.reset(seed=seed)
        self._last_filter_result = None
        observation, info = self.env.reset(seed=seed, options=options)
        self._last_observation = observation
        return observation, dict(info)

    def step(self, action: ActT) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        """Filter a nominal action before stepping the wrapped environment."""

        if self._last_observation is None:
            raise RuntimeError("SafetyWrapper.step() called before reset().")
        observation = self._last_observation
        filter_result = self.safety_filter.filter_action(observation, action, self.env.safety_state())
        self._last_filter_result = filter_result
        next_observation, reward, terminated, truncated, info = self.env.step(filter_result.safe_action)
        self._last_observation = next_observation
        wrapped_info = dict(info)
        wrapped_info["safety_filter"] = filter_result
        return next_observation, float(reward), bool(terminated), bool(truncated), wrapped_info

    def safety_state(self) -> SafetyState:
        """Return the wrapped environment safety state."""

        return self.env.safety_state()

    def safety_metrics(self) -> MetricDict:
        """Return merged environment and filter metrics."""

        metrics = dict(self.env.safety_metrics())
        metrics.update(self.safety_filter.metrics())
        if self._last_filter_result is not None:
            metrics.update(self._last_filter_result.metrics)
        return metrics

    def hard_constraint_violations(self) -> ConstraintViolations:
        """Return wrapped environment hard safety flags."""

        return self.env.hard_constraint_violations()

    def render(self) -> Any:
        """Render the wrapped environment."""

        return self.env.render()

    def close(self) -> None:
        """Close the wrapped environment."""

        self.env.close()
