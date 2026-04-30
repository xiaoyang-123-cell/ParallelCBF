"""Abstract base classes for the ParallelCBF framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Protocol, Sequence, TypeVar

import gymnasium as gym

from parallelcbf.api.types import (
    ActT,
    ArtifactCommit,
    BarrierState,
    ConstraintViolations,
    EvaluationReport,
    HiddenStateT,
    MetricDict,
    ObsT,
    Prediction,
    PreRegistrationSpec,
    SafetyFilterResult,
    SafetyState,
    WatchdogEvent,
)


ObsContraT = TypeVar("ObsContraT", contravariant=True)


class SupportsPredict(Protocol[ObsContraT, ActT, HiddenStateT]):
    """Minimal sequence-aware prediction protocol."""

    def predict(
        self,
        observation: ObsContraT,
        hidden_state: HiddenStateT,
        *,
        deterministic: bool = False,
    ) -> Prediction[ActT, HiddenStateT]:
        """Return an action and next hidden state."""


class SafeEnv(gym.Env[ObsT, ActT], ABC):
    """Base class for environments with explicit safety telemetry."""

    @abstractmethod
    def safety_state(self) -> SafetyState:
        """Return simulator-independent state needed by safety filters."""

    @abstractmethod
    def safety_metrics(self) -> MetricDict:
        """Return scalar safety metrics for logging and watchdogs."""

    @abstractmethod
    def hard_constraint_violations(self) -> ConstraintViolations:
        """Return current hard-constraint violation flags."""


class SafetyFilter(ABC, Generic[ObsT, ActT]):
    """Abstract action safety layer, including CBF/QP implementations."""

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> None:
        """Reset all filter state."""

    @abstractmethod
    def filter_action(
        self,
        observation: ObsT,
        nominal_action: ActT,
        safety_state: SafetyState,
    ) -> SafetyFilterResult[ActT]:
        """Return a safe action and diagnostics for a nominal action."""

    @abstractmethod
    def barrier_state(self) -> dict[str, BarrierState]:
        """Return the most recent barrier diagnostics."""

    @abstractmethod
    def metrics(self) -> MetricDict:
        """Return scalar diagnostics such as activation and violation rates."""


class Algorithm(ABC, Generic[ObsT, ActT, HiddenStateT]):
    """Training algorithm interface compatible with SB3/CleanRL patterns."""

    @abstractmethod
    def learn(
        self,
        env: SafeEnv[ObsT, ActT],
        *,
        total_timesteps: int,
        callback: WatchdogRegistry | None = None,
    ) -> None:
        """Train on an environment for a fixed number of timesteps."""

    @abstractmethod
    def predict(
        self,
        observation: ObsT,
        hidden_state: HiddenStateT,
        *,
        deterministic: bool = False,
    ) -> Prediction[ActT, HiddenStateT]:
        """Return an action and next hidden state for inference."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model state."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model state."""


class Watchdog(ABC):
    """Operational guard for safety, liveness, and data-quality failures."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique watchdog name."""

    @abstractmethod
    def update(self, metrics: MetricDict, *, step: int) -> WatchdogEvent | None:
        """Update the watchdog and optionally emit a halt event."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal watchdog state."""


class WatchdogRegistry(ABC):
    """Registry and dispatcher for multiple watchdogs."""

    @abstractmethod
    def register(self, watchdog: Watchdog) -> None:
        """Register a watchdog instance."""

    @abstractmethod
    def update(self, metrics: MetricDict, *, step: int) -> Sequence[WatchdogEvent]:
        """Update all watchdogs and return emitted events."""

    @abstractmethod
    def should_halt(self) -> bool:
        """Return whether any registered watchdog requested a halt."""

    @abstractmethod
    def reset(self) -> None:
        """Reset all registered watchdogs and halt state."""


class PreRegistration(ABC):
    """Pre-registered validation protocol for reproducible claims."""

    @abstractmethod
    def add_spec(self, spec: PreRegistrationSpec) -> None:
        """Register a validation claim before execution."""

    @abstractmethod
    def evaluate(self, metrics: MetricDict) -> EvaluationReport:
        """Evaluate all registered specs against observed metrics."""

    @abstractmethod
    def commit_to_artifact(self, path: str | Path) -> ArtifactCommit:
        """Write an auditable, timestamped artifact and return its digest."""
