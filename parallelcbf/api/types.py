"""Strict public data types for ParallelCBF APIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar


class AnyArray(Protocol):
    """Minimal structural array protocol for simulator-agnostic APIs."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array dimensions."""


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
HiddenStateT = TypeVar("HiddenStateT")
MetricValue: TypeAlias = int | float | bool | str
MetricDict: TypeAlias = dict[str, MetricValue]
ArrayF32: TypeAlias = AnyArray
BoolArray: TypeAlias = AnyArray
ConstraintViolations: TypeAlias = dict[str, bool]
ComparisonOp: TypeAlias = Literal["lt", "le", "eq", "ge", "gt"]
Severity: TypeAlias = Literal["info", "warning", "critical"]


@dataclass(frozen=True, slots=True)
class SafetyState:
    """Simulator-agnostic safety state consumed by filters."""

    position: ArrayF32
    velocity: ArrayF32
    goal: ArrayF32
    obstacles: ArrayF32
    robot_radius: float
    obstacle_radius: float
    arena_bounds: ArrayF32
    metadata: MetricDict


@dataclass(frozen=True, slots=True)
class BarrierState:
    """Diagnostic values for one safety barrier family."""

    h: ArrayF32
    active: BoolArray
    violation: BoolArray
    barrier_type: str
    metadata: MetricDict


@dataclass(frozen=True, slots=True)
class SafetyFilterResult(Generic[ActT]):
    """Result returned by a safety filter."""

    safe_action: ActT
    nominal_action: ActT
    modified: bool
    barrier_states: dict[str, BarrierState]
    metrics: MetricDict


@dataclass(frozen=True, slots=True)
class Prediction(Generic[ActT, HiddenStateT]):
    """Policy prediction result with sequence-model hidden state."""

    action: ActT
    hidden_state: HiddenStateT


@dataclass(frozen=True, slots=True)
class WatchdogEvent:
    """Structured event emitted when an operational guard trips."""

    name: str
    reason: str
    severity: Severity
    metrics: MetricDict
    should_halt: bool


@dataclass(frozen=True, slots=True)
class PreRegistrationSpec:
    """A pre-registered validation claim and its pass/fail rule."""

    name: str
    hypothesis: str
    metric_name: str
    threshold: float
    comparison: ComparisonOp
    sample_size: int


@dataclass(frozen=True, slots=True)
class ArtifactCommit:
    """Auditable record of a committed pre-registration artifact."""

    path: Path
    sha256: str
    committed_at: datetime


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Evaluation output for a pre-registration run."""

    status: Literal["PASS", "FAIL"]
    results: dict[str, bool]
    metrics: MetricDict
    artifact: ArtifactCommit | None
