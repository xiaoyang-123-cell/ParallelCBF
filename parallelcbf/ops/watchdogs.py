"""Concrete watchdog registry implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, cast

from parallelcbf.api import MetricDict, Watchdog, WatchdogEvent, WatchdogRegistry


SeverityValue = Literal["info", "warning", "critical"]


def _coerce_severity(value: str) -> SeverityValue:
    if value in {"info", "warning", "critical"}:
        return cast(SeverityValue, value)
    raise ValueError(f"unsupported watchdog severity: {value}")


@dataclass(slots=True)
class ThresholdWatchdog(Watchdog):
    """Emit a halt event when a named metric crosses a threshold."""

    metric_name: str
    threshold: float
    greater_than: bool = True
    severity: str = "critical"
    label: str | None = None
    _triggered: bool = False

    @property
    def name(self) -> str:
        """Return the watchdog name."""

        if self.label is not None:
            return self.label
        direction = "gt" if self.greater_than else "lt"
        return f"{self.metric_name}_{direction}_{self.threshold:g}"

    @property
    def metric(self) -> str:
        """Return the guarded metric name."""

        return self.metric_name

    @property
    def when(self) -> str:
        """Return the threshold comparison direction."""

        return ">" if self.greater_than else "<"

    def update(self, metrics: MetricDict, *, step: int) -> WatchdogEvent | None:
        """Check the latest metrics against the threshold."""

        raw_value = metrics.get(self.metric_name)
        if isinstance(raw_value, str) or raw_value is None:
            return None
        value = float(raw_value)
        crossed = value > self.threshold if self.greater_than else value < self.threshold
        if not crossed:
            return None
        self._triggered = True
        direction = ">" if self.greater_than else "<"
        return WatchdogEvent(
            name=self.name,
            reason=f"step={step}: {self.metric_name}={value:.6g} {direction} {self.threshold:.6g}",
            severity=_coerce_severity(self.severity),
            metrics=dict(metrics),
            should_halt=self.severity == "critical",
        )

    def reset(self) -> None:
        """Reset the triggered flag."""

        self._triggered = False


@dataclass(slots=True)
class SustainedPhaseWatchdog(Watchdog):
    """Halt when a phase metric remains fixed beyond a step budget."""

    phase_metric: str
    phase_value: int | float | str
    max_steps: int
    label: str
    severity: str = "critical"
    _triggered: bool = False

    @property
    def name(self) -> str:
        """Return the watchdog name."""

        return self.label

    @property
    def metric(self) -> str:
        """Return the guarded metric name."""

        return self.phase_metric

    @property
    def threshold(self) -> int:
        """Return the maximum sustained step budget."""

        return self.max_steps

    @property
    def when(self) -> str:
        """Return the halt condition."""

        return "sustained_phase"

    def update(self, metrics: MetricDict, *, step: int) -> WatchdogEvent | None:
        """Check whether the configured phase has persisted too long."""

        phase = metrics.get(self.phase_metric)
        if phase != self.phase_value or step < self.max_steps:
            return None
        self._triggered = True
        return WatchdogEvent(
            name=self.name,
            reason=f"step={step}: {self.phase_metric}={phase!r} sustained beyond {self.max_steps}",
            severity="critical",
            metrics=dict(metrics),
            should_halt=True,
        )

    def reset(self) -> None:
        """Reset the triggered flag."""

        self._triggered = False


@dataclass(slots=True)
class SustainedThresholdWatchdog(Watchdog):
    """Emit after a threshold condition remains true for a step duration."""

    metric_name: str
    threshold: float
    sustained_steps: int
    greater_than: bool = True
    severity: str = "warning"
    label: str | None = None
    _first_cross_step: int | None = None
    _triggered: bool = False

    @property
    def name(self) -> str:
        """Return the watchdog name."""

        if self.label is not None:
            return self.label
        direction = "gt" if self.greater_than else "lt"
        return f"{self.metric_name}_sustained_{direction}_{self.threshold:g}"

    @property
    def metric(self) -> str:
        """Return the guarded metric name."""

        return self.metric_name

    @property
    def when(self) -> str:
        """Return the threshold comparison direction."""

        return ">" if self.greater_than else "<"

    def update(self, metrics: MetricDict, *, step: int) -> WatchdogEvent | None:
        """Track sustained threshold crossings and emit once per episode."""

        raw_value = metrics.get(self.metric_name)
        if isinstance(raw_value, str) or raw_value is None:
            self._first_cross_step = None
            return None
        value = float(raw_value)
        crossed = value > self.threshold if self.greater_than else value < self.threshold
        if not crossed:
            self._first_cross_step = None
            return None
        if self._first_cross_step is None:
            self._first_cross_step = step
        if self._triggered or step - self._first_cross_step < self.sustained_steps:
            return None
        self._triggered = True
        direction = ">" if self.greater_than else "<"
        return WatchdogEvent(
            name=self.name,
            reason=(
                f"step={step}: {self.metric_name}={value:.6g} {direction} {self.threshold:.6g} "
                f"for {step - self._first_cross_step} steps"
            ),
            severity=_coerce_severity(self.severity),
            metrics=dict(metrics),
            should_halt=self.severity == "critical",
        )

    def reset(self) -> None:
        """Reset sustained state."""

        self._first_cross_step = None
        self._triggered = False


class DefaultWatchdogRegistry(WatchdogRegistry):
    """Simple in-process registry for operational halt guards."""

    def __init__(self) -> None:
        self._watchdogs: list[Watchdog] = []
        self._events: list[WatchdogEvent] = []
        self._halted = False

    @property
    def events(self) -> Sequence[WatchdogEvent]:
        """Return all events emitted since the last reset."""

        return tuple(self._events)

    def list_registered(self) -> Sequence[Watchdog]:
        """Return registered watchdogs in registration order."""

        return tuple(self._watchdogs)

    def register(self, watchdog: Watchdog) -> None:
        """Register a watchdog instance."""

        self._watchdogs.append(watchdog)

    def update(self, metrics: MetricDict, *, step: int) -> Sequence[WatchdogEvent]:
        """Update all watchdogs and return newly emitted events."""

        new_events: list[WatchdogEvent] = []
        for watchdog in self._watchdogs:
            event = watchdog.update(metrics, step=step)
            if event is None:
                continue
            new_events.append(event)
            self._events.append(event)
            self._halted = self._halted or event.should_halt
        return tuple(new_events)

    def should_halt(self) -> bool:
        """Return whether any watchdog requested a halt."""

        return self._halted

    def reset(self) -> None:
        """Reset registry and all registered watchdogs."""

        self._events.clear()
        self._halted = False
        for watchdog in self._watchdogs:
            watchdog.reset()
