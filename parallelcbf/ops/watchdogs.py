"""Concrete watchdog registry implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from parallelcbf.api import MetricDict, Watchdog, WatchdogEvent, WatchdogRegistry


@dataclass(slots=True)
class ThresholdWatchdog(Watchdog):
    """Emit a halt event when a named metric crosses a threshold."""

    metric_name: str
    threshold: float
    greater_than: bool = True
    severity: str = "critical"
    _triggered: bool = False

    @property
    def name(self) -> str:
        """Return the watchdog name."""

        direction = "gt" if self.greater_than else "lt"
        return f"{self.metric_name}_{direction}_{self.threshold:g}"

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
            severity="critical",
            metrics=dict(metrics),
            should_halt=True,
        )

    def reset(self) -> None:
        """Reset the triggered flag."""

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
