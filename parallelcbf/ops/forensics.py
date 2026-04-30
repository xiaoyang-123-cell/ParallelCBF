"""Failure-forensics rolling buffers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from parallelcbf.api import MetricDict


@dataclass(frozen=True, slots=True)
class ForensicRecord:
    """One timestamped telemetry snapshot."""

    step: int
    metrics: MetricDict


class FailureForensics:
    """Configurable rolling telemetry buffer for halt diagnostics."""

    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._records: deque[ForensicRecord] = deque(maxlen=capacity)

    @property
    def records(self) -> tuple[ForensicRecord, ...]:
        """Return the currently retained telemetry snapshots."""

        return tuple(self._records)

    def push(self, *, step: int, metrics: MetricDict) -> None:
        """Append one metrics snapshot to the rolling buffer."""

        self._records.append(ForensicRecord(step=step, metrics=dict(metrics)))

    def dump_to_disk(self, *, reason: str, path: str | Path) -> Path:
        """Write a timestamped JSON dump and return the final path."""

        requested_path = Path(path)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if requested_path.suffix == ".json":
            target = requested_path.with_name(f"{requested_path.stem}_{stamp}.json")
        else:
            target = requested_path / f"forensics_{stamp}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reason": reason,
            "dumped_at": datetime.now(timezone.utc).isoformat(),
            "records": [{"step": record.step, "metrics": record.metrics} for record in self._records],
        }
        tmp_path = target.with_name(f"{target.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        self._fsync_parent(target.parent)
        return target

    @staticmethod
    def _fsync_parent(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
