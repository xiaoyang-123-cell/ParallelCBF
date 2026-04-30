"""Auditable pre-registration implementation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from parallelcbf.api import (
    ArtifactCommit,
    EvaluationReport,
    MetricDict,
    PreRegistration,
    PreRegistrationSpec,
)


class JsonPreRegistration(PreRegistration):
    """JSON-backed pre-registration with atomic artifact commits."""

    def __init__(self) -> None:
        self._specs: list[PreRegistrationSpec] = []
        self._artifact: ArtifactCommit | None = None

    @property
    def specs(self) -> tuple[PreRegistrationSpec, ...]:
        """Return registered validation claims."""

        return tuple(self._specs)

    def add_spec(self, spec: PreRegistrationSpec) -> None:
        """Register a validation claim before execution."""

        self._specs.append(spec)

    def evaluate(self, metrics: MetricDict) -> EvaluationReport:
        """Evaluate all registered specs against observed metrics."""

        results: dict[str, bool] = {}
        for spec in self._specs:
            raw_value = metrics.get(spec.metric_name)
            if isinstance(raw_value, str) or raw_value is None:
                results[spec.name] = False
                continue
            value = float(raw_value)
            results[spec.name] = self._compare(value, spec.threshold, spec.comparison)
        status: Literal["PASS", "FAIL"] = "PASS" if all(results.values()) else "FAIL"
        return EvaluationReport(status=status, results=results, metrics=dict(metrics), artifact=self._artifact)

    def commit_to_artifact(self, path: str | Path) -> ArtifactCommit:
        """Atomically write specs to JSON and return a SHA-256 artifact record."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        committed_at = datetime.now(timezone.utc)
        payload = {
            "committed_at": committed_at.isoformat(),
            "specs": [asdict(spec) for spec in self._specs],
        }
        content = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        tmp_path = target.with_name(f"{target.name}.tmp")
        with tmp_path.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        self._fsync_parent(target.parent)
        commit = ArtifactCommit(path=target, sha256=hashlib.sha256(content).hexdigest(), committed_at=committed_at)
        self._artifact = commit
        return commit

    @staticmethod
    def _compare(value: float, threshold: float, comparison: str) -> bool:
        if comparison == "lt":
            return value < threshold
        if comparison == "le":
            return value <= threshold
        if comparison == "eq":
            return value == threshold
        if comparison == "ge":
            return value >= threshold
        if comparison == "gt":
            return value > threshold
        raise ValueError(f"Unsupported comparison: {comparison}")

    @staticmethod
    def _fsync_parent(path: Path) -> None:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def load_preregistration_artifact(path: str | Path) -> dict[str, Any]:
    """Load a committed pre-registration artifact for audit tests."""

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded
