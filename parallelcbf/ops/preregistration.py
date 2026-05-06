"""Auditable pre-registration implementation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from argparse import ArgumentParser
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


class ParseError(ValueError):
    """Raised when an auditable metrics/artifact payload is malformed."""


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
            value = require_metric(metrics, spec.metric_name)
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
    validate_preregistration_artifact(loaded)
    return loaded


def require_metric(metrics: MetricDict, metric_name: str) -> float:
    """Return a finite numeric metric or raise `ParseError`."""

    if metric_name not in metrics:
        raise ParseError(f"missing required metric: {metric_name}")
    raw_value = metrics[metric_name]
    if isinstance(raw_value, str):
        raise ParseError(f"metric {metric_name} must be numeric, got string")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ParseError(f"metric {metric_name} must be finite, got {value}")
    return value


def validate_preregistration_artifact(payload: dict[str, Any]) -> None:
    """Validate required fields in a committed pre-registration artifact."""

    required_top_level = ("committed_at", "specs")
    for field_name in required_top_level:
        if field_name not in payload:
            raise ParseError(f"pre-registration artifact missing field: {field_name}")
    specs = payload["specs"]
    if not isinstance(specs, list):
        raise ParseError("pre-registration artifact field 'specs' must be a list")
    required_spec_fields = (
        "name",
        "hypothesis",
        "metric_name",
        "threshold",
        "comparison",
        "sample_size",
    )
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ParseError(f"pre-registration spec {index} must be an object")
        for field_name in required_spec_fields:
            if field_name not in spec:
                raise ParseError(f"pre-registration spec {index} missing field: {field_name}")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal_path_for(path: Path) -> Path:
    return path.with_name(f"{path.stem}.seal.json")


def _commit_cli(argv: list[str]) -> int:
    parser = ArgumentParser(description="Seal a pre-registration manifest by SHA-256.")
    parser.add_argument("command", choices=("commit", "verify"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--seal-output", type=Path, default=None)
    args = parser.parse_args(argv)

    sha256 = sha256_file(args.path)
    seal_path = args.seal_output if args.seal_output is not None else _seal_path_for(args.path)
    if args.command == "verify":
        payload = json.loads(seal_path.read_text(encoding="utf-8"))
        expected = payload.get("sha256")
        if expected != sha256:
            raise ParseError(f"pre-registration seal mismatch: expected {expected}, got {sha256}")
        print(sha256)
        return 0

    if args.seal_output is not None:
        payload = {
            "path": str(args.path),
            "sha256": sha256,
            "committed_at": datetime.now(timezone.utc).isoformat(),
        }
        seal_path.parent.mkdir(parents=True, exist_ok=True)
        seal_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(sha256)
    return 0


def main() -> None:
    raise SystemExit(_commit_cli(sys.argv[1:]))


if __name__ == "__main__":
    main()
