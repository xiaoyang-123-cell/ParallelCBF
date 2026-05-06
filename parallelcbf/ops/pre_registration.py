"""Compatibility exports for pre-registration parsing utilities."""

from parallelcbf.ops.preregistration import (
    JsonPreRegistration,
    ParseError,
    load_preregistration_artifact,
    require_metric,
    validate_preregistration_artifact,
)

__all__ = [
    "JsonPreRegistration",
    "ParseError",
    "load_preregistration_artifact",
    "require_metric",
    "validate_preregistration_artifact",
]
