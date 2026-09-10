"""Minimal dependency-free validation for saved benchmark records."""

from __future__ import annotations

from typing import Any


class ResultSchemaError(ValueError):
    """Raised when a result cannot support a reproducible claim."""


_CATEGORIES = {"upstream_claim", "reproduced_result", "q_tensor_original"}


def validate_result(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "category",
        "timestamp_utc",
        "command",
        "git_commit",
        "upstream_commit",
        "hardware",
        "software",
        "parameters",
        "measurements",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ResultSchemaError(f"missing required fields: {', '.join(missing)}")
    if record["schema_version"] != 1:
        raise ResultSchemaError("schema_version must be 1")
    if record["category"] not in _CATEGORIES:
        raise ResultSchemaError("invalid result category")
    if not isinstance(record["command"], list) or not record["command"]:
        raise ResultSchemaError("command must be a non-empty list")
    for key in ("hardware", "software", "parameters", "measurements"):
        if not isinstance(record[key], dict):
            raise ResultSchemaError(f"{key} must be an object")
