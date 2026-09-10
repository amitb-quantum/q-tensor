"""Strict loading for versioned Q-Tensor experiment configurations."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration is malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ConfigError("[experiment] table is required")
    for key in ("name", "category", "seed"):
        if key not in experiment:
            raise ConfigError(f"experiment.{key} is required")
    if experiment["category"] not in {
        "upstream_claim",
        "reproduced_result",
        "q_tensor_original",
    }:
        raise ConfigError("experiment.category is not recognized")
    if not isinstance(experiment["seed"], int):
        raise ConfigError("experiment.seed must be an integer")
    return config
