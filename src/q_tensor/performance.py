"""Dependency-light helpers for reproducible performance result reduction."""

from __future__ import annotations

from statistics import median
from typing import Iterable, Mapping


def timing_summary(samples: Iterable[float]) -> dict[str, float | int]:
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("at least one timing sample is required")
    if any(value <= 0 for value in values):
        raise ValueError("timing samples must be positive")
    center = median(values)
    return {
        "repeats": len(values),
        "median_seconds": center,
        "min_seconds": min(values),
        "max_seconds": max(values),
        "median_absolute_deviation_seconds": median(abs(value - center) for value in values),
    }


def throughput(effective_shots: int, seconds: float) -> float:
    if effective_shots <= 0:
        raise ValueError("effective_shots must be positive")
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    return effective_shots / seconds


def speedup(baseline_seconds: float, tn_seconds: float) -> float:
    """Return the explicitly defined baseline/TN runtime ratio."""

    if baseline_seconds <= 0 or tn_seconds <= 0:
        raise ValueError("runtimes must be positive")
    return baseline_seconds / tn_seconds


def first_crossover(points: Iterable[Mapping[str, float | int]]) -> int | None:
    """Return the first trajectory count whose baseline/TN ratio exceeds one."""

    ordered = sorted(points, key=lambda point: int(point["trajectories"]))
    for point in ordered:
        if float(point["steady_state_speedup_baseline_over_tn"]) > 1:
            return int(point["trajectories"])
    return None
