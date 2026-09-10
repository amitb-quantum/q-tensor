import math

import pytest

from q_tensor.performance import first_crossover, speedup, throughput, timing_summary


def test_timing_summary_uses_median_and_mad() -> None:
    summary = timing_summary([1.0, 2.0, 100.0])
    assert summary["median_seconds"] == 2.0
    assert summary["median_absolute_deviation_seconds"] == 1.0
    assert summary["repeats"] == 3


def test_throughput_and_speedup_definitions() -> None:
    assert throughput(1000, 2.0) == 500
    assert speedup(4.0, 2.0) == 2
    with pytest.raises(ValueError):
        speedup(0, 1)


def test_first_crossover_sorts_by_trajectory_count() -> None:
    points = [
        {"trajectories": 64, "steady_state_speedup_baseline_over_tn": 1.2},
        {"trajectories": 16, "steady_state_speedup_baseline_over_tn": 0.8},
    ]
    assert first_crossover(points) == 64
    assert first_crossover([{**points[0], "steady_state_speedup_baseline_over_tn": 1.0}]) is None
    assert math.isclose(speedup(3, 2), 1.5)
