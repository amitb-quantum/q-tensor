from scripts.run_complexity_crossover import crossover_summary


def point(nqubits: int, depth: int, ratio: float, valid: bool = True) -> dict:
    return {
        "nqubits": nqubits,
        "entangling_depth": depth,
        "steady_state_speedup_baseline_over_tn": ratio,
        "validation": {"passed": valid},
    }


def test_crossover_summary_reports_only_measured_valid_points() -> None:
    summary = crossover_summary(
        [point(8, 4, 1.2), point(5, 2, 0.3), point(8, 2, 0.9), point(5, 4, 5.0, False)]
    )
    assert summary["first_approach_point"] == {
        "nqubits": 8,
        "entangling_depth": 2,
        "speedup_baseline_over_tn": 0.9,
    }
    assert summary["first_crossover_point"] == {
        "nqubits": 8,
        "entangling_depth": 4,
        "speedup_baseline_over_tn": 1.2,
    }
