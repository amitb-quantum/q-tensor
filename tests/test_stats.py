import math

import pytest

from q_tensor.stats import probabilities, total_variation_distance


def test_probabilities_from_counts() -> None:
    assert probabilities({"0": 3, "1": 1}) == {"0": 0.75, "1": 0.25}


def test_tvd_identical_is_zero() -> None:
    assert total_variation_distance(["0", "1"], {"0": 1, "1": 1}) == 0.0


def test_tvd_disjoint_is_one() -> None:
    assert math.isclose(total_variation_distance(["0"], ["1"]), 1.0)


def test_empty_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        probabilities([])
