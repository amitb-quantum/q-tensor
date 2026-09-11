import math

import numpy as np

from q_tensor.stats import total_variation_distance
from q_tensor.validation import (
    FROZEN_CASES,
    Gate,
    NoiseSite,
    ValidationCase,
    canonicalize_batched_counts,
    channel_outcomes,
    conditioned_distribution,
    enumerate_trajectories,
    exact_branch_distribution,
    exact_density_matrix,
    exact_distribution,
    group_trajectories,
    sample_trajectories,
    trajectory_distribution,
    upstream_broken_deduplicate,
    upstream_depolarizing_probabilities,
    upstream_noise_samples,
    z_expectation,
)
from q_tensor.validation import _apply_local


def legacy_apply_local(state: np.ndarray, matrix: np.ndarray, qubits: tuple[int, ...]) -> np.ndarray:
    result = np.zeros_like(state)
    mask = sum(1 << qubit for qubit in qubits)
    for base in range(state.size):
        if base & mask:
            continue
        values = np.array(
            [
                state[
                    base
                    | sum(((local >> bit) & 1) << qubit for bit, qubit in enumerate(qubits))
                ]
                for local in range(2 ** len(qubits))
            ]
        )
        output = matrix @ values
        for local, value in enumerate(output):
            index = base | sum(
                ((local >> bit) & 1) << qubit for bit, qubit in enumerate(qubits)
            )
            result[index] = value
    return result


def test_vectorized_local_application_preserves_qubit_order() -> None:
    rng = np.random.default_rng(94)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    for qubits in ((0,), (3,), (0, 2), (3, 1)):
        width = 2 ** len(qubits)
        matrix = rng.normal(size=(width, width)) + 1j * rng.normal(size=(width, width))
        assert np.allclose(_apply_local(state, matrix, qubits), legacy_apply_local(state, matrix, qubits))


def test_frozen_cases_cover_two_through_five_qubits() -> None:
    assert [case.nqubits for case in FROZEN_CASES] == [2, 3, 4, 5]
    for case in FROZEN_CASES:
        density = exact_density_matrix(case)
        assert density.shape == (2**case.nqubits, 2**case.nqubits)
        assert np.isclose(np.trace(density), 1)
        assert np.allclose(density, density.conj().T)
        assert math.isclose(sum(exact_distribution(case).values()), 1)
        assert total_variation_distance(exact_distribution(case), exact_branch_distribution(case)) < 1e-12


def test_exact_bell_distribution_uses_q0_first_strings() -> None:
    case = ValidationCase(
        "bell",
        2,
        (Gate("h", (0,)), Gate("cx", (0, 1))),
        (),
        (0, 1),
        1,
    )
    assert trajectory_distribution(case, ()) == {"00": 0.4999999999999999, "11": 0.4999999999999999}
    assert math.isclose(z_expectation(exact_distribution(case), (0, 1)), 1)


def test_categorical_depolarizing_channel_sums_to_one() -> None:
    site = NoiseSite(0, (0,), "depolarizing1", 0.3)
    outcomes = channel_outcomes(site)
    assert [code for code, _ in outcomes] == [0, 1, 2, 3]
    assert np.allclose([probability for _, probability in outcomes], [0.7, 0.1, 0.1, 0.1])


def test_upstream_ordered_bernoulli_is_not_categorical_depolarization() -> None:
    actual = upstream_depolarizing_probabilities(3, 0.3)
    assert np.allclose([actual[i] for i in range(4)], [0.729, 0.1, 0.09, 0.081])
    intended = {code: probability for code, probability in channel_outcomes(NoiseSite(0, (0,), "depolarizing1", 0.3))}
    assert total_variation_distance(actual, intended) > 0


def test_seeded_trajectory_grouping_preserves_multiplicity() -> None:
    case = FROZEN_CASES[0]
    draws = sample_trajectories(case, 64)
    assert draws == sample_trajectories(case, 64)
    grouped = group_trajectories(draws)
    assert sum(multiplicity for _, multiplicity in grouped) == 64
    upstream = upstream_noise_samples(case, grouped, shots_per_trajectory=7)
    assert sum(item[2] for item in upstream) == 64 * 7
    assert math.isclose(sum(conditioned_distribution(case, grouped).values()), 1)


def test_upstream_deduplication_drops_and_misweights_repeated_draws() -> None:
    draws = [(), (), (1,), (), (1,)]
    assert upstream_broken_deduplicate(draws) == [(), (1,), (1,)]
    assert group_trajectories(draws) == (((), 3), ((1,), 2))


def test_reverse_batch_layout_is_canonicalized() -> None:
    # For 5 qubits in batches [q0 q1], [q2 q3], [q4], upstream reverses
    # within each batch and prepends later batches: [q4 q3 q2 q1 q0].
    assert canonicalize_batched_counts({"10110": 4}, nqubits=5, max_free_qubits=2) == {"01101": 4}
