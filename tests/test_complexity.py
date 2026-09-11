import pytest

from q_tensor.complexity import complexity_case
from q_tensor.validation import group_trajectories, sample_trajectories, trajectory_distribution


def test_complexity_family_freezes_only_qubits_and_depth() -> None:
    shallow = complexity_case(5, 2)
    deep = complexity_case(8, 4)
    assert shallow.noise_sites[0].channel == deep.noise_sites[0].channel == "depolarizing2"
    assert shallow.noise_sites[0].probability == deep.noise_sites[0].probability == 0.22
    assert sample_trajectories(shallow, 1024) == sample_trajectories(deep, 1024)
    assert group_trajectories(sample_trajectories(shallow, 1024)) == group_trajectories(
        sample_trajectories(deep, 1024)
    )


def test_complexity_case_gate_count_and_exact_reference() -> None:
    case = complexity_case(6, 3)
    expected_entanglers = 3 + 2 + 3
    assert len(case.gates) == 2 * 6 * 3 + expected_entanglers
    distribution = trajectory_distribution(case, (0,))
    assert sum(distribution.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("nqubits,depth", [(1, 2), (5, 0)])
def test_complexity_case_rejects_invalid_shape(nqubits: int, depth: int) -> None:
    with pytest.raises(ValueError):
        complexity_case(nqubits, depth)
