from pathlib import Path

from scripts.run_figure3_proportional_control import (
    categorical_channel_audit,
    categorical_draws,
    choose_shadow_labels,
    frozen_trajectory_record,
    induced_shadow,
    parse_stim_case,
)
from q_tensor.validation import group_trajectories


def write_example(path: Path) -> None:
    path.write_text(
        "H 0\n"
        "X_ERROR(0.25) 0\n"
        "CX 0 2\n"
        "DEPOLARIZE2(0.3) 0 2\n"
        "RX(0.75) 1\n"
        "Y_ERROR(0.1) 1\n",
        encoding="utf-8",
    )


def test_parse_stim_case_preserves_gate_local_noise(tmp_path: Path) -> None:
    path = tmp_path / "example.stim"
    write_example(path)
    case = parse_stim_case(path)
    assert case.nqubits == 3
    assert [gate.name for gate in case.gates] == ["h", "cx", "rx"]
    assert case.gates[2].angle == 0.75
    assert [site.channel for site in case.noise_sites] == ["x", "depolarizing2", "y"]
    assert all(site.after_gate == index for index, site in enumerate(case.noise_sites))


def test_induced_shadow_reindexes_and_keeps_entangling_gate(tmp_path: Path) -> None:
    path = tmp_path / "example.stim"
    write_example(path)
    shadow = induced_shadow(parse_stim_case(path), (0, 2))
    assert shadow.nqubits == 2
    assert [gate.qubits for gate in shadow.gates] == [(0,), (0, 1)]
    assert [site.qubits for site in shadow.noise_sites] == [(0,), (0, 1)]


def test_shadow_selection_is_deterministic_and_retains_an_edge(tmp_path: Path) -> None:
    path = tmp_path / "example.stim"
    write_example(path)
    case = parse_stim_case(path)
    labels = choose_shadow_labels(case, 2)
    assert labels == choose_shadow_labels(case, 2)
    assert any(len(gate.qubits) == 2 for gate in induced_shadow(case, labels).gates)


def test_seeded_categorical_record_preserves_multiplicity(tmp_path: Path) -> None:
    path = tmp_path / "example.stim"
    write_example(path)
    case = induced_shadow(parse_stim_case(path), (0, 2))
    draws = categorical_draws(case, 64, seed=42)
    grouped = group_trajectories(draws)
    record = frozen_trajectory_record(draws, grouped)
    audit = categorical_channel_audit(case)
    assert len(draws) == 64
    assert record["multiplicity_sum"] == 64
    assert record["unique_count"] <= 64
    assert record["draws_sha256"] == frozen_trajectory_record(draws, grouped)["draws_sha256"]
    assert audit["passed"]
    assert audit["site_count"] == 2
