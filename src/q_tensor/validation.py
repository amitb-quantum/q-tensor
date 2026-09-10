"""Exact small-circuit references and seeded trajectory sampling.

The conventions in this module are deliberately explicit:

* logical qubit zero is the first character of every bitstring;
* a noise site is applied immediately after ``after_gate``;
* depolarizing channels are categorical Kraus mixtures, not a sequence of
  independent Bernoulli decisions; and
* repeated trajectory draws retain their multiplicity.

The largest frozen case has five qubits, so dense statevectors and density
matrices are a simple, independent reference rather than an optimization.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import cos, sin, sqrt
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    angle: float | None = None


@dataclass(frozen=True)
class NoiseSite:
    after_gate: int
    qubits: tuple[int, ...]
    channel: str
    probability: float


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    nqubits: int
    gates: tuple[Gate, ...]
    noise_sites: tuple[NoiseSite, ...]
    observable_qubits: tuple[int, ...]
    seed: int


Trajectory = tuple[int, ...]

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
PAULIS = (I2, X, Y, Z)

# Upstream/CUDA-Q code order for a two-qubit depolarizing channel.  Each pair
# gives the Pauli applied to (site.qubits[0], site.qubits[1]).
TWO_QUBIT_PAULIS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
)


FROZEN_CASES: tuple[ValidationCase, ...] = (
    ValidationCase(
        case_id="2q_asymmetric_x",
        nqubits=2,
        gates=(Gate("rx", (0,), 0.73), Gate("cx", (0, 1))),
        noise_sites=(NoiseSite(0, (0,), "x", 0.19),),
        observable_qubits=(0,),
        seed=1729,
    ),
    ValidationCase(
        case_id="3q_depolarizing1",
        nqubits=3,
        gates=(
            Gate("h", (0,)),
            Gate("cx", (0, 1)),
            Gate("ry", (2,), 0.51),
            Gate("cz", (1, 2)),
        ),
        noise_sites=(NoiseSite(2, (2,), "depolarizing1", 0.27),),
        observable_qubits=(2,),
        seed=2718,
    ),
    ValidationCase(
        case_id="4q_two_channels",
        nqubits=4,
        gates=(
            Gate("rx", (0,), 0.42),
            Gate("cx", (0, 1)),
            Gate("ry", (2,), 0.71),
            Gate("cx", (2, 3)),
            Gate("cz", (1, 2)),
        ),
        noise_sites=(
            NoiseSite(0, (0,), "x", 0.18),
            NoiseSite(2, (2,), "depolarizing1", 0.24),
        ),
        observable_qubits=(0, 2),
        seed=3141,
    ),
    ValidationCase(
        case_id="5q_depolarizing2",
        nqubits=5,
        gates=(
            Gate("rx", (0,), 0.61),
            Gate("ry", (1,), 0.37),
            Gate("cx", (0, 1)),
            Gate("h", (2,)),
            Gate("cx", (2, 3)),
            Gate("ry", (4,), 0.83),
            Gate("cz", (1, 4)),
        ),
        noise_sites=(NoiseSite(2, (0, 1), "depolarizing2", 0.22),),
        observable_qubits=(0, 1),
        seed=5772,
    ),
)


def validate_case(case: ValidationCase) -> None:
    """Reject case definitions that cannot represent upstream gate-local noise."""

    if not 2 <= case.nqubits <= 5:
        raise ValueError("validation cases must contain 2 through 5 qubits")
    seen_gate_indices: set[int] = set()
    for index, gate in enumerate(case.gates):
        if gate.name not in {"x", "y", "z", "h", "t", "rx", "ry", "rz", "cx", "cy", "cz", "ch", "crx"}:
            raise ValueError(f"unsupported gate {gate.name!r}")
        if any(q < 0 or q >= case.nqubits for q in gate.qubits):
            raise ValueError(f"gate {index} addresses an invalid qubit")
        expected_arity = 2 if gate.name.startswith("c") else 1
        if len(gate.qubits) != expected_arity:
            raise ValueError(f"gate {index} has the wrong arity")
        if gate.name in {"rx", "ry", "rz", "crx"} and gate.angle is None:
            raise ValueError(f"gate {index} requires an angle")
    for site in case.noise_sites:
        if site.after_gate in seen_gate_indices:
            raise ValueError("only one noise site per coherent gate is supported")
        seen_gate_indices.add(site.after_gate)
        if not 0 <= site.after_gate < len(case.gates):
            raise ValueError("noise site references an invalid gate")
        if not 0 <= site.probability <= 1:
            raise ValueError("noise probability must lie in [0, 1]")
        gate_arity = len(case.gates[site.after_gate].qubits)
        if len(site.qubits) != gate_arity:
            raise ValueError("noise arity must match the gate operand it is merged into")
        if site.channel == "depolarizing2" and len(site.qubits) != 2:
            raise ValueError("depolarizing2 requires two qubits")
        if site.channel != "depolarizing2" and len(site.qubits) != 1:
            raise ValueError(f"{site.channel} requires one qubit")
    if any(q < 0 or q >= case.nqubits for q in case.observable_qubits):
        raise ValueError("observable addresses an invalid qubit")


def channel_outcomes(site: NoiseSite) -> tuple[tuple[int, float], ...]:
    """Return categorical ``(operator_code, probability)`` alternatives."""

    p = site.probability
    if site.channel in {"x", "y", "z"}:
        code = {"x": 1, "y": 2, "z": 3}[site.channel]
        return ((0, 1 - p), (code, p))
    if site.channel == "depolarizing1":
        return ((0, 1 - p), (1, p / 3), (2, p / 3), (3, p / 3))
    if site.channel == "depolarizing2":
        return ((0, 1 - p),) + tuple((code, p / 15) for code in range(1, 16))
    raise ValueError(f"unsupported channel {site.channel!r}")


def enumerate_trajectories(case: ValidationCase) -> tuple[tuple[Trajectory, float], ...]:
    validate_case(case)
    alternatives = [channel_outcomes(site) for site in case.noise_sites]
    if not alternatives:
        return (((), 1.0),)
    trajectories: list[tuple[Trajectory, float]] = []
    for selected in product(*alternatives):
        trajectories.append((tuple(item[0] for item in selected), float(np.prod([item[1] for item in selected]))))
    return tuple(trajectories)


def sample_trajectories(case: ValidationCase, samples: int, seed: int | None = None) -> list[Trajectory]:
    """Draw categorical Kraus trajectories, retaining repeated draws."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    validate_case(case)
    rng = np.random.default_rng(case.seed if seed is None else seed)
    draws: list[Trajectory] = []
    outcomes = [channel_outcomes(site) for site in case.noise_sites]
    for _ in range(samples):
        draws.append(
            tuple(
                site_outcomes[int(rng.choice(len(site_outcomes), p=[item[1] for item in site_outcomes]))][0]
                for site_outcomes in outcomes
            )
        )
    return draws


def group_trajectories(draws: Iterable[Trajectory]) -> tuple[tuple[Trajectory, int], ...]:
    """Group work without discarding the number of times each trajectory occurred."""

    counts = Counter(draws)
    if not counts:
        raise ValueError("at least one trajectory draw is required")
    return tuple(sorted(counts.items()))


def trajectory_errors(case: ValidationCase, trajectory: Trajectory) -> list[tuple[int, tuple[int, ...], int]]:
    if len(trajectory) != len(case.noise_sites):
        raise ValueError("trajectory does not match the case noise sites")
    return [
        (site.after_gate, site.qubits, code)
        for site, code in zip(case.noise_sites, trajectory)
        if code != 0
    ]


def upstream_noise_samples(
    case: ValidationCase,
    grouped: Sequence[tuple[Trajectory, int]],
    shots_per_trajectory: int,
) -> list[tuple[int, list[tuple[int, tuple[int, ...], int]], int]]:
    """Translate grouped draws to upstream tuples while preserving multiplicity."""

    if shots_per_trajectory <= 0:
        raise ValueError("shots_per_trajectory must be positive")
    return [
        (serial, trajectory_errors(case, trajectory), multiplicity * shots_per_trajectory)
        for serial, (trajectory, multiplicity) in enumerate(grouped)
    ]


def _single_qubit_matrix(gate: Gate) -> np.ndarray:
    if gate.name == "x":
        return X
    if gate.name == "y":
        return Y
    if gate.name == "z":
        return Z
    if gate.name == "h":
        return H
    if gate.name == "t":
        return T
    if gate.angle is None:
        raise ValueError(f"gate {gate.name!r} requires an angle")
    half = gate.angle / 2
    if gate.name == "rx":
        return cos(half) * I2 - 1j * sin(half) * X
    if gate.name == "ry":
        return cos(half) * I2 - 1j * sin(half) * Y
    if gate.name == "rz":
        return cos(half) * I2 - 1j * sin(half) * Z
    raise ValueError(f"unsupported single-qubit gate {gate.name!r}")


def _two_qubit_matrix(gate: Gate) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    if gate.name in {"cx", "cy", "cz", "ch", "crx"}:
        if gate.name == "crx":
            if gate.angle is None:
                raise ValueError("crx requires an angle")
            target_matrix = cos(gate.angle / 2) * I2 - 1j * sin(gate.angle / 2) * X
        else:
            target_matrix = {"cx": X, "cy": Y, "cz": Z, "ch": H}[gate.name]
        for input_index in range(4):
            control = input_index & 1
            target = (input_index >> 1) & 1
            if not control:
                matrix[input_index, input_index] = 1
            else:
                for target_out in range(2):
                    output_index = 1 | (target_out << 1)
                    matrix[output_index, input_index] = target_matrix[target_out, target]
        return matrix
    raise ValueError(f"unsupported two-qubit gate {gate.name!r}")


def _apply_local(state: np.ndarray, matrix: np.ndarray, qubits: tuple[int, ...]) -> np.ndarray:
    result = np.zeros_like(state)
    mask = sum(1 << q for q in qubits)
    for base in range(state.size):
        if base & mask:
            continue
        input_values = np.array(
            [state[base | sum(((local >> bit) & 1) << q for bit, q in enumerate(qubits))] for local in range(2 ** len(qubits))]
        )
        output_values = matrix @ input_values
        for local, value in enumerate(output_values):
            index = base | sum(((local >> bit) & 1) << q for bit, q in enumerate(qubits))
            result[index] = value
    return result


def _apply_gate(state: np.ndarray, gate: Gate) -> np.ndarray:
    matrix = _single_qubit_matrix(gate) if len(gate.qubits) == 1 else _two_qubit_matrix(gate)
    return _apply_local(state, matrix, gate.qubits)


def _apply_pauli_code(state: np.ndarray, qubits: tuple[int, ...], code: int) -> np.ndarray:
    if code == 0:
        return state.copy()
    if len(qubits) == 1:
        return _apply_local(state, PAULIS[code], qubits)
    first, second = TWO_QUBIT_PAULIS[code]
    result = _apply_local(state, PAULIS[first], (qubits[0],))
    return _apply_local(result, PAULIS[second], (qubits[1],))


def trajectory_statevector(case: ValidationCase, trajectory: Trajectory) -> np.ndarray:
    """Evaluate one fixed Kraus trajectory exactly in complex128."""

    validate_case(case)
    if len(trajectory) != len(case.noise_sites):
        raise ValueError("trajectory does not match the case noise sites")
    state = np.zeros(2**case.nqubits, dtype=np.complex128)
    state[0] = 1
    sites = {site.after_gate: (position, site) for position, site in enumerate(case.noise_sites)}
    for gate_index, gate in enumerate(case.gates):
        state = _apply_gate(state, gate)
        if gate_index in sites:
            position, site = sites[gate_index]
            state = _apply_pauli_code(state, site.qubits, trajectory[position])
    return state


def bitstring(index: int, nqubits: int) -> str:
    """Render an integer in canonical logical-qubit order, q0 first."""

    return "".join(str((index >> qubit) & 1) for qubit in range(nqubits))


def statevector_distribution(state: np.ndarray, nqubits: int) -> dict[str, float]:
    probabilities = np.abs(state) ** 2
    return {bitstring(index, nqubits): float(value) for index, value in enumerate(probabilities) if value > 1e-15}


def trajectory_distribution(case: ValidationCase, trajectory: Trajectory) -> dict[str, float]:
    return statevector_distribution(trajectory_statevector(case, trajectory), case.nqubits)


def exact_density_matrix(case: ValidationCase) -> np.ndarray:
    """Return the exact density matrix by exhaustively summing Kraus branches."""

    density = np.zeros((2**case.nqubits, 2**case.nqubits), dtype=np.complex128)
    for trajectory, probability in enumerate_trajectories(case):
        state = trajectory_statevector(case, trajectory)
        density += probability * np.outer(state, state.conj())
    return density


def exact_distribution(case: ValidationCase) -> dict[str, float]:
    diagonal = np.real(np.diag(exact_density_matrix(case)))
    return {bitstring(index, case.nqubits): float(value) for index, value in enumerate(diagonal) if value > 1e-15}


def conditioned_distribution(
    case: ValidationCase,
    grouped: Sequence[tuple[Trajectory, int]],
) -> dict[str, float]:
    """Exact distribution conditioned on the finite sampled trajectory mixture."""

    total = sum(multiplicity for _, multiplicity in grouped)
    if total <= 0:
        raise ValueError("trajectory multiplicities must sum to a positive value")
    result: Counter[str] = Counter()
    for trajectory, multiplicity in grouped:
        for outcome, probability in trajectory_distribution(case, trajectory).items():
            result[outcome] += multiplicity * probability / total
    return dict(result)


def z_expectation(distribution: Mapping[str, float] | Mapping[str, int], qubits: tuple[int, ...]) -> float:
    total = float(sum(distribution.values()))
    if total <= 0:
        raise ValueError("distribution must have positive total weight")
    return sum(
        float(weight) * (-1 if sum(int(bits[q]) for q in qubits) % 2 else 1)
        for bits, weight in distribution.items()
    ) / total


def canonicalize_batched_counts(
    counts: Mapping[str, int], nqubits: int, max_free_qubits: int
) -> dict[str, int]:
    """Undo upstream's output reversal for a directly indexed Qiskit circuit.

    ``sample_batched`` reverses the bits inside every batch and prepends each
    later batch.  Together those operations reverse the complete logical
    string.  The official circuit builder compensates by reversing Qiskit's
    qubit map; this helper is for direct-Qiskit diagnostic circuits.
    """

    ranges = list(range(0, nqubits, max_free_qubits))
    if ranges[-1] != nqubits:
        ranges.append(nqubits)
    upstream_order = [q for start, stop in reversed(list(zip(ranges[:-1], ranges[1:]))) for q in reversed(range(start, stop))]
    positions = {qubit: position for position, qubit in enumerate(upstream_order)}
    canonical: Counter[str] = Counter()
    for raw_bits, count in counts.items():
        if len(raw_bits) != nqubits:
            raise ValueError("backend bitstring has the wrong width")
        canonical["".join(raw_bits[positions[q]] for q in range(nqubits))] += int(count)
    return dict(canonical)


def upstream_depolarizing_probabilities(operators: int, probability: float) -> dict[int, float]:
    """Analytic distribution produced by upstream's ordered Bernoulli loop.

    Upstream tests every non-identity Pauli independently with probability
    ``p/operators`` and keeps the first success.  This differs from the intended
    categorical depolarizing channel.
    """

    if operators not in {3, 15}:
        raise ValueError("operators must be 3 or 15")
    per_operator = probability / operators
    result = {0: (1 - per_operator) ** operators}
    for code in range(1, operators + 1):
        result[code] = (1 - per_operator) ** (code - 1) * per_operator
    return result


def upstream_broken_deduplicate(draws: Sequence[Trajectory]) -> list[Trajectory]:
    """Reproduce the equality loop in upstream ``pts_proportional``.

    This is intentionally a diagnostic model of the pinned code, not a helper
    that Q-Tensor uses to prepare trajectories.
    """

    unique: list[Trajectory] = []
    for trajectory in draws:
        same = True
        if unique == []:
            same = False
        for existing in unique:
            if existing == () and trajectory != ():
                same = False
            else:
                for trajectory_item, existing_item in zip(trajectory, existing):
                    if trajectory_item != existing_item:
                        same = False
        if not same:
            unique.append(trajectory)
    return unique


def normalized(mapping: Mapping[str, float] | Mapping[str, int]) -> dict[str, float]:
    total = float(sum(mapping.values()))
    if total <= 0:
        raise ValueError("mapping must have positive total weight")
    return {key: float(value) / total for key, value in mapping.items() if value}
