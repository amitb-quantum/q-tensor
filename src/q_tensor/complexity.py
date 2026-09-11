"""Frozen circuit family for the bounded complexity-crossover pilot."""

from __future__ import annotations

from q_tensor.validation import Gate, NoiseSite, ValidationCase, validate_case


def complexity_case(nqubits: int, entangling_depth: int, seed: int = 5772) -> ValidationCase:
    """Build a deterministic nearest-neighbor brickwork circuit.

    Every member has the same one-site categorical two-qubit depolarizing
    channel.  Therefore a shared seed produces the identical trajectory-code
    stream and multiplicities throughout the qubit/depth sweep.
    """

    if nqubits < 2:
        raise ValueError("nqubits must be at least two")
    if entangling_depth < 1:
        raise ValueError("entangling_depth must be positive")

    gates: list[Gate] = []
    noise_after_gate: int | None = None
    for layer in range(entangling_depth):
        for qubit in range(nqubits):
            angle = 0.11 + 0.037 * (((layer + 1) * (qubit + 2)) % 17)
            gates.append(Gate("ry" if (layer + qubit) % 2 else "rx", (qubit,), angle))

        parity = layer % 2
        for control in range(parity, nqubits - 1, 2):
            gates.append(Gate("cx", (control, control + 1)))
            if noise_after_gate is None:
                noise_after_gate = len(gates) - 1

        for qubit in range(nqubits):
            angle = 0.07 + 0.029 * (((layer + 3) * (qubit + 1)) % 19)
            gates.append(Gate("rz", (qubit,), angle))

    assert noise_after_gate is not None
    case = ValidationCase(
        case_id=f"brickwork_{nqubits}q_d{entangling_depth}",
        nqubits=nqubits,
        gates=tuple(gates),
        noise_sites=(NoiseSite(noise_after_gate, (0, 1), "depolarizing2", 0.22),),
        observable_qubits=(0, nqubits - 1),
        seed=seed,
    )
    validate_case(case)
    return case
