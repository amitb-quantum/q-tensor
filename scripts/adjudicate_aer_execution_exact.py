from __future__ import annotations

import copy
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import SXGate, XGate, RZGate, CZGate
from qiskit.quantum_info import DensityMatrix, Kraus, Pauli, SuperOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit_ibm_runtime.models import BackendProperties

ROOT = Path("/home/manager/q-tensor")
EXP = ROOT / "experiments/ibm_kingston_20260911"

family = json.loads((EXP / "CIRCUIT_FAMILY.json").read_text())
raw = json.loads((EXP / "inputs/IBM_PROPERTIES.json").read_text())
target = json.loads((EXP / "inputs/TARGET_INPUT.json").read_text())

PHYS = [79, 93, 94, 95]
LOCAL = {q: i for i, q in enumerate(PHYS)}

P = {c: Pauli(c).to_matrix() for c in "IXYZ"}


def flatten(x):
    if isinstance(x, list):
        for y in x:
            yield from flatten(y)
    else:
        yield x


def tvd(a, b):
    return 0.5 * np.abs(a - b).sum()


def probs(dm):
    x = np.real(np.diag(dm.data))
    x = np.maximum(x, 0)
    return x / x.sum()


# ------------------------------------------------------------
# Reduced frozen BackendProperties
# ------------------------------------------------------------
small = {
    "backend_name": raw.get("backend_name", "ibm_kingston"),
    "backend_version": raw.get("backend_version", ""),
    "last_update_date": raw.get("last_update_date"),
    "general": raw.get("general", []),
    "qubits": [],
    "gates": [],
}

for q in PHYS:
    qp = copy.deepcopy(raw["qubits"][q])
    if "frequency" not in {x.get("name") for x in qp}:
        qp.append({
            "name": "frequency",
            "value": 5.0,
            "unit": "GHz",
            "date": qp[0].get("date"),
        })
    small["qubits"].append(qp)

for g in raw["gates"]:
    qs = list(g.get("qubits", []))
    if qs and all(q in LOCAL for q in qs):
        gg = copy.deepcopy(g)
        gg["qubits"] = [LOCAL[q] for q in qs]
        small["gates"].append(gg)

props = BackendProperties.from_dict(small)

dups = Counter((g.gate, tuple(g.qubits)) for g in props.gates)
dups = {k: v for k, v in dups.items() if v > 1}

print("Duplicate BackendProperties gate entries:", dups)

# ------------------------------------------------------------
# Backend-derived no-relax model
# ------------------------------------------------------------
nm_backend = NoiseModel.from_backend_properties(
    props,
    gate_error=True,
    readout_error=False,
    thermal_relaxation=False,
    temperature=0,
)

# ------------------------------------------------------------
# Manual Aer model containing ONLY the Q-Tensor channels
# ------------------------------------------------------------
nm_manual = NoiseModel()

for q in range(4):
    r = props.gate_error("sx", [q])
    p = 1.5 * r
    err = pauli_error([
        ("I", 1-p),
        ("X", p/3),
        ("Y", p/3),
        ("Z", p/3),
    ])
    nm_manual.add_quantum_error(err, "sx", [q])

    r = props.gate_error("x", [q])
    p = 1.5 * r
    err = pauli_error([
        ("I", 1-p),
        ("X", p/3),
        ("Y", p/3),
        ("Z", p/3),
    ])
    nm_manual.add_quantum_error(err, "x", [q])

for a, b in [(0,1), (1,2), (2,3)]:
    r = props.gate_error("cz", [a,b])
    p = 1.25 * r

    terms = []
    for x, y in itertools.product("IXYZ", repeat=2):
        label = x + y
        terms.append(
            (label, 1-p if label == "II" else p/15)
        )

    nm_manual.add_quantum_error(
        pauli_error(terms), "cz", [a,b]
    )

# ------------------------------------------------------------
# Exact reference channel
# ------------------------------------------------------------
oneq_error = {}
for op in ("sx", "x"):
    for row in target["instructions"][op]:
        oneq_error[(op, int(row["qubits"][0]))] = float(row["error"])

cz_error = {}
for row in target["instructions"]["cz"]:
    a, b = map(int, row["qubits"])
    cz_error[tuple(sorted((a,b)))] = float(row["error"])


def dep1(p):
    return Kraus([
        np.sqrt(1-p) * P["I"],
        np.sqrt(p/3) * P["X"],
        np.sqrt(p/3) * P["Y"],
        np.sqrt(p/3) * P["Z"],
    ])


def dep2(p):
    ops = [np.sqrt(1-p) * np.eye(4, dtype=complex)]

    for p0, p1 in itertools.product("IXYZ", repeat=2):
        if p0 == "I" and p1 == "I":
            continue
        ops.append(
            np.sqrt(p/15) * np.kron(P[p1], P[p0])
        )

    return Kraus(ops)


def exact_distribution(spec):
    dm = DensityMatrix.from_label("0000")

    for op in flatten(spec["layers"]):
        name = op["gate"]
        pqs = tuple(int(q) for q in op["qubits"])
        lqs = tuple(LOCAL[q] for q in pqs)

        if name == "rz":
            dm = dm.evolve(
                RZGate(float(op["theta"])),
                qargs=[lqs[0]],
            )

        elif name == "sx":
            dm = dm.evolve(SXGate(), qargs=[lqs[0]])
            dm = dm.evolve(
                dep1(1.5 * oneq_error[("sx", pqs[0])]),
                qargs=[lqs[0]],
            )

        elif name == "x":
            dm = dm.evolve(XGate(), qargs=[lqs[0]])
            dm = dm.evolve(
                dep1(1.5 * oneq_error[("x", pqs[0])]),
                qargs=[lqs[0]],
            )

        elif name == "cz":
            dm = dm.evolve(CZGate(), qargs=list(lqs))
            dm = dm.evolve(
                dep2(1.25 * cz_error[tuple(sorted(pqs))]),
                qargs=list(lqs),
            )

        else:
            raise RuntimeError(name)

    return probs(dm)


# ------------------------------------------------------------
# Exact Aer superoperator execution: zero shots
# ------------------------------------------------------------
def build_circuit(spec):
    qc = QuantumCircuit(4, name=spec["id"])

    for op in flatten(spec["layers"]):
        name = op["gate"]
        qs = [LOCAL[int(q)] for q in op["qubits"]]

        if name == "rz":
            qc.rz(float(op["theta"]), qs[0])
        elif name == "sx":
            qc.sx(qs[0])
        elif name == "x":
            qc.x(qs[0])
        elif name == "cz":
            qc.cz(qs[0], qs[1])
        else:
            raise RuntimeError(name)

    qc.save_superop(label="final_superop")
    return qc


def aer_exact(spec, noise_model):
    qc = build_circuit(spec)

    sim = AerSimulator(
        method="superop",
        noise_model=noise_model,
    )

    result = sim.run(qc).result()
    data = result.data(0)["final_superop"]

    sop = data if isinstance(data, SuperOp) else SuperOp(data)

    rho0 = DensityMatrix.from_label("0000")
    rho = rho0.evolve(sop)

    return probs(rho)


print()
print("=== EXACT AER EXECUTION ADJUDICATION ===")

for spec in family:
    ref = exact_distribution(spec)
    backend = aer_exact(spec, nm_backend)
    manual = aer_exact(spec, nm_manual)

    print(
        f"{spec['id']:12s} "
        f"Exact↔BackendAer={tvd(ref, backend):.12g}  "
        f"Exact↔ManualAer={tvd(ref, manual):.12g}  "
        f"Backend↔Manual={tvd(backend, manual):.12g}"
    )
