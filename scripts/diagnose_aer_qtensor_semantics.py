from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.models import BackendProperties

ROOT = Path("/home/manager/q-tensor")
EXP = ROOT / "experiments/ibm_kingston_20260911"

PHYS = [79, 93, 94, 95]
LOCAL = {q: i for i, q in enumerate(PHYS)}
SHOTS = 500_000
SUPPORT = [format(i, "04b") for i in range(16)]

pred = json.loads((EXP / "predictions/FROZEN_PREDICTIONS.json").read_text())
family = json.loads((EXP / "CIRCUIT_FAMILY.json").read_text())
props_dict = json.loads((EXP / "inputs/IBM_PROPERTIES.json").read_text())

pred_by_id = {r["id"]: r for r in pred["circuits"]}

def arr(d):
    x = np.array([float(d.get(k, 0.0)) for k in SUPPORT], dtype=float)
    return x / x.sum()

def tvd(a, b):
    return 0.5 * np.abs(a - b).sum()

def flatten(x):
    if isinstance(x, list):
        for y in x:
            yield from flatten(y)
    else:
        yield x

def qcounts(counts):
    out = {}
    for bits, n in counts.items():
        qbits = bits.replace(" ", "")[::-1]
        out[qbits] = out.get(qbits, 0) + int(n)
    return out

# 4-qubit frozen properties.
small = {
    "backend_name": props_dict.get("backend_name", "ibm_kingston"),
    "backend_version": props_dict.get("backend_version", ""),
    "last_update_date": props_dict.get("last_update_date"),
    "general": props_dict.get("general", []),
    "qubits": [],
    "gates": [],
}

for q in PHYS:
    qp = copy.deepcopy(props_dict["qubits"][q])
    if "frequency" not in {x.get("name") for x in qp}:
        qp.append({
            "name": "frequency",
            "value": 5.0,
            "unit": "GHz",
            "date": qp[0].get("date"),
        })
    small["qubits"].append(qp)

MODEL_GATES = {"sx", "x", "rz", "cz"}

for gate in props_dict["gates"]:
    qs = list(gate.get("qubits", []))
    if (
        gate.get("gate") in MODEL_GATES
        and qs
        and all(q in LOCAL for q in qs)
    ):
        g = copy.deepcopy(gate)
        g["qubits"] = [LOCAL[q] for q in qs]
        small["gates"].append(g)

props = BackendProperties.from_dict(small)

# Full no-relax model.
full_nr = NoiseModel.from_backend_properties(
    props,
    gate_error=True,
    readout_error=True,
    thermal_relaxation=False,
    temperature=0,
)

# Gate-only no-relax.
gate_nr = NoiseModel.from_backend_properties(
    props,
    gate_error=True,
    readout_error=False,
    thermal_relaxation=False,
    temperature=0,
)

# Readout-only.
readout_only = NoiseModel.from_backend_properties(
    props,
    gate_error=False,
    readout_error=True,
    thermal_relaxation=False,
    temperature=0,
)

circuits = []
for spec in family:
    qc = QuantumCircuit(4, 4, name=spec["id"])
    for op in flatten(spec["layers"]):
        name = op["gate"]
        pqs = [int(q) for q in op["qubits"]]
        lqs = [LOCAL[q] for q in pqs]

        if name == "rz":
            qc.rz(float(op["theta"]), lqs[0])
        elif name == "sx":
            qc.sx(lqs[0])
        elif name == "x":
            qc.x(lqs[0])
        elif name == "cz":
            qc.cz(lqs[0], lqs[1])
        else:
            raise RuntimeError(name)

    qc.measure(range(4), range(4))
    circuits.append(qc)

def run(model, seed):
    sim = AerSimulator(method="density_matrix", noise_model=model)
    res = sim.run(circuits, shots=SHOTS, seed_simulator=seed).result()
    return {
        c.name: arr(qcounts(res.get_counts(c)))
        for c in circuits
    }

print("Running readout-only...")
a_ro = run(readout_only, 2026091131)

print("Running gate-only no-relax...")
a_gate = run(gate_nr, 2026091132)

print("Running full no-relax...")
a_full = run(full_nr, 2026091133)

print()
print("=== AER vs Q-TENSOR SEMANTIC DECOMPOSITION ===")

for cid in [c.name for c in circuits]:
    p = pred_by_id[cid]

    qt_ro = arr(p["ideal_plus_readout_only"])
    qt_gate = arr(p["q_tensor_gate_noise_pre_readout"])
    qt_full = arr(p["q_tensor_full_calibration_informed"])

    print(
        f"{cid:12s} "
        f"readout={tvd(qt_ro, a_ro[cid]):.6f}  "
        f"gate={tvd(qt_gate, a_gate[cid]):.6f}  "
        f"full={tvd(qt_full, a_full[cid]):.6f}"
    )

print()
print("Finite-shot diagnostic only; exact superoperator adjudication is authoritative for structural equality.")
