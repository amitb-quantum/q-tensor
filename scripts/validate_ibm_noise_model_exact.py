from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from qiskit.circuit.library import SXGate, XGate, RZGate, CZGate
from qiskit.quantum_info import DensityMatrix, Kraus, Pauli

ROOT = Path("/home/manager/q-tensor")
EXP = ROOT / "experiments/ibm_kingston_20260911"

family = json.loads((EXP / "CIRCUIT_FAMILY.json").read_text())
target = json.loads((EXP / "inputs/TARGET_INPUT.json").read_text())
props = json.loads((EXP / "inputs/IBM_PROPERTIES.json").read_text())
pred = json.loads((EXP / "predictions/FROZEN_PREDICTIONS.json").read_text())

PHYS = [79, 93, 94, 95]
LOCAL = {q: i for i, q in enumerate(PHYS)}
SUPPORT = [format(i, "04b")[::-1] for i in range(16)]
PAULI = {
    c: Pauli(c).to_matrix()
    for c in "IXYZ"
}


def flatten(x):
    if isinstance(x, list):
        for y in x:
            yield from flatten(y)
    else:
        yield x


def tvd(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(
        abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0)))
        for k in keys
    )


def normalize(d):
    s = float(sum(d.values()))
    return {k: float(v) / s for k, v in d.items()}


# Frozen target gate errors.
oneq = {}
czerr = {}

for opname in ("sx", "x"):
    for row in target["instructions"][opname]:
        oneq[(opname, int(row["qubits"][0]))] = float(row["error"])

for row in target["instructions"]["cz"]:
    a, b = map(int, row["qubits"])
    czerr[tuple(sorted((a, b)))] = float(row["error"])


# Frozen asymmetric readout.
readout = {}
for pq in PHYS:
    vals = {
        x["name"]: float(x["value"])
        for x in props["qubits"][pq]
        if "name" in x and "value" in x
    }
    readout[LOCAL[pq]] = {
        "p1_given_0": vals["prob_meas1_prep0"],
        "p0_given_1": vals["prob_meas0_prep1"],
    }


def oneq_depolarizing_nonidentity(p):
    """Exact channel: I with 1-p; X/Y/Z each p/3."""
    ops = [np.sqrt(1.0 - p) * PAULI["I"]]
    ops += [
        np.sqrt(p / 3.0) * PAULI[c]
        for c in "XYZ"
    ]
    return Kraus(ops)


def twoq_depolarizing_nonidentity(p):
    """
    Exact joint two-qubit Pauli channel.

    Identity II has probability 1-p.
    Each of the 15 non-identity tensor Paulis has probability p/15.

    Qiskit local two-qubit matrix convention is |q1 q0>, hence kron(P1,P0).
    """
    ops = [
        np.sqrt(1.0 - p) * np.eye(4, dtype=complex)
    ]

    for p0, p1 in itertools.product("IXYZ", repeat=2):
        if p0 == "I" and p1 == "I":
            continue
        mat = np.kron(PAULI[p1], PAULI[p0])
        ops.append(np.sqrt(p / 15.0) * mat)

    return Kraus(ops)


def probabilities(dm):
    diag = np.real(np.diag(dm.data))
    return normalize({
        "".join(str((i >> q) & 1) for q in range(4)): float(diag[i])
        for i in range(16)
    })


def apply_readout(dist):
    out = {b: 0.0 for b in SUPPORT}

    for true_bits, mass in dist.items():
        for obs_tuple in itertools.product("01", repeat=4):
            obs = "".join(obs_tuple)
            prob = float(mass)

            for q in range(4):
                true = true_bits[q]
                seen = obs[q]
                r = readout[q]

                if true == "0":
                    prob *= (
                        1.0 - r["p1_given_0"]
                        if seen == "0"
                        else r["p1_given_0"]
                    )
                else:
                    prob *= (
                        r["p0_given_1"]
                        if seen == "0"
                        else 1.0 - r["p0_given_1"]
                    )

            out[obs] += prob

    return normalize(out)


def evolve(spec, noisy):
    dm = DensityMatrix.from_label("0000")

    for op in flatten(spec["layers"]):
        name = op["gate"]
        pqs = tuple(int(q) for q in op["qubits"])
        lqs = tuple(LOCAL[q] for q in pqs)

        if name == "rz":
            dm = dm.evolve(RZGate(float(op["theta"])), qargs=[lqs[0]])

        elif name == "sx":
            dm = dm.evolve(SXGate(), qargs=[lqs[0]])
            if noisy:
                r = oneq[("sx", pqs[0])]
                dm = dm.evolve(
                    oneq_depolarizing_nonidentity(1.5 * r),
                    qargs=[lqs[0]],
                )

        elif name == "x":
            dm = dm.evolve(XGate(), qargs=[lqs[0]])
            if noisy:
                r = oneq[("x", pqs[0])]
                dm = dm.evolve(
                    oneq_depolarizing_nonidentity(1.5 * r),
                    qargs=[lqs[0]],
                )

        elif name == "cz":
            dm = dm.evolve(CZGate(), qargs=list(lqs))
            if noisy:
                r = czerr[tuple(sorted(pqs))]
                dm = dm.evolve(
                    twoq_depolarizing_nonidentity(1.25 * r),
                    qargs=list(lqs),
                )

        else:
            raise RuntimeError(f"unsupported gate {name}")

    return probabilities(dm)


pred_by_id = {r["id"]: r for r in pred["circuits"]}

results = []

print("=== INDEPENDENT EXACT-CHANNEL CHECK ===")
print()

for spec in family:
    cid = spec["id"]
    frozen = pred_by_id[cid]

    exact_ideal = evolve(spec, noisy=False)
    exact_gate = evolve(spec, noisy=True)

    exact_ro = apply_readout(exact_ideal)
    exact_full = apply_readout(exact_gate)

    ideal_err = tvd(exact_ideal, frozen["ideal_noiseless"])
    ro_err = tvd(exact_ro, frozen["ideal_plus_readout_only"])
    gate_err = tvd(exact_gate, frozen["q_tensor_gate_noise_pre_readout"])
    full_err = tvd(exact_full, frozen["q_tensor_full_calibration_informed"])

    results.append({
        "circuit": cid,
        "exact_vs_qtensor_ideal_tvd": ideal_err,
        "exact_vs_qtensor_readout_only_tvd": ro_err,
        "exact_vs_qtensor_gate_tvd": gate_err,
        "exact_vs_qtensor_full_tvd": full_err,
        "qtensor_half_vs_full_mc_tvd":
            frozen["monte_carlo_half_vs_full_tvd"],
    })

    print(
        f"{cid:12s} "
        f"ideal={ideal_err:.9f}  "
        f"readout={ro_err:.9f}  "
        f"gate={gate_err:.9f}  "
        f"full={full_err:.9f}  "
        f"QT_MC_half/full={frozen['monte_carlo_half_vs_full_tvd']:.9f}"
    )

out = EXP / "analysis/EXACT_CHANNEL_VALIDATION.json"
out.write_text(json.dumps({
    "schema": "q-tensor.ibm-exact-channel-validation.v1",
    "status": "POST_HOC_INDEPENDENT_CORRECTNESS_CHECK",
    "method": (
        "Exact 4-qubit density-matrix evolution with explicit categorical "
        "Pauli channels and classical asymmetric readout; no trajectory "
        "sampling and no Aer noise-model construction."
    ),
    "circuits": results,
}, indent=2, sort_keys=True) + "\n")

print()
print("wrote:", out)
