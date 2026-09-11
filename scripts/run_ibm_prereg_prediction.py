from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

from q_tensor.validation import (
    Gate,
    NoiseSite,
    ValidationCase,
    conditioned_distribution,
    exact_distribution,
    group_trajectories,
    sample_trajectories,
)

ROOT = Path("/home/manager/q-tensor")
EXP = ROOT / "experiments/ibm_kingston_20260911"

FAMILY = json.loads((EXP / "CIRCUIT_FAMILY.json").read_text())
TARGET = json.loads((EXP / "inputs/TARGET_INPUT.json").read_text())
PROPS = json.loads((EXP / "inputs/IBM_PROPERTIES.json").read_text())
PROTOCOL = json.loads((EXP / "PREDICTION_PROTOCOL.json").read_text())

PHYS = [79, 93, 94, 95]
LOCAL = {q: i for i, q in enumerate(PHYS)}

NTRAJ = int(PROTOCOL["trajectory_samples_per_circuit"])
BASE_SEED = int(PROTOCOL["base_seed"])

def flatten(x):
    if isinstance(x, list):
        for y in x:
            yield from flatten(y)
    else:
        yield x

def normalize(d):
    s = float(sum(d.values()))
    if s <= 0:
        raise RuntimeError("distribution has zero mass")
    return {k: float(v / s) for k, v in sorted(d.items()) if v > 1e-15}

def tvd(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)

# Frozen IBM target errors.
oneq_error = {}
for op in ("sx", "x"):
    for row in TARGET["instructions"][op]:
        oneq_error[(op, int(row["qubits"][0]))] = float(row["error"])

cz_error = {}
for row in TARGET["instructions"]["cz"]:
    a, b = map(int, row["qubits"])
    cz_error[tuple(sorted((a, b)))] = float(row["error"])

# Frozen asymmetric readout assignment probabilities.
readout = {}
for q in PHYS:
    fields = {
        item["name"]: item["value"]
        for item in PROPS["qubits"][q]
        if "name" in item and "value" in item
    }

    need = {"prob_meas0_prep1", "prob_meas1_prep0"}
    missing = need - fields.keys()
    if missing:
        raise RuntimeError(
            f"Missing asymmetric readout properties for physical q{q}: {sorted(missing)}"
        )

    readout[LOCAL[q]] = {
        # P(observed 0 | true 1)
        "p0_given_1": float(fields["prob_meas0_prep1"]),
        # P(observed 1 | true 0)
        "p1_given_0": float(fields["prob_meas1_prep0"]),
    }

def apply_readout(dist):
    """Independent asymmetric readout confusion in q0,q1,q2,q3 string order."""
    out = {}

    for true_bits, mass in dist.items():
        for observed_tuple in itertools.product("01", repeat=4):
            observed = "".join(observed_tuple)
            p = float(mass)

            for q in range(4):
                true = true_bits[q]
                obs = observed[q]
                r = readout[q]

                if true == "0":
                    p *= (1.0 - r["p1_given_0"]) if obs == "0" else r["p1_given_0"]
                else:
                    p *= r["p0_given_1"] if obs == "0" else (1.0 - r["p0_given_1"])

            out[observed] = out.get(observed, 0.0) + p

    return normalize(out)

def build_case(spec, seed):
    gates = []
    noise = []

    for op in flatten(spec["layers"]):
        name = op["gate"]
        pqs = tuple(int(q) for q in op["qubits"])
        lqs = tuple(LOCAL[q] for q in pqs)

        gate_index = len(gates)

        if name == "rz":
            gates.append(Gate("rz", lqs, float(op["theta"])))

        elif name == "sx":
            # IBM SX = global phase * RX(pi/2); global phase is unobservable here.
            gates.append(Gate("rx", lqs, math.pi / 2.0))

            # IBM target error is treated as average gate infidelity r.
            # For d=2 Pauli depolarizing channel: p = (d+1)/d * r = 3r/2.
            r = oneq_error[("sx", pqs[0])]
            p = min(1.0, 1.5 * r)
            noise.append(
                NoiseSite(gate_index, lqs, "depolarizing1", p)
            )

        elif name == "x":
            gates.append(Gate("x", lqs))

            r = oneq_error[("x", pqs[0])]
            p = min(1.0, 1.5 * r)
            noise.append(
                NoiseSite(gate_index, lqs, "depolarizing1", p)
            )

        elif name == "cz":
            gates.append(Gate("cz", lqs))

            r = cz_error[tuple(sorted(pqs))]
            # d=4: p = 5r/4.
            p = min(1.0, 1.25 * r)
            noise.append(
                NoiseSite(gate_index, lqs, "depolarizing2", p)
            )

        else:
            raise RuntimeError(f"Unsupported frozen operation: {name}")

    noisy = ValidationCase(
        case_id=spec["id"],
        nqubits=4,
        gates=tuple(gates),
        noise_sites=tuple(noise),
        observable_qubits=(0, 1, 2, 3),
        seed=seed,
    )

    ideal = ValidationCase(
        case_id=spec["id"] + "_ideal",
        nqubits=4,
        gates=tuple(gates),
        noise_sites=(),
        observable_qubits=(0, 1, 2, 3),
        seed=seed,
    )

    return ideal, noisy

results = {
    "schema": "q-tensor.ibm-predictions.v1",
    "status": "SEALED_BEFORE_QPU",
    "backend": TARGET["backend"],
    "calibration_snapshot": TARGET["snapshot"],
    "physical_qubits": PHYS,
    "trajectory_samples_per_circuit": NTRAJ,
    "bitstring_order": "q0q1q2q3",
    "readout_assignment": readout,
    "circuits": [],
}

for i, spec in enumerate(FAMILY):
    seed = BASE_SEED + i
    ideal_case, noisy_case = build_case(spec, seed)

    ideal = normalize(exact_distribution(ideal_case))
    readout_only = apply_readout(ideal)

    draws = sample_trajectories(noisy_case, NTRAJ, seed=seed)
    grouped = group_trajectories(draws)

    noisy_pre_readout = normalize(
        conditioned_distribution(noisy_case, grouped)
    )
    full = apply_readout(noisy_pre_readout)

    # Monte Carlo stability diagnostic fixed before QPU:
    half_grouped = group_trajectories(draws[: NTRAJ // 2])
    half_pre = normalize(
        conditioned_distribution(noisy_case, half_grouped)
    )
    half_full = apply_readout(half_pre)

    results["circuits"].append({
        "id": spec["id"],
        "seed": seed,
        "cz_count": spec["cz_count"],
        "gate_count": len(noisy_case.gates),
        "noise_site_count": len(noisy_case.noise_sites),
        "trajectory_draws": NTRAJ,
        "unique_trajectories": len(grouped),
        "monte_carlo_half_vs_full_tvd": tvd(half_full, full),

        "ideal_noiseless": ideal,
        "ideal_plus_readout_only": readout_only,
        "q_tensor_gate_noise_pre_readout": noisy_pre_readout,
        "q_tensor_full_calibration_informed": full,
    })

outdir = EXP / "predictions"
outdir.mkdir(exist_ok=True)

outfile = outdir / "FROZEN_PREDICTIONS.json"
text = json.dumps(results, indent=2, sort_keys=True) + "\n"
outfile.write_text(text)

digest = hashlib.sha256(text.encode()).hexdigest()
(outdir / "FROZEN_PREDICTIONS.sha256").write_text(
    f"{digest}  FROZEN_PREDICTIONS.json\n"
)

print("=== SEALED Q-TENSOR PREDICTIONS ===")
for row in results["circuits"]:
    print(
        row["id"],
        f"CZ={row['cz_count']}",
        f"unique_trajectories={row['unique_trajectories']}",
        f"MC_half_vs_full_TVD={row['monte_carlo_half_vs_full_tvd']:.8f}",
    )

print("\nSHA256:", digest)
print("QPU submission: NOT PERFORMED")
