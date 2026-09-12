from __future__ import annotations

import copy
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

from qiskit import QuantumCircuit, qpy
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.models import BackendProperties

ROOT = Path("/home/manager/q-tensor")
EXP = ROOT / "experiments/ibm_kingston_20260911"

PROPS_FILE = EXP / "inputs/IBM_PROPERTIES.json"
QPY_FILE = EXP / "isa/QTIBM_CIRCUITS.qpy"
PRED_FILE = EXP / "predictions/FROZEN_PREDICTIONS.json"
RAW_FILE = EXP / "hardware/RAW_HARDWARE_RESULTS.json"
FLOOR_FILE = EXP / "analysis/SHOT_NOISE_FLOOR.json"

EXPECTED_QPY_SHA256 = (
    "8b35ab3ede6f169e5129e5dbea11473d0716d662d2eceed0390d3240e2c65140"
)

SHOTS = 200_000
SEED_DEFAULT = 2026091122
SEED_NO_RELAX = 2026091123
BOOT = 10_000
BOOT_SEED = 2026091124

SUPPORT = [format(i, "04b") for i in range(16)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arr(d):
    x = np.array([float(d.get(k, 0.0)) for k in SUPPORT], dtype=float)
    return x / x.sum()


def tvd(a, b):
    return float(0.5 * np.abs(a - b).sum())


def counts_to_qorder(counts):
    # Qiskit display order is c3c2c1c0.
    out = {}
    for bits, n in counts.items():
        bits = bits.replace(" ", "")
        if len(bits) != 4:
            raise RuntimeError(f"Unexpected count key: {bits!r}")
        qbits = bits[::-1]
        out[qbits] = out.get(qbits, 0) + int(n)
    return out


if sha256(QPY_FILE) != EXPECTED_QPY_SHA256:
    raise SystemExit("STOP: frozen QPY checksum mismatch")

props_dict = json.loads(PROPS_FILE.read_text())

# Build a reduced BackendProperties object containing only the four physical
# qubits used by the frozen experiment. This avoids depending on incomplete
# calibration fields elsewhere on the 156-qubit processor.
PHYS = [79, 93, 94, 95]
LOCAL = {q: i for i, q in enumerate(PHYS)}

small_props = {
    "backend_name": props_dict.get("backend_name", "ibm_kingston"),
    "backend_version": props_dict.get("backend_version", ""),
    "last_update_date": props_dict.get("last_update_date"),
    "general": props_dict.get("general", []),
    "qubits": [],
    "gates": [],
}

synthetic_frequency_count = 0

for q in PHYS:
    qprops = copy.deepcopy(props_dict["qubits"][q])
    names = {item.get("name") for item in qprops}

    # Frozen IBM properties omit frequency. Aer needs it only for thermal
    # excited-state population. At T=0 K that population is zero, so the
    # numerical value cannot affect the relaxation channel.
    if "frequency" not in names:
        date = (
            qprops[0].get("date")
            if qprops
            else props_dict.get("last_update_date")
        )
        qprops.append({
            "name": "frequency",
            "value": 5.0,
            "unit": "GHz",
            "date": date,
        })
        synthetic_frequency_count += 1

    small_props["qubits"].append(qprops)

# Keep only gate operations represented by the Q-Tensor model.
#
# IBM BackendProperties also contains measurement-family entries carrying
# values numerically equal to readout calibration. Passing those through
# NoiseModel.from_backend_properties causes Aer to attach a quantum error
# to measurement in addition to its separate ReadoutError, which is not
# equivalent to the frozen Q-Tensor model.
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
        small_props["gates"].append(g)

props = BackendProperties.from_dict(small_props)

print("Reduced frozen properties qubits =", len(small_props["qubits"]))
print("Reduced frozen properties gates  =", len(small_props["gates"]))
print("Synthetic frequency placeholders  =", synthetic_frequency_count)

pred = json.loads(PRED_FILE.read_text())
raw = json.loads(RAW_FILE.read_text())
floor = json.loads(FLOOR_FILE.read_text())

pred_by_id = {x["id"]: x for x in pred["circuits"]}
raw_by_id = {x["circuit"]: x for x in raw["results"]}
floor_by_id = {x["circuit"]: x for x in floor["circuits"]}

expected_names = [
    "QTIBM_CZ03",
    "QTIBM_CZ06",
    "QTIBM_CZ09",
    "QTIBM_CZ12",
]

family = json.loads((EXP / "CIRCUIT_FAMILY.json").read_text())

def flatten(x):
    if isinstance(x, list):
        for y in x:
            yield from flatten(y)
    else:
        yield x

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
            raise RuntimeError(f"Unsupported frozen gate: {name}")

    qc.measure(range(4), range(4))
    circuits.append(qc)

if [c.name for c in circuits] != expected_names:
    raise SystemExit(
        f"STOP: unexpected frozen circuits: {[c.name for c in circuits]}"
    )

print("Local Aer circuits =", [c.name for c in circuits])

# ----------------------------------------------------------------------
# Model A: standard Aer device model.
#
# Aer composes thermal relaxation derived from T1/T2 and gate duration
# with a depolarizing component chosen so the resulting average gate
# infidelity matches the reported backend gate error.
# ----------------------------------------------------------------------

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")

    aer_default_noise = NoiseModel.from_backend_properties(
        props,
        gate_error=True,
        readout_error=True,
        thermal_relaxation=True,
        temperature=0,
    )

default_warnings = [str(w.message) for w in caught]

# ----------------------------------------------------------------------
# Model B: no-relaxation cross-validation.
#
# This is NOT treated as a second scientific baseline. It is a
# cross-check against Q-Tensor's all-depolarizing gate-error model.
# ----------------------------------------------------------------------

aer_no_relax_noise = NoiseModel.from_backend_properties(
    props,
    gate_error=True,
    readout_error=True,
    thermal_relaxation=False,
    temperature=0,
)

def run_aer(noise_model, seed):
    sim = AerSimulator(
        method="density_matrix",
        noise_model=noise_model,
    )

    job = sim.run(
        circuits,
        shots=SHOTS,
        seed_simulator=seed,
    )

    result = job.result()

    distributions = {}
    for circ in circuits:
        counts = result.get_counts(circ)
        qcounts = counts_to_qorder(counts)
        distributions[circ.name] = arr(qcounts)

    return distributions


print("Running Aer default baseline...")
aer_default = run_aer(aer_default_noise, SEED_DEFAULT)

print("Running Aer no-relaxation cross-validation...")
aer_no_relax = run_aer(aer_no_relax_noise, SEED_NO_RELAX)

rng = np.random.default_rng(BOOT_SEED)

rows = []

for cid in expected_names:
    qt = arr(pred_by_id[cid]["q_tensor_full_calibration_informed"])

    raw_counts = counts_to_qorder(
        raw_by_id[cid]["counts_raw_qiskit_order"]
    )
    ibm = arr(raw_counts)

    ad = aer_default[cid]
    an = aer_no_relax[cid]

    qt_ibm = tvd(qt, ibm)
    ad_ibm = tvd(ad, ibm)
    an_ibm = tvd(an, ibm)

    # Primary implementation cross-check:
    # Q-Tensor versus Aer with thermal relaxation disabled.
    qt_vs_an = tvd(qt, an)

    # Exploratory paired bootstrap:
    # positive delta means Q-Tensor is closer to the observed IBM sample.
    delta_obs = ad_ibm - qt_ibm

    boot_delta = np.empty(BOOT)
    for i in range(BOOT):
        sample = rng.multinomial(4096, ibm) / 4096.0
        boot_delta[i] = tvd(ad, sample) - tvd(qt, sample)

    lo, med, hi = np.percentile(boot_delta, [2.5, 50, 97.5])

    ibm_floor_hi = (
        floor_by_id[cid]
        ["one_sample_floor_under_ibm_empirical"]
        ["p97_5"]
    )

    rows.append({
        "circuit": cid,
        "qtensor_vs_ibm_tvd": qt_ibm,
        "aer_default_vs_ibm_tvd": ad_ibm,
        "aer_no_relax_vs_ibm_tvd": an_ibm,

        "qtensor_vs_aer_no_relax_tvd": qt_vs_an,

        "aer_default_minus_qtensor_delta": delta_obs,
        "paired_bootstrap_delta_95pct": [
            float(lo), float(hi)
        ],
        "paired_bootstrap_delta_median": float(med),

        "ibm_empirical_4096_floor_p97_5": ibm_floor_hi,
        "qtensor_inside_4096_floor": bool(qt_ibm <= ibm_floor_hi),
        "aer_default_inside_4096_floor": bool(ad_ibm <= ibm_floor_hi),
    })

output = {
    "schema": "q-tensor.ibm-aer-baseline.v2",
    "status": "POST_HOC_EXPLORATORY_BASELINE",
    "comparator_construction": (
        "Reduced frozen BackendProperties restricted to sx, x, rz, and cz "
        "gate entries; measurement-family gate_error entries excluded; "
        "asymmetric readout calibration retained separately."
    ),
    "correction_note": (
        "Supersedes the initial exploratory comparator that retained "
        "measurement-family gate_error entries while also enabling separate "
        "readout error."
    ),
    "backend": "ibm_kingston",
    "hardware_job_id": raw["job_id"],
    "aer_simulation_shots": SHOTS,
    "aer_default_seed": SEED_DEFAULT,
    "aer_no_relax_seed": SEED_NO_RELAX,
    "paired_bootstrap_replicates": BOOT,
    "paired_bootstrap_seed": BOOT_SEED,
    "default_model": (
        "Aer backend-properties model with gate error, asymmetric "
        "readout error, and thermal relaxation enabled."
    ),
    "no_relax_model": (
        "Aer backend-properties model with thermal relaxation disabled; "
        "used only as implementation cross-validation against Q-Tensor."
    ),
    "default_model_warnings": default_warnings,
    "circuits": rows,
}

out = EXP / "analysis/AER_BASELINE.json"
text = json.dumps(output, indent=2, sort_keys=True) + "\n"
out.write_text(text)

digest = hashlib.sha256(text.encode()).hexdigest()

(EXP / "analysis/AER_BASELINE.sha256").write_text(
    f"{digest}  AER_BASELINE.json\n"
)

print()
print("=== AER EXPLORATORY BASELINE ===")
print("Aer default construction warnings:", len(default_warnings))
for w in default_warnings:
    print("WARNING:", w)

print()

for r in rows:
    lo, hi = r["paired_bootstrap_delta_95pct"]

    print(
        f"{r['circuit']:12s} "
        f"QT→IBM={r['qtensor_vs_ibm_tvd']:.6f}  "
        f"AerDef→IBM={r['aer_default_vs_ibm_tvd']:.6f}  "
        f"AerNoRel→IBM={r['aer_no_relax_vs_ibm_tvd']:.6f}  "
        f"QT↔AerNoRel={r['qtensor_vs_aer_no_relax_tvd']:.6f}  "
        f"Δ(AerDef-QT)={r['aer_default_minus_qtensor_delta']:+.6f} "
        f"CI=[{lo:+.6f},{hi:+.6f}]  "
        f"floor(QT/Aer)="
        f"{'IN' if r['qtensor_inside_4096_floor'] else 'OUT'}/"
        f"{'IN' if r['aer_default_inside_4096_floor'] else 'OUT'}"
    )

print()
print("SHA256 =", digest)
print("No QPU execution was performed.")
