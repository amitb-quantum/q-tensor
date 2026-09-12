from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

ROOT = Path("/home/manager/q-tensor")
EXP = ROOT / "experiments/ibm_kingston_20260911"

SHOTS = 4096
REPS = 100_000
SEED = 2026091141

# Reuse the independently constructed exact-channel implementation.
# This executes its validation first, then exposes its functions/data.
ns = runpy.run_path(
    str(ROOT / "scripts/validate_ibm_noise_model_exact.py")
)

family = ns["family"]
evolve = ns["evolve"]
apply_readout = ns["apply_readout"]

raw = json.loads(
    (EXP / "hardware/RAW_HARDWARE_RESULTS.json").read_text()
)
pred = json.loads(
    (EXP / "predictions/FROZEN_PREDICTIONS.json").read_text()
)

raw_by_id = {r["circuit"]: r for r in raw["results"]}
pred_by_id = {r["id"]: r for r in pred["circuits"]}

SUPPORT = [format(i, "04b")[::-1] for i in range(16)]


def qorder_counts(counts):
    out = {}
    for bits, n in counts.items():
        bits = bits.replace(" ", "")
        out[bits[::-1]] = out.get(bits[::-1], 0) + int(n)
    return out


def arr(d):
    x = np.array([float(d.get(k, 0.0)) for k in SUPPORT])
    return x / x.sum()


def tvd(a, b):
    return float(0.5 * np.abs(a - b).sum())


rng = np.random.default_rng(SEED)
rows = []

print()
print("=== EXACT-MODEL 4096-SHOT RESOLUTION ===")

for spec in family:
    cid = spec["id"]

    exact_dict = apply_readout(evolve(spec, noisy=True))
    exact = arr(exact_dict)

    ibm_counts = qorder_counts(
        raw_by_id[cid]["counts_raw_qiskit_order"]
    )
    ibm = arr(ibm_counts)

    qt = arr(
        pred_by_id[cid]["q_tensor_full_calibration_informed"]
    )

    observed = tvd(exact, ibm)
    qt_exact = tvd(qt, exact)

    draws = rng.multinomial(
        SHOTS,
        exact,
        size=REPS,
    )

    simulated = 0.5 * np.abs(
        draws / SHOTS - exact
    ).sum(axis=1)

    p2_5, median, p97_5 = np.percentile(
        simulated,
        [2.5, 50, 97.5],
    )

    percentile = 100.0 * np.mean(simulated <= observed)
    upper_tail = np.mean(simulated >= observed)

    rows.append({
        "circuit": cid,
        "shots": SHOTS,
        "exact_model_vs_ibm_tvd": observed,
        "frozen_qtensor_vs_exact_tvd": qt_exact,
        "exact_null": {
            "p2_5": float(p2_5),
            "median": float(median),
            "p97_5": float(p97_5),
        },
        "observed_percentile_under_exact_null": float(percentile),
        "upper_tail_fraction": float(upper_tail),
        "inside_95pct_exact_null": bool(
            observed <= p97_5 and observed >= p2_5
        ),
    })

    print(
        f"{cid:12s} "
        f"Exact→IBM={observed:.6f}  "
        f"QT→Exact={qt_exact:.6f}  "
        f"null95=[{p2_5:.6f},{p97_5:.6f}]  "
        f"percentile={percentile:6.2f}%  "
        f"upper-tail={upper_tail:.4f}"
    )

out = EXP / "analysis/EXACT_SHOT_NOISE_FLOOR.json"

payload = {
    "schema": "q-tensor.ibm-exact-shot-noise-floor.v1",
    "status": "POST_HOC_EXACT_MODEL_RESOLUTION_DIAGNOSTIC",
    "hardware_job_id": "dai8c4hhvn6c73cstebg",
    "shots": SHOTS,
    "replicates": REPS,
    "seed": SEED,
    "generator": (
        "Independent exact 4-qubit density-matrix implementation of the "
        "specified Q-Tensor Pauli-plus-readout channel."
    ),
    "circuits": rows,
}

out.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)

digest = hashlib.sha256(out.read_bytes()).hexdigest()
sha = out.with_suffix(".sha256")
sha.write_text(f"{digest}  {out.name}\n")

print()
print("SHA256 =", digest)
print("No QPU execution was performed.")
