from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/manager/q-tensor")
EXP = ROOT / "experiments/ibm_kingston_20260911"

PRED = EXP / "predictions/FROZEN_PREDICTIONS.json"
RAW = EXP / "hardware/RAW_HARDWARE_RESULTS.json"

SHOTS = 4096
REPS = 10000
SEED = 2026091121

support = [format(i, "04b") for i in range(16)]


def normalize(d):
    s = float(sum(d.values()))
    return {k: float(v) / s for k, v in d.items()}


def arr(d):
    x = np.array([float(d.get(k, 0.0)) for k in support], dtype=float)
    return x / x.sum()


def tvd(a, b):
    return 0.5 * np.abs(a - b).sum()


def interval(x):
    lo, med, hi = np.percentile(x, [2.5, 50.0, 97.5])
    return {
        "p2_5": float(lo),
        "median": float(med),
        "p97_5": float(hi),
    }


pred = json.loads(PRED.read_text())
raw = json.loads(RAW.read_text())

pred_by_id = {x["id"]: x for x in pred["circuits"]}
raw_by_id = {x["circuit"]: x for x in raw["results"]}

rng = np.random.default_rng(SEED)
rows = []

for cid in ["QTIBM_CZ03", "QTIBM_CZ06", "QTIBM_CZ09", "QTIBM_CZ12"]:
    q = arr(pred_by_id[cid]["q_tensor_full_calibration_informed"])

    # IBM stores Qiskit display order c3c2c1c0.
    counts = raw_by_id[cid]["counts_raw_qiskit_order"]
    ibm_d = {}
    for bits, n in counts.items():
        qb = bits.replace(" ", "")[::-1]
        ibm_d[qb] = ibm_d.get(qb, 0) + int(n)

    ibm = arr(normalize(ibm_d))
    observed = tvd(q, ibm)

    # Primary resolution floor:
    # one finite 4096-shot observation compared with its true distribution.
    q_one = np.empty(REPS)
    ibm_one = np.empty(REPS)

    # Supplementary two-independent-sample floor.
    q_two = np.empty(REPS)
    ibm_two = np.empty(REPS)

    for i in range(REPS):
        qa = rng.multinomial(SHOTS, q) / SHOTS
        ia = rng.multinomial(SHOTS, ibm) / SHOTS

        q_one[i] = tvd(qa, q)
        ibm_one[i] = tvd(ia, ibm)

        qb = rng.multinomial(SHOTS, q) / SHOTS
        ib = rng.multinomial(SHOTS, ibm) / SHOTS

        q_two[i] = tvd(qa, qb)
        ibm_two[i] = tvd(ia, ib)

    q_int = interval(q_one)
    i_int = interval(ibm_one)

    rows.append({
        "circuit": cid,
        "shots": SHOTS,
        "observed_qtensor_vs_ibm_tvd": float(observed),

        "one_sample_floor_under_qtensor": q_int,
        "one_sample_floor_under_ibm_empirical": i_int,

        "two_sample_floor_under_qtensor": interval(q_two),
        "two_sample_floor_under_ibm_empirical": interval(ibm_two),

        "observed_above_qtensor_95pct_floor": bool(observed > q_int["p97_5"]),
        "observed_above_ibm_95pct_floor": bool(observed > i_int["p97_5"]),

        "qtensor_floor_exceedance_fraction":
            float(np.mean(q_one >= observed)),
        "ibm_floor_exceedance_fraction":
            float(np.mean(ibm_one >= observed)),
    })

result = {
    "schema": "q-tensor.ibm-shot-noise-floor.v1",
    "status": "POST_HOC_RESOLUTION_DIAGNOSTIC",
    "hardware_job_id": raw["job_id"],
    "shots": SHOTS,
    "replicates": REPS,
    "seed": SEED,
    "interpretation_rule":
        "If observed Q-Tensor-to-IBM TVD lies within the 95% one-sample "
        "floor under the relevant distribution, the experiment cannot "
        "resolve that residual from ordinary 4096-shot sampling noise.",
    "circuits": rows,
}

out = EXP / "analysis/SHOT_NOISE_FLOOR.json"
text = json.dumps(result, indent=2, sort_keys=True) + "\n"
out.write_text(text)

digest = hashlib.sha256(text.encode()).hexdigest()
(EXP / "analysis/SHOT_NOISE_FLOOR.sha256").write_text(
    f"{digest}  SHOT_NOISE_FLOOR.json\n"
)

print("\n=== IBM 4096-SHOT RESOLUTION FLOOR ===")
for r in rows:
    qf = r["one_sample_floor_under_qtensor"]
    ibf = r["one_sample_floor_under_ibm_empirical"]

    print(
        f"{r['circuit']:12s} "
        f"observed={r['observed_qtensor_vs_ibm_tvd']:.6f}  "
        f"QTensor-floor95=[{qf['p2_5']:.6f},{qf['p97_5']:.6f}]  "
        f"IBM-floor95=[{ibf['p2_5']:.6f},{ibf['p97_5']:.6f}]  "
        f"above_Q={'YES' if r['observed_above_qtensor_95pct_floor'] else 'NO'} "
        f"above_IBM={'YES' if r['observed_above_ibm_95pct_floor'] else 'NO'}"
    )

print()
print("SHA256 =", digest)
print("This is a post-hoc resolution diagnostic; it does not alter the preregistered result.")
