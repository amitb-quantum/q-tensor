from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/manager/q-tensor")
EXP = ROOT / "experiments/ibm_kingston_20260911"

PRED = EXP / "predictions/FROZEN_PREDICTIONS.json"
RAW = EXP / "hardware/RAW_HARDWARE_RESULTS.json"
PROTOCOL = EXP / "ANALYSIS_PROTOCOL.json"

EXPECTED_PRED_SHA = "40887db2dd269101023bd05435563281eebd6a60de63e0b0a7616162698c4700"
EXPECTED_RAW_SHA = "b875a74cfdf9c8a66ed4ba559ed274f09623ea7c21f465cbad9f23f0d23166e0"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if sha(PRED) != EXPECTED_PRED_SHA:
    raise SystemExit("STOP: frozen prediction checksum mismatch")

if sha(RAW) != EXPECTED_RAW_SHA:
    raise SystemExit("STOP: raw IBM result checksum mismatch")

pred = json.loads(PRED.read_text())
raw = json.loads(RAW.read_text())
protocol = json.loads(PROTOCOL.read_text())

BOOT = int(protocol["uncertainty"]["replicates"])
SEED = int(protocol["uncertainty"]["seed"])

def normalize(d):
    s = float(sum(d.values()))
    return {k: float(v) / s for k, v in d.items()}

def tvd(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)

# Predictions by circuit id.
pred_by_id = {r["id"]: r for r in pred["circuits"]}

rows = []
rng = np.random.default_rng(SEED)

for hw in raw["results"]:
    cid = hw["circuit"]
    p = pred_by_id[cid]

    # Qiskit counts display c3c2c1c0.
    # Q-Tensor frozen convention is q0q1q2q3.
    normalized_counts = {}
    for bits, count in hw["counts_raw_qiskit_order"].items():
        bits = bits.replace(" ", "")
        if len(bits) != 4:
            raise RuntimeError(f"{cid}: unexpected bitstring {bits!r}")
        qorder = bits[::-1]
        normalized_counts[qorder] = normalized_counts.get(qorder, 0) + int(count)

    shots = sum(normalized_counts.values())
    if shots != 4096:
        raise RuntimeError(f"{cid}: expected 4096 shots, found {shots}")

    ibm = normalize(normalized_counts)

    ideal = p["ideal_noiseless"]
    ro = p["ideal_plus_readout_only"]
    full = p["q_tensor_full_calibration_informed"]

    d_ideal = tvd(ideal, ibm)
    d_ro = tvd(ro, ibm)
    d_full = tvd(full, ibm)

    delta = d_ideal - d_full
    control_delta = d_ro - d_full

    # Preregistered nonparametric bootstrap of observed IBM outcomes.
    support = [format(i, "04b") for i in range(16)]
    probs = np.array([ibm.get(k, 0.0) for k in support], dtype=float)
    probs /= probs.sum()

    boot_delta = np.empty(BOOT, dtype=float)
    boot_control = np.empty(BOOT, dtype=float)

    for i in range(BOOT):
        bc = rng.multinomial(shots, probs)
        bd = {k: int(v) for k, v in zip(support, bc) if v}
        bd = normalize(bd)

        boot_delta[i] = tvd(ideal, bd) - tvd(full, bd)
        boot_control[i] = tvd(ro, bd) - tvd(full, bd)

    lo, hi = np.percentile(boot_delta, [2.5, 97.5])
    clo, chi = np.percentile(boot_control, [2.5, 97.5])

    rows.append({
        "circuit": cid,
        "cz_count": p["cz_count"],
        "shots": shots,

        "tvd_ideal_vs_ibm": d_ideal,
        "tvd_readout_only_vs_ibm": d_ro,
        "tvd_qtensor_full_vs_ibm": d_full,

        "delta_ideal_minus_qtensor": delta,
        "delta_95pct_ci": [float(lo), float(hi)],
        "qtensor_beats_ideal": bool(delta > 0),
        "strong_predictive_win": bool(lo > 0),

        "delta_readout_only_minus_qtensor": control_delta,
        "control_delta_95pct_ci": [float(clo), float(chi)],
        "gate_noise_adds_value": bool(control_delta > 0),
        "strong_gate_noise_value": bool(clo > 0),
    })

wins = sum(r["qtensor_beats_ideal"] for r in rows)
strong = sum(r["strong_predictive_win"] for r in rows)
control_wins = sum(r["gate_noise_adds_value"] for r in rows)

primary_pass = wins >= 3

result = {
    "schema": "q-tensor.ibm-preregistered-analysis.v1",
    "backend": "ibm_kingston",
    "job_id": raw["job_id"],
    "prediction_sha256": EXPECTED_PRED_SHA,
    "raw_hardware_sha256": EXPECTED_RAW_SHA,
    "bootstrap_replicates": BOOT,
    "bootstrap_seed": SEED,
    "circuits": rows,
    "summary": {
        "qtensor_beats_ideal_count": wins,
        "strong_predictive_win_count": strong,
        "gate_noise_beats_readout_only_count": control_wins,
        "primary_success_rule": "Q-Tensor beats ideal on at least 3 of 4 circuits",
        "primary_result": "PASS" if primary_pass else "FAIL",
    },
}

outdir = EXP / "analysis"
outdir.mkdir(exist_ok=True)

outfile = outdir / "PREREGISTERED_ANALYSIS.json"
text = json.dumps(result, indent=2, sort_keys=True) + "\n"
outfile.write_text(text)

digest = hashlib.sha256(text.encode()).hexdigest()
(outdir / "PREREGISTERED_ANALYSIS.sha256").write_text(
    f"{digest}  PREREGISTERED_ANALYSIS.json\n"
)

print("\n=== PREREGISTERED IBM HARDWARE TEST ===")
for r in rows:
    print(
        f"{r['circuit']:12s} CZ={r['cz_count']:2d} "
        f"ideal→IBM={r['tvd_ideal_vs_ibm']:.6f} "
        f"QTensor→IBM={r['tvd_qtensor_full_vs_ibm']:.6f} "
        f"delta={r['delta_ideal_minus_qtensor']:+.6f} "
        f"95%CI=[{r['delta_95pct_ci'][0]:+.6f},"
        f"{r['delta_95pct_ci'][1]:+.6f}] "
        f"{'STRONG_WIN' if r['strong_predictive_win'] else ('WIN' if r['qtensor_beats_ideal'] else 'LOSS')}"
    )

print()
print("Q-Tensor beats ideal:", wins, "/ 4")
print("Strong predictive wins:", strong, "/ 4")
print("Gate-noise model beats readout-only:", control_wins, "/ 4")
print("PRIMARY RESULT =", result["summary"]["primary_result"])
print("ANALYSIS SHA256 =", digest)
