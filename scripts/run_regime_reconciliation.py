#!/usr/bin/env python3
"""Reproduce one official paper regime and reconcile it with Q-Tensor results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from q_tensor.schema import validate_result
from run_fair_benchmark import SmiMonitor
from run_statistical_validation import UPSTREAM_COMMIT, git_head, gpu_properties, package_versions


OFFICIAL_QUBITS = 50
OFFICIAL_GATES = 200
OFFICIAL_CIRCUIT_ID = 0
OFFICIAL_BATCH_QUBITS = "10,10,2,28"
OFFICIAL_BATCH_SHOTS = "1,1,1,100"
OFFICIAL_PTSBE_HYPERSAMPLES = 100
OFFICIAL_TRAJECTORIES = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def circuit_summary(stim_path: Path) -> dict[str, Any]:
    coherent = []
    one_noise = []
    two_noise = []
    for raw in stim_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("(", 1)[0].split()[0]
        if name in {"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"}:
            one_noise.append(line)
        elif name == "DEPOLARIZE2":
            two_noise.append(line)
        else:
            coherent.append(line)
    two_qubit = [
        line
        for line in coherent
        if line.split("(", 1)[0].split()[0] in {"CX", "CY", "CZ", "CH", "CRX"}
    ]
    separations = []
    for line in two_qubit:
        operands = [int(value) for value in re.findall(r"\d+", line.split(")", 1)[-1])]
        if len(operands) >= 2:
            separations.append(abs(operands[-1] - operands[-2]))
    probabilities = [
        float(match.group(1))
        for line in one_noise + two_noise
        if (match := re.search(r"\(([^)]+)\)", line))
    ]
    return {
        "coherent_gate_count": len(coherent),
        "single_qubit_gate_count": len(coherent) - len(two_qubit),
        "two_qubit_gate_count": len(two_qubit),
        "single_qubit_noise_sites": len(one_noise),
        "two_qubit_noise_sites": len(two_noise),
        "noise_probability_min": min(probabilities),
        "noise_probability_max": max(probabilities),
        "two_qubit_label_separation_min": min(separations),
        "two_qubit_label_separation_max": max(separations),
        "two_qubit_label_separation_median": sorted(separations)[len(separations) // 2],
    }


def qpy_summary(qpy_path: Path) -> dict[str, Any]:
    """Return structural metadata from the exact retained Qiskit artifact."""
    import qiskit.qpy

    with qpy_path.open("rb") as handle:
        circuit = qiskit.qpy.load(handle)[0]
    return {
        "nqubits": circuit.num_qubits,
        "coherent_gate_count": circuit.size(),
        "depth": circuit.depth(),
        "operation_counts": dict(circuit.count_ops()),
    }


def retained_circuit_block(text: str, circuit_id: int) -> str:
    marker = f"RUN 1/1 | circuit_id={circuit_id}:"
    start = text.index(marker)
    next_match = re.search(r"\nRUN 1/1 \| circuit_id=\d+:", text[start + len(marker) :])
    end = len(text) if next_match is None else start + len(marker) + next_match.start()
    return text[start:end]


def retained_h100_metrics(upstream: Path) -> dict[str, Any]:
    base = upstream / "scaling/data_collection/figure_03_data_collection_speedup"
    pts_text = (base / "50q_200g_100hs_10nfbs_28fbs_ptsbe.txt").read_text(encoding="utf-8")
    cudaq_text = (base / "50q_200g_100hs_cudaq.txt").read_text(encoding="utf-8")
    pts = retained_circuit_block(pts_text, OFFICIAL_CIRCUIT_ID)
    cudaq = retained_circuit_block(cudaq_text, OFFICIAL_CIRCUIT_ID)

    def number(pattern: str, value: str) -> float:
        match = re.search(pattern, value)
        if match is None:
            raise ValueError(f"retained metric not found: {pattern}")
        return float(match.group(1))

    ptsbe_seconds = number(r"PTSBE total:\s+([\d.]+)s", pts)
    ptsbe_records = int(number(r"Total number of overall PTSBE shots collected:\s+(\d+)", pts))
    cudaq_seconds = number(r"Sample time:\s+([\d.]+)s", cudaq)
    throughput_advantage = (ptsbe_records / ptsbe_seconds) / (1.0 / cudaq_seconds)
    return {
        "hardware": "NVIDIA H100 80GB HBM3",
        "circuit_id": OFFICIAL_CIRCUIT_ID,
        "ptsbe_contraction_loop_seconds": ptsbe_seconds,
        "ptsbe_unique_records": ptsbe_records,
        "ptsbe_unique_records_per_second": ptsbe_records / ptsbe_seconds,
        "ptsbe_num_contractions": int(number(r"num_contractions:\s+(\d+)", pts)),
        "ptsbe_path_build_seconds": number(r"Build expr/operands \(4 batches\):\s+([\d.]+)s", pts),
        "cudaq_sample_seconds_one_shot": cudaq_seconds,
        "official_metric_throughput_advantage": throughput_advantage,
        "source": "retained upstream Figure 3 public logs",
    }


def common_command(upstream: Path, workdir: Path, output: Path, hypersamples: int) -> list[str]:
    return [
        sys.executable,
        str(upstream / "scaling/scaling_comparison_avg.py"),
        "--num_circuits",
        "1",
        "--nnoise_samples",
        str(OFFICIAL_TRAJECTORIES),
        "--cudaq_nshots",
        "1",
        "--smart-opt-off",
        "both",
        "--verbose_level",
        "3",
        "--unique_sampling",
        "--take_all_final",
        "--batch_shots_policy",
        "custom",
        "--batch_qubit_policy",
        "custom",
        "--cudaq_timeout",
        "4500",
        "--ptsbe_sample_timeout",
        "3600",
        "--nqubits",
        str(OFFICIAL_QUBITS),
        "--ngates",
        str(OFFICIAL_GATES),
        "--num_hyper_samples",
        str(hypersamples),
        "--shots_per_batch",
        OFFICIAL_BATCH_SHOTS,
        "--qubits_per_batch",
        OFFICIAL_BATCH_QUBITS,
        "--circuit_dir",
        str(upstream / "scaling/data_collection/circuits/50q_200g"),
        "--circuit_id",
        str(OFFICIAL_CIRCUIT_ID),
        "--work_dir",
        str(workdir),
        "--json_output",
        str(output),
    ]


def run_stage(label: str, command: list[str], output: Path, timeout: float) -> dict[str, Any]:
    upstream_root = str(Path(command[1]).resolve().parents[1])
    python_path = upstream_root + os.pathsep + os.environ.get("PYTHONPATH", "")
    executable_path = str(Path(sys.executable).resolve().parent)
    path = executable_path + os.pathsep + os.environ.get("PATH", "")
    monitor = SmiMonitor()
    monitor.start()
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": python_path,
                "PATH": path,
            },
        )
    finally:
        elapsed = time.perf_counter() - started
        telemetry = monitor.stop()
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"{label} failed (returncode={completed.returncode}): "
            f"{completed.stdout[-3000:]}{completed.stderr[-3000:]}"
        )
    return {
        "label": label,
        "command": command,
        "subprocess_wall_seconds": elapsed,
        "telemetry": telemetry,
        "upstream_result": json.loads(output.read_text(encoding="utf-8")),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def reduction(ptsbe: dict[str, Any], cudaq_artifact: dict[str, Any], cudaq_text: dict[str, Any]) -> dict[str, Any]:
    pts = ptsbe["upstream_result"]["ptsbe"]
    records = int(pts["num_total_shots_mean"])
    contraction_seconds = float(pts["time_contraction_loop_mean"])
    path_seconds = float(pts["time_build_expr_operands_mean"])
    cold_seconds = float(pts["time_execution_mean"])

    individual_pts = ptsbe["upstream_result"]["individual_results"][0]["ptsbe"]

    def comparison(cudaq_stage: dict[str, Any]) -> dict[str, float]:
        cudaq = cudaq_stage["upstream_result"]["cudaq"]
        cudaq_sample = float(cudaq["time_sample_mean"])
        return {
            "cudaq_sample_seconds_one_shot": cudaq_sample,
            "raw_runtime_speedup_cudaq_over_ptsbe": cudaq_sample / contraction_seconds,
            "output_count_ratio_ptsbe_over_cudaq": float(records),
            "ptsbe_unique_records_per_contraction_second": records / contraction_seconds,
            "official_metric_throughput_advantage": (records / contraction_seconds) / (1.0 / cudaq_sample),
            "path_inclusive_ptsbe_execution_throughput_advantage": (
                (records / cold_seconds) / (1.0 / float(cudaq["time_total_mean"]))
            ),
            "full_subprocess_wall_throughput_advantage": (
                (records / float(ptsbe["subprocess_wall_seconds"]))
                / (1.0 / float(cudaq_stage["subprocess_wall_seconds"]))
            ),
        }

    batches = [10, 10, 2, 28]
    batch_profile = []
    for index, batch_qubits in enumerate(batches):
        batch_profile.append(
            {
                "batch": index,
                "qubits": batch_qubits,
                "population_entries": 2**batch_qubits,
                "contractions": int(pts[f"num_contractions_batch_{index}_mean"]),
                "gpu_contraction_seconds": float(pts[f"gpu_contract_batch_{index}_mean"]),
            }
        )
    return {
        "ptsbe_unique_records": records,
        "ptsbe_path_build_seconds": path_seconds,
        "ptsbe_contraction_loop_seconds": contraction_seconds,
        "ptsbe_path_inclusive_execution_seconds": cold_seconds,
        "ptsbe_stim_to_pts_seconds": float(individual_pts["time_stim_to_pts"]),
        "ptsbe_trajectory_sampling_seconds": float(individual_pts["time_pts_sampling"]),
        "ptsbe_gpu_contraction_seconds": float(pts["time_contract_gpu_total_mean"]),
        "ptsbe_apply_errors_seconds": float(pts["time_apply_errors_total_mean"]),
        "ptsbe_loop_minus_gpu_and_error_seconds": (
            contraction_seconds
            - float(pts["time_contract_gpu_total_mean"])
            - float(pts["time_apply_errors_total_mean"])
        ),
        "ptsbe_contractions": int(pts["num_contractions_mean"]),
        "path_sets_built": len(batches),
        "path_reuse_per_batch": OFFICIAL_TRAJECTORIES,
        "final_batch_fraction_of_gpu_contraction_time": (
            float(pts["gpu_contract_batch_3_mean"]) / float(pts["time_contract_gpu_total_mean"])
        ),
        "mean_unique_records_per_final_batch_contraction": records / OFFICIAL_TRAJECTORIES,
        "mean_final_population_occupancy": records / (OFFICIAL_TRAJECTORIES * 2**28),
        "batch_profile": batch_profile,
        "public_artifact_cudaq_100_hypersamples": comparison(cudaq_artifact),
        "manuscript_text_cudaq_1_hypersample_control": comparison(cudaq_text),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--q-tensor-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    upstream = args.upstream.resolve()
    if git_head(upstream) != UPSTREAM_COMMIT:
        raise SystemExit("upstream revision mismatch")

    circuits = upstream / "scaling/data_collection/circuits/50q_200g"
    qpy = circuits / "circuit_0.qpy"
    stim = circuits / "circuit_0.stim"
    baseline = json.loads(args.q_tensor_baseline.read_text(encoding="utf-8"))
    if not baseline["measurements"]["all_timing_points_passed_validation"]:
        raise SystemExit("Q-Tensor comparison baseline did not pass validation")

    with tempfile.TemporaryDirectory(prefix="q-tensor-regime-reconciliation-") as temporary:
        temp = Path(temporary)
        pts_output = temp / "ptsbe.json"
        pts_command = common_command(upstream, temp / "ptsbe-work", pts_output, OFFICIAL_PTSBE_HYPERSAMPLES)
        pts_command.append("--skip_cudaq")
        ptsbe = run_stage("official_ptsbe", pts_command, pts_output, args.timeout)

        cudaq100_output = temp / "cudaq-hs100.json"
        cudaq100_command = common_command(
            upstream, temp / "cudaq-hs100-work", cudaq100_output, OFFICIAL_PTSBE_HYPERSAMPLES
        )
        cudaq100_command.append("--skip_ptsbe")
        cudaq100 = run_stage("public_artifact_cudaq_hs100", cudaq100_command, cudaq100_output, args.timeout)

        cudaq1_output = temp / "cudaq-hs1.json"
        cudaq1_command = common_command(upstream, temp / "cudaq-hs1-work", cudaq1_output, 1)
        cudaq1_command.append("--skip_ptsbe")
        cudaq1 = run_stage("manuscript_text_cudaq_hs1", cudaq1_command, cudaq1_output, args.timeout)

    record = {
        "schema_version": 1,
        "experiment_id": "official-50q-200g-regime-reconciliation-v1",
        "category": "q_tensor_original",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, str(Path(__file__)), *sys.argv[1:]],
        "git_commit": git_head(root),
        "upstream_commit": UPSTREAM_COMMIT,
        "hardware": gpu_properties(),
        "software": {"python": platform.python_version(), "packages": package_versions()},
        "parameters": {
            "paper_figure": "Figure 3 non-proportional data-collection speedup",
            "official_workload": {
                "circuit_family": "public Figure 3 noisy random-circuit data-collection artifact",
                "topology": (
                    "exact retained gate graph; two-qubit operand label separations 1-44 "
                    "(median 11), not assumed nearest-neighbor geometry"
                ),
                "nqubits": OFFICIAL_QUBITS,
                "ngates": OFFICIAL_GATES,
                "circuit_id": OFFICIAL_CIRCUIT_ID,
                "circuit_qpy_sha256": sha256(qpy),
                "circuit_stim_sha256": sha256(stim),
                "qpy_summary": qpy_summary(qpy),
                "circuit_summary": circuit_summary(stim),
                "noise_trajectories": OFFICIAL_TRAJECTORIES,
                "trajectory_seed": None,
                "trajectory_seed_note": "the public upstream Figure 3 driver exposes no seed",
                "precision": "complex128",
                "batch_qubits": [10, 10, 2, 28],
                "batch_shots": [1, 1, 1, 100],
                "population_sampling": "non-proportional unique prefixes; exhaustive nonzero final batch",
                "ptsbe_hypersamples": OFFICIAL_PTSBE_HYPERSAMPLES,
                "cudaq_backend": "tensornet",
                "public_artifact_cudaq_hypersamples": 100,
                "manuscript_stated_cudaq_hypersamples": 1,
                "contraction_optimizer": {
                    "smart_option": 0,
                    "cache_reuse_nruns": 1000000,
                },
            },
            "metric": "PTSBE unique labeled records/contraction-loop second divided by CUDA-Q shots/sample second",
            "statistical_validation": {
                "applicable": False,
                "reason": "official Figure 3 exhaustively harvests a non-proportional final-batch dataset, not IID shots",
            },
        },
        "measurements": {
            "retained_h100_circuit_0": retained_h100_metrics(upstream),
            "rtx_5090": {
                "reduction": reduction(ptsbe, cudaq100, cudaq1),
                "stages": [ptsbe, cudaq100, cudaq1],
            },
            "validated_q_tensor_baseline": {
                "source": str(args.q_tensor_baseline),
                "experiment_id": baseline["experiment_id"],
                "all_points_passed_validation": True,
                "parameters": baseline["parameters"],
                "crossover": baseline["measurements"]["crossover"],
                "points": [
                    {
                        "nqubits": point["nqubits"],
                        "entangling_depth": point["entangling_depth"],
                        "coherent_gate_count": point["coherent_gate_count"],
                        "speedup_baseline_over_tn": point["steady_state_speedup_baseline_over_tn"],
                    }
                    for point in baseline["measurements"]["points"]
                ],
            },
        },
    }
    validate_result(record)
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
