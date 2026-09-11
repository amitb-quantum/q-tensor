#!/usr/bin/env python3
"""Run a validation-gated qubit/depth crossover pilot on one trajectory stream."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from q_tensor.complexity import complexity_case
from q_tensor.performance import speedup, throughput
from q_tensor.schema import validate_result
from q_tensor.stats import total_variation_distance
from q_tensor.validation import (
    conditioned_distribution,
    exact_branch_distribution,
    group_trajectories,
    sample_trajectories,
    z_expectation,
)
from run_fair_benchmark import (
    CUDAQ_OPTION,
    CUDAQ_TARGET,
    TN_DTYPE,
    monitored,
    prepare_cudaq,
    sample_cudaq_kernels,
    summarize_cudaq,
    summarize_tn,
    timed_tn,
)
from run_statistical_validation import (
    UPSTREAM_COMMIT,
    backend_comparison,
    case_definition,
    git_head,
    gpu_properties,
    install_cuquantum_gate_order_workaround,
    package_versions,
    two_backend_comparison,
    write_qiskit_circuit,
)

APPROACH_RATIO = 0.8


def point_key(point: dict[str, Any]) -> tuple[int, int]:
    return int(point["nqubits"]), int(point["entangling_depth"])


def crossover_summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [point for point in sorted(points, key=point_key) if point["validation"]["passed"]]
    crossover = next(
        (point for point in valid if point["steady_state_speedup_baseline_over_tn"] >= 1.0),
        None,
    )
    approach = next(
        (point for point in valid if point["steady_state_speedup_baseline_over_tn"] >= APPROACH_RATIO),
        None,
    )
    best = max(valid, key=lambda point: point["steady_state_speedup_baseline_over_tn"], default=None)

    def identify(point):
        if point is None:
            return None
        return {
            "nqubits": point["nqubits"],
            "entangling_depth": point["entangling_depth"],
            "speedup_baseline_over_tn": point["steady_state_speedup_baseline_over_tn"],
        }

    return {
        "approach_definition": f"CUDA-Q/TN steady runtime ratio >= {APPROACH_RATIO}",
        "first_approach_point": identify(approach),
        "first_crossover_point": identify(crossover),
        "best_measured_point": identify(best),
    }


def run_worker(args) -> dict[str, Any]:
    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream))
    install_cuquantum_gate_order_workaround()
    if git_head(upstream) != UPSTREAM_COMMIT:
        raise RuntimeError("upstream revision mismatch")

    case = complexity_case(args.worker_qubits, args.worker_depth)
    draws = sample_trajectories(case, args.trajectory_count, seed=case.seed)
    grouped = group_trajectories(draws)
    conditioned = conditioned_distribution(case, grouped)
    channel = exact_branch_distribution(case)
    seed = case.seed * 100000 + case.nqubits * 1000 + args.worker_depth

    with tempfile.TemporaryDirectory(prefix="q-tensor-complexity-worker-") as temporary:
        workdir = Path(temporary)
        input_started = time.perf_counter()
        write_qiskit_circuit(case, workdir / f"{case.case_id}-reversed.qpy", reverse_qubits=True)
        input_preparation_seconds = time.perf_counter() - input_started

        _, tn_cold_first_use = timed_tn(case, grouped, args.shots_per_trajectory, seed - 1000, workdir)
        prepared, cudaq_setup = prepare_cudaq(case, grouped, seed + 20000)
        cudaq_setup["context_note"] = "TN initialized the process CUDA context before this measurement"
        tn_counts_for_validation = None
        cudaq_counts_for_validation = None
        tn_repeats = []
        cudaq_sample_seconds = []
        cudaq_postprocess_seconds = []

        for repeat in range(args.repeats):
            order = ("tn", "cudaq") if repeat % 2 == 0 else ("cudaq", "tn")
            for backend in order:
                if backend == "tn":
                    counts, timing = timed_tn(
                        case, grouped, args.shots_per_trajectory, seed + repeat * 1000, workdir
                    )
                    if tn_counts_for_validation is None:
                        tn_counts_for_validation = counts
                    tn_repeats.append(timing)
                else:
                    counts, sample_seconds, postprocess_seconds = sample_cudaq_kernels(
                        prepared,
                        case.nqubits,
                        args.shots_per_trajectory,
                        seed + 30000 + repeat * 1000,
                    )
                    if cudaq_counts_for_validation is None:
                        cudaq_counts_for_validation = counts
                    cudaq_sample_seconds.append(sample_seconds)
                    cudaq_postprocess_seconds.append(postprocess_seconds)

        assert tn_counts_for_validation is not None and cudaq_counts_for_validation is not None
        tn_validation = backend_comparison(
            case,
            grouped,
            args.shots_per_trajectory,
            tn_counts_for_validation,
            conditioned,
            args.null_trials,
            seed + 40000,
        )
        cudaq_validation = backend_comparison(
            case,
            grouped,
            args.shots_per_trajectory,
            cudaq_counts_for_validation,
            conditioned,
            args.null_trials,
            seed + 50000,
        )
        pair_validation = two_backend_comparison(
            case,
            grouped,
            args.shots_per_trajectory,
            tn_counts_for_validation,
            cudaq_counts_for_validation,
            args.null_trials,
            seed + 60000,
        )
        valid = tn_validation["passed"] and cudaq_validation["passed"] and pair_validation["passed"]

        telemetry = None
        if args.telemetry:
            telemetry = {
                "tn": monitored(
                    lambda: timed_tn(
                        case, grouped, args.shots_per_trajectory, seed + 70000, workdir
                    )
                ),
                "cudaq": monitored(
                    lambda: [
                        sample_cudaq_kernels(
                            prepared,
                            case.nqubits,
                            args.shots_per_trajectory,
                            seed + 80000 + repeat * 1000,
                        )
                        for repeat in range(3)
                    ]
                ),
            }

    tn_summary = summarize_tn(tn_repeats)
    tn_summary["cold_first_use"] = tn_cold_first_use
    cudaq_summary = summarize_cudaq(cudaq_setup, cudaq_sample_seconds, cudaq_postprocess_seconds)
    effective_shots = args.trajectory_count * args.shots_per_trajectory
    tn_steady = tn_summary["components"]["steady_total_seconds"]["median_seconds"]
    cudaq_steady = cudaq_summary["steady_total"]["median_seconds"]
    components = {
        name: tn_summary["components"][name]["median_seconds"]
        for name in (
            "gpu_contraction_seconds",
            "apply_errors_seconds",
            "host_sampling_seconds",
            "host_postprocess_seconds",
        )
    }
    return {
        "case": case_definition(case),
        "nqubits": case.nqubits,
        "entangling_depth": args.worker_depth,
        "coherent_gate_count": len(case.gates),
        "trajectories": args.trajectory_count,
        "unique_trajectories": len(grouped),
        "trajectory_multiplicities": {str(trajectory): multiplicity for trajectory, multiplicity in grouped},
        "shots_per_trajectory": args.shots_per_trajectory,
        "total_effective_shots": effective_shots,
        "common_input_preparation_seconds": input_preparation_seconds,
        "validation": {
            "passed": valid,
            "reference": "exact conditioned complex128 Kraus-branch statevector mixture",
            "tn_vs_conditioned": tn_validation,
            "cudaq_vs_conditioned": cudaq_validation,
            "tn_vs_cudaq": pair_validation,
            "trajectory_vs_exact_channel_tvd": total_variation_distance(conditioned, channel),
            "trajectory_vs_exact_channel_z_expectation_error": abs(
                z_expectation(conditioned, case.observable_qubits)
                - z_expectation(channel, case.observable_qubits)
            ),
        },
        "tn": tn_summary,
        "cudaq": cudaq_summary,
        "tn_dominant_measured_component": max(components, key=components.get),
        "tn_measured_component_medians_seconds": components,
        "steady_state_speedup_baseline_over_tn": speedup(cudaq_steady, tn_steady),
        "tn_effective_shots_per_second": throughput(effective_shots, tn_steady),
        "cudaq_effective_shots_per_second": throughput(effective_shots, cudaq_steady),
        "telemetry": telemetry,
    }


def worker_main(args) -> int:
    result = run_worker(args)
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["validation"]["passed"] else 1


def parent_main(args) -> int:
    root = Path(__file__).resolve().parents[1]
    upstream = args.upstream.resolve()
    if git_head(upstream) != UPSTREAM_COMMIT:
        raise SystemExit("upstream revision mismatch")
    requested = [(nqubits, depth) for nqubits in args.qubits for depth in args.depths]
    points = []
    with tempfile.TemporaryDirectory(prefix="q-tensor-complexity-parent-") as temporary:
        for nqubits, depth in requested:
            worker_output = Path(temporary) / f"worker-{nqubits}q-d{depth}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-qubits",
                str(nqubits),
                "--worker-depth",
                str(depth),
                "--worker-output",
                str(worker_output),
                "--upstream",
                str(upstream),
                "--trajectory-count",
                str(args.trajectory_count),
                "--shots-per-trajectory",
                str(args.shots_per_trajectory),
                "--repeats",
                str(args.repeats),
                "--null-trials",
                str(args.null_trials),
            ]
            if (nqubits, depth) == requested[-1]:
                command.append("--telemetry")
            completed = subprocess.run(
                command,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            if not worker_output.is_file():
                raise RuntimeError(
                    f"benchmark worker {nqubits}q/d{depth} failed before writing output: "
                    f"{completed.stdout[-2000:]}{completed.stderr[-2000:]}"
                )
            point = json.loads(worker_output.read_text(encoding="utf-8"))
            point["worker_process_returncode"] = completed.returncode
            point["worker_stdout_tail"] = completed.stdout[-1000:]
            point["worker_stderr_tail"] = completed.stderr[-1000:]
            points.append(point)

    all_valid = all(
        point["validation"]["passed"] and point["worker_process_returncode"] == 0
        for point in points
    )
    record = {
        "schema_version": 1,
        "experiment_id": "validated-complexity-crossover-pilot-v1",
        "category": "q_tensor_original",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, str(Path(__file__)), *sys.argv[1:]],
        "git_commit": git_head(root),
        "upstream_commit": UPSTREAM_COMMIT,
        "hardware": gpu_properties(),
        "software": {"python": platform.python_version(), "packages": package_versions()},
        "parameters": {
            "qubits": args.qubits,
            "entangling_depths": args.depths,
            "trajectory_count": args.trajectory_count,
            "shots_per_trajectory": args.shots_per_trajectory,
            "total_effective_shots": args.trajectory_count * args.shots_per_trajectory,
            "repeats": args.repeats,
            "null_trials_per_validation_check": args.null_trials,
            "tn_dtype": TN_DTYPE,
            "cudaq_target": f"{CUDAQ_TARGET} {CUDAQ_OPTION}",
            "max_free_qubits": 2,
            "trajectory_stream": "identical categorical code list and multiplicities from seed 5772",
            "speedup_definition": "CUDA-Q explicit steady total / TN proportional steady total",
            "process_isolation": "one fresh worker process per qubit/depth point",
            "timing_order": "TN cold first-use, CUDA-Q build/compile warmup, alternating steady repeats",
        },
        "measurements": {
            "conclusion": "PASS" if all_valid else "FAIL",
            "all_timing_points_passed_validation": all_valid,
            "crossover": crossover_summary(points) if all_valid else None,
            "points": points,
        },
    }
    validate_result(record)
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if all_valid else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("upstream/Accelerated_TN_PTSBE"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qubits", nargs="+", type=int, default=[5, 8, 11])
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--trajectory-count", type=int, default=1024)
    parser.add_argument("--shots-per-trajectory", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--null-trials", type=int, default=1000)
    parser.add_argument("--worker-qubits", type=int)
    parser.add_argument("--worker-depth", type=int)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--telemetry", action="store_true")
    args = parser.parse_args()
    worker_mode = args.worker_qubits is not None or args.worker_depth is not None
    if worker_mode and (args.worker_qubits is None or args.worker_depth is None or args.worker_output is None):
        parser.error("worker mode requires --worker-qubits, --worker-depth, and --worker-output")
    if not worker_mode and args.output is None:
        parser.error("--output is required for the parent benchmark")
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if args.trajectory_count <= 0 or args.shots_per_trajectory <= 0 or args.null_trials <= 0:
        parser.error("trajectory count, shots, and null trials must be positive")
    if any(value < 2 for value in args.qubits) or any(value < 1 for value in args.depths):
        parser.error("qubits must be >= 2 and depths must be >= 1")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    worker = parsed.worker_qubits is not None
    raise SystemExit(worker_main(parsed) if worker else parent_main(parsed))
