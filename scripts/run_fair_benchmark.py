#!/usr/bin/env python3
"""Benchmark validated TN proportional batching against explicit CUDA-Q trajectories."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from q_tensor.performance import first_crossover, speedup, throughput, timing_summary
from q_tensor.schema import validate_result
from q_tensor.stats import total_variation_distance
from q_tensor.validation import (
    FROZEN_CASES,
    TWO_QUBIT_PAULIS,
    conditioned_distribution,
    exact_distribution,
    group_trajectories,
    sample_trajectories,
    z_expectation,
)
from run_statistical_validation import (
    UPSTREAM_COMMIT,
    append_cudaq_error,
    append_cudaq_gate,
    backend_comparison,
    case_definition,
    git_head,
    gpu_properties,
    install_cuquantum_gate_order_workaround,
    package_versions,
    run_tn,
    two_backend_comparison,
    write_qiskit_circuit,
)

CASE_ID = "5q_depolarizing2"
TN_DTYPE = "complex128"
CUDAQ_TARGET = "nvidia"
CUDAQ_OPTION = "fp64"


def representative_case():
    return next(case for case in FROZEN_CASES if case.case_id == CASE_ID)


def build_cudaq_kernels(case, grouped):
    import cudaq

    sites = {site.after_gate: (position, site) for position, site in enumerate(case.noise_sites)}
    prepared = []
    for trajectory, multiplicity in grouped:
        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(case.nqubits)
        for gate_index, gate in enumerate(case.gates):
            append_cudaq_gate(kernel, qubits, gate)
            if gate_index in sites:
                position, site = sites[gate_index]
                append_cudaq_error(kernel, qubits, site.qubits, trajectory[position])
        kernel.mz(qubits)
        prepared.append((kernel, multiplicity))
    return prepared


def sample_cudaq_kernels(prepared, nqubits: int, shots_per_trajectory: int, seed: int):
    import cudaq

    counts: Counter[str] = Counter()
    sample_seconds = 0.0
    postprocess_seconds = 0.0
    for serial, (kernel, multiplicity) in enumerate(prepared):
        cudaq.set_random_seed(seed + serial)
        started = time.perf_counter()
        sampled = cudaq.sample(kernel, shots_count=multiplicity * shots_per_trajectory)
        sample_seconds += time.perf_counter() - started
        started = time.perf_counter()
        counts.update({str(bits).zfill(nqubits): int(value) for bits, value in sampled.items()})
        postprocess_seconds += time.perf_counter() - started
    return dict(counts), sample_seconds, postprocess_seconds


def prepare_cudaq(
    case,
    grouped,
    seed: int,
    target: str = CUDAQ_TARGET,
    option: str = CUDAQ_OPTION,
):
    import cudaq

    started = time.perf_counter()
    cudaq.set_target(target, option=option)
    target_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prepared = build_cudaq_kernels(case, grouped)
    build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for serial, (kernel, _) in enumerate(prepared):
        cudaq.set_random_seed(seed + serial)
        cudaq.sample(kernel, shots_count=1)
    compile_seconds = time.perf_counter() - started
    return prepared, {
        "target": target,
        "option": option,
        "target_precision": str(cudaq.get_target().get_precision()),
        "target_initialization_seconds": target_seconds,
        "kernel_build_seconds": build_seconds,
        "compile_warmup_seconds": compile_seconds,
        "total_seconds": target_seconds + build_seconds + compile_seconds,
    }


def timed_tn(
    case,
    grouped,
    shots_per_trajectory: int,
    seed: int,
    workdir: Path,
    max_free_qubits: int = 2,
):
    choice_seconds = 0.0
    original_choice = np.random.choice

    def timed_choice(*args, **kwargs):
        nonlocal choice_seconds
        started = time.perf_counter()
        result = original_choice(*args, **kwargs)
        choice_seconds += time.perf_counter() - started
        return result

    np.random.choice = timed_choice
    try:
        started = time.perf_counter()
        counts, profile = run_tn(
            case,
            grouped,
            shots_per_trajectory,
            seed,
            workdir,
            reverse_qubits=True,
            max_free_qubits=max_free_qubits,
            dtype=TN_DTYPE,
            profile=True,
            prepare_circuit=False,
        )
        wall_seconds = time.perf_counter() - started
    finally:
        np.random.choice = original_choice

    setup_seconds = (
        profile["time_load_circuit"]
        + profile["time_circuit_to_einsum"]
        + profile["time_build_expr_operands"]
    )
    postprocess_seconds = profile["q_tensor_postprocess_seconds"]
    loop_seconds = profile["time_contraction_loop"]
    return counts, {
        "wall_seconds": wall_seconds,
        "setup_path_planning_seconds": setup_seconds,
        "steady_loop_seconds": loop_seconds,
        "steady_compute_orchestration_seconds": max(0.0, loop_seconds - choice_seconds),
        "host_sampling_seconds": choice_seconds,
        "host_postprocess_seconds": postprocess_seconds,
        "steady_total_seconds": loop_seconds + postprocess_seconds,
        "gpu_contraction_seconds": profile["time_contract_gpu_total"],
        "apply_errors_seconds": profile["time_apply_errors_total"],
        "contractions": profile["num_contractions"],
        "upstream_profile": profile,
    }


def summarize_tn(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "wall_seconds",
        "setup_path_planning_seconds",
        "steady_loop_seconds",
        "steady_compute_orchestration_seconds",
        "host_sampling_seconds",
        "host_postprocess_seconds",
        "steady_total_seconds",
        "gpu_contraction_seconds",
        "apply_errors_seconds",
    )
    return {
        "components": {field: timing_summary(repeat[field] for repeat in repeats) for field in fields},
        "first_repeat": repeats[0],
        "raw_repeats": repeats,
        "contractions": repeats[0]["contractions"],
    }


def summarize_cudaq(
    setup: dict[str, float], sample_seconds: list[float], postprocess_seconds: list[float]
) -> dict[str, Any]:
    steady_totals = [sample + post for sample, post in zip(sample_seconds, postprocess_seconds)]
    return {
        "setup_compilation": setup,
        "sample_call_including_device_sampling": timing_summary(sample_seconds),
        "host_postprocess": timing_summary(postprocess_seconds),
        "steady_total": timing_summary(steady_totals),
        "first_steady_total_seconds": steady_totals[0],
        "raw_sample_call_seconds": sample_seconds,
        "raw_postprocess_seconds": postprocess_seconds,
        "sampling_split_available": False,
        "sampling_split_note": "CUDA-Q 0.13 cudaq.sample combines trajectory execution and device sampling",
    }


class SmiMonitor:
    """Collect coarse global GPU telemetry outside the timed repetitions."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "--loop-ms=50",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        def read_output() -> None:
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                try:
                    utilization, memory = (float(value.strip()) for value in line.split(","))
                    self.samples.append((utilization, memory))
                except (ValueError, TypeError):
                    continue

        self.thread = threading.Thread(target=read_output, daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def stop(self) -> dict[str, Any]:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if not self.samples:
            return {"available": False, "reason": "nvidia-smi produced no samples"}
        utilizations = [sample[0] for sample in self.samples]
        memories = [sample[1] for sample in self.samples]
        return {
            "available": True,
            "scope": "global device counters; separate untimed telemetry run",
            "interval_ms": 50,
            "samples": len(self.samples),
            "gpu_utilization_percent_mean": float(np.mean(utilizations)),
            "gpu_utilization_percent_p90": float(np.quantile(utilizations, 0.9)),
            "gpu_utilization_percent_peak": max(utilizations),
            "gpu_utilization_percent_median": float(np.median(utilizations)),
            "memory_used_mib_min": min(memories),
            "memory_used_mib_peak": max(memories),
            "memory_used_mib_delta": max(memories) - min(memories),
        }


def monitored(workload: Callable[[], Any]) -> dict[str, Any]:
    monitor = SmiMonitor()
    monitor.start()
    try:
        workload()
    finally:
        telemetry = monitor.stop()
    return telemetry


def run_worker(args) -> dict[str, Any]:
    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream))
    install_cuquantum_gate_order_workaround()
    if git_head(upstream) != UPSTREAM_COMMIT:
        raise RuntimeError("upstream revision mismatch")
    case = representative_case()
    draws = sample_trajectories(case, args.trajectory_count, seed=case.seed)
    grouped = group_trajectories(draws)
    conditioned = conditioned_distribution(case, grouped)
    channel = exact_distribution(case)
    seed = case.seed * 100000 + args.trajectory_count

    with tempfile.TemporaryDirectory(prefix="q-tensor-benchmark-worker-") as temporary:
        workdir = Path(temporary)
        input_started = time.perf_counter()
        write_qiskit_circuit(
            case,
            workdir / f"{case.case_id}-reversed.qpy",
            reverse_qubits=True,
        )
        input_preparation_seconds = time.perf_counter() - input_started

        _, tn_cold_first_use = timed_tn(
            case, grouped, args.shots_per_trajectory, seed - 1000, workdir
        )
        prepared, cudaq_setup = prepare_cudaq(case, grouped, seed + 20000)
        cudaq_setup["context_note"] = "TN initialized the process CUDA context before this measurement"
        tn_counts_for_validation = None
        cudaq_counts_for_validation = None
        tn_repeats = []
        cudaq_sample_seconds = []
        cudaq_postprocess_seconds = []

        def collect_tn(repeat: int) -> None:
            nonlocal tn_counts_for_validation
            counts, timing = timed_tn(
                case, grouped, args.shots_per_trajectory, seed + repeat * 1000, workdir
            )
            if tn_counts_for_validation is None:
                tn_counts_for_validation = counts
            tn_repeats.append(timing)

        def collect_cudaq(repeat: int) -> None:
            nonlocal cudaq_counts_for_validation
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

        for repeat in range(args.repeats):
            if repeat % 2 == 0:
                collect_tn(repeat)
                collect_cudaq(repeat)
            else:
                collect_cudaq(repeat)
                collect_tn(repeat)

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
                            seed + 80000 + index * 1000,
                        )
                        for index in range(5)
                    ]
                ),
            }

    tn_summary = summarize_tn(tn_repeats)
    tn_summary["cold_first_use"] = tn_cold_first_use
    cudaq_summary = summarize_cudaq(cudaq_setup, cudaq_sample_seconds, cudaq_postprocess_seconds)
    effective_shots = args.trajectory_count * args.shots_per_trajectory
    tn_steady = tn_summary["components"]["steady_total_seconds"]["median_seconds"]
    cudaq_steady = cudaq_summary["steady_total"]["median_seconds"]
    return {
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
        "steady_state_speedup_baseline_over_tn": speedup(cudaq_steady, tn_steady),
        "cold_setup_comparison_note": (
            "reported as components only; no cold speedup is computed because TN initializes the shared GPU "
            "context before CUDA-Q within each isolated worker"
        ),
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
    points = []
    with tempfile.TemporaryDirectory(prefix="q-tensor-benchmark-parent-") as temporary:
        for trajectory_count in args.trajectory_counts:
            worker_output = Path(temporary) / f"worker-{trajectory_count}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-trajectory-count",
                str(trajectory_count),
                "--worker-output",
                str(worker_output),
                "--upstream",
                str(upstream),
                "--shots-per-trajectory",
                str(args.shots_per_trajectory),
                "--repeats",
                str(args.repeats),
                "--null-trials",
                str(args.null_trials),
            ]
            if trajectory_count == max(args.trajectory_counts):
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
                    f"benchmark worker {trajectory_count} failed before writing output: "
                    f"{completed.stdout[-2000:]}{completed.stderr[-2000:]}"
                )
            point = json.loads(worker_output.read_text(encoding="utf-8"))
            point["worker_process_returncode"] = completed.returncode
            point["worker_stdout_tail"] = completed.stdout[-1000:]
            point["worker_stderr_tail"] = completed.stderr[-1000:]
            points.append(point)

    all_valid = all(point["validation"]["passed"] and point["worker_process_returncode"] == 0 for point in points)
    crossover = first_crossover(points) if all_valid else None
    record = {
        "schema_version": 1,
        "experiment_id": "fair-5q-trajectory-sweep-v1",
        "category": "q_tensor_original",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, str(Path(__file__)), *sys.argv[1:]],
        "git_commit": git_head(root),
        "upstream_commit": UPSTREAM_COMMIT,
        "hardware": gpu_properties(),
        "software": {"python": platform.python_version(), "packages": package_versions()},
        "parameters": {
            "case": case_definition(representative_case()),
            "trajectory_counts": args.trajectory_counts,
            "shots_per_trajectory": args.shots_per_trajectory,
            "repeats": args.repeats,
            "null_trials_per_validation_check": args.null_trials,
            "tn_dtype": TN_DTYPE,
            "cudaq_target": f"{CUDAQ_TARGET} {CUDAQ_OPTION}",
            "max_free_qubits": 2,
            "trajectory_stream": "nested prefixes from frozen case seed",
            "speedup_definition": "CUDA-Q explicit steady total / TN proportional steady total",
            "process_isolation": "one fresh worker process per trajectory count",
        },
        "measurements": {
            "conclusion": "PASS" if all_valid else "FAIL",
            "all_timing_points_passed_validation": all_valid,
            "first_crossover_trajectories": crossover,
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
    parser.add_argument("--trajectory-counts", nargs="+", type=int, default=[16, 64, 256, 1024, 4096])
    parser.add_argument("--shots-per-trajectory", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--null-trials", type=int, default=1000)
    parser.add_argument("--worker-trajectory-count", dest="trajectory_count", type=int)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--telemetry", action="store_true")
    args = parser.parse_args()
    if args.trajectory_count is None and args.output is None:
        parser.error("--output is required for the parent benchmark")
    if args.trajectory_count is not None and args.worker_output is None:
        parser.error("--worker-output is required in worker mode")
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if args.shots_per_trajectory <= 0 or args.null_trials <= 0:
        parser.error("shots and null trials must be positive")
    if any(value <= 0 for value in args.trajectory_counts):
        parser.error("trajectory counts must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(worker_main(parsed) if parsed.trajectory_count is not None else parent_main(parsed))
