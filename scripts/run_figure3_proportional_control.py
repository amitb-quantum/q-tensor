#!/usr/bin/env python3
"""Validation-gated proportional control on the exact Figure-3 circuit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from q_tensor.performance import speedup, throughput
from q_tensor.schema import validate_result
from q_tensor.validation import (
    Gate,
    NoiseSite,
    ValidationCase,
    group_trajectories,
)
from q_tensor.validation import channel_outcomes, conditioned_distribution, z_expectation
from run_fair_benchmark import (
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


FIGURE3_RELATIVE = Path("scaling/data_collection/circuits/50q_200g/circuit_0.stim")
FULL_SEED = 50902026
SHADOW_SEED = 80509026
FULL_TRAJECTORIES = 10
FULL_SHOTS_PER_DRAW = 100
SHADOW_TRAJECTORIES = 128
SHADOW_SHOTS_PER_DRAW = 32
SHADOW_QUBITS = 8
TN_BATCH_QUBITS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_stim_case(path: Path) -> ValidationCase:
    """Parse the exact alternating coherent-gate/noise Figure-3 artifact."""

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 2:
        raise ValueError("expected each coherent gate to be followed by one noise instruction")
    gates: list[Gate] = []
    sites: list[NoiseSite] = []
    for offset in range(0, len(lines), 2):
        gate_match = re.fullmatch(r"([A-Z]+)(?:\(([^)]+)\))?\s+(.+)", lines[offset])
        noise_match = re.fullmatch(r"([A-Z0-9_]+)\(([^)]+)\)\s+(.+)", lines[offset + 1])
        if gate_match is None or noise_match is None:
            raise ValueError(f"unsupported Stim pair at lines {offset + 1}-{offset + 2}")
        gate_name, angle_text, operand_text = gate_match.groups()
        operands = tuple(int(value) for value in operand_text.split())
        angle = None if angle_text is None else float(angle_text)
        gate = Gate(gate_name.lower(), operands, angle)
        noise_name, probability_text, target_text = noise_match.groups()
        targets = tuple(int(value) for value in target_text.split())
        if targets != operands:
            raise ValueError("noise targets do not match the preceding gate")
        channel = {
            "X_ERROR": "x",
            "Y_ERROR": "y",
            "Z_ERROR": "z",
            "DEPOLARIZE1": "depolarizing1",
            "DEPOLARIZE2": "depolarizing2",
        }.get(noise_name)
        if channel is None:
            raise ValueError(f"unsupported noise instruction {noise_name}")
        gates.append(gate)
        sites.append(NoiseSite(len(gates) - 1, targets, channel, float(probability_text)))
    nqubits = 1 + max(qubit for gate in gates for qubit in gate.qubits)
    return ValidationCase(
        case_id="figure3_50q_200g_circuit0",
        nqubits=nqubits,
        gates=tuple(gates),
        noise_sites=tuple(sites),
        observable_qubits=(0,),
        seed=FULL_SEED,
    )


def choose_shadow_labels(case: ValidationCase, count: int = SHADOW_QUBITS) -> tuple[int, ...]:
    """Choose a deterministic connected/high-retention induced qubit subset."""

    if not 2 <= count <= case.nqubits:
        raise ValueError("shadow size must be between two and the full circuit size")
    single_counts = Counter(gate.qubits[0] for gate in case.gates if len(gate.qubits) == 1)
    incident_counts = Counter(qubit for gate in case.gates if len(gate.qubits) == 2 for qubit in gate.qubits)
    first = max(range(case.nqubits), key=lambda qubit: (single_counts[qubit] + incident_counts[qubit], -qubit))
    selected = {first}
    while len(selected) < count:
        def score(candidate: int) -> tuple[int, int, int]:
            internal_edges = sum(
                1
                for gate in case.gates
                if len(gate.qubits) == 2
                and candidate in gate.qubits
                and any(qubit in selected for qubit in gate.qubits if qubit != candidate)
            )
            return internal_edges, single_counts[candidate] + incident_counts[candidate], -candidate

        selected.add(max((q for q in range(case.nqubits) if q not in selected), key=score))
    return tuple(sorted(selected))


def induced_shadow(case: ValidationCase, labels: Sequence[int]) -> ValidationCase:
    mapping = {label: index for index, label in enumerate(labels)}
    gates: list[Gate] = []
    sites: list[NoiseSite] = []
    for gate, site in zip(case.gates, case.noise_sites):
        if all(qubit in mapping for qubit in gate.qubits):
            mapped = tuple(mapping[qubit] for qubit in gate.qubits)
            gates.append(Gate(gate.name, mapped, gate.angle))
            sites.append(NoiseSite(len(gates) - 1, mapped, site.channel, site.probability))
    if not any(len(gate.qubits) == 2 for gate in gates):
        raise ValueError("shadow selection retained no entangling gates")
    return ValidationCase(
        case_id=f"figure3_shadow_{len(labels)}q",
        nqubits=len(labels),
        gates=tuple(gates),
        noise_sites=tuple(sites),
        observable_qubits=(0,),
        seed=SHADOW_SEED,
    )


def categorical_draws(case: ValidationCase, count: int, seed: int) -> list[tuple[int, ...]]:
    """Draw corrected categorical Kraus codes without imposing the reference-size cap."""

    if count <= 0:
        raise ValueError("trajectory count must be positive")
    rng = np.random.default_rng(seed)
    alternatives = [channel_outcomes(site) for site in case.noise_sites]
    return [
        tuple(
            choices[int(rng.choice(len(choices), p=[probability for _, probability in choices]))][0]
            for choices in alternatives
        )
        for _ in range(count)
    ]


def categorical_channel_audit(case: ValidationCase) -> dict[str, Any]:
    probability_sum_errors = []
    total_error_errors = []
    for site in case.noise_sites:
        outcomes = channel_outcomes(site)
        probability_sum_errors.append(abs(sum(probability for _, probability in outcomes) - 1.0))
        total_error_errors.append(
            abs(sum(probability for code, probability in outcomes if code != 0) - site.probability)
        )
    max_sum_error = max(probability_sum_errors, default=0.0)
    max_total_error_error = max(total_error_errors, default=0.0)
    return {
        "site_count": len(case.noise_sites),
        "max_probability_sum_error": max_sum_error,
        "max_total_error_probability_error": max_total_error_error,
        "passed": max(max_sum_error, max_total_error_error) <= 1e-12,
    }


def frozen_trajectory_record(draws, grouped) -> dict[str, Any]:
    encoded = json.dumps(draws, separators=(",", ":"))
    return {
        "draws": [list(draw) for draw in draws],
        "draws_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "grouped": [
            {"codes": list(trajectory), "multiplicity": multiplicity}
            for trajectory, multiplicity in grouped
        ],
        "draw_count": len(draws),
        "unique_count": len(grouped),
        "multiplicity_sum": sum(multiplicity for _, multiplicity in grouped),
    }


def count_z_vector(counts: dict[str, int], nqubits: int) -> list[float]:
    return [z_expectation(counts, (qubit,)) for qubit in range(nqubits)]


def exclusive_tn_components(summary: dict[str, Any]) -> dict[str, float]:
    component = summary["components"]
    loop = component["steady_loop_seconds"]["median_seconds"]
    gpu = component["gpu_contraction_seconds"]["median_seconds"]
    errors = component["apply_errors_seconds"]["median_seconds"]
    sampling = component["host_sampling_seconds"]["median_seconds"]
    post = component["host_postprocess_seconds"]["median_seconds"]
    return {
        "gpu_contraction_seconds": gpu,
        "apply_errors_seconds": errors,
        "host_sampling_seconds": sampling,
        "host_postprocess_seconds": post,
        "other_loop_seconds": max(0.0, loop - gpu - errors - sampling),
    }


def run_shadow(args) -> dict[str, Any]:
    case = parse_stim_case(args.stim)
    labels = choose_shadow_labels(case, args.shadow_qubits)
    shadow = induced_shadow(case, labels)
    draws = categorical_draws(shadow, args.shadow_trajectories, SHADOW_SEED)
    grouped = group_trajectories(draws)
    reference = conditioned_distribution(shadow, grouped)
    channel_audit = categorical_channel_audit(shadow)
    with tempfile.TemporaryDirectory(prefix="q-tensor-figure3-shadow-") as temporary:
        workdir = Path(temporary)
        write_qiskit_circuit(shadow, workdir / f"{shadow.case_id}-reversed.qpy", reverse_qubits=True)
        tn_counts, _ = timed_tn(
            shadow, grouped, args.shadow_shots, SHADOW_SEED + 1000, workdir, max_free_qubits=2
        )
        prepared, cudaq_setup = prepare_cudaq(
            shadow, grouped, SHADOW_SEED + 2000, target="nvidia", option="fp64"
        )
        cudaq_counts, _, _ = sample_cudaq_kernels(
            prepared, shadow.nqubits, args.shadow_shots, SHADOW_SEED + 3000
        )
    tn_check = backend_comparison(
        shadow, grouped, args.shadow_shots, tn_counts, reference, args.null_trials, SHADOW_SEED + 4000
    )
    cudaq_check = backend_comparison(
        shadow,
        grouped,
        args.shadow_shots,
        cudaq_counts,
        reference,
        args.null_trials,
        SHADOW_SEED + 5000,
    )
    pair_check = two_backend_comparison(
        shadow,
        grouped,
        args.shadow_shots,
        tn_counts,
        cudaq_counts,
        args.null_trials,
        SHADOW_SEED + 6000,
    )
    expected_shots = args.shadow_trajectories * args.shadow_shots
    passed = (
        tn_check["passed"]
        and cudaq_check["passed"]
        and pair_check["passed"]
        and channel_audit["passed"]
        and sum(tn_counts.values()) == expected_shots
        and sum(cudaq_counts.values()) == expected_shots
    )
    return {
        "passed": passed,
        "source_qubit_labels": list(labels),
        "construction": "chronological induced subcircuit; retain exact gates/noise wholly inside labels",
        "case": case_definition(shadow),
        "coherent_gate_count": len(shadow.gates),
        "entangling_gate_count": sum(len(gate.qubits) == 2 for gate in shadow.gates),
        "noise_site_count": len(shadow.noise_sites),
        "trajectory_seed": SHADOW_SEED,
        "categorical_channel_audit": channel_audit,
        "trajectories": frozen_trajectory_record(draws, grouped),
        "shots_per_draw": args.shadow_shots,
        "total_effective_shots": expected_shots,
        "exact_reference": "complex128 statevector mixture conditioned on frozen trajectories/multiplicities",
        "reference_distribution": reference,
        "tn_counts": tn_counts,
        "cudaq_counts": cudaq_counts,
        "tn_vs_exact_conditioned": tn_check,
        "cudaq_vs_exact_conditioned": cudaq_check,
        "tn_vs_cudaq": pair_check,
        "cudaq_setup": cudaq_setup,
    }


def run_full(args) -> dict[str, Any]:
    case = parse_stim_case(args.stim)
    draws = categorical_draws(case, args.trajectories, FULL_SEED)
    grouped = group_trajectories(draws)
    effective_shots = args.trajectories * args.shots_per_draw
    with tempfile.TemporaryDirectory(prefix="q-tensor-figure3-proportional-") as temporary:
        workdir = Path(temporary)
        started = time.perf_counter()
        write_qiskit_circuit(case, workdir / f"{case.case_id}-reversed.qpy", reverse_qubits=True)
        input_seconds = time.perf_counter() - started
        _, tn_cold = timed_tn(
            case,
            grouped,
            args.shots_per_draw,
            FULL_SEED + 1000,
            workdir,
            max_free_qubits=TN_BATCH_QUBITS,
        )
        prepared, cudaq_setup = prepare_cudaq(
            case, grouped, FULL_SEED + 2000, target="tensornet", option="fp64"
        )
        cudaq_setup["context_note"] = "TN initialized the process CUDA context first"
        tn_repeats: list[dict[str, Any]] = []
        cudaq_sample: list[float] = []
        cudaq_post: list[float] = []
        tn_first_counts = None
        cudaq_first_counts = None
        for repeat in range(args.repeats):
            order = ("tn", "cudaq") if repeat % 2 == 0 else ("cudaq", "tn")
            for backend in order:
                if backend == "tn":
                    counts, timing = timed_tn(
                        case,
                        grouped,
                        args.shots_per_draw,
                        FULL_SEED + 10000 + repeat * 1000,
                        workdir,
                        max_free_qubits=TN_BATCH_QUBITS,
                    )
                    tn_repeats.append(timing)
                    if tn_first_counts is None:
                        tn_first_counts = counts
                else:
                    counts, sample_seconds, post_seconds = sample_cudaq_kernels(
                        prepared,
                        case.nqubits,
                        args.shots_per_draw,
                        FULL_SEED + 20000 + repeat * 1000,
                    )
                    cudaq_sample.append(sample_seconds)
                    cudaq_post.append(post_seconds)
                    if cudaq_first_counts is None:
                        cudaq_first_counts = counts
        assert tn_first_counts is not None and cudaq_first_counts is not None
        telemetry = {
            "tn": monitored(
                lambda: timed_tn(
                    case,
                    grouped,
                    args.shots_per_draw,
                    FULL_SEED + 70000,
                    workdir,
                    max_free_qubits=TN_BATCH_QUBITS,
                )
            ),
            "cudaq": monitored(
                lambda: sample_cudaq_kernels(
                    prepared, case.nqubits, args.shots_per_draw, FULL_SEED + 80000
                )
            ),
        }
    tn = summarize_tn(tn_repeats)
    tn["cold_first_use"] = tn_cold
    cudaq = summarize_cudaq(cudaq_setup, cudaq_sample, cudaq_post)
    tn_steady = tn["components"]["steady_total_seconds"]["median_seconds"]
    cudaq_steady = cudaq["steady_total"]["median_seconds"]
    tn_path_inclusive = tn["components"]["wall_seconds"]["median_seconds"]
    cudaq_cold_total = cudaq_setup["total_seconds"] + cudaq["first_steady_total_seconds"]
    exclusive = exclusive_tn_components(tn)
    contraction_repeats = [repeat["contractions"] for repeat in tn_repeats]
    tn_z = count_z_vector(tn_first_counts, case.nqubits)
    cudaq_z = count_z_vector(cudaq_first_counts, case.nqubits)
    z_differences = [abs(left - right) for left, right in zip(tn_z, cudaq_z)]
    familywise_alpha = 0.001
    familywise_z_limit = math.sqrt(
        4 * math.log(2 * case.nqubits / familywise_alpha) / effective_shots
    )
    totals_ok = all(
        repeat_total == effective_shots
        for repeat_total in [
            sum(tn_first_counts.values()),
            sum(cudaq_first_counts.values()),
        ]
    )
    return {
        "passed_input_and_shot_invariants": totals_ok,
        "case": case_definition(case),
        "stim_sha256": sha256(args.stim),
        "coherent_gate_count": len(case.gates),
        "entangling_gate_count": sum(len(gate.qubits) == 2 for gate in case.gates),
        "noise_site_count": len(case.noise_sites),
        "trajectory_seed": FULL_SEED,
        "trajectories": frozen_trajectory_record(draws, grouped),
        "shots_per_draw": args.shots_per_draw,
        "total_effective_shots": effective_shots,
        "tn_batch_qubits": TN_BATCH_QUBITS,
        "tn_contractions_repeats": contraction_repeats,
        "tn_contractions_median": float(np.median(contraction_repeats)),
        "precision": "complex128 / fp64",
        "input_preparation_seconds": input_seconds,
        "tn": tn,
        "cudaq": cudaq,
        "cold_total_seconds": {
            "tn_first_use_including_path_and_equal_shots": tn_cold["wall_seconds"],
            "cudaq_setup_compile_plus_first_equal_shots": cudaq_cold_total,
        },
        "steady_state": {
            "definition": "execution and requested-shot postprocessing; TN path planning excluded",
            "tn_seconds": tn_steady,
            "cudaq_seconds": cudaq_steady,
            "speedup_cudaq_over_tn": speedup(cudaq_steady, tn_steady),
            "tn_effective_shots_per_second": throughput(effective_shots, tn_steady),
            "cudaq_effective_shots_per_second": throughput(effective_shots, cudaq_steady),
        },
        "path_inclusive": {
            "tn_seconds": tn_path_inclusive,
            "cudaq_seconds": cudaq_steady,
            "speedup_cudaq_over_tn": speedup(cudaq_steady, tn_path_inclusive),
        },
        "tn_exclusive_component_medians_seconds": exclusive,
        "tn_dominant_runtime_component": max(exclusive, key=exclusive.get),
        "first_repeat_cross_backend_diagnostics": {
            "note": "diagnostic only; 50-qubit output has no tractable exact reference",
            "tn_total_shots": sum(tn_first_counts.values()),
            "cudaq_total_shots": sum(cudaq_first_counts.values()),
            "per_qubit_z_tn": tn_z,
            "per_qubit_z_cudaq": cudaq_z,
            "per_qubit_z_absolute_difference_mean": float(np.mean(z_differences)),
            "per_qubit_z_absolute_difference_max": max(z_differences),
            "familywise_hoeffding_alpha": familywise_alpha,
            "familywise_hoeffding_z_difference_limit": familywise_z_limit,
            "familywise_hoeffding_check_passed": max(z_differences) <= familywise_z_limit,
        },
        "telemetry": telemetry,
        "non_proportional_unique_record_metric_used": False,
    }


def run_worker(args) -> int:
    sys.path.insert(0, str(args.upstream.resolve()))
    install_cuquantum_gate_order_workaround()
    if git_head(args.upstream.resolve()) != UPSTREAM_COMMIT:
        raise RuntimeError("upstream revision mismatch")
    result = run_shadow(args) if args.worker == "shadow" else run_full(args)
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    passed = result["passed"] if args.worker == "shadow" else result["passed_input_and_shot_invariants"]
    return 0 if passed else 1


def invoke_worker(args, worker: str, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        worker,
        "--worker-output",
        str(output),
        "--upstream",
        str(args.upstream.resolve()),
        "--stim",
        str(args.stim.resolve()),
        "--trajectories",
        str(args.trajectories),
        "--shots-per-draw",
        str(args.shots_per_draw),
        "--shadow-qubits",
        str(args.shadow_qubits),
        "--shadow-trajectories",
        str(args.shadow_trajectories),
        "--shadow-shots",
        str(args.shadow_shots),
        "--repeats",
        str(args.repeats),
        "--null-trials",
        str(args.null_trials),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=args.timeout,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    metadata = {
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }
    if not output.is_file():
        raise RuntimeError(f"{worker} worker produced no result: {metadata}")
    return json.loads(output.read_text(encoding="utf-8")), metadata


def parent_main(args) -> int:
    root = Path(__file__).resolve().parents[1]
    if git_head(root) != args.expected_commit:
        raise SystemExit(f"Q-Tensor revision mismatch: expected {args.expected_commit}")
    if git_head(args.upstream.resolve()) != UPSTREAM_COMMIT:
        raise SystemExit("upstream revision mismatch")
    with tempfile.TemporaryDirectory(prefix="q-tensor-figure3-control-parent-") as temporary:
        temp = Path(temporary)
        shadow, shadow_process = invoke_worker(args, "shadow", temp / "shadow.json")
        full = None
        full_process = None
        if shadow["passed"] and shadow_process["returncode"] == 0:
            full, full_process = invoke_worker(args, "full", temp / "full.json")
    passed = (
        shadow["passed"]
        and shadow_process["returncode"] == 0
        and full is not None
        and full["passed_input_and_shot_invariants"]
        and full_process is not None
        and full_process["returncode"] == 0
    )
    record = {
        "schema_version": 1,
        "experiment_id": "figure3-50q-proportional-control-v1",
        "category": "q_tensor_original",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, str(Path(__file__)), *sys.argv[1:]],
        "git_commit": git_head(root),
        "upstream_commit": UPSTREAM_COMMIT,
        "hardware": gpu_properties(),
        "software": {"python": platform.python_version(), "packages": package_versions()},
        "parameters": {
            "source_workload": "exact public Figure-3 50q/200g circuit 0 Stim artifact",
            "stim_path": str(args.stim),
            "stim_sha256": sha256(args.stim),
            "trajectory_method": "seeded categorical Kraus alternatives, grouped with exact multiplicity",
            "trajectory_count": args.trajectories,
            "shots_per_draw": args.shots_per_draw,
            "total_effective_shots": args.trajectories * args.shots_per_draw,
            "tn_dtype": "complex128",
            "cudaq_target": "tensornet fp64",
            "tn_batch_qubits": TN_BATCH_QUBITS,
            "repeats": args.repeats,
            "timing_discipline": "isolated workers; cold setup separated; alternating steady repeats",
            "comparison_metric": "equal effective shots per second and runtime",
            "non_proportional_unique_record_metric_used": False,
        },
        "measurements": {
            "conclusion": "PASS" if passed else "FAIL",
            "shadow_validation_hard_gate": shadow,
            "shadow_worker": shadow_process,
            "full_50q_proportional": full,
            "full_worker": full_process,
            "proportional_crossover_exists": bool(
                passed and full is not None and full["steady_state"]["speedup_cudaq_over_tn"] >= 1.0
            ),
        },
    }
    validate_result(record)
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if passed else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--stim", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit", default="50d933e8cb7060663c1e5921f1a15f62f2e2df53")
    parser.add_argument("--trajectories", type=int, default=FULL_TRAJECTORIES)
    parser.add_argument("--shots-per-draw", type=int, default=FULL_SHOTS_PER_DRAW)
    parser.add_argument("--shadow-qubits", type=int, default=SHADOW_QUBITS)
    parser.add_argument("--shadow-trajectories", type=int, default=SHADOW_TRAJECTORIES)
    parser.add_argument("--shadow-shots", type=int, default=SHADOW_SHOTS_PER_DRAW)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--null-trials", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--worker", choices=("shadow", "full"))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.stim is None:
        args.stim = args.upstream / FIGURE3_RELATIVE
    if args.worker is None and args.output is None:
        parser.error("--output is required")
    if args.worker is not None and args.worker_output is None:
        parser.error("--worker-output is required in worker mode")
    if args.repeats < 3 or min(
        args.trajectories,
        args.shots_per_draw,
        args.shadow_trajectories,
        args.shadow_shots,
        args.null_trials,
    ) <= 0:
        parser.error("repeats must be >= 3 and counts must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(run_worker(parsed) if parsed.worker else parent_main(parsed))
