#!/usr/bin/env python3
"""Run the seeded 2--5 qubit proportional-sampling validation suite."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from q_tensor.schema import validate_result
from q_tensor.stats import total_variation_distance
from q_tensor.validation import (
    FROZEN_CASES,
    TWO_QUBIT_PAULIS,
    Gate,
    ValidationCase,
    conditioned_distribution,
    exact_distribution,
    group_trajectories,
    normalized,
    sample_trajectories,
    trajectory_distribution,
    upstream_depolarizing_probabilities,
    upstream_broken_deduplicate,
    upstream_noise_samples,
    z_expectation,
)

UPSTREAM_COMMIT = "0569f848d9a3e385d6c20162c534a8031f6a69c5"
NULL_QUANTILE = 0.999


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def package_versions() -> dict[str, str | None]:
    packages = (
        "cuquantum-python-cu12",
        "cupy-cuda12x",
        "cuda-quantum-cu12",
        "qiskit",
        "numpy",
        "scipy",
        "mpi4py",
    )
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def gpu_properties() -> dict[str, Any]:
    import cupy

    props = cupy.cuda.runtime.getDeviceProperties(0)
    return {
        "name": props["name"].decode(),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "sm_count": props["multiProcessorCount"],
        "total_global_memory_bytes": props["totalGlobalMem"],
        "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_api": cupy.cuda.runtime.driverGetVersion(),
    }


def install_cuquantum_gate_order_workaround() -> None:
    """Use the same converter workaround as the pinned upstream tests."""

    from cuquantum.tensornet._internal import circuit_parser_utils_qiskit
    from qiskit.circuit import Measure

    def remove_measurements(circuit):
        for instruction in circuit.data:
            if isinstance(instruction.operation, Measure):
                raise ValueError("the input circuit cannot contain measurements")
        return circuit

    circuit_parser_utils_qiskit.remove_measurements = remove_measurements


def write_qiskit_circuit(case: ValidationCase, path: Path, reverse_qubits: bool) -> None:
    """Write the noiseless Qiskit circuit used by the upstream TN adapter."""

    import qiskit
    import qiskit.qpy

    circuit = qiskit.QuantumCircuit(case.nqubits)
    mapped = lambda qubit: case.nqubits - 1 - qubit if reverse_qubits else qubit
    for gate in case.gates:
        qubits = tuple(mapped(qubit) for qubit in gate.qubits)
        operation = getattr(circuit, gate.name)
        if gate.angle is None:
            operation(*qubits)
        else:
            operation(gate.angle, *qubits)
    with path.open("wb") as handle:
        qiskit.qpy.dump(circuit, handle)


def run_tn(
    case: ValidationCase,
    grouped,
    shots_per_trajectory: int,
    seed: int,
    workdir: Path,
    reverse_qubits: bool = True,
    max_free_qubits: int = 2,
    dtype: str = "complex128",
    profile: bool = False,
    prepare_circuit: bool = True,
) -> dict[str, int] | tuple[dict[str, int], dict[str, Any]]:
    import cupy
    from utils_circuit import get_noisy_shots_batched

    circuit_path = workdir / f"{case.case_id}-{'reversed' if reverse_qubits else 'direct'}.qpy"
    if prepare_circuit:
        write_qiskit_circuit(case, circuit_path, reverse_qubits=reverse_qubits)
    elif not circuit_path.is_file():
        raise FileNotFoundError(f"prepared circuit not found: {circuit_path}")
    np.random.seed(seed)
    cupy.random.seed(seed)
    noise_samples = upstream_noise_samples(case, grouped, shots_per_trajectory)
    result = get_noisy_shots_batched(
        str(circuit_path),
        noise_samples,
        case.nqubits,
        max_free_qubits,
        shots_per_trajectory,
        dtype,
        proportional_sampling=True,
        full_rdm=False,
        enable_profiling=profile,
        verbose_level=0,
    )
    if profile:
        per_trajectory, profiling = result
    else:
        per_trajectory = result
    postprocess_started = time.perf_counter()
    combined: Counter[str] = Counter()
    for counts in per_trajectory:
        combined.update(counts)
    postprocess_seconds = time.perf_counter() - postprocess_started
    if profile:
        profiling["q_tensor_postprocess_seconds"] = postprocess_seconds
        return dict(combined), profiling
    return dict(combined)


def run_tn_fixed(
    case: ValidationCase,
    grouped,
    shots_per_trajectory: int,
    seed: int,
    workdir: Path,
    fixed_logical_qubits: int = 1,
) -> dict[str, int]:
    """Exercise the fixed-qubit recombination path used by ``test_ptsbe.py``."""

    import cupy
    from utils_noisy_shots import get_noisy_shots2

    circuit_path = workdir / f"{case.case_id}-fixed.qpy"
    write_qiskit_circuit(case, circuit_path, reverse_qubits=True)
    np.random.seed(seed)
    cupy.random.seed(seed)
    # The official builder maps final logical qubits to the first Qiskit indices.
    fixed_qiskit = tuple(range(fixed_logical_qubits))
    per_trajectory = get_noisy_shots2(
        str(circuit_path),
        upstream_noise_samples(case, grouped, shots_per_trajectory),
        fixed_qiskit,
        shots_per_trajectory,
        full_rdm=False,
        verbose_level=0,
    )
    combined: Counter[str] = Counter()
    for fixed_branches in per_trajectory:
        for _, shots in fixed_branches:
            combined.update(shots)
    return dict(combined)


def append_cudaq_gate(kernel, qubits, gate: Gate) -> None:
    operation = getattr(kernel, gate.name)
    operands = [qubits[index] for index in gate.qubits]
    if gate.angle is None:
        operation(*operands)
    else:
        operation(gate.angle, *operands)


def append_cudaq_error(kernel, qubits, targets: tuple[int, ...], code: int) -> None:
    if code == 0:
        return
    if len(targets) == 1:
        getattr(kernel, ("i", "x", "y", "z")[code])(qubits[targets[0]])
        return
    first, second = TWO_QUBIT_PAULIS[code]
    if first:
        getattr(kernel, ("i", "x", "y", "z")[first])(qubits[targets[0]])
    if second:
        getattr(kernel, ("i", "x", "y", "z")[second])(qubits[targets[1]])


def run_cudaq(
    case: ValidationCase,
    grouped,
    shots_per_trajectory: int,
    seed: int,
) -> dict[str, int]:
    import cudaq

    combined: Counter[str] = Counter()
    sites = {site.after_gate: (position, site) for position, site in enumerate(case.noise_sites)}
    for serial, (trajectory, multiplicity) in enumerate(grouped):
        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(case.nqubits)
        for gate_index, gate in enumerate(case.gates):
            append_cudaq_gate(kernel, qubits, gate)
            if gate_index in sites:
                position, site = sites[gate_index]
                append_cudaq_error(kernel, qubits, site.qubits, trajectory[position])
        kernel.mz(qubits)
        cudaq.set_random_seed(seed + serial)
        counts = cudaq.sample(kernel, shots_count=multiplicity * shots_per_trajectory)
        combined.update({str(bits).zfill(case.nqubits): int(count) for bits, count in counts.items()})
    return dict(combined)


def _support(case: ValidationCase) -> list[str]:
    return ["".join(str((value >> qubit) & 1) for qubit in range(case.nqubits)) for value in range(2**case.nqubits)]


def _sample_stratified(case: ValidationCase, grouped, shots_per_trajectory: int, rng) -> dict[str, int]:
    return _stratified_sampler(case, grouped, shots_per_trajectory)(rng)


def _stratified_sampler(case: ValidationCase, grouped, shots_per_trajectory: int):
    """Precompute exact branch probabilities for repeated seeded null draws."""

    support = _support(case)
    plan = []
    for trajectory, multiplicity in grouped:
        distribution = trajectory_distribution(case, trajectory)
        probabilities = np.array([distribution.get(bits, 0.0) for bits in support])
        probabilities /= probabilities.sum()
        plan.append((multiplicity * shots_per_trajectory, probabilities))

    def sample(rng) -> dict[str, int]:
        counts = np.zeros(len(support), dtype=np.int64)
        for shots, probabilities in plan:
            counts += rng.multinomial(shots, probabilities)
        return {bits: int(value) for bits, value in zip(support, counts) if value}

    return sample


def _metrics(observed: Mapping[str, int] | Mapping[str, float], reference, observable) -> tuple[float, float]:
    return (
        total_variation_distance(observed, reference),
        abs(z_expectation(observed, observable) - z_expectation(reference, observable)),
    )


def backend_comparison(
    case: ValidationCase,
    grouped,
    shots_per_trajectory: int,
    observed: Mapping[str, int],
    reference: Mapping[str, float],
    null_trials: int,
    seed: int,
) -> dict[str, Any]:
    observed_tvd, observed_expectation = _metrics(observed, reference, case.observable_qubits)
    rng = np.random.default_rng(seed)
    sample_null = _stratified_sampler(case, grouped, shots_per_trajectory)
    null = [
        _metrics(sample_null(rng), reference, case.observable_qubits)
        for _ in range(null_trials)
    ]
    tvd_limit = float(np.quantile([item[0] for item in null], NULL_QUANTILE, method="higher"))
    expectation_limit = float(np.quantile([item[1] for item in null], NULL_QUANTILE, method="higher"))
    passed = observed_tvd <= tvd_limit + 1e-12 and observed_expectation <= expectation_limit + 1e-12
    return {
        "tvd": observed_tvd,
        "z_expectation_error": observed_expectation,
        "null_quantile": NULL_QUANTILE,
        "null_tvd_limit": tvd_limit,
        "null_z_expectation_error_limit": expectation_limit,
        "passed": passed,
    }


def two_backend_comparison(
    case: ValidationCase,
    grouped,
    shots_per_trajectory: int,
    left: Mapping[str, int],
    right: Mapping[str, int],
    null_trials: int,
    seed: int,
) -> dict[str, Any]:
    observed_tvd, observed_expectation = _metrics(left, right, case.observable_qubits)
    rng = np.random.default_rng(seed)
    sample_null = _stratified_sampler(case, grouped, shots_per_trajectory)
    null: list[tuple[float, float]] = []
    for _ in range(null_trials):
        first = sample_null(rng)
        second = sample_null(rng)
        null.append(_metrics(first, second, case.observable_qubits))
    tvd_limit = float(np.quantile([item[0] for item in null], NULL_QUANTILE, method="higher"))
    expectation_limit = float(np.quantile([item[1] for item in null], NULL_QUANTILE, method="higher"))
    return {
        "tvd": observed_tvd,
        "z_expectation_difference": observed_expectation,
        "null_quantile": NULL_QUANTILE,
        "null_tvd_limit": tvd_limit,
        "null_z_expectation_difference_limit": expectation_limit,
        "passed": observed_tvd <= tvd_limit + 1e-12 and observed_expectation <= expectation_limit + 1e-12,
    }


def case_definition(case: ValidationCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "nqubits": case.nqubits,
        "seed": case.seed,
        "gates": [
            {"name": gate.name, "qubits": list(gate.qubits), "angle": gate.angle} for gate in case.gates
        ],
        "noise_sites": [
            {
                "after_gate": site.after_gate,
                "qubits": list(site.qubits),
                "channel": site.channel,
                "probability": site.probability,
            }
            for site in case.noise_sites
        ],
        "observable": "".join(f"Z{qubit}" for qubit in case.observable_qubits),
    }


def ordering_audit(workdir: Path, shots: int) -> dict[str, Any]:
    cases = (
        ValidationCase("ordering_x_q0", 2, (Gate("x", (0,)),), (), (0,), 901),
        ValidationCase("ordering_bell", 2, (Gate("h", (0,)), Gate("cx", (0, 1))), (), (0, 1), 902),
    )
    results = []
    for case in cases:
        grouped = (((), 1),)
        reference = exact_distribution(case)
        official_tn = run_tn(case, grouped, shots, case.seed, workdir, reverse_qubits=True)
        direct_tn = run_tn(case, grouped, shots, case.seed, workdir, reverse_qubits=False)
        cudaq_counts = run_cudaq(case, grouped, shots, case.seed)
        results.append(
            {
                "case_id": case.case_id,
                "exact": reference,
                "official_reversed_qiskit_tn": normalized(official_tn),
                "direct_qiskit_tn": normalized(direct_tn),
                "cudaq": normalized(cudaq_counts),
                "official_tn_vs_exact_tvd": total_variation_distance(official_tn, reference),
                "direct_tn_vs_exact_tvd": total_variation_distance(direct_tn, reference),
                "cudaq_vs_exact_tvd": total_variation_distance(cudaq_counts, reference),
            }
        )
    return {"shots_per_backend": shots, "cases": results}


def minimized_stale_cache_audit(workdir: Path, shots: int = 4096) -> dict[str, Any]:
    """Minimize upstream's serial-number-only cache failure to two qubits."""

    cached_case = ValidationCase("cached_x_q0", 2, (Gate("x", (0,)),), (), (0,), 930)
    current_case = ValidationCase("current_x_q1", 2, (Gate("x", (1,)),), (), (1,), 931)
    grouped = (((), 1),)
    cached_cudaq = run_cudaq(cached_case, grouped, shots, cached_case.seed)
    current_tn = run_tn_fixed(current_case, grouped, shots, current_case.seed, workdir)
    fresh_cudaq = run_cudaq(current_case, grouped, shots, current_case.seed)
    reference = exact_distribution(current_case)
    return {
        "nqubits": 2,
        "shots": shots,
        "cached_circuit": "X(q0)",
        "current_circuit": "X(q1)",
        "cache_key_collision": "ptsa_example_serialnumber_0.npy",
        "cached_cudaq_distribution": normalized(cached_cudaq),
        "current_tn_distribution": normalized(current_tn),
        "fresh_cudaq_distribution": normalized(fresh_cudaq),
        "current_exact_distribution": reference,
        "current_tn_vs_stale_cudaq_tvd": total_variation_distance(current_tn, cached_cudaq),
        "current_tn_vs_fresh_cudaq_tvd": total_variation_distance(current_tn, fresh_cudaq),
        "current_tn_vs_exact_tvd": total_variation_distance(current_tn, reference),
    }


def fixed_path_audit(workdir: Path, trajectory_count: int, shots_per_trajectory: int) -> dict[str, Any]:
    """Compare the official test's TN sampler with the same CUDA-Q trajectories."""

    results = []
    for case in FROZEN_CASES:
        draws = sample_trajectories(case, trajectory_count, seed=case.seed)
        grouped = group_trajectories(draws)
        reference = conditioned_distribution(case, grouped)
        seed = case.seed * 10000 + 991
        fixed_logical_qubits = min(2, case.nqubits - 1)
        tn_counts = run_tn_fixed(
            case,
            grouped,
            shots_per_trajectory,
            seed,
            workdir,
            fixed_logical_qubits=fixed_logical_qubits,
        )
        cudaq_counts = run_cudaq(case, grouped, shots_per_trajectory, seed + 1000)
        results.append(
            {
                "case_id": case.case_id,
                "trajectories": trajectory_count,
                "unique_trajectories": len(grouped),
                "requested_shots_per_trajectory": shots_per_trajectory,
                "fixed_logical_qubits": fixed_logical_qubits,
                "tn_actual_shots": sum(tn_counts.values()),
                "cudaq_actual_shots": sum(cudaq_counts.values()),
                "tn_vs_conditioned_tvd": total_variation_distance(tn_counts, reference),
                "cudaq_vs_conditioned_tvd": total_variation_distance(cudaq_counts, reference),
                "tn_vs_cudaq_tvd": total_variation_distance(tn_counts, cudaq_counts),
                "tn_z_expectation_error": abs(
                    z_expectation(tn_counts, case.observable_qubits)
                    - z_expectation(reference, case.observable_qubits)
                ),
                "cudaq_z_expectation_error": abs(
                    z_expectation(cudaq_counts, case.observable_qubits)
                    - z_expectation(reference, case.observable_qubits)
                ),
            }
        )
    return {
        "fixed_logical_qubits": "min(2, nqubits - 1)",
        "trajectory_count": trajectory_count,
        "shots_per_trajectory": shots_per_trajectory,
        "cases": results,
    }


def gate_semantics_audit(workdir: Path, shots: int = 16384) -> dict[str, Any]:
    """Cover every coherent gate family used by upstream's random E2E test."""

    gate_cases = [
        ValidationCase(
            "one_qubit_gate_family",
            2,
            (
                Gate("h", (0,)),
                Gate("t", (0,)),
                Gate("x", (1,)),
                Gate("y", (1,)),
                Gate("z", (1,)),
                Gate("rx", (0,), 0.63),
                Gate("h", (0,)),
            ),
            (),
            (0,),
            1200,
        )
    ]
    for offset, controlled in enumerate(("cx", "cy", "cz", "ch", "crx")):
        gate_cases.append(
            ValidationCase(
                f"controlled_{controlled}",
                2,
                (
                    Gate("rx", (0,), 0.77),
                    Gate("rx", (1,), 0.39),
                    Gate(controlled, (0, 1), 0.58 if controlled == "crx" else None),
                    Gate("h", (0,)),
                    Gate("rx", (1,), 0.31),
                ),
                (),
                (0, 1),
                1201 + offset,
            )
        )
    results = []
    for case in gate_cases:
        grouped = (((), 1),)
        reference = exact_distribution(case)
        tn_counts = run_tn_fixed(case, grouped, shots, case.seed, workdir)
        cudaq_counts = run_cudaq(case, grouped, shots, case.seed + 100)
        results.append(
            {
                "case_id": case.case_id,
                "nqubits": case.nqubits,
                "gates": [gate.name for gate in case.gates],
                "tn_vs_exact_tvd": total_variation_distance(tn_counts, reference),
                "cudaq_vs_exact_tvd": total_variation_distance(cudaq_counts, reference),
                "tn_vs_cudaq_tvd": total_variation_distance(tn_counts, cudaq_counts),
            }
        )
    return {"shots_per_backend": shots, "cases": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("upstream/Accelerated_TN_PTSBE"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-counts", nargs="+", type=int, default=[16, 64, 256, 1024])
    parser.add_argument("--shots-per-trajectory", type=int, default=64)
    parser.add_argument("--null-trials", type=int, default=2000)
    args = parser.parse_args()
    if sorted(args.trajectory_counts) != args.trajectory_counts or min(args.trajectory_counts) <= 0:
        raise SystemExit("trajectory counts must be positive and increasing")

    root = Path(__file__).resolve().parents[1]
    upstream = args.upstream.resolve()
    actual_upstream = git_head(upstream)
    if actual_upstream != UPSTREAM_COMMIT:
        raise SystemExit(f"upstream revision mismatch: {actual_upstream}")
    sys.path.insert(0, str(upstream))
    install_cuquantum_gate_order_workaround()

    import cudaq

    cudaq.set_target("nvidia", option="fp32")
    all_case_results: list[dict[str, Any]] = []
    all_backend_checks: list[bool] = []
    with tempfile.TemporaryDirectory(prefix="q-tensor-validation-") as temporary:
        workdir = Path(temporary)
        audit = ordering_audit(workdir, shots=4096)
        stale_cache_audit = minimized_stale_cache_audit(workdir)
        for case in FROZEN_CASES:
            channel_reference = exact_distribution(case)
            master_draws = sample_trajectories(case, max(args.trajectory_counts), seed=case.seed)
            points: list[dict[str, Any]] = []
            for trajectory_count in args.trajectory_counts:
                grouped = group_trajectories(master_draws[:trajectory_count])
                conditioned = conditioned_distribution(case, grouped)
                backend_seed = case.seed * 10000 + trajectory_count
                tn_counts = run_tn(
                    case,
                    grouped,
                    args.shots_per_trajectory,
                    backend_seed,
                    workdir,
                    reverse_qubits=True,
                )
                cudaq_counts = run_cudaq(case, grouped, args.shots_per_trajectory, backend_seed + 1000)
                expected_shots = trajectory_count * args.shots_per_trajectory
                if sum(tn_counts.values()) != expected_shots or sum(cudaq_counts.values()) != expected_shots:
                    raise RuntimeError("a backend did not preserve the requested proportional shot total")

                tn_conditioned = backend_comparison(
                    case,
                    grouped,
                    args.shots_per_trajectory,
                    tn_counts,
                    conditioned,
                    args.null_trials,
                    backend_seed + 2000,
                )
                cudaq_conditioned = backend_comparison(
                    case,
                    grouped,
                    args.shots_per_trajectory,
                    cudaq_counts,
                    conditioned,
                    args.null_trials,
                    backend_seed + 3000,
                )
                tn_cudaq = two_backend_comparison(
                    case,
                    grouped,
                    args.shots_per_trajectory,
                    tn_counts,
                    cudaq_counts,
                    args.null_trials,
                    backend_seed + 4000,
                )
                checks = [tn_conditioned["passed"], cudaq_conditioned["passed"], tn_cudaq["passed"]]
                all_backend_checks.extend(checks)
                trajectory_tvd, trajectory_expectation = _metrics(
                    conditioned, channel_reference, case.observable_qubits
                )
                tn_channel_tvd, tn_channel_expectation = _metrics(
                    tn_counts, channel_reference, case.observable_qubits
                )
                cudaq_channel_tvd, cudaq_channel_expectation = _metrics(
                    cudaq_counts, channel_reference, case.observable_qubits
                )
                points.append(
                    {
                        "trajectories": trajectory_count,
                        "unique_trajectories": len(grouped),
                        "shots_per_trajectory": args.shots_per_trajectory,
                        "total_backend_shots": expected_shots,
                        "conditioned_distribution": conditioned,
                        "tn_counts": tn_counts,
                        "cudaq_counts": cudaq_counts,
                        "trajectory_vs_exact_channel": {
                            "tvd": trajectory_tvd,
                            "z_expectation_error": trajectory_expectation,
                        },
                        "tn_vs_conditioned_exact": tn_conditioned,
                        "cudaq_vs_conditioned_exact": cudaq_conditioned,
                        "tn_vs_cudaq": tn_cudaq,
                        "tn_vs_exact_channel": {
                            "tvd": tn_channel_tvd,
                            "z_expectation_error": tn_channel_expectation,
                        },
                        "cudaq_vs_exact_channel": {
                            "tvd": cudaq_channel_tvd,
                            "z_expectation_error": cudaq_channel_expectation,
                        },
                        "diagnosis": "finite_trajectory_and_shot_variance" if all(checks) else "semantic_mismatch",
                    }
                )
            tvds = [point["trajectory_vs_exact_channel"]["tvd"] for point in points]
            slope = float(np.polyfit(np.log(args.trajectory_counts), np.log(np.maximum(tvds, 1e-15)), 1)[0])
            all_case_results.append(
                {
                    **case_definition(case),
                    "reference_method": "exact complex128 Kraus-branch density matrix",
                    "exact_channel_distribution": channel_reference,
                    "exact_z_expectation": z_expectation(channel_reference, case.observable_qubits),
                    "sweep": points,
                    "convergence": {
                        "trajectory_tvd_log_log_slope": slope,
                        "endpoint_improved_over_start": tvds[-1] < tvds[0],
                    },
                }
            )
        fixed_audit = fixed_path_audit(workdir, trajectory_count=64, shots_per_trajectory=512)
        gate_audit = gate_semantics_audit(workdir)

    depolarizing_audit = []
    for operators, probability in ((3, 0.27), (15, 0.22)):
        intended = {0: 1 - probability} | {code: probability / operators for code in range(1, operators + 1)}
        actual = upstream_depolarizing_probabilities(operators, probability)
        depolarizing_audit.append(
            {
                "non_identity_operators": operators,
                "probability_parameter": probability,
                "intended_total_error_probability": probability,
                "upstream_effective_total_error_probability": 1 - actual[0],
                "intended_vs_upstream_tvd": total_variation_distance(intended, actual),
                "upstream_operator_probabilities": {str(key): value for key, value in actual.items()},
            }
        )

    dedup_draws = sample_trajectories(FROZEN_CASES[0], 64, seed=FROZEN_CASES[0].seed)
    dedup_retained = upstream_broken_deduplicate(dedup_draws)
    intended_grouped = group_trajectories(dedup_draws)
    broken_grouped = group_trajectories(dedup_retained)
    dedup_audit = {
        "case_id": FROZEN_CASES[0].case_id,
        "draws": len(dedup_draws),
        "intended_multiplicities": {str(key): value for key, value in intended_grouped},
        "retained_entries": len(dedup_retained),
        "retained_multiplicities": {str(key): value for key, value in broken_grouped},
        "distribution_tvd_from_reweighting": total_variation_distance(
            conditioned_distribution(FROZEN_CASES[0], intended_grouped),
            conditioned_distribution(FROZEN_CASES[0], broken_grouped),
        ),
    }

    corrected_passed = all(all_backend_checks) and all(
        item["convergence"]["endpoint_improved_over_start"] for item in all_case_results
    )
    conclusion = "PASS" if corrected_passed else "FAIL"
    record = {
        "schema_version": 1,
        "experiment_id": "seeded-proportional-validation-v1",
        "category": "q_tensor_original",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [
            sys.executable,
            str(Path(__file__)),
            "--upstream",
            str(upstream),
            "--output",
            str(args.output),
            "--trajectory-counts",
            *[str(value) for value in args.trajectory_counts],
            "--shots-per-trajectory",
            str(args.shots_per_trajectory),
            "--null-trials",
            str(args.null_trials),
        ],
        "git_commit": git_head(root),
        "upstream_commit": actual_upstream,
        "hardware": gpu_properties(),
        "software": {"python": platform.python_version(), "packages": package_versions()},
        "parameters": {
            "trajectory_counts": args.trajectory_counts,
            "shots_per_trajectory": args.shots_per_trajectory,
            "max_free_qubits": 2,
            "tn_dtype": "complex128",
            "cudaq_target": "nvidia fp32",
            "null_trials": args.null_trials,
            "null_quantile": NULL_QUANTILE,
            "bitstring_convention": "q0 first",
            "trajectory_sampling": "categorical Kraus alternatives; grouped with exact multiplicity",
        },
        "measurements": {
            "conclusion": conclusion,
            "all_backend_statistical_checks_passed": all(all_backend_checks),
            "all_convergence_endpoints_improved": all(
                item["convergence"]["endpoint_improved_over_start"] for item in all_case_results
            ),
            "cases": all_case_results,
            "qubit_order_audit": audit,
            "minimized_stale_cache_audit": stale_cache_audit,
            "upstream_fixed_path_audit": fixed_audit,
            "upstream_gate_semantics_audit": gate_audit,
            "upstream_depolarizing_sampler_audit": depolarizing_audit,
            "upstream_trajectory_deduplication_audit": dedup_audit,
        },
    }
    validate_result(record)
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if conclusion == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
