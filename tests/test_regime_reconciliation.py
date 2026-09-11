from pathlib import Path

from scripts.run_regime_reconciliation import (
    OFFICIAL_BATCH_QUBITS,
    circuit_summary,
    common_command,
    reduction,
    retained_circuit_block,
)


def test_retained_circuit_block_stops_at_next_circuit() -> None:
    text = "RUN 1/1 | circuit_id=0:\nvalue 0\nRUN 1/1 | circuit_id=1:\nvalue 1\n"
    assert "value 0" in retained_circuit_block(text, 0)
    assert "value 1" not in retained_circuit_block(text, 0)


def test_official_command_freezes_public_artifact_configuration(tmp_path: Path) -> None:
    command = common_command(Path("/upstream"), tmp_path, tmp_path / "out.json", 100)
    assert command[command.index("--nqubits") + 1] == "50"
    assert command[command.index("--ngates") + 1] == "200"
    assert command[command.index("--nnoise_samples") + 1] == "10"
    assert command[command.index("--qubits_per_batch") + 1] == OFFICIAL_BATCH_QUBITS
    assert command[command.index("--num_hyper_samples") + 1] == "100"


def test_circuit_summary_counts_noise_sites(tmp_path: Path) -> None:
    stim = tmp_path / "circuit.stim"
    stim.write_text("H 0\nX_ERROR(0.1) 0\nCX 0 3\nDEPOLARIZE2(0.2) 0 3\n", encoding="utf-8")
    summary = circuit_summary(stim)
    assert summary["coherent_gate_count"] == 2
    assert summary["single_qubit_noise_sites"] == 1
    assert summary["two_qubit_noise_sites"] == 1
    assert summary["two_qubit_label_separation_max"] == 3


def test_reduction_decomposes_throughput_into_yield_and_runtime() -> None:
    pts_metrics = {
        "num_total_shots_mean": 200.0,
        "time_contraction_loop_mean": 10.0,
        "time_build_expr_operands_mean": 2.0,
        "time_execution_mean": 12.0,
        "time_contract_gpu_total_mean": 8.0,
        "time_apply_errors_total_mean": 1.0,
        "num_contractions_mean": 40.0,
    }
    for index in range(4):
        pts_metrics[f"num_contractions_batch_{index}_mean"] = 10.0
        pts_metrics[f"gpu_contract_batch_{index}_mean"] = 2.0
    ptsbe = {
        "subprocess_wall_seconds": 15.0,
        "upstream_result": {
            "ptsbe": pts_metrics,
            "individual_results": [
                {"ptsbe": {"time_stim_to_pts": 0.2, "time_pts_sampling": 0.3}}
            ],
        },
    }
    cudaq = {
        "subprocess_wall_seconds": 7.0,
        "upstream_result": {
            "cudaq": {"time_sample_mean": 5.0, "time_total_mean": 6.0}
        }
    }
    result = reduction(ptsbe, cudaq, cudaq)
    comparison = result["public_artifact_cudaq_100_hypersamples"]
    assert comparison["raw_runtime_speedup_cudaq_over_ptsbe"] == 0.5
    assert comparison["official_metric_throughput_advantage"] == 100.0
    assert result["ptsbe_loop_minus_gpu_and_error_seconds"] == 1.0
