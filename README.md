# Q-Tensor

Independent reproduction, correction, benchmarking, and prospective hardware validation of GPU-accelerated noisy quantum trajectory simulation with tensor networks.

Q-Tensor started from NVIDIA Research's open-source PTSBE work and asks three practical questions:

1. **Are the sampling semantics statistically correct?**
2. **Where does tensor-network batching actually become faster?**
3. **Can a frozen calibration-informed trajectory model predict real quantum hardware better than an ideal noiseless model?**

## Results at a glance

| Result | Observation | Status |
|---|---|---:|
| **Statistical validation** | Corrected proportional sampling passed **48/48** seeded TN/reference, CUDA-Q/reference, and TN/CUDA-Q finite-shot checks on 2–5 qubit exact-reference cases | **PASS** |
| **Small-workload regime** | No proportional crossover through the validated 11-qubit sweep; CUDA-Q remained faster | **NO CROSSOVER** |
| **50q proportional crossover** | Exact public 50-qubit / 200-gate Figure-3 workload: TN **40.419 s** vs CUDA-Q TensorNet **277.865 s** for the same frozen 1,000 effective shots | **6.875× TN speedup** |
| **Non-proportional regime** | Figure-3 PTSBE generated **24,378,496 distinct labeled records from 40 contractions**; this is a unique-data-harvest result, **not** a raw execution speedup | **REPRODUCED** |
| **Prospective IBM hardware test** | Within one preregistered four-depth circuit family, Q-Tensor had lower TVD to IBM Kingston than the ideal model at all four tested depths; 3 depth-wise bootstrap intervals were entirely above zero | **PREREGISTERED PASS** |
| **4096-shot resolution diagnostic** | Every observed Q-Tensor→IBM residual fell inside the 95% finite-shot TVD envelope under both the Q-Tensor and empirical IBM distributions | **RESOLUTION-LIMITED** |
| **Exploratory Aer baseline** | Q-Tensor was numerically closer at 2 depths and Aer default at 2; every paired 95% CI crossed zero and both models remained inside the 4096-shot resolution floor | **INDISTINGUISHABLE** |

The central finding is not that tensor networks always win. They do not. Q-Tensor measured a clear regime split: small proportional workloads favored CUDA-Q, while the exact 50q/200g workload produced a measured **6.875× equal-shot proportional crossover**. Separately, a prospectively frozen calibration-informed model had lower TVD than the ideal noiseless circuit at all four tested depths within one preregistered circuit family.

## Prospective IBM hardware validation

Q-Tensor's first real-QPU experiment was executed on IBM `ibm_kingston` using physical qubits `[79, 93, 94, 95]`.

Before the QPU run, Q-Tensor froze and hashed:

- the physical-qubit selection;
- calibration inputs;
- exact IBM ISA circuits;
- the noise-model construction rules;
- 100,000-trajectory predictions for each circuit;
- the TVD success criterion and bootstrap analysis protocol.

No model parameter was tuned after observing hardware results.

| Circuit | CZs | Ideal → IBM TVD | Q-Tensor → IBM TVD | Δ | Result |
|---|---:|---:|---:|---:|---|
| `QTIBM_CZ03` | 3 | 0.027627 | **0.022535** | +0.005092 | Win |
| `QTIBM_CZ06` | 6 | 0.041201 | **0.031467** | +0.009734 | **Strong win** |
| `QTIBM_CZ09` | 9 | 0.048038 | **0.030458** | +0.017580 | **Strong win** |
| `QTIBM_CZ12` | 12 | 0.046108 | **0.026653** | +0.019455 | **Strong win** |

Primary preregistered result: **PASS** — within this single four-depth circuit family, Q-Tensor had lower TVD than the ideal model at every tested depth, with the complete 95% bootstrap interval above zero at three depths.

A post-hoc resolution analysis showed that all four Q-Tensor→IBM residual TVDs were inside the expected 95% finite-shot envelope for a 4096-shot observation. An exploratory comparison with Qiskit Aer's standard backend-derived noise model was likewise unresolved at this shot count: Q-Tensor was numerically closer at two depths, Aer at two, every paired confidence interval crossed zero, and both models were inside the same finite-shot resolution floor.

The IBM job used **6 quantum seconds**.

See the full [IBM hardware validation report](reports/IBM_HARDWARE_VALIDATION.md).

## 50-qubit proportional crossover

On the exact public Figure-3 circuit-0 workload — 50 qubits, 200 coherent gates, 200 retained noise sites — Q-Tensor gave both methods the same ten frozen categorical trajectories and exactly 1,000 effective shots per backend.

| Timing / throughput | TN proportional | CUDA-Q TensorNet | CUDA-Q / TN |
|---|---:|---:|---:|
| Steady execution | **40.419 s** | 277.865 s | **6.875×** |
| Path-inclusive TN / CUDA-Q steady | 40.848 s | 277.865 s | **6.802×** |
| Effective shots/s | **24.74** | 3.60 | **6.875×** |

This establishes a crossover at this measured 50q/200g point only. Q-Tensor does **not** infer where the crossover boundary lies between the earlier 11-qubit measurements and this workload.

See the [Figure-3 proportional control](reports/FIGURE3_PROPORTIONAL_CONTROL.md).

## Why the sampling correction matters

The upstream artifact can produce misleading validation results because CUDA-Q reference files are cached by trajectory serial number rather than by the circuit/noise configuration that generated them. A clean run can pass while an immediate rerun against stale cached reference data can fail.

Q-Tensor also separates two sampling semantics that should not be conflated:

- **Proportional sampling:** trajectory multiplicities are preserved and outputs represent physical sampling weights.
- **Non-proportional unique-record harvesting:** deliberately expands distinct labeled outcomes for dataset generation; useful, but not an IID proportional sample and not a raw simulator speedup.

Q-Tensor uses categorical Kraus sampling and preserves trajectory multiplicity for its scientific and benchmark claims.

See the [statistical validation report](reports/STATISTICAL_VALIDATION.md) and [regime reconciliation](reports/REGIME_RECONCILIATION.md).

## Hardware

Primary local test system:

- NVIDIA GeForce RTX 5090, 32 GiB, compute capability 12.0, 170 SMs
- Intel Core Ultra 9 285K
- Ubuntu 26.04.1 under WSL2

Real-QPU validation:

- IBM Quantum `ibm_kingston`
- 156-qubit processor
- selected physical path `[79, 93, 94, 95]`
- 4 × 4,096 hardware shots
- 6 IBM-reported quantum seconds

See [`reports/gpu.txt`](reports/gpu.txt) and [`reports/environment.txt`](reports/environment.txt).

## Reproduce the foundation

```bash
conda env create -f environment/environment.yml
conda activate q-tensor
python scripts/fetch_upstream.py
pytest
python scripts/run_upstream_smoke.py \
  --output results/smoke/latest_upstream_core.json
PYTHONPATH=src python scripts/run_statistical_validation.py \
  --upstream upstream/Accelerated_TN_PTSBE \
  --output results/statistical_validation/latest.json
```

The upstream checkout is pinned at:

```text
0569f848d9a3e385d6c20162c534a8031f6a69c5
```

GPU tests are intentionally kept out of the normal unit suite:

```bash
Q_TENSOR_UPSTREAM="$PWD/upstream/Accelerated_TN_PTSBE" pytest -m gpu
```

## Reports

- [IBM prospective hardware validation](reports/IBM_HARDWARE_VALIDATION.md)
- [Figure-3 proportional crossover](reports/FIGURE3_PROPORTIONAL_CONTROL.md)
- [Statistical validation](reports/STATISTICAL_VALIDATION.md)
- [Complexity crossover sweep](reports/COMPLEXITY_CROSSOVER.md)
- [Regime reconciliation](reports/REGIME_RECONCILIATION.md)
- [Performance analysis](reports/PERFORMANCE_ANALYSIS.md)
- [Executive summary](reports/EXECUTIVE_SUMMARY.md)
- [Smoke-test record](reports/SMOKE_TEST.md)

## Methodology and provenance

- [Paper notes and sampling semantics](docs/PAPER_NOTES.md)
- [RTX 5090 / WSL feasibility](docs/FEASIBILITY.md)
- [Upstream source, revision, dependencies, and licenses](docs/UPSTREAM.md)
- [Project worklog](docs/WORKLOG.md)

Saved results distinguish `UPSTREAM CLAIM`, `REPRODUCED RESULT`, and `Q-TENSOR ORIGINAL RESULT`. Machine-readable result records capture hardware, software, parameters, commands, revisions, timestamps, and measurements.

The IBM experiment additionally preserves separate hashes for calibration selection, exact circuits, frozen predictions, raw hardware results, and final preregistered analysis.

## Upstream NVIDIA work

Q-Tensor uses NVIDIA Research's Apache-2.0 [`NVlabs/Accelerated_TN_PTSBE`](https://github.com/NVlabs/Accelerated_TN_PTSBE), associated with [arXiv:2604.08467v1](https://arxiv.org/abs/2604.08467).

Upstream source and paper results are not relabeled as Q-Tensor results.

## Limitations

The results are deliberately bounded to measured workloads and hardware.

- The original H100 80 GB campaigns were not rerun.
- The 6.875× proportional speedup is established only at the measured 50q/200g point; the crossover boundary was not located.
- The IBM experiment covers one processor, one four-qubit path, one nested four-depth circuit family, and one calibration regime; the four depth points are not independent replications.
- At 4096 hardware shots, Q-Tensor and the standard Aer backend-derived model cannot be statistically ranked from these data because both lie inside the measured finite-shot resolution envelope.
- An exploratory Aer no-relaxation cross-check did not reproduce Q-Tensor's gate-noise distribution; decomposition localized the difference to gate-noise semantics/application rather than readout. This remains an open implementation diagnostic.
- The IBM model uses independent stochastic Pauli gate noise and asymmetric readout error; it does not model all coherent, correlated, leakage, crosstalk, or time-dependent effects.
- The IBM result validates Q-Tensor's calibration-informed prediction methodology under the tested conditions; it is **not** a claim that PTSBE itself was directly validated by IBM hardware.

## Repository structure

`src/` contains independent utilities; `scripts/` reproducible runners; `configs/` versioned inputs; `benchmarks/` schemas; `results/` raw machine-readable observations; `reports/` interpreted results; `docs/` provenance and method notes; and `upstream/` the revision lock and fetch policy.
