# Q-Tensor

Independent reproduction, correction, benchmarking, and prospective hardware study of GPU-accelerated noisy quantum trajectory simulation with tensor networks.

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
| **Prospective IBM model study** | Within one preregistered four-depth circuit family, Q-Tensor had lower TVD to IBM Kingston than the ideal model at all four tested depths; 3 depth-wise bootstrap intervals were entirely above zero | **PREREGISTERED PASS** |
| **Exact-channel check** | Frozen 100k-trajectory predictions reproduced an independent exact 4-qubit implementation of the specified channel to TVD **1.1e-4–2.7e-4** | **CONSISTENT** |
| **4096-shot resolution** | Exact-model residual percentiles were **61.35%, 97.24%, 96.21%, 83.73%**; all remained inside pointwise 95% envelopes, with two near the upper boundary | **RESOLUTION-LIMITED** |
| **Corrected Aer baseline** | Aer default and Q-Tensor differed by **<0.001 TVD** in hardware fit at every tested depth; every paired 95% CI crossed zero | **INDISTINGUISHABLE** |

The central finding is not that tensor networks always win. They do not. Q-Tensor measured a clear regime split: small proportional workloads favored CUDA-Q, while the exact 50q/200g workload produced a measured **6.875× equal-shot proportional crossover**. Separately, a prospectively frozen calibration-informed model had lower TVD than the ideal noiseless circuit at all four tested depths within one preregistered circuit family.

## Prospective IBM hardware study

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

Post-hoc exact-channel adjudication showed that the frozen 100,000-trajectory Q-Tensor predictions reproduce the independently constructed specified Pauli-plus-readout channel to TVD 1.1e-4–2.7e-4. This validates implementation consistency of the sampler with the stated model; it does not independently establish that the chosen conversion from IBM-reported gate error is the unique physical interpretation of that calibration quantity.

Using the exact model as the finite-shot null, the IBM residuals fell at the 61.35th, 97.24th, 96.21st, and 83.73rd percentiles. All are inside the pointwise 95% envelopes, but all four are above the null median and two are near the upper boundary, suggesting a small residual model discrepancy near the experiment's resolution limit.

A corrected exploratory Aer comparison was also statistically unresolved. Q-Tensor was numerically closer at two depths and Aer default at two, the hardware-fit TVDs differed by less than 0.001 at every depth, and every paired 95% confidence interval crossed zero.

The IBM job used **6 quantum seconds**.

See the full [IBM Kingston noise-model study](reports/IBM_KINGSTON_NOISE_MODEL_STUDY.md).

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

- [IBM Kingston calibration-informed noise-model study](reports/IBM_KINGSTON_NOISE_MODEL_STUDY.md)
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
- At 4096 hardware shots, Q-Tensor and the corrected standard Aer backend-derived model cannot be statistically ranked from these data.
- All four exact-model hardware residuals lie above their null medians and two lie near the pointwise 95% upper boundary; this is suggestive of a small residual model discrepancy near the available resolution limit.
- The exact-channel check validates that Q-Tensor samples its stated Pauli-plus-readout model correctly; it does not independently prove that the chosen interpretation of IBM's reported average gate error is uniquely correct.
- An initial exploratory Aer reconstruction retained measurement-family `gate_error` entries while also enabling separate readout error, effectively applying the same measurement calibration twice. The corrected comparator excludes those measurement-family gate entries.
- The IBM model uses independent stochastic Pauli gate noise and asymmetric readout error; it does not model all coherent, correlated, leakage, crosstalk, or time-dependent effects.
- The IBM study is a bounded calibration-informed noise-model result on one circuit family and one physical path; it is **not** a claim that PTSBE itself was directly validated by IBM hardware.

## Repository structure

`src/` contains independent utilities; `scripts/` reproducible runners; `configs/` versioned inputs; `benchmarks/` schemas; `results/` raw machine-readable observations; `reports/` interpreted results; `docs/` provenance and method notes; and `upstream/` the revision lock and fetch policy.
