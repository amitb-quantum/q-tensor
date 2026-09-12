# Prospective Calibration-Informed Noise-Model Study on IBM Kingston

**Date:** September 11, 2026  
**Backend:** IBM Quantum `ibm_kingston`  
**Hardware:** 156-qubit IBM quantum processor  
**Physical qubits:** `[79, 93, 94, 95]`  
**IBM Runtime job:** `dai8c4hhvn6c73cstebg`  
**Q-Tensor repository:** `amitb-quantum/q-tensor`

## Summary

Q-Tensor performed a prospective, calibration-informed prediction of four quantum circuits before those circuits were executed on IBM Quantum hardware.

The prediction model, physical qubits, circuit family, calibration inputs, statistical success criterion, and model parameters were frozen before observing any QPU results.

Across all four preregistered circuits, the frozen Q-Tensor noisy prediction was closer to the observed IBM hardware distribution than the corresponding ideal noiseless prediction.

Within this single preregistered four-depth circuit family:

- Q-Tensor had lower TVD to IBM than the ideal model at **all four tested depths**
- the complete preregistered 95% bootstrap interval was above zero at **three depths**
- the full gate-noise model was numerically closer than the preregistered readout-only control at **three depths**

The preregistered primary success criterion therefore **PASSed**.

These four nested depth points share one physical path and circuit family and should not be interpreted as four independent experimental replications.

Post-hoc exact-channel analysis confirmed that the frozen trajectory prediction reproduces its specified noise channel to Monte Carlo accuracy. The remaining exact-model discrepancy to IBM is near the 4096-shot resolution limit: all four residuals are above the corresponding null median and two lie near the pointwise 95% upper boundary.

A corrected exploratory Qiskit Aer comparator produced hardware-fit TVDs within 0.001 of Q-Tensor at every tested depth, with no statistically resolved difference between the two models.

This experiment does not establish that the model generalizes to arbitrary circuits, processors, or noise regimes. It demonstrates that, for this prospectively frozen four-circuit experiment on IBM Kingston, a simple calibration-informed stochastic noise model predicted the physical device more accurately than an ideal noiseless model.

---

## Motivation

Q-Tensor began as an independent reproduction and characterization of NVIDIA's accelerated tensor-network trajectory work.

Earlier Q-Tensor experiments established two distinct computational regimes:

1. For small proportional-sampling workloads, explicit CUDA-Q execution was substantially faster than tensor-network batching.
2. On the NVIDIA Figure-3 50-qubit / 200-gate workload, Q-Tensor measured a **6.875× proportional equal-shot speedup** from tensor-network batching over the matched CUDA-Q TensorNet baseline on an RTX 5090.

Those results answered a performance question:

> When does accelerated tensor-network trajectory simulation become computationally useful?

The IBM experiment asked a different question:

> Can a frozen, calibration-informed proportional trajectory model predict the behavior of a real quantum processor better than the ideal noiseless circuit predicts it?

The hardware experiment was intentionally designed as a prospective test rather than a post-hoc fit.

---

## Experimental Integrity

The sequence was fixed before examining IBM hardware outcomes:

1. Audit currently available IBM processors and calibration history.
2. Select a robust physical-qubit path.
3. Freeze the relevant calibration and topology inputs.
4. Freeze the circuit family.
5. Freeze the exact IBM ISA circuits.
6. Define the Q-Tensor noise model.
7. Generate and hash the Q-Tensor predictions.
8. Verify that the hardware calibration inputs consumed by the model had not changed.
9. Submit the frozen circuits to IBM Kingston.
10. Freeze the raw IBM result before comparison.
11. Apply the pre-specified statistical analysis.

No model parameter was tuned after observing the QPU results.

No circuit was rerun because of an unfavorable scientific outcome.

---

## Physical-Qubit Selection

A five-day IBM calibration/topology audit was performed before submission.

The selected path was:

`79 -- 93 -- 94 -- 95`

At selection time:

- backend: `ibm_kingston`
- path: `[79, 93, 94, 95]`
- robust across the audit history: **yes**
- estimated circuit error: **8.63%**
- path-selection score: **0.10200018**
- pending jobs: **0**

The selected CZ calibration errors were approximately:

| Edge | CZ error |
|---|---:|
| 79–93 | 0.1169% |
| 93–94 | 0.1195% |
| 94–95 | 0.1319% |

Selected-qubit readout errors were:

| Qubit | Readout error |
|---:|---:|
| 79 | 0.5737% |
| 93 | 0.6836% |
| 94 | 1.2817% |
| 95 | 0.5859% |

The relevant coherence values were also preserved in the frozen calibration record.

---

## Calibration Change Guard

The backend-wide calibration timestamp changed after the initial preregistration was created.

The experiment therefore stopped automatically before QPU submission.

A stricter comparison was then performed against every calibration parameter actually consumed by the frozen prediction model on physical qubits `79,93,94,95` and edges:

- `79-93`
- `93-94`
- `94-95`

Although IBM's global backend calibration timestamp had changed, none of the gate errors, durations, or asymmetric readout probabilities used by the frozen model had changed.

The final pre-submission decision was:

`FROZEN_PREDICTION_STILL_VALID`

A second live guard was executed immediately before submission and again found:

`changed_model_inputs = 0`  
`FINAL PREFLIGHT = PASS`

This distinction matters because a large processor can receive calibration updates unrelated to the physical qubits used by a particular experiment.

---

## Frozen Circuit Family

Four circuits used the same four physical qubits and progressively increased entangling-gate exposure.

| Circuit | CZ count | Hardware shots |
|---|---:|---:|
| `QTIBM_CZ03` | 3 | 4096 |
| `QTIBM_CZ06` | 6 | 4096 |
| `QTIBM_CZ09` | 9 | 4096 |
| `QTIBM_CZ12` | 12 | 4096 |

Total hardware measurements:

`4 × 4096 = 16,384 shots`

The logical-to-physical mapping was fixed:

- `q0 -> 79`
- `q1 -> 93`
- `q2 -> 94`
- `q3 -> 95`

The circuits were constructed from IBM-native operations and frozen as exact QPY/QASM artifacts before QPU execution.

No readout-error mitigation was applied to the hardware results.

---

## Q-Tensor Prediction Model

Q-Tensor used its previously validated categorical Kraus trajectory machinery.

### Coherent circuit

- IBM `SX` was represented as `RX(pi/2)`.
- The two differ only by a global phase for the purposes of these measurement probabilities.
- `RZ` was modeled as an ideal virtual operation.
- `CZ` was modeled directly.

### Gate noise

Each calibrated physical gate was followed by an independent categorical Pauli depolarizing channel.

IBM-reported average gate error `r` was converted to a Pauli-channel probability using:

- 1-qubit: `p = (3/2) r`
- 2-qubit: `p = (5/4) r`

Repeated sampled trajectories retained their exact multiplicities.

The post-hoc exact-channel check validates that the implementation correctly samples this **stated** channel construction. It does not independently establish that converting IBM-reported average gate error using these Pauli-channel formulas is the unique or complete physical interpretation of the calibration quantity.

### Readout

Independent asymmetric assignment errors were constructed from the frozen IBM properties:

- `P(meas 0 | prep 1)`
- `P(meas 1 | prep 0)`

for each of the four physical qubits.

### Relaxation

T1/T2 values were preserved as provenance but explicit gate-time relaxation was not added to the primary prediction.

This was intentional: IBM's calibrated gate-error values already incorporate physical error contributions, and independently adding another relaxation model risked double counting decoherence.

### Frozen trajectory sampling

Each circuit used **100,000 proportional trajectory draws** with fixed seeds.

Monte Carlo stability was checked by comparing the 50,000-trajectory prefix against the full 100,000-trajectory prediction.

| Circuit | 50k vs 100k TVD |
|---|---:|
| CZ03 | 0.000394 |
| CZ06 | 0.000237 |
| CZ09 | 0.000130 |
| CZ12 | 0.000348 |

This stochastic-model uncertainty was small relative to the observed QPU discrepancies.

---

## Pre-Specified Hypothesis

The primary metric was **total variation distance (TVD)**.

For distributions \(P\) and \(Q\):

`TVD(P,Q) = 1/2 Σ |P(x) - Q(x)|`

The primary success rule was fixed before hardware analysis:

> Q-Tensor's full calibration-informed prediction must have lower TVD to IBM hardware than the ideal noiseless prediction on at least 3 of the 4 circuits.

Per-circuit improvement was defined as:

`Δ = TVD(Ideal, IBM) - TVD(Q-Tensor, IBM)`

Therefore, `Δ > 0` means Q-Tensor predicted the hardware more accurately than the ideal circuit did.

Uncertainty was evaluated with 10,000 nonparametric bootstrap resamples of the observed IBM outcomes using a fixed seed and a 95% percentile interval.

A **strong predictive win** required the entire 95% interval for Δ to remain above zero.

---

## Prospective Hardware Comparison

IBM Runtime job:

`dai8c4hhvn6c73cstebg`

Each circuit returned the requested 4096 shots.

The raw hardware result was frozen and hashed before comparison with the Q-Tensor prediction.

### Primary comparison

| Circuit | CZ | Ideal → IBM TVD | Q-Tensor → IBM TVD | Δ | 95% CI for Δ | Result |
|---|---:|---:|---:|---:|---:|---|
| CZ03 | 3 | 0.027627 | **0.022535** | +0.005092 | [-0.003348, +0.012888] | Win |
| CZ06 | 6 | 0.041201 | **0.031467** | +0.009734 | [+0.003030, +0.013630] | **Strong win** |
| CZ09 | 9 | 0.048038 | **0.030458** | +0.017580 | [+0.005964, +0.024128] | **Strong win** |
| CZ12 | 12 | 0.046108 | **0.026653** | +0.019455 | [+0.004195, +0.021950] | **Strong win** |

Summary:

- Q-Tensor beats ideal: **4 / 4**
- Statistically strong predictive wins: **3 / 4**
- Primary preregistered result: **PASS**

The shallowest three-CZ circuit favored Q-Tensor directionally, but its bootstrap interval crossed zero and therefore was not classified as statistically strong.

---

## Readout-Only Control

A second prediction contained IBM's calibrated readout errors but omitted modeled gate noise.

This control tested whether Q-Tensor's advantage could be explained simply by incorporating known measurement error.

Result:

**Full gate-noise model beat readout-only model: 3 / 4 circuits**

This indicates that modeled gate noise added predictive information beyond readout calibration alone on most of the preregistered circuit family.

---

## Exact-Channel Correctness Check

After the preregistered hardware analysis, the four-qubit prediction model was independently reconstructed as an exact density-matrix channel.

This adjudicator used:

- deterministic four-qubit density-matrix evolution;
- the stated joint categorical Pauli channels;
- the frozen IBM calibration values;
- the stated `p = (3/2)r` and `p = (5/4)r` conversion rules;
- the frozen asymmetric readout confusion matrices;
- no Q-Tensor trajectory sampling; and
- no Aer noise-model construction.

Results:

| Circuit | Ideal exact ↔ frozen | Readout-only exact ↔ frozen | Gate-noise exact ↔ Q-Tensor | Full exact ↔ Q-Tensor |
|---|---:|---:|---:|---:|
| CZ03 | 0 | 0 | 0.000141 | 0.000134 |
| CZ06 | 0 | 0 | 0.000118 | 0.000114 |
| CZ09 | 0 | 0 | 0.000278 | 0.000265 |
| CZ12 | 0 | 0 | 0.000179 | 0.000172 |

The ideal and readout-only comparisons agree to machine precision. The full noisy predictions agree at TVD `1.1e-4`–`2.7e-4`, the same scale as the trajectory Monte Carlo stability diagnostics.

This establishes that Q-Tensor's 100,000-trajectory implementation is sampling its **specified** Pauli-plus-readout channel correctly to finite-trajectory accuracy.

It does **not** independently establish that this channel is a complete physical model of IBM Kingston or that the selected conversion from IBM-reported average gate error is the only valid interpretation of the calibration quantity.

At four qubits, an exact density-matrix calculation is simpler and more accurate than trajectory simulation. The purpose of this check is therefore implementation adjudication, not computational advantage.

Machine-readable result: `experiments/ibm_kingston_20260911/analysis/EXACT_CHANNEL_VALIDATION.json`.

---

## Post-Hoc Exact-Model Resolution Diagnostic

The finite-shot analysis was recomputed using the independently constructed exact model rather than the 100,000-trajectory approximation.

For each depth, 100,000 synthetic 4096-shot observations were drawn from the exact specified model.

| Circuit | Exact model → IBM TVD | Exact-model 95% null envelope | Observed percentile |
|---|---:|---:|---:|
| CZ03 | 0.022545 | [0.013170, 0.031261] | 61.35% |
| CZ06 | 0.031504 | [0.013552, 0.031720] | 97.24% |
| CZ09 | 0.030483 | [0.013307, 0.031520] | 96.21% |
| CZ12 | 0.026673 | [0.013668, 0.031879] | 83.73% |

Every individual residual remains inside its pointwise 95% finite-shot envelope.

However, the family-level pattern is not centered on the null: all four residuals are above the corresponding null median, and CZ06 and CZ09 lie close to the upper boundary.

The appropriate interpretation is therefore:

> The remaining model discrepancy is at or just below the resolution available from 4096 hardware shots. The consistent one-sided pattern is suggestive of a small residual model error, but these four nested depths are correlated observations from one circuit family and are not treated as four independent hypothesis tests.

This post-hoc resolution analysis does not alter the preregistered primary result.

Machine-readable result: `experiments/ibm_kingston_20260911/analysis/EXACT_SHOT_NOISE_FLOOR.json`.

---

## Corrected Exploratory Qiskit Aer Baseline

A post-hoc comparison was performed against Qiskit Aer's backend-properties noise construction using the same frozen Kingston calibration data.

The initial exploratory reconstruction inadvertently retained IBM measurement-family `gate_error` entries while simultaneously enabling Aer's separate asymmetric readout-error model. Those measurement gate-error values duplicated the same calibration quantities already represented by readout error and therefore did not provide an apples-to-apples comparison with Q-Tensor.

The comparator was corrected by restricting the reconstructed gate model to the operations represented by Q-Tensor:

- `sx`
- `x`
- `rz`
- `cz`

Asymmetric readout calibration remained enabled separately.

Independent channel checks then established:

- Aer and Q-Tensor use equivalent no-relaxation SX and CZ depolarizing channels for the stated conversion rules;
- an exact Aer superoperator execution agrees with the independently constructed exact specified channel to approximately `1e-16`;
- the earlier large no-relaxation discrepancy was caused by the comparator construction, not by Q-Tensor's trajectory sampler.

Corrected exploratory hardware comparison:

| Circuit | Q-Tensor → IBM TVD | Aer default → IBM TVD | Aer − Q-Tensor | Paired 95% CI |
|---|---:|---:|---:|---:|
| CZ03 | 0.022535 | 0.021571 | -0.000964 | [-0.002347, +0.001323] |
| CZ06 | 0.031467 | 0.031194 | -0.000273 | [-0.001832, +0.001120] |
| CZ09 | 0.030458 | 0.031011 | +0.000553 | [-0.001029, +0.002689] |
| CZ12 | 0.026653 | 0.026857 | +0.000204 | [-0.001420, +0.001786] |

Aer default was numerically closer at CZ03 and CZ06; Q-Tensor was numerically closer at CZ09 and CZ12. The difference in hardware-fit TVD was below `0.001` at every depth, and every paired 95% confidence interval crossed zero.

The defensible conclusion is:

> Q-Tensor and the corrected standard Aer backend-derived model are statistically indistinguishable with the available 4096-shot IBM dataset.

The initial exploratory Aer numbers committed before the measurement-entry correction are superseded and are not used for scientific conclusions.

Machine-readable result: `experiments/ibm_kingston_20260911/analysis/AER_BASELINE.json`.

---

## Increasing-CZ Observation

Within this specific four-circuit family, the advantage of Q-Tensor over the ideal prediction increased as the number of CZ gates increased:

- 3 CZ  → Δ = +0.0051
- 6 CZ  → Δ = +0.0097
- 9 CZ  → Δ = +0.0176
- 12 CZ → Δ = +0.0195

This is an observation from this experiment, not a claimed general scaling law.

A larger prospectively designed experiment would be required to determine whether this behavior generalizes across circuit families, qubit paths, processors, and calibration regimes.

---

## Provenance and Cryptographic Record

The experiment preserved separate hashes for selection, calibration, circuits, prediction, hardware result, and analysis.

### Physical-path selection

`4c0a3087cbb1a908451c224680bac31c296d2ded451475943b2024baff03823c`

### Calibration input

`53e9c30f9e94fae025fcf65a8f262c29ffba4ab3f12704f482e3c4ef92cff70d`

### Frozen target input

`fe9a33349c2df33aae55800ef6abc57800ba880905322b5eb9f1df257e8ac51a`

### Circuit-family specification

`160cdf82b3c3637ff8331f8a7448818903c94cc8e9a26cf0529194510a5b571d`

### Preregistration manifest

`2c5b009fa86c2847af3747219eca0c9d5ae269475c0e0262c21cddeb54eb96a8`

### Frozen IBM QPY circuits

`8b35ab3ede6f169e5129e5dbea11473d0716d662d2eceed0390d3240e2c65140`

### Frozen Q-Tensor predictions

`40887db2dd269101023bd05435563281eebd6a60de63e0b0a7616162698c4700`

### IBM submission record

`ef527c691fedfaab846a0ff3b268ad771a0274922d6ffacc2fab3cdc902c9054`

### Frozen raw IBM hardware result

`b875a74cfdf9c8a66ed4ba559ed274f09623ea7c21f465cbad9f23f0d23166e0`

### Final preregistered analysis

`e4df94d8aea6a532fc9eaf17c36ecfd0d7ac11e0dc0c2b4bd7a2c63fe8319a83`

These hashes establish artifact identity and integrity within the recorded workflow. Because the repository was first published after QPU execution, they do **not** provide an independent public timestamp proving that the prediction artifact existed before the hardware run.

The prediction is nevertheless constrained by reproducibility: it regenerates from the frozen circuit family, calibration inputs, fixed seeds, and explicitly stated gate-error conversion rules. This limits opportunities for arbitrary post-hoc fitting, but it is weaker evidence than an external pre-execution timestamp. Future prospective runs should externally timestamp the preregistration manifest before submission.

---

## Analysis-Protocol Recovery Note

The statistical analysis protocol was specified before QPU submission, including:

- primary TVD metric
- 3-of-4 success criterion
- delta definition
- bootstrap method
- 10,000 replicates
- fixed bootstrap seed
- 95% interval
- readout-only secondary control

The corresponding local `ANALYSIS_PROTOCOL.json` file was inadvertently absent when the first post-run analysis command was invoked.

That invocation terminated immediately with `FileNotFoundError` before reading, displaying, or comparing the IBM counts.

The file was then reconstructed from the already specified pre-submission protocol, and a recovery note was recorded before analysis proceeded.

No metric, threshold, statistical test, model parameter, or success criterion was changed after the QPU result existed.

This procedural detail is retained explicitly rather than omitted.

---

## Interpretation

The experiment supports a narrow but meaningful conclusion:

> For four prospectively frozen circuits executed on IBM Kingston using physical qubits 79, 93, 94, and 95, Q-Tensor's calibration-informed stochastic prediction had lower TVD to the observed hardware distribution than the ideal noiseless model at every tested depth within the single preregistered circuit family.

Three of those four improvements remained positive across the complete preregistered 95% bootstrap interval.

This demonstrates predictive value from a simple calibration-informed trajectory model under the tested conditions.

It does **not** demonstrate:

- universal accuracy across IBM devices
- general accuracy for arbitrary circuit families
- full modeling of coherent or correlated hardware error
- quantum advantage
- that independent Pauli noise is a complete physical description
- that PTSBE itself has been directly validated by IBM hardware

The hardware experiment validates the Q-Tensor prediction methodology used here.

---

## Relationship to the NVIDIA PTSBE Work

Q-Tensor was motivated by NVIDIA's open-source work on accelerated tensor-network trajectory simulation.

Q-Tensor independently reproduced and characterized that work on an RTX 5090 and found two distinct regimes.

### Proportional sampling

On NVIDIA's published 50-qubit / 200-gate workload, Q-Tensor measured:

- TN proportional runtime: **40.419 s**
- CUDA-Q TensorNet runtime: **277.865 s**
- Equal-shot speedup: **6.875×**
- Path-inclusive speedup: **6.802×**

### Non-proportional data generation

For the paper's unique-record harvesting regime, Q-Tensor reproduced the extremely high data-generation efficiency enabled by batched tensor-network sampling.

However, Q-Tensor explicitly distinguishes this record-generation metric from raw wall-clock acceleration.

### IBM extension

The IBM Kingston experiment adds a separate dimension:

> not only whether accelerated trajectory methods can generate simulation data efficiently, but whether a prospectively frozen calibration-informed trajectory model carries useful predictive information about real quantum hardware.

The IBM experiment itself used the validated Q-Tensor proportional trajectory machinery on a four-qubit circuit family rather than requiring PTSBE acceleration.

---

## Limitations

This first hardware experiment intentionally remained small.

Important limitations include:

- one IBM processor
- one four-qubit physical path
- four related circuits
- one calibration regime
- 4096 hardware shots per circuit
- independent stochastic Pauli gate noise
- independent asymmetric readout error
- no explicit coherent-error model
- no crosstalk model
- no leakage model
- no correlated-error model
- no explicit gate-time T1/T2 channel
- no readout mitigation
- no post-hardware fitting

The experiment should therefore be viewed as a prospective proof of predictive value, not a complete validation of a device-noise model.

---

## Recommended Next Experiment

The highest-value replication would be a fully prospective test on a second independent IBM processor, such as `ibm_fez`, using:

1. a newly audited robust physical path,
2. the same frozen model-construction rules,
3. a new preregistered circuit family,
4. predictions sealed before hardware execution,
5. the same TVD and bootstrap methodology.

Successful replication across a second processor would provide substantially stronger evidence that the methodology generalizes beyond a single favorable Kingston path.

No additional hardware experiment is required to interpret the present Kingston result.

---

## Bottom Line

Q-Tensor's first prospectively frozen hardware experiment passed its preregistered criterion.

- **All four tested depths:** Q-Tensor had lower TVD to IBM than the ideal model
- **Three tested depths:** preregistered bootstrap interval entirely above zero
- **4096-shot resolution:** every Q-Tensor residual remained inside the finite-shot 95% envelope
- **Aer exploratory baseline:** Q-Tensor and Aer default were statistically indistinguishable at this resolution

The result establishes a concrete connection between Q-Tensor's classical noisy-trajectory modeling and measurements from a real IBM quantum processor, while preserving a cryptographic provenance chain from calibration selection through final analysis.

---

## QPU Usage

**IBM-reported QPU usage:** 6 quantum seconds

**IBM usage status:** complete

**Runtime timestamps:**
- running: `2026-09-11T22:52:03.951154Z`
- finished: `2026-09-11T22:54:04.034308Z`

The experiment used only six seconds of IBM-reported quantum execution time.
