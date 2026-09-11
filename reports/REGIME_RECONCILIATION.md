# PTSBE regime reconciliation

## Conclusion

**CONDITIONAL PASS.** A meaningful PTSBE regime exists on the RTX 5090 for the
paper's non-proportional data-collection objective: the reproduced public
Figure 3 configuration delivered **24,378,496 unique labeled records from 40
contractions**, or **563,717 records/s**. By the paper's throughput definition,
that is **1,468,244x** the public-artifact CUDA-Q control (100 optimizer
hypersamples), and **881,660x** the manuscript-stated CUDA-Q control (one
hypersample).

This is not an execution-time crossover and is not a proportional physical-shot
speedup. The PTSBE contraction loop took **43.246 s**, while one CUDA-Q sample
took **2.605 s** (100 hypersamples) or **1.564 s** (one hypersample). Thus the
raw CUDA-Q/PTSBE runtime ratios were 0.0602 and 0.0362: PTSBE was respectively
**16.6x and 27.7x slower**. The apparent contradiction with Q-Tensor's earlier
no-crossover result disappears once output semantics and the compared unit of
work are matched.

Machine-readable evidence is in
[`results/performance/2026-09-11_regime_reconciliation.json`](../results/performance/2026-09-11_regime_reconciliation.json).

## 1. Closest paper workload reproduced

The bounded target is circuit 0 from the public Figure 3 non-proportional
data-collection benchmark, the smallest official workload retained in the
upstream artifact. The harness runs the artifact itself at pinned commit
`0569f848d9a3e385d6c20162c534a8031f6a69c5`.

| Property | Reproduced configuration |
|---|---|
| Circuit | Exact retained `circuit_0.qpy` / `circuit_0.stim` |
| Artifact hashes | QPY `0a7257b...99c82`; Stim `962886a...ffa7` |
| Size | 50 qubits, 200 coherent gates, Qiskit depth 10 |
| Gate mix | 160 one-qubit and 40 two-qubit gates |
| Noise | Pauli X/Y/Z after one-qubit gates; two-qubit depolarizing after two-qubit gates; retained probabilities 0.02017--0.19967 |
| Trajectories | 10, as in the artifact; no seed option is exposed by the public driver |
| Precision/backend | complex128; CUDA-Q `tensornet` |
| PTSBE batches | qubits `[10,10,2,28]`, shots `[1,1,1,100]`; unique sampling and exhaustive nonzero final-batch harvest |
| Contraction planning | 100 PTSBE optimizer hypersamples, smart option disabled, reuse cache `nruns=1,000,000` |
| CUDA-Q planning | both the public script's 100-hypersample setting and the paper text's one-hypersample setting |

The retained circuit's two-qubit operands have label separations 1--44 (median
11), so the exact artifact is reported here rather than assuming a geometric
nearest-neighbor mapping from the paper's general circuit-family description.
Likewise, the upstream trajectory generator's ordered per-Pauli Bernoulli
behavior is preserved. Neither semantics nor implementation was corrected for
this reproduction.

The official metric is:

`PTSBE unique labeled records / contraction-loop second`

divided by:

`CUDA-Q returned shots / sample second`.

Path planning is excluded from that primary metric, matching the paper. The
harness also records path-inclusive and subprocess-wall controls.

## 2. Measured RTX 5090 result

| Measurement | PTSBE | CUDA-Q, artifact 100 hs | CUDA-Q, text 1 hs |
|---|---:|---:|---:|
| Compared execution time | 43.246 s | 2.605 s | 1.564 s |
| Path-inclusive execution | 45.618 s | recorded in machine result | recorded in machine result |
| Full subprocess wall | 47.457 s | 5.830 s | 4.433 s |
| Output records/shots | 24,378,496 | 1 | 1 |
| Paper-style throughput advantage | -- | 1,468,244x | 881,660x |
| Raw runtime ratio, CUDA-Q/PTSBE | -- | 0.0602x | 0.0362x |
| Mean / p90 / peak GPU utilization | 5.5% / 11% / 95% | 3.0% / 6% / 42% | 4.4% / 6% / 66% |
| Peak VRAM (increment over observed floor) | 17,770 MiB (+13,249) | 21,023 MiB (+16,502) | 21,023 MiB (+16,502) |

The 50-qubit workload fits comfortably on the 32 GB RTX 5090. Device counters
are global 50 ms samples and therefore describe the whole subprocess, including
CPU-heavy intervals; they are not kernel-only occupancy measurements.

### Per-stage PTSBE profile

| Stage | Time | Share / interpretation |
|---|---:|---|
| Stim-to-PTS conversion | 0.016 s | setup |
| Trajectory sampling | 0.081 s | unseeded artifact behavior |
| Path construction (4 path sets) | 2.311 s | reused over 10 trajectories per batch |
| Apply trajectory errors | 0.803 s | inside contraction loop |
| GPU contraction calls | 28.112 s | 65.0% of contraction loop |
| Other loop work | 14.332 s | 33.1% of contraction loop |
| Entire contraction loop | 43.246 s | official PTSBE denominator |

All four batches perform 10 contractions, for 40 total. GPU contraction time by
batch was 0.152, 0.171, 0.125, and **27.663 s**. The final 28-qubit population
therefore consumed **98.4%** of GPU contraction time. It contains 268,435,456
possible entries per trajectory, but only 2,437,850 unique records on average,
an observed **0.908% population occupancy**. This exhaustive sparse harvest is
the dominant regime-defining operation.

## 3. Why this differs from the paper/H100 and Q-Tensor baseline

The throughput ratio factorizes directly:

`24,378,496 output records per CUDA-Q shot x 0.0602 CUDA-Q/PTSBE runtime`

`= 1,468,244x paper-style advantage`.

Thus the result comes from amortizing a contraction over a very large
non-proportional labeled dataset, not from making the same stochastic workload
execute faster.

The retained H100 circuit-0 log reports 20,005,920 records in 49.286 s and an
11.237 s CUDA-Q sample, giving 4,561,421x. On this RTX 5090 reproduction:

- PTSBE record throughput was 1.39x the retained H100 value.
- CUDA-Q sampling was 4.31x faster than the retained H100 value when both use
  the artifact's 100-hypersample command, which reduces the relative advantage.
- The unseeded trajectory draw yielded 24.4 million rather than 20.0 million
  records. Because the public driver exposes no seed, this yield is not expected
  to reproduce bit-for-bit even when the benchmark configuration is unchanged.
- Including PTSBE path construction lowers the RTX artifact comparison from
  1.468 million x to 1.392 million x, a 5.2% reduction. Path reuse helps, but
  it is not the main explanation.
- The public Figure 3 driver passes 100 hypersamples to CUDA-Q even though the
  manuscript describes one. On this machine that discrepancy changes the
  reported-style advantage by 1.67x.

The validated Q-Tensor crossover pilot instead used 5--11 qubit circuits,
categorical proportional trajectories, 1,024 frozen trajectories with 64 shots
each, equal effective work, exact-reference validation, and
`max_free_qubits=2`. Every validation gate passed, but the best measured
CUDA-Q/TN steady-state ratio was only 0.0310. It asks whether both methods can
produce the same physical distribution for the same effective shots; Figure 3
asks how many distinct labeled records an exhaustive final population can emit
per contraction. Those are different scientific questions.

Statistical validation is deliberately marked not applicable to the reproduced
Figure 3 metric: its exhaustive final-batch records are not IID samples from the
physical noise distribution. This run therefore cannot supersede Q-Tensor's
proportional-validation gate.

## 4. Does a meaningful PTSBE speedup regime exist on RTX 5090?

**Yes, conditionally:** for non-proportional training-data collection where each
distinct error/population record has value regardless of physical probability,
the RTX 5090 reproduces a roughly million-fold paper-style throughput regime.
The large final batch fits in memory, paths are reused, and each final
contraction yields about 2.44 million unique records in the committed run.

**No measured proportional crossover follows from this result.** For equal
execution units the same PTSBE run is slower, and the previously validated
proportional baseline remains a no-crossover result. Speedup claims must always
state which output semantics and denominator are used.

## 5. Next recommended experiment

Run one pre-registered **proportional control on this exact 50-qubit/200-gate
circuit**: freeze a corrected categorical trajectory list, use identical
multiplicities and total effective shots, complex128, CUDA-Q `tensornet`, and
the same contraction-planning policy. First validate that control on a tractable
reduced shadow circuit against an exact reference, then time the 50-qubit point
while reporting both equal-shot throughput and unique-record throughput. This
single ablation will quantify how much of the million-fold regime survives when
the population-harvest semantics—not circuit complexity or hardware—are
changed.

## Reproduction

From the pinned Q-Tensor environment:

```bash
python scripts/run_regime_reconciliation.py \
  --upstream /home/manager/q-tensor/upstream/Accelerated_TN_PTSBE \
  --q-tensor-baseline results/performance/2026-09-10_complexity_crossover_pilot.json \
  --output results/performance/2026-09-11_regime_reconciliation.json
```

Primary sources: the [PTSBE paper](https://arxiv.org/html/2604.08467v1), the
[NVIDIA technical blog](https://nvidia.github.io/cuda-quantum/blogs/blog/2026/05/21/ptsbe-million-fold-speedups/),
and the [public benchmark artifact](https://github.com/NVlabs/Accelerated_TN_PTSBE).
