# Figure-3 proportional control

## Conclusion: PASS

The corrected categorical, multiplicity-preserving method passed its exact-reference
shadow validation gate. On the exact public 50-qubit/200-gate Figure-3 circuit, TN
proportional batching then completed the same frozen 1,000 effective shots as CUDA-Q
TensorNet in **40.419 s versus 277.865 s** (steady medians). The measured
CUDA-Q/TN runtime ratio is **6.875x**, or **24.74 versus 3.60 effective shots/s**.

This is the first measured proportional crossover in Q-Tensor. It establishes a
crossover at this exact 50q/200g point only. The boundary between the prior largest
validated 11-qubit workload and this point was not measured, so none is inferred.

Machine-readable evidence is in
[`results/performance/2026-09-11_figure3_50q_proportional_control.json`](../results/performance/2026-09-11_figure3_50q_proportional_control.json).

## 1. Shadow-circuit validation

The harness parsed the exact retained Stim artifact and deterministically selected
source qubit labels `[1,15,22,25,27,29,35,46]`. It retained, in original chronological
order, every gate and its immediately following noise site whose operands were wholly
inside that set, then reindexed the result to eight qubits.

The shadow contains 35 coherent gates, including 10 entangling gates, and all 35
corresponding noise sites. Each site uses the retained probability and a corrected
categorical Kraus choice: identity has probability `1-p`, and the non-identity
alternatives sum to `p`. The actual-site probability audit passed with maximum
normalization error `7.77e-16` and maximum total-error-probability error `2.78e-17`.

From seed `80509026`, 128 trajectory draws produced 126 unique trajectories. Grouping
retained total multiplicity 128, directly exercising the duplicate-preservation path.
Each draw received 32 shots, for 4,096 effective shots per backend. The trusted
reference was the independent complex128 statevector mixture conditioned on this exact
finite trajectory list and its multiplicities.

| Check | Observed TVD | Seeded 99.9% null limit | Z error/difference | Z null limit | Result |
|---|---:|---:|---:|---:|---|
| TN vs exact conditioned | 0.06911 | 0.08704 | 0.02718 | 0.04801 | PASS |
| CUDA-Q vs exact conditioned | 0.07039 | 0.08607 | 0.01400 | 0.05160 | PASS |
| TN vs CUDA-Q | 0.10571 | 0.12012 | 0.01318 | 0.07764 | PASS |

The gate validates the finite-trajectory execution and multiplicity semantics. It does
not claim an exhaustive exact density matrix for the 50-qubit circuit.

## 2. Exact 50-qubit proportional result

The full workload is the exact public Figure-3 circuit-0 Stim artifact, SHA-256
`962886ae3f65526b3218221e9196cbf9006be31ab6078237e5493d047480ffa7`: 50
qubits, 200 coherent gates, 40 entangling gates, and 200 retained noise sites.

Both methods received the same ten categorical trajectories from seed `50902026`.
Their frozen draw hash is
`62aa25b1bcca9fb1be5e31e91191c49b76d4190168249c355562f328aeab48d2`.
All ten were unique with multiplicity one; the complete code arrays are stored in the
machine result. Each draw received 100 shots, so each backend returned exactly 1,000
shots. TN used complex128 with 10-qubit proportional batches; CUDA-Q used TensorNet
with its target reporting fp64. No noise model or trajectory was sampled internally by
CUDA-Q.

| Timing/throughput | TN proportional | CUDA-Q TensorNet | CUDA-Q/TN |
|---|---:|---:|---:|
| Cold total | 41.742 s | 287.563 s | 6.889x |
| Steady execution + requested-shot postprocess | 40.419 s | 277.865 s | **6.875x** |
| TN path-inclusive / CUDA-Q steady | 40.848 s | 277.865 s | 6.802x |
| Effective shots/s | **24.74** | 3.60 | 6.875x |

Cold CUDA-Q includes target initialization, construction and warmup of all ten fixed
trajectory kernels, followed by the first equal-shot sample. Cold TN includes first-use
path construction and the equal-shot run. Steady values are medians of three
alternating repeats. TN path-inclusive time is also shown because its public API
rebuilds the paths on each invocation even though paths are reused across trajectories
inside an invocation.

The first 50-qubit output repeat is a diagnostic, not an exact validation claim. Both
backends returned 1,000 shots. Across the 50 single-qubit Z expectations, their mean
absolute difference was 0.0213 and maximum was 0.076. The maximum is below the
conservative simultaneous 0.1%-alpha Hoeffding limit of 0.2146.

## 3. Proportional crossover

**Yes, one proportional crossover is measured:** TN beats CUDA-Q TensorNet by 6.875x
at this exact 50q/200g, 10-trajectory, 100-shots-per-draw point. This conclusion uses
equal effective shots and identical frozen physical trajectories.

No crossover location is claimed. Earlier measurements ended at 11 qubits and favored
CUDA-Q; the intervening circuit sizes, depths, trajectory counts, and shot allocations
were not swept in this experiment. The result is not extrapolated to any unmeasured
workload.

## 4. Dominant runtime component

TN performed a median 3,783 conditional contractions across the three repeats. Its
exclusive median components were:

| TN component | Time | Share of 40.419 s steady total |
|---|---:|---:|
| GPU contraction | **39.436 s** | **97.6%** |
| Apply trajectory errors | 0.776 s | 1.9% |
| Host conditional sampling | 0.213 s | 0.5% |
| Host post-processing | 0.00007 s | <0.01% |
| Path planning, reported separately | 0.426 s | not in steady denominator |

GPU contraction is unambiguously dominant. In separate 50 ms global telemetry runs,
TN showed 40.3% mean, 49% p90, and 53% peak utilization; CUDA-Q showed 26.5% mean,
44% p90, and 61% peak. The shared process already held both CUDA contexts, so the
observed 20.8--21.0 GB global VRAM level is informative for fit but the small within-run
memory deltas are not backend-isolated allocation measurements.

## 5. Proportional versus non-proportional regimes

This control never invokes unique-prefix harvesting and never counts distinct labeled
records as extra shots. Its numerator is exactly 1,000 physical measurement outcomes
for each backend, and its metric is equal-shot runtime/throughput.

The prior Figure-3 reproduction answered a different question. Its non-proportional
final 28-qubit population emitted 24,378,496 distinct labeled records from 40
contractions and therefore produced a roughly million-fold *data-record throughput*
ratio, even though its raw PTSBE contraction loop was slower than a one-shot CUDA-Q
call. Those records are not IID proportional samples and that metric is not used here.

The two findings are compatible:

- non-proportional PTSBE has a very large unique-data-harvest regime;
- proportional TN has a measured 6.875x equal-shot crossover on this 50-qubit point;
- neither result licenses a claim outside its measured circuit, trajectories, shots,
  semantics, precision, or hardware.

## Reproduce

```bash
PYTHONPATH=src:scripts python scripts/run_figure3_proportional_control.py \
  --upstream /home/manager/q-tensor/upstream/Accelerated_TN_PTSBE \
  --output results/performance/2026-09-11_figure3_50q_proportional_control.json \
  --repeats 3 \
  --null-trials 1000
```
