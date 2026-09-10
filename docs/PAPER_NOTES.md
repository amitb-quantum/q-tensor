# Paper notes: optimized tensor-network PTSBE

Source read: arXiv:2604.08467v1 in full, accessed 2026-09-10:
<https://arxiv.org/html/2604.08467v1>. These notes are an independent explanation;
reported performance values in this document are explicitly labeled **UPSTREAM CLAIM**.

## Conventional noisy trajectories

A density matrix represents general Markovian noise exactly but stores roughly
`2^(2n)` complex values. A trajectory method instead samples Kraus choices and evolves
a pure `2^n` state for each trajectory. Averaging many independent trajectories
approximates the noisy state. In a conventional tensor-network implementation, each
shot can trigger three expensive repetitions: choose errors, construct the resulting
network and find contraction paths, then contract conditional marginals batch by batch
to sample one complete bitstring.

Tensor-network contraction order matters enormously. Path search is a CPU optimization
over possible contraction orders and slices; the selected path controls floating-point
work and peak memory. Re-running it for networks that differ only in error values is
therefore avoidable overhead if their topology can be held constant.

## Pre-Trajectory Sampling with Batched Execution

PTSBE moves all random Kraus choices before state evolution. Identical sampled error
patterns can be deduplicated, and multiple measurements can be allocated to one
prepared trajectory. The earlier method cached paths per distinct error set, but still
found new paths for different error sets and gathered tensor-network shots mostly one
at a time.

This changes cost from “one expensive state/path operation per reported shot” toward
“one expensive operation per distinct trajectory or conditional prefix.” It does not
make those reported records statistically interchangeable in every allocation mode.

## Unified Path Variations (UPV)

UPV assumes an error is adjacent to a coherent gate with the same arity and location.
The error tensor is contracted into a copy of that gate tensor. If full ranks (or dummy
ranks) preserve operand count, shape, rank, and network topology, the same path can be
computed once on the error-free structure and reused across error patterns and shots.
The eliminated work is repeated *path optimization*, not the numerical effect of the
error and not every tensor contraction.

The assumption limits which circuit/noise transformations fit without additional
engineering. Mid-circuit measurement and feed-forward are outside the presented static
circuit workflow.

## Non-Degenerate Batched Sampling (NBS)

Sampling a large output in batches conditions each later marginal on the already
selected prefix. Conventional code repeats contractions even when many shots share a
prefix. NBS contracts once per unique prefix, samples/counts all children for that
prefix, and only branches when the prefixes diverge. The first batch needs one
unconditioned contraction; later work scales with unique prefixes rather than raw shot
count.

Batch size trades fewer batches against exponentially larger marginal vectors and
more costly contractions. Exposing each `b_j` makes that trade-off tunable; the paper's
CUDA-Q reference used a fixed batch size of 24 while the optimized random-circuit runs
typically used 10 for non-final batches and 28 for the final batch.

## Proportional and non-proportional semantics

There are two separable probability choices:

1. **Error/trajectory allocation**: how often each Kraus pattern is represented.
2. **Population sampling**: how bitstrings are drawn from the state conditioned on a
   trajectory.

In proportional NBS, shots are drawn from every conditional marginal with their
multiplicities carried forward. Shared prefixes remove duplicate contractions while
preserving the circuit distribution, subject to normal Monte Carlo error and the
trajectory allocation being proportional.

In the paper's non-proportional benchmark, non-final prefixes are deliberately explored
and the final batch is exhaustively harvested above a probability threshold. Those
unique labeled bitstrings are a coverage-oriented dataset, not IID quantum shots and
not a sample from the original output distribution. They can be useful when a learner
needs examples and error labels or when rare/varied outcomes are intentionally
oversampled, provided training/evaluation accounts for the changed measure. They cannot
support ordinary expectation estimates, likelihoods, goodness-of-fit claims, or
hardware-like shot comparisons without appropriate weights and an independently
validated reweighting scheme.

## What “million shots for the price of one” means

It is an amortization/data-yield statement, not free independent sampling. A prepared
trajectory can yield many measurements; shared prefixes reuse contractions; and the
final exhaustive batch can emit many unique non-negligible outcomes after one final
marginal is computed. The numerator in the paper's non-proportional throughput is
therefore “unique labeled bitstrings,” whose statistical semantics differ from IID
shots. The paper also excludes reusable path-finding time from optimized steady-state
throughput. Q-Tensor will report setup/path planning separately and will never put
proportional trajectories and exhaustive unique outcomes in an unlabeled shots/s column.

## Experimental design and upstream claims

The paper uses H100 80 GB GPUs, CUDA 12.9, complex128, cuQuantum 26.1.0 /
cuTensorNet 2.11.0, CUDA-Q 0.13.0, random nearest-neighbor circuits with 50–200
qubits and 200–1,000 gates (plus a 1,200-gate case), 20% two-qubit gates, and noise
probabilities uniform on `[0.02, 0.2]`. Ten retained circuit instances are used per
configuration. Summary speedups are geometric means across per-circuit throughput
ratios.

- **UPSTREAM CLAIM:** non-proportional unique-data collection reaches more than
  `10^8x` versus traditional CUDA-Q trajectories in selected H100 configurations.
- **UPSTREAM CLAIM:** proportional sampling reaches up to roughly `10^3x`.
- **UPSTREAM CLAIM:** path finding can take tens or hundreds of seconds while a
  reused contraction is much shorter, and batch size 10 substantially outperformed
  24 for the reported 100-qubit/600-gate random circuits.

None of those speedups is a Q-Tensor result. The paper's optimized throughput omits
path-finding on the argument that it is amortized over large campaigns; Q-Tensor will
show cold-start and amortized figures separately. The full upstream campaign is
hundreds of serial H100 GPU-hours and is deliberately outside the Phase-1 smoke test.
