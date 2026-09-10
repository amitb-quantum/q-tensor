# Statistical validation

Date: 2026-09-10. Q-Tensor source under test:
`589f2978fbbdf98f6df86141144b39ce15b1f081`. Pinned upstream source:
`NVlabs/Accelerated_TN_PTSBE@0569f848d9a3e385d6c20162c534a8031f6a69c5`.

## Conclusion: PASS

Q-Tensor's corrected proportional-sampling path passes the small-circuit validation
gate. Across four frozen 2--5 qubit cases and four trajectory counts, every tensor
network (TN) and CUDA-Q comparison remained inside a seeded 99.9th-percentile finite-
shot null envelope. The conditioned trajectory mixture converged toward the exact
channel in every case. This result validates the bounded harness, not arbitrary
circuits or upstream's unmodified trajectory generator.

The previously observed upstream TVD 0.3039 failure was a semantic cache mismatch,
not statistical variance and not a disagreement between fresh TN and CUDA-Q
simulations. The generated CUDA-Q program skips an output file whenever a file with
the same trajectory serial number already exists. The official test creates a new
unseeded random circuit and new noise set on every invocation but does not clear
`shot_sets/`. A rerun can therefore compare fresh TN samples with CUDA-Q samples from
the preceding circuit.

Q-Tensor resolves the validation gate by:

- running the upstream workflow in an isolated clean directory;
- treating its printed PASS/FAIL result as authoritative instead of trusting exit 0;
- defining canonical `q0...q(n-1)` output order explicitly;
- sampling each noise site's Kraus alternatives categorically;
- grouping repeated trajectories while preserving their exact multiplicities; and
- comparing both backends with exact channel and finite-trajectory references.

No speedup or throughput claim was measured in this work.

## Frozen experiment

All circuit gates, noise sites, probabilities, seeds, trajectory counts, and output
shots are embedded in `src/q_tensor/validation.py` and copied into the result JSON.
Noise is applied immediately after the named coherent gate. Q-Tensor's independent
reference exhaustively enumerates every Kraus branch in complex128 and sums the pure-
state outer products into a dense density matrix.

| Case | Qubits | Seed | Noise model | Observable |
|---|---:|---:|---|---|
| `2q_asymmetric_x` | 2 | 1729 | X error, p=0.19 | Z0 |
| `3q_depolarizing1` | 3 | 2718 | one-qubit depolarizing, p=0.27 | Z2 |
| `4q_two_channels` | 4 | 3141 | X p=0.18; depolarizing p=0.24 | Z0Z2 |
| `5q_depolarizing2` | 5 | 5772 | two-qubit depolarizing, p=0.22 | Z0Z1 |

The trajectory sweep is the nested prefix `[16, 64, 256, 1024]` from each case's
seeded draw stream. Each draw contributes 64 measurement shots, so backend sample
totals are `[1024, 4096, 16384, 65536]`. Repeated trajectories are contracted once but
receive `multiplicity * 64` shots. TN uses upstream's proportional batched sampler with
at most two free qubits; CUDA-Q executes the identical grouped trajectory list on its
NVIDIA target.

For every sweep point, the saved artifact separates:

1. finite-trajectory error: exact conditioned mixture versus exact noise channel;
2. population-shot error: each backend versus the exact conditioned mixture; and
3. semantic disagreement: TN versus CUDA-Q under the same trajectory allocations.

The null distribution for (2) and (3) uses 2,000 seeded stratified replicates with the
same per-trajectory shot allocation. A check passes only when both TVD and the selected
Z-observable error are no larger than their empirical 99.9th-percentile null limits.
This is a conservative regression gate, not a universal goodness-of-fit certificate.

## Results

All 48 backend checks passed: 4 cases x 4 sweep points x (TN/reference,
CUDA-Q/reference, TN/CUDA-Q).

| Case | Trajectory TVD, 16 -> 1024 | Log-log slope | TN/conditioned TVD at 1024 | CUDA-Q/conditioned TVD at 1024 | TN/CUDA-Q TVD at 1024 |
|---|---:|---:|---:|---:|---:|
| 2 qubits | 0.044710 -> 0.002591 | -0.702 | 0.001284 | 0.000929 | 0.002213 |
| 3 qubits | 0.061092 -> 0.000580 | -0.933 | 0.001389 | 0.002655 | 0.002029 |
| 4 qubits | 0.164356 -> 0.015427 | -0.541 | 0.002682 | 0.002362 | 0.003952 |
| 5 qubits | 0.046997 -> 0.004013 | -0.640 | 0.004626 | 0.004537 | 0.006378 |

The negative slopes and lower endpoints show convergence at approximately the expected
Monte Carlo scale. Individual prefixes are not required to improve monotonically: the
3-qubit trajectory TVD, for example, moved from 0.007091 at 64 trajectories to
0.020182 at 256 before reaching 0.000580 at 1024. That fluctuation is consistent with
the explicitly measured finite-trajectory component and is not evidence of semantic
mismatch.

| Case | Trajectory Z error, 16 -> 1024 | TN/conditioned Z error at 1024 | CUDA-Q/conditioned Z error at 1024 |
|---|---:|---:|---:|
| 2 qubits | 0.089421 -> 0.005181 | 0.002568 | 0.001857 |
| 3 qubits | 0.122184 -> 0.001159 | 0.001502 | 0.001916 |
| 4 qubits | 0.041686 -> 0.011932 | 0.002233 | 0.004725 |
| 5 qubits | 0.014296 -> 0.002093 | 0.000552 | 0.001132 |

The fresh fixed-qubit path used by upstream's official test also agreed with the exact
conditioned distribution. Its TN TVDs were 0.000005, 0.000123, 0.000411, and 0.002765
for the 2--5 qubit cases; corresponding fresh TN/CUDA-Q TVDs were 0.000107, 0.002223,
0.004629, and 0.006049. A separate audit covered every coherent gate family in the
official random generator (`x`, `y`, `z`, `h`, `t`, `rx`, `cx`, `cy`, `cz`, `ch`,
and `crx`) and found no material fresh-backend mismatch.

## Smallest passing and failing examples

The smallest passing main case is the 2-qubit `RX(0.73) q0; CX q0,q1` circuit with an
X error after RX. At 1,024 trajectories, its trajectory-to-channel TVD was 0.002591,
TN-to-conditioned TVD 0.001284, CUDA-Q-to-conditioned TVD 0.000929, and TN/CUDA-Q TVD
0.002213. A deterministic 2-qubit `X(q0)` ordering probe also gave TN/exact TVD 0 and
CUDA-Q/exact TVD 0 when upstream's reversed Qiskit mapping was retained.

The smallest failing workflow is also two qubits. Cache CUDA-Q results for `X(q0)`
under serial number 0, then generate a fresh `X(q1)` circuit while retaining that
file. The fresh TN and exact distributions are both `01` with TVD 0; fresh CUDA-Q is
also `01`. The stale CUDA-Q file contains `10`, producing TN/stale-CUDA-Q TVD 1. This
is the minimized form of the upstream end-to-end cache failure.

## Root cause and upstream semantic findings

Three independent issues were found at the pinned revision.

1. **Stale CUDA-Q result reuse caused the observed failure.** Generated `be_output.py`
   names output only by trajectory serial number and samples only when that path does
   not already exist. The circuit, noise model, seed, or shot count is not part of the
   cache key. In an isolated two-run reproduction, run 1 started empty and passed at
   TVD 0.0643. Run 2 generated a new random workload, reused run-1 CUDA-Q files,
   failed at TVD 0.2101, and still returned process status 0.

2. **The test ignores its returned verdict.** `verify_distributions` correctly returns
   `(passed, tvd)`, but `test_ptsbe.py` assigns those values and reaches end of file
   without asserting or calling `sys.exit`. Q-Tensor's strict runner parses the metric
   and printed verdict and returns 1 for a failed or incomplete check.

3. **The upstream PTS helper does not implement the intended proportional channel.**
   It tests the 3 or 15 non-identity depolarizing operators as ordered independent
   Bernoulli events and keeps the first success. For p=0.27 one-qubit depolarization,
   the intended total error probability is 0.27 but the implemented value is 0.246429
   (TVD 0.023571). For p=0.22 two-qubit depolarization, it is 0.198788 (TVD 0.021212).
   Its duplicate-removal loop also does not preserve sampled multiplicity. In the
   frozen 64-draw audit, that alone shifted the output distribution by TVD 0.005726.

Issue 3 does not explain a fresh TN/CUDA-Q difference because both upstream backends
consume the same generated error list. It does mean their agreement is not, by itself,
validation against the requested physical depolarizing channel. Q-Tensor therefore
uses categorical alternatives and explicit multiplicities before sending the same
trajectory list to either backend.

## Strict failure behavior

`scripts/run_upstream_e2e_strict.py` copies the pinned source to a clean temporary
directory, runs the official script, parses TVD/threshold/PASS/FAIL, and exits nonzero
if the process fails, output is incomplete, or the printed statistical criterion
fails. `--runs 2` intentionally retains files between the two isolated invocations and
reproduces the cache defect. The saved record reports `strict_returncode: 1` even
though both official subprocesses returned 0.

## Recommended first fair CUDA-Q baseline

Validation now permits the first baseline, but it has not been timed. Use the frozen
Q-Tensor cases and exact grouped trajectory list. Compare:

- TN: upstream proportional batched sampling, complex128, `max_free_qubits=2`;
- CUDA-Q: NVIDIA target, one execution per unique explicit trajectory;
- identical circuit, operator code, trajectory multiplicity, and total measurement
  shots for both paths;
- separate cold setup/compilation/path-planning time from steady execution; and
- reject any run whose distributions fail the same exact-conditioned null gate.

Do not use upstream's non-proportional mode, unseeded random test, or serial-number
cache in this first comparison. Do not publish a speedup until the baseline runner and
timing boundaries are reviewed.

## Reproduce

```bash
conda activate q-tensor
python scripts/fetch_upstream.py
PYTHONPATH=src python scripts/run_statistical_validation.py \
  --upstream upstream/Accelerated_TN_PTSBE \
  --output results/statistical_validation/2026-09-10_seeded_proportional.json \
  --trajectory-counts 16 64 256 1024 \
  --shots-per-trajectory 64 \
  --null-trials 2000
```

Reproduce strict cache/failure behavior (expected Q-Tensor exit status: 1):

```bash
PYTHONPATH=src python scripts/run_upstream_e2e_strict.py \
  --upstream upstream/Accelerated_TN_PTSBE \
  --runs 2 \
  --output results/statistical_validation/upstream_cache_reproduction.json
```

Machine-readable evidence:

- `results/statistical_validation/2026-09-10_seeded_proportional.json`
- `results/statistical_validation/2026-09-10_upstream_cache_reproduction.json`

## Limitations

The exact references stop at five qubits and computational-basis distributions. The
cases cover all coherent gate families through additional audits but not every circuit
topology or channel composition. CUDA-Q used fp32 while TN and the exact reference
used complex128; their observed agreement bounds this effect only for these cases.
The empirical 99.9% limits use 2,000 replicates and are intended as deterministic
regression thresholds, not exact frequentist confidence intervals. The unmodified
upstream PTS helper remains unsuitable for physical-channel claims without repair.
