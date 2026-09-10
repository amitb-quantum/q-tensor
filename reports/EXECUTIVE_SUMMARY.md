# Q-Tensor executive summary — statistical gate

## Problem

Noisy tensor-network simulations repeat expensive path planning and conditional
contractions across trajectories and shots, limiting useful reference-data generation.

## What NVIDIA introduced

NVIDIA's paper combines error-independent Unified Path Variations, non-degenerate
batched sampling, and tunable batch sizes. Its largest speedups are upstream H100 claims,
especially for non-proportional unique-data harvesting.

## What Q-Tensor independently tested

Q-Tensor ran the pinned public code's gate mapping and merged-error checks, then built
an independent exact density-matrix suite for four frozen 2--5 qubit circuits.

## RTX 5090 result

The exact complex128 merged-error contraction passed on Blackwell. The seeded
proportional suite passed all 48 calibrated TN/reference, CUDA-Q/reference, and
TN/CUDA-Q comparisons.

## What caused the performance difference

Not yet measured by Q-Tensor. Performance work remains intentionally blocked until the
validated trajectory and cache semantics are used by a fair baseline.

## Statistical finding

The prior TVD failure came from CUDA-Q files cached by trajectory serial number and
reused for a newly randomized circuit. A clean run passed at TVD 0.0643; a stale-cache
rerun failed at 0.2101 while still exiting 0. Q-Tensor now fails that condition strictly.

## Downstream demonstration

Not attempted; sample semantics must pass first.

## What this means

The corrected Q-Tensor proportional path is a valid small-circuit reference workflow.
This clears a carefully semantics-matched first baseline, not a speedup claim.

## Limitations

The H100 paper configurations were not rerun; validation stops at five qubits; and the
upstream PTS helper still samples depolarizing alternatives and duplicates incorrectly.

## Next experiment

Time CUDA-Q explicit trajectories against TN proportional batching using the exact
same frozen circuit, trajectory multiplicities, and shots, with validation as a gate.
