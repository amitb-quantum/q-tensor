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

For the first five-qubit comparison, explicit CUDA-Q remained cheaper than TN's 37--208
conditional contractions. At 4,096 trajectories, GPU contractions accounted for about
90% of TN steady time; path planning and host sampling were not dominant.

## Statistical finding

The prior TVD failure came from CUDA-Q files cached by trajectory serial number and
reused for a newly randomized circuit. A clean run passed at TVD 0.0643; a stale-cache
rerun failed at 0.2101 while still exiting 0. Q-Tensor now fails that condition strictly.

## Downstream demonstration

Not attempted; sample semantics must pass first.

## What this means

The corrected path is statistically valid, but the first fair benchmark found no
crossover. At 4,096 trajectories, baseline/TN was 0.08512x: TN was 11.75x slower on
this small circuit. This is a bounded negative result, not a general simulator claim.

## Limitations

The H100 paper configurations were not rerun; validation stops at five qubits; and the
upstream PTS helper still samples depolarizing alternatives and duplicates incorrectly.

## Next experiment

Run one bounded higher-complexity pilot with exact validation. Do not expand the sweep
unless TN approaches or crosses the CUDA-Q baseline.
