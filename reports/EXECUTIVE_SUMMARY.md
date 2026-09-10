# Q-Tensor executive summary — Phase 0/1

## Problem

Noisy tensor-network simulations repeat expensive path planning and conditional
contractions across trajectories and shots, limiting useful reference-data generation.

## What NVIDIA introduced

NVIDIA's paper combines error-independent Unified Path Variations, non-degenerate
batched sampling, and tunable batch sizes. Its largest speedups are upstream H100 claims,
especially for non-proportional unique-data harvesting.

## What Q-Tensor independently tested

Q-Tensor ran the pinned public code's gate mapping, merged-error contraction, and small
end-to-end comparison on a local RTX 5090 under WSL2.

## RTX 5090 result

The exact complex128 merged-error contraction passed on Blackwell. The full small
workflow executed, but its distribution criterion failed (TVD 0.3039 > 0.15).

## What caused the performance difference

Not yet measured by Q-Tensor. Upstream attributes its gains to path reuse, removal of
duplicate prefix contractions, and batch-size optimization.

## Statistical finding

The supplied end-to-end script can exit successfully after printing statistical
failure. Q-Tensor therefore treats execution and distribution validity as separate gates.

## Downstream demonstration

Not attempted; sample semantics must pass first.

## What this means

The RTX 5090 is software-compatible with the core upstream stack, but no Q-Tensor
speedup or scientific sampling claim is justified yet.

## Limitations

The H100 paper configurations were not rerun; Docker is not integrated in this WSL;
the local GPU has much less memory; and the upstream smoke circuit is unseeded.

## Next experiment

Use a frozen small circuit with an exact density-matrix reference, sweep seeded
trajectory count, establish convergence, then time a semantics-matched CUDA-Q baseline.
