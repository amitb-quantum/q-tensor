# Q-Tensor

Independent reproduction and characterization of GPU-accelerated noisy quantum
trajectory simulation with tensor networks.

## Why this exists

Noisy simulations need many stochastic trajectories. Repeating tensor-network path
planning and conditional contractions can dominate useful data generation. Q-Tensor
tests what NVIDIA's UPV/NBS implementation actually reuses, how it performs on one
local RTX 5090, and which sampling guarantees survive each acceleration mode.

## Current finding

**Core Blackwell execution works; end-to-end statistical agreement is not yet
established.** The pinned NVIDIA code ran its complex128 gate-map and merged-error
contraction checks on the RTX 5090. Its unseeded end-to-end comparison also executed,
but the captured distribution check failed. No speedup claim is published.

| Category | Local RTX 5090 observation | Status |
|---|---|---:|
| **REPRODUCED RESULT** | Official gate-map check, 3.24 s process wall time | PASS |
| **REPRODUCED RESULT** | Official separated-vs-merged error contraction, `allclose=True`, 3.15 s | PASS |
| **REPRODUCED DIAGNOSTIC** | 10,015 TN vs 10,000 CUDA-Q samples; TVD 0.3039 vs 0.15 threshold | FAIL |

The timings include interpreter startup/imports and are smoke diagnostics, not GPU
kernel benchmarks. See [the complete smoke record](reports/SMOKE_TEST.md).

## What Q-Tensor tests

- reproduction of the official public implementation on non-H100 hardware;
- cold-start/path-planning, contraction, sampling, orchestration, and memory costs;
- proportional sampling against exact small-circuit references;
- explicitly separate coverage-oriented non-proportional data semantics;
- one bounded downstream QEC/validation demonstration after statistical validation.

## Hardware

Initial host: RTX 5090 (Blackwell, compute capability 12.0, 170 SMs, 32,607 MiB
visible VRAM), Intel Core Ultra 9 285K, Ubuntu 26.04.1 under WSL2. See
[`reports/gpu.txt`](reports/gpu.txt) and [`reports/environment.txt`](reports/environment.txt).

## Reproduce the foundation

```bash
conda env create -f environment/environment.yml
conda activate q-tensor
python scripts/fetch_upstream.py
pytest
python scripts/run_upstream_smoke.py \
  --output results/smoke/latest_upstream_core.json
```

The upstream checkout is detached at
`0569f848d9a3e385d6c20162c534a8031f6a69c5`. GPU tests are kept out of the normal
unit suite; run them explicitly with:

```bash
Q_TENSOR_UPSTREAM="$PWD/upstream/Accelerated_TN_PTSBE" pytest -m gpu
```

## Methodology

- [Paper notes and sampling semantics](docs/PAPER_NOTES.md)
- [RTX 5090 / WSL feasibility and decision gates](docs/FEASIBILITY.md)
- [Upstream source, revision, dependencies, and licenses](docs/UPSTREAM.md)
- [Project worklog](docs/WORKLOG.md)
- [Executive summary](reports/EXECUTIVE_SUMMARY.md)

Every saved result uses one of three labels: `UPSTREAM CLAIM`, `REPRODUCED RESULT`, or
`Q-TENSOR ORIGINAL RESULT`. The machine-readable schema requires hardware, software,
parameters, commands, both Git revisions, timestamps, and measurements.

## Upstream NVIDIA work

Q-Tensor uses NVIDIA Research's Apache-2.0
[`NVlabs/Accelerated_TN_PTSBE`](https://github.com/NVlabs/Accelerated_TN_PTSBE), the
artifact associated with [arXiv:2604.08467v1](https://arxiv.org/abs/2604.08467).
Upstream source and paper results are not vendored or relabeled as Q-Tensor results.

## Limitations

The paper's H100 80 GB campaigns have not been rerun. This WSL distribution has no
Docker integration, no system CUDA toolkit, and no Nsight Compute. The wheel-based core
path works, but the first end-to-end statistical gate failed and must be understood
before baseline timing, scaling, or downstream claims.

## Repository structure

`src/` contains independent utilities; `scripts/` reproducible runners; `configs/`
versioned inputs; `benchmarks/` schemas; `results/` raw machine-readable observations;
`reports/` interpreted results; `docs/` provenance/method notes; and `upstream/` the
revision lock and fetch policy.
