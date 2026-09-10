# RTX 5090 / WSL feasibility audit

Audit date: 2026-09-10. Status labels below are Q-Tensor decisions, not upstream claims.

## Gate 1 — can official upstream code run on RTX 5090?

**GO for the core implementation.** The pinned CUDA-12 wheels loaded on the RTX 5090,
reported compute capability 12.0, and executed the official gate-map and merged-error
tensor-contraction checks successfully. cuTensorNet 2.11.0 returned version 21100 and
CuPy observed CUDA runtime 12.9 with driver API 13.3.

This is consistent with NVIDIA's cuTensorNet 26.01 documentation, which lists Blackwell
as supported and requires a compatible CUDA 12.x/13.x driver:
<https://docs.nvidia.com/cuda/cuquantum/26.01.0/cutensornet/index.html>.
NVIDIA's CUDA 12.9 release notes also identify Blackwell support:
<https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/>.

It is not yet evidence that every paper-sized workload fits or performs well. The paper
used H100 80 GB; this RTX 5090 exposes 32,607 MiB. The paper says a 28-qubit final batch
was its largest single-H100 population vector. Q-Tensor will pilot smaller final batches
and measure peak memory before scaling.

## Host and toolchain findings

- WSL distribution: Ubuntu 26.04.1 LTS on WSL 2.7.10.0; kernel
  `6.18.33.2-microsoft-standard-WSL2`.
- GPU: NVIDIA GeForce RTX 5090, Blackwell, compute capability 12.0, 170 SMs,
  32,607 MiB visible VRAM.
- Windows NVIDIA driver/KMD: 610.74; `nvidia-smi` 610.53; CUDA UMD 13.3.
- CPU: Intel Core Ultra 9 285K; WSL exposes 24 logical CPUs.
- WSL RAM: 19 GiB plus 16 GiB swap. Host physical RAM could not be independently
  captured from the sandbox, so no claim is made about the WSL memory cap.
- GCC 15.2.0, Git 2.53.0, system Python 3.14.4.
- No system `nvcc` or CMake. They are not needed for the wheel-based smoke path.
- Nsight Systems CLI 2026.4.1 is installed; Nsight Compute is absent.
- Docker is not integrated into this WSL distribution, so the upstream's validated NGC
  container path is currently unavailable. Enabling it is a host configuration change,
  not required for the direct-wheel feasibility check.

NVIDIA documents WSL 2 CUDA operation and notes a limited `nvidia-smi` feature set:
<https://docs.nvidia.com/cuda/wsl-user-guide/>. Native CUDA 13.3 documentation lists
Ubuntu 26.04 as supported, but Q-Tensor's Phase-1 execution uses CUDA 12.9 runtime
wheels rather than a system toolkit.

## Dedicated environment

The user-local conda environment `q-tensor` uses Python 3.11.16 and the exact direct
Python versions in the upstream Dockerfile. Direct setup also needs Open MPI 5.0.10;
the upstream lists `mpi4py` but relies on the appliance base image to provide `libmpi`.
Without Open MPI, its CUDA-Q subprocess fails before producing reference samples.

Recreate with:

```bash
conda env create -f environment/environment.yml
conda activate q-tensor
python scripts/fetch_upstream.py
python scripts/run_upstream_smoke.py
```

## Smoke evidence and limitation

Two official core checks pass: gate operand mapping and equality of an explicitly
error-separated contraction with UPV-style error tensors merged into coherent gates.
The latter printed `True` in complex128 and used the GPU.

The official seven-qubit end-to-end script executed after adding Open MPI, but its own
distribution check did **not** pass on the captured run: 10,015 tensor-network samples,
10,000 CUDA-Q reference samples, TVD 0.3039, threshold 0.15. The upstream script returns
process status 0 even after printing `FAIL`, and it does not seed its randomly generated
circuit or all sampling paths. Therefore Q-Tensor records “execution reproduced;
statistical agreement not reproduced.” This is not a stable benchmark point and is not
used as a performance claim.

## Remaining decision gates

- **Gate 2 — meaningful accelerated workload:** CONDITIONAL GO. Core UPV operations and
  end-to-end data generation execute, but a deterministic, statistically validated
  proportional workload is still required.
- **Gate 3 — fair baseline:** NO-GO. No speedup ratio will be published until semantics,
  seeds, precision, path effort, and timing boundaries match.
- **Gate 4 — sample semantics:** NO-GO. The captured official validation failed; Q-Tensor
  must construct a seeded exact-reference circuit and test convergence.
- **Gate 5 — useful downstream experiment:** NOT EVALUATED in Phase 0/1.

## Recommended next experiment

Freeze a two- to five-qubit circuit and Pauli/depolarizing channel with a tractable exact
density-matrix distribution. Run seeded proportional PTSBE over increasing trajectory
counts, separately record path, contraction, and sampling time, and compare TVD plus
observable confidence intervals. Only after convergence passes should the same circuit
be timed against conventional CUDA-Q trajectories. Start with final batch sizes 4–10;
do not begin with a paper-scale 28-qubit final batch.
