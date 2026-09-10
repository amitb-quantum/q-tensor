# Worklog

## 2026-09-10 — Phase 0 and Phase 1 foundation

- Inspected `C:\Users\amitb\Documents\Q-Tensor`: empty and not a Git repository.
- Inspected `/home/manager`: `/home/manager/q-tensor` did not exist; no unrelated
  directories were modified.
- Confirmed GitHub CLI is already authenticated to `amitb-quantum` as the active
  account. No remote repository was created and nothing was pushed.
- Located the official paper source as `NVlabs/Accelerated_TN_PTSBE`; rejected similarly
  named third-party repositories as provenance sources.
- Read arXiv:2604.08467v1, its predecessor arXiv:2504.16297, NVIDIA's PTSBE overview,
  the official repository, license, Docker setup, source, and reproduction guide.
- Pinned upstream commit `0569f848d9a3e385d6c20162c534a8031f6a69c5`
  (2026-09-02). Chose fetch-on-setup instead of vendoring or a submodule.
- Captured actual host/GPU/toolchain details. Confirmed RTX 5090 compute capability
  12.0, 170 SMs, and 32,607 MiB visible VRAM.
- Found Docker unavailable in this WSL distribution. Avoided changing Docker Desktop or
  system configuration.
- Created user-local conda environment `q-tensor` with Python 3.11 and exact upstream
  package pins.
- Official end-to-end attempt 1: optimized tensor-network sampling ran, but CUDA-Q
  failed because no MPI shared library was present. Root cause: upstream's pip list
  includes `mpi4py`, while the container base supplies `libmpi` implicitly.
- Installed Open MPI 5.0.10 in the dedicated environment. `mpi4py` then reported its
  library successfully.
- Official core checks: gate-map test passed; separated-versus-merged error contraction
  printed `True` in complex128. These establish basic Blackwell execution and the core
  error-merge identity, not paper-scale performance.
- Official end-to-end audit run: 10,015 tensor-network samples and 10,000 CUDA-Q
  samples; TVD 0.3039 versus threshold 0.15 (`FAIL`). Process exit was nevertheless 0.
  Decision: do not claim statistical validation; add strict Q-Tensor failure handling
  before using this workflow as a gate.
- Created independent config/result schemas, provenance/statistics utilities, upstream
  fetch/smoke scripts, and CPU-first tests.

Next: freeze a seeded exact-reference circuit and explain the TVD failure before any
speedup benchmark.

## 2026-09-10 — Seeded statistical validation

- Added exact complex128 Kraus-branch density-matrix references for frozen 2--5 qubit
  circuits and seeded nested trajectory sweeps at 16, 64, 256, and 1,024 draws.
- Preserved duplicate trajectory multiplicities and used categorical Kraus alternatives
  before sending the identical grouped list to TN and CUDA-Q.
- All 48 calibrated TN/reference, CUDA-Q/reference, and TN/CUDA-Q checks passed. Every
  case's 1,024-trajectory TVD improved over its 16-trajectory endpoint.
- Verified the upstream fixed-qubit path and every coherent random-gate family against
  exact references; found no fresh-backend semantic mismatch in the bounded audits.
- Identified the earlier end-to-end failure as stale CUDA-Q files: generated output is
  keyed only by trajectory serial number and skipped when present, while the test creates
  a new unseeded circuit and noise model on every run.
- Reproduced clean-run PASS at TVD 0.0643 followed by cached-run FAIL at TVD 0.2101;
  both upstream processes returned 0. Q-Tensor's strict wrapper returned 1.
- Minimized the cache defect to two qubits: stale `X(q0)` CUDA-Q data compared with a
  fresh `X(q1)` TN circuit has TVD 1; fresh backends both match the exact result.
- Audited upstream PTS semantics: ordered Bernoulli depolarization changes the requested
  channel, and duplicate filtering can change trajectory weights. The Q-Tensor harness
  does not use those behaviors.
- Gate 4 passes for the corrected bounded proportional path. No performance measurement
  or speedup claim was made.
