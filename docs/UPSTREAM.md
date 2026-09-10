# Upstream provenance

Access date: 2026-09-10.

## Paper implementation

- Project: **Optimized Tensor Network PTSBE**
- Official repository: <https://github.com/NVlabs/Accelerated_TN_PTSBE>
- Pinned revision: `0569f848d9a3e385d6c20162c534a8031f6a69c5`
- Revision date/message: 2026-09-02, “align figure campaigns with revised manuscript”
- Source license: Apache License 2.0
- Release state: no tagged releases at the access date; the repository has two commits.
- Paper: Patti et al., “Accelerating Quantum Tensor Network Simulations with Unified
  Path Variations and Non-Degenerate Batched Sampling,” arXiv:2604.08467v1,
  9 April 2026, CC BY 4.0: <https://arxiv.org/abs/2604.08467>
- Predecessor: Patti et al., “Augmenting Simulated Noisy Quantum Data Collection by
  Orders of Magnitude Using Pre-Trajectory Sampling with Batched Execution,”
  arXiv:2504.16297 / SC '25: <https://arxiv.org/abs/2504.16297>
- NVIDIA overview: <https://nvidia.github.io/cuda-quantum/blogs/blog/2026/05/21/ptsbe-million-fold-speedups/>

The lock is machine-readable in `upstream/LOCK.json`. Q-Tensor fetches this exact
revision into an ignored directory rather than copying the implementation or its
retained paper results. This keeps upstream history, copyright, Apache notice, and
third-party notices intact. Q-Tensor calls the public scripts as an external reference
and adds its own validation, metadata, failure handling, and later benchmark harness.

## What upstream implements

The corresponding repository is a compact research artifact rather than an installed
Python package. Its central implementation is `utils_circuit.py`, with noise sampling
in `pts.py`, code generation from Stim-like input, retained circuits/results, and
CUDA-Q comparison scripts. It uses cuQuantum/cuTensorNet for contraction, CuPy arrays,
Qiskit as circuit IR, and CUDA-Q as the conventional-trajectory reference.

The paper stack in the repository's validated container is CUDA 12.9.1,
cuQuantum Python 26.1.0 / cuTensorNet 2.11.0, CuPy 13.6.0, CUDA-Q 0.13.0,
Qiskit 2.2.3, NumPy 1.26.4, pandas 2.3.2, Matplotlib 3.10.6, SciPy 1.16.2,
nvmath-python 0.7.0, and mpi4py 4.1.0. The container base is pinned by digest.

The arXiv HTML states “CuPy v2.2.3”; that does not match a plausible CuPy release in
this stack. The public repository pins **CuPy 13.6.0** and **Qiskit 2.2.3**. Q-Tensor
uses the executable repository pins and records this source discrepancy rather than
silently choosing one.

## Related but distinct NVIDIA source

CUDA-Q has a general PTSBE API in `NVIDIA/cuda-quantum`; current documentation is at
<https://nvidia.github.io/cuda-quantum/latest/using/examples/ptsbe.html>. That API
productionizes the earlier statevector-oriented PTSBE work. The new UPV/NBS paper's
official artifact is `NVlabs/Accelerated_TN_PTSBE`, and Q-Tensor does not treat the two
implementations as interchangeable.

The pinned PyPI CUDA-Q 0.13.0 binary identifies its source revision as
`b66c5bb7fd8c08e5014e2f03e97e7b0e92691650`.

## License decision

Q-Tensor itself uses Apache-2.0. Q-Tensor does not incorporate upstream files, so its
license need not be inherited. Apache-2.0 is nevertheless the best fit: it aligns with
the official research code, preserves an explicit patent grant, and permits later
small attributed adaptations without a license mismatch. NVIDIA binary libraries and
the NGC appliance remain under their separate NVIDIA terms; fetching or installing
them does not relicense them as Q-Tensor code.
