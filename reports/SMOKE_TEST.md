# Phase-1 smoke-test record

Date: 2026-09-10. Hardware and software are recorded in `gpu.txt` and
`environment.txt`. Upstream revision:
`NVlabs/Accelerated_TN_PTSBE@0569f848d9a3e385d6c20162c534a8031f6a69c5`.

## Reproduced result: official core checks

The unmodified official `test_gate_map/test_gate_map.py` completed with no assertions.
The unmodified `test_contract/test_contract.py` compared a tensor network containing
explicit separate error gates with the UPV-style network whose error tensors were
merged into adjacent coherent gates. `cupy.allclose` printed `True` in complex128.

Observed process-level wall times from `/usr/bin/time` (including Python import/startup):

| Check | Result | Wall time | Max RSS |
|---|---:|---:|---:|
| Gate map | PASS | 3.24 s | 1,377,612 KiB |
| Merged-error contraction | PASS | 3.15 s | 2,074,900 KiB |

These are smoke measurements, not steady-state GPU timings and not speedup results.

## Reproduced execution, failed statistical criterion

After adding the missing Open MPI runtime, the official `test_ptsbe/test_ptsbe.py`
completed both tensor-network and CUDA-Q data generation. A captured run reported:

- tensor-network samples: 10,015;
- CUDA-Q reference samples: 10,000;
- total variation distance: 0.3039;
- upstream threshold: 0.15;
- printed result: `FAIL: Distributions differ beyond threshold`;
- process exit: 0;
- process wall time: 7.99 s;
- max RSS: 2,178,224 KiB.

The script generates an unseeded random 7-qubit, 160-gate circuit with ten sampled
noise patterns and requests 1,000 shots per pattern. Because it neither seeds the run
nor exits nonzero when `passed` is false, the wall time and TVD are diagnostic only.
They are retained to explain Gate 4; they are not a validated scientific result.

## Reproduce the bounded core smoke

```bash
conda activate q-tensor
python scripts/fetch_upstream.py
python scripts/run_upstream_smoke.py \
  --output results/smoke/latest_upstream_core.json
```

The Q-Tensor runner rejects the wrong upstream revision and requires explicit success
text as well as a zero exit status.
