#!/usr/bin/env python3
"""Run the official bounded core checks and emit claim-ready JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "0569f848d9a3e385d6c20162c534a8031f6a69c5"


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in (
        "cuquantum-python-cu12",
        "cupy-cuda12x",
        "cuda-quantum-cu12",
        "qiskit",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "mpi4py",
        "nvmath-python",
    ):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def gpu_properties() -> dict[str, Any]:
    import cupy

    props = cupy.cuda.runtime.getDeviceProperties(0)
    return {
        "name": props["name"].decode(),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "sm_count": props["multiProcessorCount"],
        "total_global_memory_bytes": props["totalGlobalMem"],
        "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_api": cupy.cuda.runtime.driverGetVersion(),
    }


def run_check(upstream: Path, relative_script: str, success_text: str) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, relative_script],
        cwd=upstream,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    passed = completed.returncode == 0 and success_text in output
    return {
        "script": relative_script,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "success_evidence": success_text,
        "passed": passed,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("upstream/Accelerated_TN_PTSBE"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    actual_commit = git_head(upstream)
    if actual_commit != UPSTREAM_COMMIT:
        raise SystemExit(f"upstream revision mismatch: {actual_commit}")

    checks = [
        run_check(upstream, "test_gate_map/test_gate_map.py", "No assertions failed in the testing suite."),
        run_check(upstream, "test_contract/test_contract.py", "\nTrue\n"),
    ]
    record = {
        "schema_version": 1,
        "experiment_id": "upstream-core-smoke",
        "category": "reproduced_result",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": [sys.executable, str(Path(__file__)), "--upstream", str(upstream)],
        "git_commit": git_head(Path(__file__).resolve().parents[1]),
        "upstream_commit": actual_commit,
        "hardware": gpu_properties(),
        "software": {"python": platform.python_version(), "packages": package_versions()},
        "parameters": {"precision": "complex128", "official_checks": [item["script"] for item in checks]},
        "measurements": {"checks": checks, "all_passed": all(item["passed"] for item in checks)},
    }
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if record["measurements"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
