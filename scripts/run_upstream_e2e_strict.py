#!/usr/bin/env python3
"""Run the official end-to-end test in isolation and enforce its printed verdict."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from q_tensor.upstream_verdict import evaluate_upstream_verification

UPSTREAM_COMMIT = "0569f848d9a3e385d6c20162c534a8031f6a69c5"


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path("upstream/Accelerated_TN_PTSBE"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    upstream = args.upstream.resolve()
    revision = git_head(upstream)
    if revision != UPSTREAM_COMMIT:
        raise SystemExit(f"upstream revision mismatch: {revision}")

    with tempfile.TemporaryDirectory(prefix="q-tensor-upstream-e2e-") as temporary:
        isolated = Path(temporary) / "upstream"
        shutil.copytree(
            upstream,
            isolated,
            ignore=shutil.ignore_patterns(".git", "shot_sets", "*.png", "*_output.py", "error_sets.pickle"),
        )
        test_dir = isolated / "test_ptsbe"
        for filename in (
            "stim_to_pts.py",
            "stim_to_be.py",
            "plot_cudaq.py",
            "utils_circuit.py",
            "utils_noisy_shots.py",
            "utils_verification.py",
        ):
            shutil.copy2(isolated / filename, test_dir / filename)
        runs = []
        for run_index in range(args.runs):
            completed = subprocess.run(
                [sys.executable, "test_ptsbe.py"],
                cwd=test_dir,
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            output = completed.stdout + completed.stderr
            verdict = evaluate_upstream_verification(output, completed.returncode)
            runs.append(
                {
                    "run_index": run_index + 1,
                    "cache_state_before_run": "empty" if run_index == 0 else "CUDA-Q files retained from prior run",
                    "command_returncode": completed.returncode,
                    "strict_returncode": 0 if verdict.passed else 1,
                    "verdict": verdict.as_dict(),
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                }
            )

    all_passed = all(run["verdict"]["passed"] for run in runs)
    record = {
        "upstream_commit": revision,
        "runs_requested": args.runs,
        "all_runs_passed": all_passed,
        "strict_returncode": 0 if all_passed else 1,
        "runs": runs,
    }
    rendered = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
