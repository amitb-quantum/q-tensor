#!/usr/bin/env python3
"""Render performance plots exclusively from a saved benchmark JSON record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from q_tensor.schema import validate_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = json.loads(args.result.read_text(encoding="utf-8"))
    validate_result(record)
    points = record["measurements"]["points"]
    if not record["measurements"]["all_timing_points_passed_validation"]:
        raise SystemExit("refusing to plot timing points that failed validation")

    trajectories = [point["trajectories"] for point in points]
    tn_seconds = [point["tn"]["components"]["steady_total_seconds"]["median_seconds"] for point in points]
    cudaq_seconds = [point["cudaq"]["steady_total"]["median_seconds"] for point in points]
    speedups = [point["steady_state_speedup_baseline_over_tn"] for point in points]

    fig, (runtime_ax, speedup_ax) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    runtime_ax.plot(trajectories, tn_seconds, "o-", label="TN proportional")
    runtime_ax.plot(trajectories, cudaq_seconds, "s-", label="CUDA-Q explicit")
    runtime_ax.set_xscale("log", base=2)
    runtime_ax.set_yscale("log")
    runtime_ax.set_xlabel("Trajectory draws")
    runtime_ax.set_ylabel("Median steady total (s)")
    runtime_ax.grid(True, which="both", alpha=0.25)
    runtime_ax.legend(frameon=False)

    speedup_ax.plot(trajectories, speedups, "o-", color="#6a3d9a")
    speedup_ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    speedup_ax.set_xscale("log", base=2)
    speedup_ax.set_xlabel("Trajectory draws")
    speedup_ax.set_ylabel("Speedup = CUDA-Q / TN runtime")
    speedup_ax.grid(True, which="both", alpha=0.25)
    fig.suptitle("RTX 5090 validated 5-qubit trajectory sweep")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
