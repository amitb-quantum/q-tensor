#!/usr/bin/env python3
"""Plot measured validation-gated qubit/depth crossover points."""

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
    if not record["measurements"]["all_timing_points_passed_validation"]:
        raise SystemExit("refusing to plot timing points that failed validation")

    points = record["measurements"]["points"]
    depths = sorted({point["entangling_depth"] for point in points})
    fig, (runtime_ax, ratio_ax) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for depth in depths:
        series = sorted(
            (point for point in points if point["entangling_depth"] == depth),
            key=lambda point: point["nqubits"],
        )
        qubits = [point["nqubits"] for point in series]
        runtime_ax.plot(
            qubits,
            [point["tn"]["components"]["steady_total_seconds"]["median_seconds"] for point in series],
            "o-",
            label=f"TN depth {depth}",
        )
        runtime_ax.plot(
            qubits,
            [point["cudaq"]["steady_total"]["median_seconds"] for point in series],
            "s--",
            label=f"CUDA-Q depth {depth}",
        )
        ratio_ax.plot(
            qubits,
            [point["steady_state_speedup_baseline_over_tn"] for point in series],
            "o-",
            label=f"depth {depth}",
        )

    runtime_ax.set_yscale("log")
    runtime_ax.set_xlabel("Qubits")
    runtime_ax.set_ylabel("Median steady total (s)")
    runtime_ax.grid(True, which="both", alpha=0.25)
    runtime_ax.legend(frameon=False, fontsize=8)
    ratio_ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ratio_ax.axhline(0.8, color="gray", linewidth=1, linestyle=":")
    ratio_ax.set_yscale("log")
    ratio_ax.set_xlabel("Qubits")
    ratio_ax.set_ylabel("Speedup = CUDA-Q / TN runtime")
    ratio_ax.grid(True, alpha=0.25)
    ratio_ax.legend(frameon=False)
    fig.suptitle("RTX 5090 validated complexity-crossover pilot")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
