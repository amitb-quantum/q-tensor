"""Command-line entry point for Q-Tensor diagnostics and artifact validation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from .config import load_config
from .provenance import command_record
from .schema import validate_result


def _doctor() -> int:
    packages = {}
    for name in (
        "cuquantum-python-cu12",
        "cupy-cuda12x",
        "cuda-quantum-cu12",
        "qiskit",
        "numpy",
        "pandas",
        "scipy",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "nvidia_smi": command_record(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"]),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["nvidia_smi"]["returncode"] == 0 else 1


def _validate_result(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        validate_result(json.load(handle))
    print(f"valid: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="q-tensor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="capture runtime and GPU diagnostics")
    config_parser = subparsers.add_parser("check-config", help="validate an experiment TOML file")
    config_parser.add_argument("path", type=Path)
    result_parser = subparsers.add_parser("check-result", help="validate a saved result JSON file")
    result_parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    if args.command == "check-config":
        load_config(args.path)
        print(f"valid: {args.path}")
        return 0
    return _validate_result(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
