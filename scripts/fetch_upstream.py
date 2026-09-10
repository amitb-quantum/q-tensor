#!/usr/bin/env python3
"""Fetch the exact upstream revision without vendoring it in Q-Tensor."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("upstream/LOCK.json"))
    parser.add_argument("--target", type=Path, default=Path("upstream/Accelerated_TN_PTSBE"))
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    repository = lock["repository"]
    revision = lock["commit"]

    if args.target.exists():
        if not (args.target / ".git").exists():
            raise SystemExit(f"refusing to replace non-git path: {args.target}")
        status = run("git", "status", "--porcelain", cwd=args.target)
        if status:
            raise SystemExit(f"refusing to update dirty upstream checkout: {args.target}")
        origin = run("git", "remote", "get-url", "origin", cwd=args.target)
        if origin.rstrip("/").removesuffix(".git") != repository.rstrip("/").removesuffix(".git"):
            raise SystemExit(f"unexpected upstream origin: {origin}")
        run("git", "fetch", "origin", revision, cwd=args.target)
    else:
        args.target.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", repository, str(args.target))

    run("git", "checkout", "--detach", revision, cwd=args.target)
    actual = run("git", "rev-parse", "HEAD", cwd=args.target)
    if actual != revision:
        raise SystemExit(f"checkout mismatch: expected {revision}, found {actual}")
    print(f"upstream ready: {args.target} @ {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
