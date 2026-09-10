"""Deterministic provenance helpers used by experiment runners."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def experiment_id(parameters: dict[str, Any]) -> str:
    """Return a stable 16-hex-character identifier for experiment parameters."""
    return hashlib.sha256(canonical_json(parameters).encode("utf-8")).hexdigest()[:16]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_revision(path: str | Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def command_record(command: Sequence[str]) -> dict[str, Any]:
    """Run a read-only diagnostic command and return a serializable record."""
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
