"""Strict interpretation of the upstream end-to-end verification output."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class UpstreamVerdict:
    process_returncode: int
    printed_status: str | None
    tvd: float | None
    threshold: float | None
    passed: bool
    reason: str

    def as_dict(self) -> dict[str, int | str | float | bool | None]:
        return asdict(self)


_TVD_RE = re.compile(r"Total Variation Distance:\s*([0-9.eE+-]+)")
_THRESHOLD_RE = re.compile(r"Threshold:\s*([0-9.eE+-]+)")
_STATUS_RE = re.compile(r"^\s*(PASS|FAIL):", re.MULTILINE)


def evaluate_upstream_verification(output: str, process_returncode: int) -> UpstreamVerdict:
    """Require process success, a parseable metric, and an actual PASS verdict."""

    status_match = _STATUS_RE.search(output)
    tvd_match = _TVD_RE.search(output)
    threshold_match = _THRESHOLD_RE.search(output)
    status = status_match.group(1) if status_match else None
    tvd = float(tvd_match.group(1)) if tvd_match else None
    threshold = float(threshold_match.group(1)) if threshold_match else None

    if process_returncode != 0:
        return UpstreamVerdict(process_returncode, status, tvd, threshold, False, "subprocess exited nonzero")
    if status is None or tvd is None or threshold is None:
        return UpstreamVerdict(process_returncode, status, tvd, threshold, False, "verification output was incomplete")
    metric_passed = tvd < threshold
    if status != "PASS" or not metric_passed:
        return UpstreamVerdict(
            process_returncode,
            status,
            tvd,
            threshold,
            False,
            "upstream statistical criterion failed despite process exit zero",
        )
    return UpstreamVerdict(process_returncode, status, tvd, threshold, True, "verification criterion passed")
