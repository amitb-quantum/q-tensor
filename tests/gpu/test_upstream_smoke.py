import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.gpu
def test_official_upstream_core_checks() -> None:
    configured = os.environ.get("Q_TENSOR_UPSTREAM")
    if not configured:
        pytest.skip("set Q_TENSOR_UPSTREAM to the pinned checkout")
    subprocess.run(
        [sys.executable, "scripts/run_upstream_smoke.py", "--upstream", configured],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
