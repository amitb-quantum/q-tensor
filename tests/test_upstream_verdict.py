from q_tensor.upstream_verdict import evaluate_upstream_verification


def test_printed_failure_overrides_zero_exit() -> None:
    output = """Total Variation Distance: 0.3039
Threshold: 0.15
FAIL: Distributions differ beyond threshold
"""
    verdict = evaluate_upstream_verification(output, process_returncode=0)
    assert not verdict.passed
    assert "despite process exit zero" in verdict.reason


def test_printed_pass_and_metric_are_required() -> None:
    output = """Total Variation Distance: 0.0200
Threshold: 0.15
PASS: Distributions match within threshold
"""
    assert evaluate_upstream_verification(output, process_returncode=0).passed
    assert not evaluate_upstream_verification(output, process_returncode=2).passed
    assert not evaluate_upstream_verification("PASS", process_returncode=0).passed
