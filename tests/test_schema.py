import pytest

from q_tensor.schema import ResultSchemaError, validate_result


def valid_record() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "abc",
        "category": "reproduced_result",
        "timestamp_utc": "2026-09-10T20:00:00Z",
        "command": ["python", "run.py"],
        "git_commit": "deadbeef",
        "upstream_commit": "feedface",
        "hardware": {},
        "software": {},
        "parameters": {},
        "measurements": {},
    }


def test_valid_result() -> None:
    validate_result(valid_record())


def test_missing_provenance_is_rejected() -> None:
    record = valid_record()
    del record["hardware"]
    with pytest.raises(ResultSchemaError, match="hardware"):
        validate_result(record)


def test_unknown_category_is_rejected() -> None:
    record = valid_record()
    record["category"] = "benchmark"
    with pytest.raises(ResultSchemaError, match="category"):
        validate_result(record)
