from q_tensor.provenance import canonical_json, experiment_id


def test_canonical_json_ignores_mapping_order() -> None:
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_experiment_id_is_stable_and_short() -> None:
    first = experiment_id({"shots": 10, "seed": 7})
    second = experiment_id({"seed": 7, "shots": 10})
    assert first == second
    assert len(first) == 16
