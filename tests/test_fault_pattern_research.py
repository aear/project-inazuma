import math

import pytest

from fault_pattern_research import (
    apply_fault_map, create_challenge_experiment, extract_fault_map, fault_features, generate_fault_map,
    position_capacity_bits, score_candidate, total_variation,
)


def test_seeded_fault_maps_are_bounded_reproducible_and_reversible():
    faults = generate_fault_map(8192, 20, seed=34, cluster_probability=0.4)
    assert faults == generate_fault_map(8192, 20, seed=34, cluster_probability=0.4)
    assert 20 <= len(faults) <= 80
    carrier = bytes(range(256)) * 4
    observed = apply_fault_map(carrier, faults)
    assert extract_fault_map(carrier, observed) == faults
    assert apply_fault_map(observed, faults) == carrier


def test_features_and_distances_are_transparent():
    features = fault_features([1, 2, 18], 64, bank_bits=16)
    assert features["fault_count"] == 3
    assert features["adjacent_fraction"] == 0.5
    assert features["occupied_banks"] == 2
    assert total_variation([1, 1], [2, 2]) == 0.0
    assert score_candidate(features, features)["starter_detectability"] == 0.0


def test_capacity_is_an_upper_bound_not_a_crypto_claim():
    assert position_capacity_bits(8, 1) == pytest.approx(3.0)
    assert position_capacity_bits(8, 0) == pytest.approx(0.0)
    assert math.isfinite(position_capacity_bits(1_000_000, 100))


def test_invalid_or_unbounded_inputs_fail_closed():
    with pytest.raises(ValueError):
        generate_fault_map(7, 1, seed=1)
    with pytest.raises(ValueError):
        apply_fault_map(b"x", [8])
    with pytest.raises(ValueError):
        extract_fault_map(b"a", b"ab")


def test_challenge_helper_supplies_tools_without_project_access():
    class Lab:
        def create(self, **kwargs):
            return kwargs
    created = create_challenge_experiment(
        Lab(), hypothesis="A constrained map will recover.", code="pass", dataset={"seed": 34},
    )
    assert created["room"] == "python-scratch"
    assert "fault_pattern_research.py" in created["support_files"]
    assert created["autonomous_continuation_budget"] == 0
