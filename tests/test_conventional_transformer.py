import math

from cognitive_benchmarks.core import BenchmarkCase, run_benchmark
from module_benchmarks import benchmark_module
from transformers.conventional_transformer import (
    ConventionalTransformer, ConventionalTransformerConfig,
)


def _model(seed=11):
    config = ConventionalTransformerConfig(
        model_width=8, heads=2, layers=1, feed_forward_width=12, max_sequence=24,
    )
    return ConventionalTransformer(config, seed=seed)


def test_forward_is_deterministic_bounded_and_causally_masked():
    model = _model()
    short = model.forward([0, 66], return_attention=True)
    long = model.forward([0, 66, 67], return_attention=True)
    assert short["logits"] == model.forward([0, 66])["logits"]
    assert short["logits"][1] == long["logits"][1]
    assert len(long["attention"][0]) == 2
    assert [len(row) for row in long["attention"][0][0]] == [1, 2, 3]


def test_state_round_trip_and_shape_validation():
    model = _model()
    restored = ConventionalTransformer(model.config, state=model.state_dict())
    assert restored.forward([0, 66])["logits"] == model.forward([0, 66])["logits"]
    broken = model.state_dict()
    broken["output_projection"] = broken["output_projection"][:-1]
    try:
        ConventionalTransformer(model.config, state=broken)
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("invalid state was accepted")


def test_choice_scorer_participates_without_claiming_training_or_council_role():
    model = _model()
    result = run_benchmark([
        BenchmarkCase("smoke", "composition", "A", ("B", "C"), 0),
    ], model)
    evidence = model.promotion_evidence()
    assert result.total == 1 and math.isfinite(result.mean_margin)
    assert evidence["trained_weights"] is False
    assert evidence["deployment_status"] == "benchmark_only"
    assert evidence["council_member"] is False


def test_versioned_capability_benchmark_records_absent_and_candidate_versions():
    v1, v2 = benchmark_module("conventional_transformer")
    assert (v1.version, v1.correct, v1.total) == ("V1", 0, 4)
    assert (v2.version, v2.correct, v2.total) == ("V2", 4, 4)
