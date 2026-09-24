from experience_cognition import (
    adaptive_computation_plan,
    assess_uncertainty,
    build_multi_horizon_predictions,
    gate_transient_state,
    inspect_attention_lenses,
    plan_experience_cognition,
    route_experience,
)


def test_sparse_router_selects_relevant_routes_and_preserves_rejections():
    routed = route_experience({"signals": {
        "contradiction": .95, "causal": .8, "uncertainty": .7, "social": .1,
    }}, max_routes=2)
    assert [row["route"] for row in routed["selected"]] == ["hindsight", "prediction"]
    assert len(routed["selected"]) == 2
    assert {row["reason"] for row in routed["rejected"]} <= {"below_threshold", "capacity_limit"}


def test_router_can_abstain_instead_of_forcing_activity():
    routed = route_experience({"signals": {}}, minimum_score=.2)
    assert routed["abstained"] is True
    assert routed["selected"] == []


def test_quantity_routes_to_actual_counting_capability():
    routed = route_experience({"signals": {"quantity": 1.0, "sensory": .5}}, max_routes=2)
    assert routed["selected"][0]["route"] == "counting"


def test_attention_lenses_keep_disagreement_visible():
    lenses = inspect_attention_lenses({
        "signals": {"causal": .9, "social": .1, "contradiction": .8},
        "evidence": {"causal": ["event:a"], "contradiction": ["witness:b"]},
    })
    indexed = {row["lens"]: row for row in lenses}
    assert indexed["causal"]["salience"] == .9
    assert indexed["social"]["salience"] == .1
    assert indexed["causal"]["evidence_references"] == ["event:a"]
    assert "aggregate" not in indexed["causal"]


def test_transient_gate_never_mutates_source_experience():
    gated = gate_transient_state([
        {"state_id": "state:a", "relevance": .9, "confidence": .9, "salience": .8,
         "novelty": .4, "independent_witnesses": 2, "source_references": ["experience:1"]},
        {"state_id": "state:b", "relevance": .1, "confidence": .1,
         "source_references": ["experience:2"]},
    ])
    assert [row["action"] for row in gated["candidates"]] == ["propagate", "decay"]
    assert all(row["source_experience_unchanged"] for row in gated["candidates"])
    assert gated["mutates_memory"] is False


def test_transient_state_without_provenance_cannot_propagate():
    gated = gate_transient_state([
        {"state_id": "unsupported", "relevance": 1, "confidence": 1,
         "salience": 1, "novelty": 1, "independent_witnesses": 4},
    ])
    assert gated["candidates"][0]["action"] == "hold"
    assert gated["candidates"][0]["hold_reason"] == "missing_source_reference"


def test_adaptive_computation_is_finite_and_can_stop_early():
    ordinary = adaptive_computation_plan({"signals": {}})
    difficult = adaptive_computation_plan({"signals": {
        "contradiction": .9, "uncertainty": .9, "stakes": .9, "novelty": .9,
    }}, max_steps=3)
    assert ordinary["steps"] == 1
    assert difficult["steps"] == 3
    assert difficult["may_stop_early"] is True
    assert difficult["autonomous_continuation"] is False


def test_unknown_is_valid_and_points_to_missing_evidence_without_forcing_work():
    result = assess_uncertainty({
        "signals": {"uncertainty": .9, "causal": .8, "sensory": .7},
        "required_evidence": ["causal", "sensory"],
        "evidence": {"sensory": ["camera:frame-1"]},
        "candidate_answer": "the switch caused it",
    })
    assert result["status"] == "unknown"
    assert result["answer"] is None
    assert result["missing_evidence"] == ["causal"]
    assert "proposed cause" in result["optional_observations"][0]
    assert result["continuation_required"] is False


def test_conflicting_witnesses_remain_uncertain_instead_of_collapsing():
    result = assess_uncertainty({
        "signals": {"contradiction": .8, "uncertainty": .6},
        "evidence": {"contradiction": ["witness:a", "witness:b"]},
        "candidate_answer": "one interpretation",
    })
    assert result["status"] == "uncertain"
    assert result["conflict_retained"] is True
    assert result["confidence"] <= .4


def test_multi_horizon_predictions_require_provenance_and_disconfirmation():
    predictions = build_multi_horizon_predictions([
        {"horizon": "immediate", "prediction": "the object moves", "confidence": .7,
         "source_references": ["event:1"], "disconfirming_observations": ["object remains still"]},
        {"horizon": "later", "prediction": "unsupported guess"},
    ])
    assert predictions["horizons"]["immediate"][0]["prediction"] == "the object moves"
    assert predictions["horizons"]["later"] == []
    assert predictions["rejected"][0]["reason"].startswith("requires_known_horizon")
    assert predictions["writes_memory"] is False


def test_composed_plan_respects_custom_memory_boundary():
    plan = plan_experience_cognition(
        {"signals": {"novelty": .8, "sensory": .7}},
        transient_candidates=[{"state_id": "visible", "salience": .8,
                               "relevance": .8, "confidence": .8,
                               "source_references": ["observation:1"]}],
        prediction_candidates=[{"horizon": "near", "prediction": "contact",
                                "source_references": ["observation:1"],
                                "disconfirming_observations": ["distance increases"]}],
    )
    assert plan["routing"]["selected"]
    assert len(plan["attention_lenses"]) == 5
    assert plan["memory_boundary"] == {
        "reads_fragment_store": False, "writes_memory": False, "trains_parameters": False,
    }
