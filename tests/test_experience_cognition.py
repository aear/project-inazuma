from experience_cognition import (
    adaptive_computation_plan,
    assess_uncertainty,
    build_multi_horizon_predictions,
    gate_transient_state,
    inspect_attention_lenses,
    plan_experience_cognition,
    route_experience,
    compare_hypotheses,
)


def test_empty_and_malformed_references_do_not_count_as_evidence():
    for references in ([None], ['  '], {'not': 'a witness'}, 1):
        result = assess_uncertainty({'candidate_answer': 'guess', 'evidence': {'causal': references}})
        assert result['status'] == 'unknown'
        assert result['available_evidence'] == []


def test_counterevidence_overrides_omitted_contradiction_score():
    event = {'candidate_answer': 'candidate', 'evidence': {'causal': ['a'], 'sensory': ['b']},
        'evidence_origins': {'a': 'origin:a', 'b': 'origin:b'}, 'counterevidence_references': ['opposing:1']}
    result = assess_uncertainty(event)
    assert result['status'] == 'uncertain'
    assert result['conflict_retained']
    assert result['counterevidence_references'] == ['opposing:1']


def test_hypothesis_check_distinguishes_instead_of_repeating_confirmation():
    event = {'hypotheses': [{'id': 'a', 'claim': 'sensor failed'}, {'id': 'b', 'claim': 'object moved'}],
        'observation_candidates': [
            {'id': 'repeat', 'question': 'Read same sensor?', 'expected_outcomes': {'a': 'changed', 'b': 'changed'}},
            {'id': 'independent', 'question': 'Inspect with separate sensor?',
             'expected_outcomes': {'a': 'unchanged', 'b': 'changed'}}]}
    result = compare_hypotheses(event)
    assert result['suggested_check'] == 'independent'
    assert result['checks'][0]['indistinguishable_pairs'] == [['a', 'b']]
    assert not result['automatic_execution']
    from experience_engine import ExperienceCycleEngine
    assert ExperienceCycleEngine.plan_cognition(event)['hypothesis_comparison'] == result


def test_missing_outcomes_and_duplicate_ids_cannot_fake_discrimination():
    result = compare_hypotheses({'hypotheses': [{'id': 'a', 'claim': 'A'}, {'id': 'b', 'claim': 'B'},
        {'id': 'b', 'claim': 'duplicate'}], 'observation_candidates': [
        {'id': 'x', 'question': 'Check?', 'expected_outcomes': {'a': 'yes'}}]})
    assert result['suggested_check'] is None
    assert result['checks'][0]['missing_prediction_pairs'] == [['a', 'b']]
    assert 'missing_or_duplicate_id' in result['rejected']


def test_alternative_budget_and_empty_case_are_explicit():
    assert compare_hypotheses({})['suggested_check'] is None
    result = compare_hypotheses({'hypotheses': [{'id': str(i), 'claim': 'Maybe'} for i in range(10)]})
    assert result['input_truncated'] and len(result['alternatives']) == 8


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


def test_shared_origin_does_not_become_independent_by_repetition():
    event = {'candidate_answer': 'hypothesis', 'evidence': {'causal': ['a'], 'sensory': ['b']},
             'evidence_origins': {'a': 'same source', 'b': 'same source'}}
    assert assess_uncertainty(event)['status'] == 'uncertain'
    event['evidence_origins']['b'] = 'independent source'
    assert assess_uncertainty(event)['status'] == 'known'
    event['signals'] = {'contradiction': .9}
    assert assess_uncertainty(event)['status'] == 'uncertain'


def test_bounded_cognition_does_not_consume_unbounded_iterators():
    def candidates():
        for index in range(64):
            yield {'state_id': str(index)}
        raise AssertionError('input budget exceeded')
    assert len(gate_transient_state(candidates(), limit=64)['candidates']) == 64
    assert len(build_multi_horizon_predictions(candidates())['rejected']) == 64
