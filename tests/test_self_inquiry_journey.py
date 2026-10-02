import pytest

from self_inquiry_journey import (
    begin_self_inquiry, continue_self_inquiry, current_inquiry_request, begin_intuition_inquiry,
)


def test_terminal_inquiry_cannot_be_reopened():
    for choice in ('stop', 'remain_uncertain'):
        journey = begin_self_inquiry('Investigate?', trigger_references=['event:1'], depth_budget=3)
        stopped = continue_self_inquiry(journey, choice=choice)
        with pytest.raises(PermissionError, match='terminal'):
            continue_self_inquiry(stopped, choice='deeper')


def test_intuition_preserves_error_correction_without_certifying_truth():
    original = begin_intuition_inquiry('Something about this matters', question='Does the contrast help?',
        countercheck='Compare a version without the contrast', trigger_references=['drawing:1'], depth_budget=2)
    assert not original['intuition']['explanation_required']
    assert current_inquiry_request(original)['countercheck']
    first = continue_self_inquiry(original, choice='deeper', hypotheses=[
        {'hypothesis': 'Contrast helps', 'confidence': .8, 'evidence_references': ['review:a']}])
    revised = continue_self_inquiry(first, choice='remain_uncertain', observation_references=['review:b'],
        hypotheses=[{'hypothesis': 'It may depend on context', 'confidence': .3,
                     'evidence_references': ['review:a'], 'counterevidence_references': ['review:b']}])
    assert revised['hypothesis_history'][-1]['previous'][0]['confidence'] == .8
    assert revised['hypotheses'][0]['counterevidence_references'] == ['review:b']
    assert not revised['hypotheses'][0]['authoritative']
    assert revised['intuition']['truth_status'] == 'unresolved'
    assert original['hypotheses'] == []
    assert first['hypotheses'][0]['confidence'] == .8


def test_inquiry_does_not_overconsume_duplicate_witnesses_or_candidates():
    def refs():
        for _ in range(32):
            yield 'same'
        raise AssertionError('overconsumed references')
    journey = begin_self_inquiry('Bounded?', trigger_references=refs())
    def candidates():
        for _ in range(8):
            yield {'hypothesis': 'Maybe'}
        raise AssertionError('overconsumed hypotheses')
    result = continue_self_inquiry(journey, choice='stop', hypotheses=candidates())
    assert len(result['hypotheses']) == 8


def test_no_revision_preserves_candidates_and_nonfinite_confidence_is_rejected():
    journey = begin_self_inquiry('Question?', trigger_references=['event:1'], depth_budget=2)
    first = continue_self_inquiry(journey, choice='deeper', hypotheses=[{'hypothesis': 'Maybe'}])
    result = continue_self_inquiry(first, choice='stop')
    assert result['hypotheses'] == first['hypotheses']
    with pytest.raises(ValueError, match='finite'):
        continue_self_inquiry(journey, choice='stop', hypotheses=[{'hypothesis': 'Maybe', 'confidence': float('nan')}])


def test_self_inquiry_is_voluntary_bounded_and_code_is_a_late_witness():
    journey = begin_self_inquiry(
        "Why did I choose that expression?", trigger_references=["selection:1"],
        depth_budget=5, include_code=True,
    )
    assert [stage["name"] for stage in journey["stages"]] == [
        "experience", "near_context", "patterns", "architecture", "hypotheses",
    ]
    assert current_inquiry_request(journey)["evidence_route"] == "activity_witness"
    for expected_route in ("reflection_witness", "memory_index_query", "self_read_code"):
        journey = continue_self_inquiry(
            journey, choice="deeper", observation_references=[f"witness:{expected_route}"],
        )
        assert current_inquiry_request(journey)["evidence_route"] == expected_route
    assert current_inquiry_request(journey)["limits"] == {
        "records": 16, "code_files": 4, "may_defer": True,
    }


def test_self_inquiry_can_remain_uncertain_without_confabulation():
    journey = begin_self_inquiry(
        "What moved me?", trigger_references=["experience:1"], depth_budget=2,
    )
    journey = continue_self_inquiry(journey, choice="remain_uncertain", hypotheses=[
        {"hypothesis": "Perhaps familiarity mattered", "confidence": .35,
         "evidence_references": ["reflection:1"]},
        {"hypothesis": "Perhaps playfulness mattered", "confidence": .4,
         "evidence_references": ["affect:1"]},
    ])
    assert journey["status"] == "remain_uncertain"
    assert len(journey["hypotheses"]) == 2
    assert all(item["authoritative"] is False for item in journey["hypotheses"])
    assert current_inquiry_request(journey) is None


def test_self_inquiry_cannot_exceed_its_chosen_depth():
    journey = begin_self_inquiry(
        "Look closer", trigger_references=["choice:1"], depth_budget=1, include_code=True,
    )
    with pytest.raises(PermissionError, match="exhausted"):
        continue_self_inquiry(journey, choice="deeper")


def test_meditation_admits_explicit_request_without_auto_continuation(monkeypatch):
    import meditation_state

    state = {"self_inquiry_request": {
        "requested": True, "question": "Why this gesture?",
        "trigger_references": ["selection:gesture-1"], "depth_budget": 4,
        "include_code": True,
    }}
    monkeypatch.setattr(meditation_state, "get_inastate", lambda key, default=None: state.get(key, default))
    monkeypatch.setattr(meditation_state, "update_inastate", lambda key, value: state.__setitem__(key, value))
    monkeypatch.setattr(meditation_state, "log_to_statusbox", lambda message: None)
    journey = meditation_state._begin_requested_self_inquiry()
    assert journey["stage_index"] == 0
    assert journey["remaining_continuations"] == 3
    assert state["self_inquiry_request"]["status"] == "started"
    assert state["self_inquiry_evidence_request"]["stage"] == "experience"

    state["self_inquiry_continue_request"] = {
        "requested": True, "choice": "deeper",
        "observation_references": ["activity:gesture-1"],
    }
    updated = meditation_state._continue_requested_self_inquiry()
    assert updated["stage_index"] == 1
    assert state["self_inquiry_evidence_request"]["stage"] == "near_context"
    assert state["self_inquiry_continue_request"]["status"] == "applied"

    assert meditation_state._continue_requested_self_inquiry() is None
