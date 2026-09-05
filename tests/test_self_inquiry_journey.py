import pytest

from self_inquiry_journey import (
    begin_self_inquiry, continue_self_inquiry, current_inquiry_request,
)


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
