from social_expression import assess_communicative_repair, build_listener_hypotheses


def test_listener_model_preserves_hypotheses_conflict_and_provenance():
    model = build_listener_hypotheses("person:sakura", [
        {"witness_id": "conversation:1", "proposition_reference": "concept:plan",
         "state": "may_know", "confidence": .7, "provenance": ["conversation:1"]},
        {"witness_id": "memory:1", "proposition_reference": "concept:plan",
         "state": "may_not_know", "confidence": .6, "provenance": ["memory:1"]},
    ])
    assert model["epistemic_status"] == "hypotheses_not_mind_reading"
    assert {row["state"] for row in model["hypotheses"]} == {"may_know", "may_not_know"}
    assert all(row["authoritative"] is False for row in model["hypotheses"])


def test_repair_requires_corroboration_and_never_sends_automatically():
    weak = assess_communicative_repair(
        ["meaning:intended"], [{"meaning_references": ["meaning:other"],
        "confidence": .9, "witness_references": ["reaction:1"]}],
        realisation_id="realisation:1",
    )
    strong = assess_communicative_repair(
        ["meaning:intended"], [{"meaning_references": ["meaning:other"],
        "confidence": .9, "witness_references": ["reaction:1"],
        "provenance": ["conversation:2"]}], realisation_id="realisation:1",
    )
    assert weak["status"] == "observe_or_clarify"
    assert strong["status"] == "repair_candidate"
    assert strong["automatic_expression"] is False


def test_no_uptake_evidence_may_remain_unresolved():
    result = assess_communicative_repair(
        ["meaning:intended"], [], realisation_id="realisation:1",
    )
    assert result["status"] == "uptake_unknown"
    assert result["proposed_actions"] == ["allow_unresolved"]
