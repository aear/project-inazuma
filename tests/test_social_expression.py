from expression_core import create_expression_intent
from social_expression import (
    assess_communicative_repair, assess_mutual_intelligibility, build_listener_hypotheses,
)


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


def _clarity_fixture(native_recoverability=.3, english_fidelity=.9):
    intent = create_expression_intent(
        "share a grounded meaning", meaning_references=["meaning:care"],
        audience_references=["person:sakura"], allowed_media=["native_symbol", "text"],
    )
    listener = build_listener_hypotheses("person:sakura", [
        {"witness_id": "conversation:1", "proposition_reference": "language:english",
         "state": "may_know", "confidence": .9, "provenance": ["conversation:1"]},
        {"witness_id": "history:1", "proposition_reference": "language:english",
         "state": "may_know", "confidence": .9, "provenance": ["history:1"]},
    ])
    assessments = [
        {"realisation_id": "native:1", "medium": "native_symbol",
         "fidelity": .95, "recoverability": native_recoverability,
         "uncertainty_preservation": .9,
         "witnesses": {key: [f"{key}:native:1", f"{key}:native:2"]
                       for key in ("fidelity", "recoverability", "uncertainty_preservation")}},
        {"realisation_id": "text:1", "medium": "text", "language": "english",
         "fidelity": english_fidelity, "recoverability": .9,
         "uncertainty_preservation": .8,
         "witnesses": {key: [f"{key}:text:1", f"{key}:text:2"]
                       for key in ("fidelity", "recoverability", "uncertainty_preservation")}},
    ]
    return intent, listener, assessments


def test_low_listener_recoverability_offers_voluntary_english_bridge():
    intent, listener, assessments = _clarity_fixture()
    result = assess_mutual_intelligibility(
        intent, assessments, listener_model=listener, bridge_willingness=.8,
        willingness_witnesses=["choice:current", "preference:stable"],
    )
    assert result["mode"] == "native_with_english_bridge"
    assert result["english_is_internal_representation"] is False
    assert result["automatic_expression"] is False


def test_clear_native_expression_needs_no_english_bridge():
    intent, listener, assessments = _clarity_fixture(native_recoverability=.9)
    result = assess_mutual_intelligibility(
        intent, assessments, listener_model=listener, bridge_willingness=.8,
        willingness_witnesses=["choice:current", "preference:stable"],
    )
    assert result["mode"] == "native_only"


def test_bridge_is_not_forced_and_single_signal_cannot_establish_clarity():
    intent, listener, assessments = _clarity_fixture()
    declined = assess_mutual_intelligibility(
        intent, assessments, listener_model=listener, bridge_willingness=.1,
        willingness_witnesses=["choice:current", "preference:stable"],
    )
    for key in assessments[1]["witnesses"]:
        assessments[1]["witnesses"][key] = ["one:model"]
    unsupported = assess_mutual_intelligibility(
        intent, assessments, listener_model=listener, bridge_willingness=.8,
        willingness_witnesses=["choice:current", "preference:stable"],
    )
    assert declined["mode"] == "native_only"
    assert declined["reason"] == "bridge_not_voluntarily_chosen"
    assert unsupported["mode"] == "clarify"
    assert unsupported["engagement_is_understanding_evidence"] is False


def test_english_bridge_requires_corroborated_listener_language_evidence():
    intent, listener, assessments = _clarity_fixture()
    listener["hypotheses"][0]["support"] = listener["hypotheses"][0]["support"][:1]
    result = assess_mutual_intelligibility(
        intent, assessments, listener_model=listener, bridge_willingness=.8,
        willingness_witnesses=["choice:current", "preference:stable"],
    )
    assert result["mode"] == "clarify"
    assert len(result["english_listener_evidence_origins"]) == 1
