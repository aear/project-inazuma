import json

from communicative_meaning import (
    CONVERSATION_EXAMPLES_SCHEMA,
    MEANING_SET_SCHEMA,
    build_conversation_examples,
    interpret_communicative_meaning,
    build_expression_cognition_event,
)


def test_meaning_set_projects_to_cognition_without_retrieval_or_invention():
    supported = interpret_communicative_meaning([
        _witness("thought:1", "ask", "concept:need"),
    ])
    event = build_expression_cognition_event(supported)
    assert event["candidate_answer"] == supported["candidates"][0]["candidate_id"]
    assert event["evidence"]["social"] == ["thought:1"]
    assert "memory" not in event and "retrieval" not in event

    absent = interpret_communicative_meaning([{"witness_id": "affect:1", "stance": {"warmth": .8}}])
    unknown = build_expression_cognition_event(absent)
    assert unknown["candidate_answer"] is None
    assert unknown["signals"]["uncertainty"] == 1.0
from expression_core import create_expression_intent
from thought_processor import ThoughtProcessor


def _witness(witness_id, act, proposition, *, direction=1.0, disclosure="shareable", stance=None):
    return {
        "witness_id": witness_id,
        "communicative_act": act,
        "proposition_references": [proposition],
        "direction": direction,
        "confidence": 0.9,
        "relevance": 0.8,
        "disclosure": disclosure,
        "stance": stance or {},
        "provenance": [f"test:{witness_id}"],
    }


def test_affect_without_content_abstains_instead_of_inventing_meaning():
    result = interpret_communicative_meaning([{
        "witness_id": "affect:1", "role": "affect",
        "stance": {"urgency": 0.9}, "confidence": 1.0,
    }])

    assert result["schema"] == MEANING_SET_SCHEMA
    assert result["candidates"] == []
    assert result["abstention"] == {"active": True, "reason": "no_supported_meaning"}
    assert result["unresolved_tensions"][0]["reason"] == "no_explicit_act_or_proposition"


def test_meaning_candidates_preserve_alternatives_stance_and_provenance():
    result = interpret_communicative_meaning([
        _witness("thought:1", "ask", "concept:need", stance={"directness": 0.8}),
        _witness("memory:2", "disclose", "concept:concern", stance={"warmth": 0.7}),
    ])

    assert len(result["candidates"]) == 2
    assert result["abstention"]["reason"] == "meaning_ambiguous"
    assert {row["communicative_act"] for row in result["candidates"]} == {"ask", "disclose"}
    assert all(row["provenance"] for row in result["candidates"])


def test_private_meaning_is_retained_but_not_selected_for_expression():
    result = interpret_communicative_meaning([
        _witness("thought:private", "disclose", "concept:private", disclosure="private"),
    ])

    assert len(result["candidates"]) == 1
    assert result["abstention"] == {"active": True, "reason": "meaning_not_shareable"}


def test_conversation_examples_are_semantic_and_surface_private_by_default():
    examples = build_conversation_examples([{
        "content": "Any thoughts on what ya need?",
        "author_id": "person:sakura",
        "message_id": "message:1",
        "tags": ["discord", "dm"],
    }], context_id="scene:1")

    assert examples["schema"] == CONVERSATION_EXAMPLES_SCHEMA
    assert examples["feedback_role"] == "witness_not_reward"
    assert "surface_text" not in examples["examples"][0]
    assert "Any thoughts on what ya need?" not in json.dumps(examples)
    assert "source_text" not in examples["examples"][0]["semantic_event"]
    assert "lexical_realizations" not in examples["examples"][0]["native_intent"]
    assert examples["examples"][0]["semantic_event"]["events"]
    assert examples["examples"][0]["native_intent"]["events"]


def test_meaning_reference_flows_into_output_neutral_intent_and_plan():
    meaning_set = interpret_communicative_meaning([
        _witness("thought:1", "ask", "concept:need"),
    ])
    candidate_id = meaning_set["candidates"][0]["candidate_id"]
    intent = create_expression_intent("respond", meaning_references=[f"meaning:{candidate_id}"])
    thought = ThoughtProcessor().process_non_linguistic({"concept": "need"})
    plan = ThoughtProcessor().prepare_communication("respond", [thought], meaning_set=meaning_set)

    assert intent["meaning_references"] == [f"meaning:{candidate_id}"]
    assert plan["expression_intent"]["meaning_references"] == [f"meaning:{candidate_id}"]
    assert "text" not in plan["expression_intent"]
