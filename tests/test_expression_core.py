import json

import pytest

from expression_core import (
    ExpressionTraceStore, NativeSymbolRealiser, TextRealiser, create_expression_intent, create_realisation,
    create_reaction_interpretation, create_reaction_observation,
)


def test_intent_is_output_neutral_and_realisers_are_medium_specific():
    intent = create_expression_intent(
        "offer reassurance", concept_references=["semantic:event-1"],
        dimensions={"intensity": 0.4, "confidence": 0.7},
        allowed_media=["text", "voice", "gesture"], provenance=["emotion:snapshot-1"],
    )
    text = create_realisation(
        intent, medium="text", content={"text": "I'm here."},
        conventions=["discord:dm"], realiser="test.text", version="V1",
    )
    gesture = create_realisation(
        intent, medium="gesture", content={"candidate": "steady gaze"},
        conventions=["embodied"], realiser="test.gesture", version="V1",
    )
    assert text["intent_id"] == gesture["intent_id"] == intent["intent_id"]
    assert "text" not in intent and "gesture" not in intent
    with pytest.raises(ValueError, match="medium-specific"):
        create_expression_intent("care", dimensions={"emoji": 0.5})


def test_intent_can_reference_meaning_without_embedding_a_realisation():
    intent = create_expression_intent(
        "respond", meaning_references=["meaning:candidate-1"],
        allowed_media=["text", "native_symbol"],
    )
    assert intent["meaning_references"] == ["meaning:candidate-1"]
    assert "text" not in intent and "native_text" not in intent


def test_reaction_is_observation_then_uncertain_interpretation_not_reward(tmp_path):
    intent = create_expression_intent("greet", allowed_media=["text"])
    realised = create_realisation(
        intent, medium="text", content={"text": "<3"},
        conventions=["discord"], realiser="test.text",
    )
    reaction = create_reaction_observation(
        realised["realisation_id"], {"kind": "reply", "delay_seconds": 12},
        source="discord:event-2", causal_confidence=0.3,
    )
    interpretation = create_reaction_interpretation(reaction["reaction_id"], [
        {"meaning": "welcomed", "confidence": 0.55},
        {"meaning": "unrelated timing", "confidence": 0.45},
    ])
    store = ExpressionTraceStore(tmp_path / "expression.jsonl")
    for record in (intent, realised, reaction, interpretation):
        store.append(record)
    rows = [json.loads(line) for line in (tmp_path / "expression.jsonl").read_text().splitlines()]
    assert [row["schema"] for row in rows] == [
        intent["schema"], realised["schema"], reaction["schema"], interpretation["schema"],
    ]
    assert len(interpretation["candidates"]) == 2
    with pytest.raises(ValueError, match="not rewards"):
        create_reaction_observation(realised["realisation_id"], {"reward": 1}, source="bad")


def test_realisation_must_be_allowed_by_intent():
    intent = create_expression_intent("speak", allowed_media=["voice"])
    with pytest.raises(PermissionError):
        create_realisation(intent, medium="text", content={}, realiser="test")


def test_concrete_realisers_translate_one_intent_independently():
    intent = create_expression_intent("care", allowed_media=["text", "native_symbol"])
    text = TextRealiser().realise(intent, content={"text": "I care."}, conventions=["discord"])
    native = NativeSymbolRealiser().realise(intent, content={"native_text": "glyph_care"})
    assert text["medium"] == "text" and native["medium"] == "native_symbol"
    assert text["intent_id"] == native["intent_id"] == intent["intent_id"]
    assert text["content"] != native["content"]
