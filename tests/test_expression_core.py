import json

import pytest

from expression_core import (
    ExpressionTraceStore, NativeSymbolRealiser, TextRealiser, create_expression_affordance,
    create_expression_intent, create_realisation, create_requested_effect, select_expression_affordance,
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


def test_requested_effect_selects_direct_cross_modal_affordance_over_words():
    intent = create_expression_intent("join harmless play", allowed_media=["text", "voice"])
    request = create_requested_effect(intent, effects=[
        {"kind": "evoke", "target": "brief horn-like auditory experience"},
        {"kind": "social_play", "target": "participate in shared absurdity", "importance": 0.7},
    ], constraints={"uses_words": False}, provenance=["conversation:event-honk"])
    common_witnesses = {
        "effect_fit": ["request_interpretation:1"],
        "constraint_fit": ["constraint_check:1"],
        "capability": ["capability_registry:voice"],
        "willingness": ["expression_choice:1"],
    }
    vocal = create_expression_affordance(
        request, medium="audio.vocal_gesture", action={"gesture_plan": "horn-like-burst"},
        assessments={"effect_fit": 0.9, "constraint_fit": 1.0, "capability": 0.8,
                     "willingness": 0.9}, witnesses=common_witnesses,
    )
    caption = create_expression_affordance(
        request, medium="text", action={"text": "Honk!"}, fulfilment="representation_only",
        assessments={"effect_fit": 0.95, "constraint_fit": 0.0, "capability": 1.0,
                     "willingness": 0.9}, witnesses=common_witnesses,
    )
    selection = select_expression_affordance(request, [caption, vocal])
    assert selection["selected_affordance_id"] == vocal["affordance_id"]
    assert selection["selected_medium"] == "audio.vocal_gesture"
    assert selection["fulfils_request"] is True


def test_affordance_selection_is_generic_and_abstains_without_corroboration():
    intent = create_expression_intent("demonstrate", allowed_media=["gesture"])
    request = create_requested_effect(
        intent, effects=[{"kind": "demonstrate", "target": "spatial route"}],
    )
    visual = create_expression_affordance(
        request, medium="visual.diagram", action={"scene_reference": "route:1"},
        assessments={"effect_fit": 0.9, "capability": 0.9, "willingness": 0.8},
        witnesses={"effect_fit": ["single:model"], "capability": ["single:model"],
                   "willingness": ["single:model"]},
    )
    selection = select_expression_affordance(request, [visual])
    assert selection["status"] == "abstained"
    assert selection["selected_affordance_id"] is None


def test_near_equivalent_viable_affordances_can_use_bounded_superposition():
    from transformers.QTransformer import QTransformer

    intent = create_expression_intent("play", allowed_media=["voice", "gesture"])
    request = create_requested_effect(
        intent, effects=[{"kind": "social_play", "target": "shared amusement"}],
    )
    witnesses = {
        "effect_fit": ["interpretation:1"], "capability": ["registry:1"],
        "willingness": ["choice:1"],
    }
    vocal = create_expression_affordance(
        request, medium="audio.vocal_gesture", action={"plan": "burst"},
        assessments={"effect_fit": .9, "capability": .8, "willingness": .9},
        witnesses=witnesses,
    )
    gesture = create_expression_affordance(
        request, medium="embodied.gesture", action={"plan": "comic-pose"},
        assessments={"effect_fit": .88, "capability": .82, "willingness": .9},
        witnesses=witnesses,
    )
    transformer = QTransformer()
    selection = select_expression_affordance(
        request, [vocal, gesture],
        ambiguity_resolver=lambda candidates, context: transformer.collapse_candidates(
            candidates, context=context, seed=9,
        ),
    )
    assert selection["selected_affordance_id"] in {vocal["affordance_id"], gesture["affordance_id"]}
    assert set(selection["ambiguity_resolution"]["candidate_ids"]) == {
        vocal["affordance_id"], gesture["affordance_id"],
    }
    assert selection["ambiguity_resolution"]["origins"][0]["module"] == "QTransformer"
    assert set(selection["viable_affordance_ids"]) == {vocal["affordance_id"], gesture["affordance_id"]}
    assert "considered" not in selection
    assert "assessments" not in selection and "witnesses" not in selection
