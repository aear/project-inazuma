from cognition_runtime import CognitiveContext
from thought_processor import ThoughtProcessor
from expression_core import create_reaction_interpretation, create_reaction_observation, create_realisation


def test_non_linguistic_thought_keeps_structure_without_language_conversion():
    processor = ThoughtProcessor()
    content = {"spatial_vector": [0.2, -0.8], "affect": {"valence": -0.3}}
    thought = processor.process_non_linguistic(content, provenance=["sensor:vestibular"])
    assert thought.mode == "non_linguistic"
    assert thought.content["spatial_vector"] == (0.2, -0.8)
    assert "source_text" not in thought.content
    assert thought.provenance == ("sensor:vestibular",)


def test_linguistic_thought_uses_semantic_event_boundary_and_context():
    processor = ThoughtProcessor()
    context = CognitiveContext.build(
        discourse={"resolutions": {"you": {"role": "addressee", "referents": [{"id": "ina"}]}}},
        provenance=["conversation:7"],
    )
    thought = processor.process_linguistic("You did not move the key.", context=context)
    assert thought.mode == "linguistic"
    event = thought.content["semantic_event"]["events"][0]
    assert event["predicate"] == "do"
    assert event["arguments"]["theme"]["surface"] == "move the key"
    assert event["negated"] is True
    assert thought.content["native_intent"]["grammar"]
    assert thought.context_id == context.context_id


def test_decision_can_mix_linguistic_and_non_linguistic_evidence():
    processor = ThoughtProcessor()
    seen = processor.process_non_linguistic({"door": "blocked"}, confidence=0.95)
    heard = processor.process_linguistic("The left route is shorter.", confidence=0.7)
    decision = processor.decide(
        ["left", "right"], [seen, heard], evidence=[
            {"option": "left", "thought_id": heard.thought_id, "weight": 0.8, "reason": "shorter"},
            {"option": "left", "thought_id": seen.thought_id, "weight": -1.0, "reason": "blocked"},
            {"option": "right", "thought_id": seen.thought_id, "weight": 0.8, "reason": "clear alternative"},
        ],
    )
    assert decision.selected == "right"
    assert decision.modalities == ("linguistic", "non_linguistic")
    assert len(decision.evidence) == 3


def test_decision_may_remain_unresolved_when_evidence_is_tied_or_missing():
    processor = ThoughtProcessor()
    thought = processor.process_non_linguistic({"signal": "uncertain"}, confidence=0.5)
    tied = processor.decide(
        ["wait", "act"], [thought], evidence=[
            {"option": "wait", "thought_id": thought.thought_id, "weight": 1.0},
            {"option": "act", "thought_id": thought.thought_id, "weight": 1.0},
        ],
    )
    missing = processor.decide(["wait", "act"], [thought], evidence=[])
    assert tied.status == "undecided" and tied.reason == "evidence_too_close"
    assert missing.status == "undecided" and missing.reason == "no_applicable_evidence"


def test_guided_decision_integrates_four_sources_without_hidden_priority():
    processor = ThoughtProcessor()
    context = CognitiveContext.build(goals=["choose a route"], provenance=["cycle:9"])
    result = processor.guided_decision(
        ["cross", "wait"],
        {
            "emotion": [{"content": {"risk": 0.8}, "confidence": 0.8}],
            "instinct": [{"content": {"urge": "wait"}, "confidence": 0.9}],
            "cognition": [{"content": "Crossing now is quicker.", "linguistic": True, "confidence": 0.7}],
            "memory": [{"content": {"reference": "memory://fragment/7", "outcome": "unsafe"}, "confidence": 0.9}],
        },
        evidence=[
            {"option": "wait", "source": "emotion", "weight": 0.6, "reason": "risk"},
            {"option": "wait", "source": "instinct", "weight": 0.7, "reason": "urge"},
            {"option": "cross", "source": "cognition", "weight": 0.7, "reason": "shorter"},
            {"option": "wait", "source": "memory", "weight": 0.8, "reason": "prior outcome"},
        ],
        context=context,
    )
    assert result["decision"]["selected"] == "wait"
    assert result["guidance_coverage"] == {
        "emotion": 1, "instinct": 1, "cognition": 1, "memory": 1,
    }
    assert all(item["context_id"] == context.context_id for item in result["thoughts"])
    assert {item["metadata"]["guidance_source"] for item in result["thoughts"]} == {
        "emotion", "instinct", "cognition", "memory",
    }


def test_model_manager_exposes_guided_decision_through_stable_facade():
    import model_manager as mm
    result = mm.guide_thought_decision(
        ["rest", "continue"],
        {"emotion": [{"content": {"stress": 0.9}}]},
        evidence=[{"option": "rest", "source": "emotion", "weight": 1.0}],
        provenance=["runtime:test"],
    )
    assert result["decision"]["selected"] == "rest"
    assert result["thoughts"][0]["provenance"] == ["runtime:test", "guidance:emotion"]


def test_thoughts_prepare_communication_without_exposing_internal_content():
    processor = ThoughtProcessor()
    feeling = processor.process_non_linguistic(
        {"care": 0.9}, confidence=0.8,
        metadata={"guidance_source": "emotion"}, provenance=["emotion:snapshot-2"],
    )
    meaning = processor.process_linguistic("I want to understand what changed.", confidence=0.7)
    plan = processor.prepare_communication(
        "ask for clarification", [feeling, meaning],
        audience_references=["person:sakura"], allowed_media=["text", "voice"],
        dimensions={"care": 0.9, "directness": 0.6},
    )
    intent = plan["expression_intent"]
    assert set(plan["selected_thought_ids"]) == {feeling.thought_id, meaning.thought_id}
    assert intent["affect_references"] == [f"affect:{feeling.thought_id}"]
    assert intent["semantic_references"] == [f"semantic:{meaning.thought_id}"]
    assert feeling.content not in intent.values()
    assert "text" not in intent


def test_communication_plan_integrates_cognition_before_text_realisation():
    processor = ThoughtProcessor()
    thought = processor.process_non_linguistic(
        {"candidate": "cause:a"}, confidence=.5, provenance=["observation:1"],
    )
    plan = processor.prepare_communication(
        "answer causal question", [thought], allowed_media=["text"],
        cognition_event={
            "signals": {"uncertainty": .9, "causal": .8},
            "candidate_answer": "cause:a", "required_evidence": ["causal", "temporal"],
            "evidence": {"causal": ["observation:1"]},
        },
    )
    assert plan["experience_cognition"]["epistemic_state"]["status"] == "unknown"
    assert plan["text_expression_guidance"]["allowed_response_kinds"] == [
        "acknowledge_unknown", "ask_for_evidence", "silence",
    ]
    assert plan["expression_intent"]["uncertainty"]["missing_evidence"] == ["temporal"]


def test_communication_feedback_becomes_evidence_for_revision_not_reward():
    processor = ThoughtProcessor()
    original = processor.process_linguistic("The plan is clear.", confidence=0.6)
    plan = processor.prepare_communication("check shared understanding", [original], allowed_media=["text"])
    realised = create_realisation(
        plan["expression_intent"], medium="text", content={"text": "Does that make sense?"},
        realiser="test.text",
    )
    reaction = create_reaction_observation(
        realised["realisation_id"], {"kind": "clarifying_question"},
        source="conversation:event-3", causal_confidence=0.8,
    )
    interpretation = create_reaction_interpretation(reaction["reaction_id"], [
        {"meaning": "explanation was ambiguous", "confidence": 0.75},
        {"meaning": "listener wants detail", "confidence": 0.25},
    ])
    feedback = processor.process_communication_feedback(plan, reaction, interpretation)
    revised = processor.revise_thought(
        original, "The plan may need a clearer explanation.", evidence=[feedback], confidence=0.75,
    )
    assert feedback.metadata["revision_candidate"] is True
    assert feedback.confidence == 0.6
    assert "reward" not in feedback.content
    assert revised.metadata["revision_of"] == original.thought_id
    assert revised.metadata["evidence_thought_ids"] == (feedback.thought_id,)
