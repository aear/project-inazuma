from utterance_memory import evaluate_utterance_memory_hook


def _usage(text="I want to keep this sentence."):
    return {"utterance": text, "speaker": "Sakura", "entity_links": []}


def test_ordinary_sentence_and_resonance_alone_are_not_promoted():
    ordinary = evaluate_utterance_memory_hook(_usage(), {
        "id": "event:1", "situation_tags": ["conversation"], "internal_state": {},
    })
    resonance_only = evaluate_utterance_memory_hook(_usage(), {
        "id": "event:2", "situation_tags": ["conversation"],
        "internal_state": {"communicative_resonance": .95},
    })
    assert ordinary["decision"] == "do_not_promote"
    assert resonance_only["decision"] == "do_not_promote"
    assert resonance_only["text"] is None


def test_resonance_and_independent_relational_signal_can_promote_sentence():
    result = evaluate_utterance_memory_hook(_usage("I know my memory loves quotes."), {
        "id": "event:3", "situation_tags": ["conversation", "relationship"],
        "internal_state": {"emotional_resonance": .9},
    })
    assert result["decision"] == "retain_projection"
    assert result["independent_origins"] == 2
    assert result["text"] == "I know my memory loves quotes."
    assert result["source_unchanged"] is True


def test_edit_and_novelty_can_be_retained_for_later_repair():
    result = evaluate_utterance_memory_hook({
        **_usage("Actually, I meant developmental memory."),
        "entity_links": [{"type": "discord_message_edit"}],
    }, {
        "id": "event:4", "situation_tags": ["message_edit"],
        "internal_state": {"novelty": .8},
    })
    assert result["retained"] is True
    assert {row["signal"] for row in result["signals"]} == {"repair_value", "novelty"}


def test_corroborated_meaning_preserving_rephrase_can_be_the_projection():
    usage = {
        **_usage("The thing I mean is that this history is memory in a way."),
        "meaning_references": ["meaning:developmental-history"],
    }
    event = {
        "id": "event:5", "situation_tags": ["relationship"],
        "internal_state": {"communicative_resonance": .9},
    }
    result = evaluate_utterance_memory_hook(usage, event, rephrasing_candidates=[{
        "text": "Project history is part of developmental memory.",
        "meaning_references": ["meaning:developmental-history"],
        "confidence": .85, "purpose": "clarity",
        "provenance": ["semantic:event-5", "listener-model:2"],
    }])
    assert result["surface_kind"] == "rephrased"
    assert result["text"] == "Project history is part of developmental memory."
    assert result["rephrasing"]["status"] == "selected"
    assert result["source_unchanged"] is True


def test_rephrase_with_changed_meaning_or_one_origin_is_rejected():
    usage = {**_usage(), "meaning_references": ["meaning:keep"]}
    event = {"id": "event:6", "situation_tags": ["relationship"],
             "internal_state": {"communicative_resonance": .9}}
    result = evaluate_utterance_memory_hook(usage, event, rephrasing_candidates=[{
        "text": "Forget it.", "meaning_references": ["meaning:discard"],
        "confidence": .99, "purpose": "compression", "provenance": ["one:model"],
    }])
    assert result["surface_kind"] == "source"
    assert result["rephrasing"]["status"] == "source_retained"
