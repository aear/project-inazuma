from creative_experience import (
    NEXT_CHOICES, begin_cultivation, begin_experience, choose_next, experience_command_fields,
    experience_path, record_experiment, save_experience,
)
from emergence_capture import CAPTURE_SCHEMA, MAX_CONTENT_BYTES, append_record, capture_emergence, propose_interpretation
import pytest


def test_creative_experience_is_one_attempt_then_an_explicit_choice(tmp_path):
    session = begin_experience("daw", "Try a quiet pulse", hypothesis="space may help the rhythm")
    session = record_experiment(session, {"id": "daw-command-60"}, observation="one pulse heard")

    assert session["stage"] == "observation"
    assert session["experiment_count"] == 1
    assert session["experiments"][0]["choice"] is None
    assert session["next_choices"] == list(NEXT_CHOICES)
    assert session["may_pause"] is True
    assert choose_next(session, "revisit", reflection="listen again later")["stage"] == "revisit"

    save_experience("Ina", session, root=tmp_path, history_limit=2)
    assert experience_path("Ina", tmp_path).exists()


def test_drawing_and_motor_commands_share_non_forcing_experience_metadata():
    drawing = begin_experience("drawing", "Try one line")
    motor = begin_experience("motor", "Try one step")

    for session in (drawing, motor):
        fields = experience_command_fields(session)["creative_experience"]
        assert fields["may_pause"] is True
        assert fields["may_stop"] is True
        assert set(fields["next_choices"]) == set(NEXT_CHOICES)


def test_emergence_can_be_caught_without_intent_or_explanation(tmp_path):
    capture = capture_emergence(
        "We Had Fun on The Deathstar", modality="text", salience=0.8,
        context_references=[{"id": "conversation-1", "kind": "conversation"}],
    )
    assert capture["schema"] == CAPTURE_SCHEMA
    assert capture["meaning_status"] == "unresolved"
    assert capture["meaning"] is None
    assert capture["requires_interpretation"] is False
    assert "intention" not in capture

    original = dict(capture)
    interpretation = propose_interpretation(capture, "cheerful institutional satire", confidence=0.3)
    assert interpretation["capture_id"] == capture["capture_id"]
    assert capture == original

    ledger = append_record(tmp_path / "emergence.jsonl", capture)
    append_record(ledger, interpretation)
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2


def test_cultivation_links_capture_without_recasting_it_as_intent():
    capture = capture_emergence("a crooked green rhythm", modality="mixed")
    session = begin_cultivation(capture, "drawing", "Try one visual variation")
    assert session["emergence_capture_id"] == capture["capture_id"]
    assert session["intention"] == "Try one visual variation"
    assert session["payload_references"][-1]["role"] == "surfaced_material"


def test_emergence_rejects_unbounded_or_unserializable_inline_payloads():
    with pytest.raises(ValueError, match="externally and reference it"):
        capture_emergence("x" * (MAX_CONTENT_BYTES + 1), modality="text")
    with pytest.raises(ValueError, match="JSON-compatible"):
        capture_emergence({"opaque": object()}, modality="mixed")
