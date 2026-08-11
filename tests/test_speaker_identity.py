from types import SimpleNamespace

import backend_discord
import lm_studio_adapter as lms


def test_other_discord_bot_is_external_speaker_not_ina():
    message = SimpleNamespace(
        author=SimpleNamespace(id=22, display_name="Proxy Speaker", bot=True)
    )
    external = backend_discord.make_sender_info_from_discord(message, self_user_id=11)
    ina = backend_discord.make_sender_info_from_discord(
        SimpleNamespace(author=SimpleNamespace(id=11, display_name="Ina", bot=True)),
        self_user_id=11,
    )
    assert external.display_name == "Proxy Speaker"
    assert external.is_self is False
    assert ina.is_self is True


def test_legacy_operator_narrative_uses_recorded_speaker_without_rewrite():
    record = {
        "narrative": "Conversation with the operator: Rowan said 'hello'",
        "speaker": "Rowan",
        "utterance": "hello",
    }
    assert lms._speaker_aware_narrative(record) == "Conversation: Rowan said 'hello'"
    assert record["narrative"].startswith("Conversation with the operator:")


def test_unknown_word_question_names_actual_speaker(monkeypatch, tmp_path):
    questions = []
    adapter = lms.LMStudioAdapter("TestChild", base_path=tmp_path)
    monkeypatch.setattr(adapter, "_load_known_words", lambda: {})
    monkeypatch.setattr(lms, "seed_self_question", questions.append)
    adapter._compose_reply("zabble", speaker="Rowan")
    assert questions == [
        "What experience grounds the word 'zabble' mentioned by Rowan?"
    ]
