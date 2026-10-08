from benchmarks.benchmark_selected_expression_resilience import load_rendering_functions
db = load_rendering_functions('discord_bridge.py')
import pytest


def test_permission_denial_is_not_bypassed(monkeypatch):
    def deny(*args, **kwargs):
        raise PermissionError('not authorized')
    monkeypatch.setattr(db, 'generate_symbolic_reply_from_text', deny)
    with pytest.raises(PermissionError):
        db.encode_selected_text_expression('text', child='fixture', language_preference='auto', max_symbols=16)


def test_selected_english_survives_symbolic_encoder_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('fixture')
    monkeypatch.setattr(db, 'generate_symbolic_reply_from_text', fail)
    text = "I meant 'meanings', not certainty."
    rendered, metadata = db.encode_selected_text_expression(
        text, child='fixture', language_preference='auto', max_symbols=16)
    assert rendered == text
    assert metadata['selected_wording_preserved']
    assert not metadata['native_translation_complete']


def test_selected_text_survives_gloss_failure_without_fake_translation(monkeypatch):
    monkeypatch.setattr(db, 'generate_symbolic_reply_from_text', lambda *a, **k: {'symbols': ['s']})
    def fail(*args, **kwargs):
        raise ValueError('fixture')
    monkeypatch.setattr(db, 'build_dual_symbolic_message', fail)
    rendered, metadata = db.encode_selected_text_expression(
        'Perhaps.', child='fixture', language_preference='native', max_symbols=16)
    assert rendered == 'Perhaps.'
    assert metadata['requested_language_mode'] == 'native'
    assert metadata['effective_language_mode'] == 'selected_text_rendering_unavailable'


def test_empty_selection_does_not_invent_a_reply():
    rendered, metadata = db.encode_selected_text_expression(
        '', child='fixture', language_preference='english', max_symbols=16)
    assert rendered == ''
    assert metadata['effective_language_mode'] == 'none'
