from language_processing import select_symbolic_message_text
from benchmarks.benchmark_selected_expression_resilience import load_rendering_functions


def test_native_only_does_not_require_english_gloss():
    message = {'native_text': '∘⊙', 'gloss_text': 'candidate', 'text': 'Native: ∘⊙\nHuman guess: candidate'}
    assert select_symbolic_message_text(message, 'native_only') == ('∘⊙', 'native')
    assert select_symbolic_message_text(message, 'english') == ('candidate', 'english')
    assert select_symbolic_message_text(message, 'mixed')[0] == message['text']


def test_discord_respects_assessed_abstention_and_native_choice(monkeypatch):
    db = load_rendering_functions('discord_bridge.py')
    monkeypatch.setattr(db, 'select_symbolic_message_text', select_symbolic_message_text, raising=False)
    monkeypatch.setattr(db, '_is_emotion_or_sound_symbol', lambda symbol: False, raising=False)
    monkeypatch.setattr(db, 'generate_symbolic_reply_from_text', lambda *a, **k: {'symbols': ['s']})
    for mode in ('abstain', 'clarify', 'native_only'):
        monkeypatch.setattr(db, 'build_dual_symbolic_message', lambda *a, **k: {
            'native_text': '∘⊙', 'native_tokens': ['∘⊙'], 'gloss_tokens': ['candidate'],
            'native_sources': {'s':'fixture'}, 'gloss_sources': {'s':'fixture'},
            'mutual_intelligibility_assessment': {'schema':'ina.mutual_intelligibility_assessment/V1', 'mode':mode}})
        text, meta = db.encode_selected_text_expression('candidate', child='fixture',
            language_preference='auto', max_symbols=8)
        assert text == ('∘⊙' if mode == 'native_only' else '')
        assert meta['language_choice_preserved']


def test_rendering_failure_does_not_force_english(monkeypatch):
    db = load_rendering_functions('discord_bridge.py')
    def fail(*a, **k):
        raise RuntimeError('fixture')
    monkeypatch.setattr(db, 'generate_symbolic_reply_from_text', fail)
    text, meta = db.encode_selected_text_expression('candidate', child='fixture',
        language_preference='native_only', max_symbols=8)
    assert text == '' and meta['english_fallback_declined']
