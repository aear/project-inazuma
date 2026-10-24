import io
import subprocess
import wave

import pytest

from language_intelligence import ConstructionLearner, DiscourseEntityMemory, analyze_utterance
from ina_desktop.files import VirtualDrive, VirtualFileSystem
from speech_bridge import render_private_speech


def test_language_transfer_and_learning_assessment():
    from benchmarks.benchmark_language_speech import language_v1, language_v2
    old, new = language_v1(), language_v2()
    assert new['correct'] > old['correct']
    assert new['correct'] == new['total']


def test_learning_bounds_and_replay_window(tmp_path):
    learner = ConstructionLearner(tmp_path/'patterns.json', max_patterns=8)
    clauses = [{'predicate': str(index), 'arguments': {}} for index in range(30)]
    assert len(learner.observe(iter(clauses), source_event='first')) == 8
    assert len(learner.counts) <= 8
    for index in range(520):
        learner.observe([], source_event=str(index))
    assert len(learner.source_events) == 512
    with pytest.raises(ValueError):
        learner.observe([], source_event='x'*161)


def test_authoritative_unresolved_route_never_uses_recent_person():
    from discourse_context import build_discourse_context
    memory = DiscourseEntityMemory(); memory.mention('Distractor')
    context = build_discourse_context('You found your key.', speaker='Mira', addressee=None)
    result = analyze_utterance('You found your key.', discourse=memory, context={'discourse': context})
    assert all(ref['resolved'] is None and not ref['candidates'] for ref in result['referents'])


def test_speech_contract_and_stdin(tmp_path, monkeypatch):
    import speech_bridge
    captured = []
    def run(args, **kwargs):
        captured.append((args, kwargs))
        with wave.open(args[-1], 'wb') as wav:
            wav.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
            wav.writeframes(b'\0\0'*2205)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(speech_bridge.shutil, 'which', lambda name: '/usr/bin/espeak-ng')
    monkeypatch.setattr(speech_bridge.subprocess, 'run', run)
    fs = VirtualFileSystem((VirtualDrive('ina_hdd', 'Fixture', tmp_path/'personal', True, 'personal'),))
    text = '--help <speak>literal chosen text</speak>'
    result = render_private_speech(fs, text=text)
    assert result['status'] == 'rendered' and result['duration_seconds'] == 0.1
    assert not result['played'] and not result['published'] and not result['native_voice']
    args, kwargs = captured[0]
    assert text not in args and kwargs['input'] == text.encode()
    assert kwargs['timeout'] == 12 and '-w' in args and not kwargs.get('shell')
    assert (tmp_path/'personal'/result['relative_path']).is_file()


def test_invalid_speech_never_starts_backend(tmp_path, monkeypatch):
    import speech_bridge
    def unexpected(*args, **kwargs):
        raise AssertionError('backend should not start')
    monkeypatch.setattr(speech_bridge.shutil, 'which', unexpected)
    for text, voice in (('', 'en-gb'), ('x'*513, 'en-gb'), ('\x00', 'en-gb'), ('hello', '../voice')):
        with pytest.raises(ValueError):
            render_private_speech(None, text=text, voice=voice)


def test_missing_and_timed_out_backend(tmp_path, monkeypatch):
    import speech_bridge
    monkeypatch.setattr(speech_bridge.shutil, 'which', lambda _: None)
    assert render_private_speech(None, text='Hello')['status'] == 'unavailable'
    monkeypatch.setattr(speech_bridge.shutil, 'which', lambda _: '/usr/bin/espeak-ng')
    def expired(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 12)
    monkeypatch.setattr(speech_bridge.subprocess, 'run', expired)
    assert render_private_speech(None, text='Hello')['reason'] == 'TimeoutExpired'


def test_truncated_backend_wav_is_rejected(monkeypatch):
    import speech_bridge
    monkeypatch.setattr(speech_bridge.shutil, 'which', lambda _: '/usr/bin/espeak-ng')
    def run(args, **kwargs):
        from pathlib import Path
        data = io.BytesIO()
        with wave.open(data, 'wb') as wav:
            wav.setparams((1, 2, 22050, 0, 'NONE', 'not compressed')); wav.writeframes(b'\0\0'*2205)
        Path(args[-1]).write_bytes(data.getvalue()[:-20])
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(speech_bridge.subprocess, 'run', run)
    with pytest.raises(ValueError):
        render_private_speech(None, text='Hello')


def test_language_context_passes_authoritative_participants():
    from language_context import build_language_context_snapshot
    result = build_language_context_snapshot({"source_text": "I remember your lantern.",
        "current_speaker": {"id": "mira", "name": "Mira"}, "learned_media_guidance": {},
        "language_state_signals": {"current_prediction": {}, "machine_semantics": {}, "emotion_snapshot": {}}}, child="Fixture", logic_reader=False)
    analysis = result['linguistic_analysis']
    table = result['referent_table']
    assert table['speaker']['id'] == 'mira'
    assert next(ref for ref in analysis['referents'] if ref['surface'] == 'i')['resolved'] == table['speaker']['id']
    assert next(ref for ref in analysis['referents'] if ref['surface'] == 'your')['resolved'] == table['addressee']['id']
