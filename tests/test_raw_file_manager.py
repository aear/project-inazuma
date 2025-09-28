import wave
from pathlib import Path

import pytest


from transformers.fractal_multidimensional_transformers import FractalTransformer


import raw_file_manager as rfm


def _create_wav_file(path: Path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"\x00\x00" * 200)


def test_fragment_audio_wav(tmp_path):
    wav_path = tmp_path / "sample.wav"
    _create_wav_file(wav_path)

    transformer = FractalTransformer()
    fragments = rfm.fragment_audio(wav_path, transformer)

    assert fragments, "Expected a fragment for WAV input"
    frag = fragments[0]
    assert frag["modality"] == "audio"
    assert frag["source"] == str(wav_path)
    assert "importance" in frag
    assert "self_read" in frag["tags"]


def test_fragment_audio_mp3(monkeypatch, tmp_path):
    mp3_path = tmp_path / "clip.mp3"
    mp3_path.write_bytes(b"fake mp3 data")

    analysis = {
        "summary": "Synthetic MP3 analysis",
        "clarity": 0.432187,
        "tags": ["audio", "digest", "synthetic"],
        "emotions": {"focus": 0.2},
    }

    monkeypatch.setattr(rfm, "analyze_audio_clip", lambda path, transformer: analysis)

    transformer = FractalTransformer()
    fragments = rfm.fragment_audio(mp3_path, transformer)

    assert fragments, "Expected a fragment for MP3 input"
    frag = fragments[0]
    assert frag["modality"] == "audio"
    assert "self_read" in frag["tags"]
    assert "synthetic" in frag["tags"]
    assert frag["importance"] == pytest.approx(0.4322, rel=0, abs=1e-4)
