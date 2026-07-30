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


def test_fragment_audio_opus(monkeypatch, tmp_path):
    opus_path = tmp_path / "clip.opus"
    opus_path.write_bytes(b"fake opus data")

    analysis = {
        "summary": "Synthetic MP3 analysis",
        "clarity": 0.432187,
        "tags": ["audio", "digest", "synthetic"],
        "emotions": {"focus": 0.2},
    }

    monkeypatch.setattr(rfm, "analyze_audio_clip", lambda path, transformer: analysis)

    transformer = FractalTransformer()
    fragments = rfm.fragment_audio(opus_path, transformer)

    assert fragments, "Expected a fragment for MP3 input"
    frag = fragments[0]
    assert frag["modality"] == "audio"
    assert "self_read" in frag["tags"]
    assert "synthetic" in frag["tags"]
    assert frag["importance"] == pytest.approx(0.4322, rel=0, abs=1e-4)


def test_rapidcrest_music_is_external_signed_artist(tmp_path):
    fragment = {"tags": ["self_read"], "metadata": {"flags": []}}

    rfm.annotate_fragment_source(
        fragment,
        "music",
        "Rapidcrest: Da Drums Beat/master.wav",
        tmp_path,
    )

    assert "external_music" in fragment["tags"]
    assert "signed_artist" in fragment["tags"]
    assert "ina_music" not in fragment["tags"]
    assert "self_voice" not in fragment["tags"]
    assert fragment["provenance"] == "signed_artist_catalog"
    assert fragment["source_context"]["ownership_hint"] == "external_artist"
    assert fragment["source_context"]["external_artist_hint"] == "rapidcrest"


def test_ina_music_keeps_self_voice_ownership(tmp_path):
    fragment = {"tags": ["self_read"], "metadata": {"flags": []}}

    rfm.annotate_fragment_source(
        fragment,
        "music",
        "Ina Sings: Soft Orbit/master.wav",
        tmp_path,
    )

    assert "ina_music" in fragment["tags"]
    assert "self_voice" in fragment["tags"]
    assert "external_music" not in fragment["tags"]
    assert fragment["source_context"]["ownership_hint"] == "self_voice"
