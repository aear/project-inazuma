from pathlib import Path

import raw_file_manager as rfm
from module_benchmarks import benchmark_module
from self_read_language import (
    annotate_music_language_evidence,
    media_seek_fraction,
    video_language_kind,
)


class _Encoder:
    def encode_audio_fragment(self, _fragment):
        return {"importance": 0.4}

    def encode_video_fragment(self, _fragment):
        return {"importance": 0.5}


def test_vocal_stems_and_lyrics_share_language_alignment_without_claiming_transcription():
    vocal = {
        "modality": "audio", "tags": ["self_read", "audio", "music_stem"],
        "source_context": {"stem_label": "01 Lead Vocals"},
    }
    lyrics = {"modality": "text", "tags": ["self_read"], "source_context": {}}
    instrumental = {
        "modality": "audio", "tags": ["self_read", "audio", "music_stem"],
        "source_context": {"stem_label": "02 Guitar"},
    }
    annotate_music_language_evidence(vocal, "Song/01 Lead Vocals.wav")
    annotate_music_language_evidence(lyrics, "Song/lyrics.txt")
    annotate_music_language_evidence(instrumental, "Song/02 Guitar.wav")

    assert vocal["language_learning"]["role"] == "isolated_vocal_stem"
    assert vocal["language_learning"]["supports_cadence"] is True
    assert instrumental["language_learning"]["role"] == "instrumental_contrast"
    assert instrumental["language_learning"]["supports_pronunciation"] is False
    assert "song" in vocal["language_learning"]["alignment_keys"]
    assert "song" in lyrics["language_learning"]["alignment_keys"]
    assert lyrics["language_learning"]["token_alignment_claimed"] is False


def test_video_essay_boundary_excludes_cadence_and_links_matching_script():
    essay = {"modality": "video", "duration": 600.001, "tags": ["self_read", "video"], "source_context": {}}
    script = {"modality": "text", "tags": ["self_read"], "source_context": {}}
    annotate_music_language_evidence(essay, "Essays/Memory Garden.mp4")
    annotate_music_language_evidence(script, "Essays/Memory Garden transcript.srt")

    assert video_language_kind(600) == "channel_video"
    assert video_language_kind(600.001) == "video_essay"
    assert essay["language_learning"]["supports_cadence"] is False
    assert essay["language_learning"]["supports_written_alignment"] is True
    assert set(essay["language_learning"]["alignment_keys"]) & set(script["language_learning"]["alignment_keys"])


def test_seekable_video_uses_bounded_audio_excerpt_and_records_watching(monkeypatch, tmp_path):
    calls = []
    path = tmp_path / "essay.mp4"
    path.write_bytes(b"video")
    monkeypatch.setattr(rfm, "cv2", None)
    monkeypatch.setattr(rfm, "_VIDEO_IMPORT_ERROR", None)
    monkeypatch.setattr(rfm, "_extract_audio_metadata", lambda _path: {"technical": {"duration_seconds": 900}})

    def analyze(_path, _transformer, **kwargs):
        calls.append(kwargs)
        return {"duration": 30, "embedding": [0.1], "symbols": ["snd_a"], "proto_words": ["pair"]}

    monkeypatch.setattr(rfm, "analyze_audio_clip", analyze)
    fragment = rfm.fragment_video(path, _Encoder(), seek_seconds=120)[0]
    rfm.annotate_fragment_source(fragment, "music", "Essays/essay.mp4", tmp_path)

    assert calls[0]["max_seconds"] == 30
    assert calls[0]["start_seconds"] == 105
    assert fragment["media_experience"]["mode"] == "watching"
    assert fragment["media_experience"]["observed_spans"] == [{"start_seconds": 105.0, "end_seconds": 135.0}]
    assert fragment["media_experience"]["controls"]["can_skip"] is True
    assert fragment["language_learning"]["role"] == "video_essay"


def test_seekable_audio_revisits_an_alternate_bounded_listening_span(monkeypatch, tmp_path):
    calls = []
    path = tmp_path / "song.mp3"
    path.write_bytes(b"audio")
    monkeypatch.setattr(rfm, "_extract_audio_metadata", lambda _path: {"technical": {"duration_seconds": 300}})

    def analyze(_path, _transformer, **kwargs):
        calls.append(kwargs)
        return {"duration": 60, "embedding": [0.2], "symbols": ["snd_a"], "proto_words": ["pair"], "clarity": 0.5}

    monkeypatch.setattr(rfm, "analyze_audio_clip", analyze)
    fraction = media_seek_fraction("revisit", {"read_count": 2})
    fragment = rfm.fragment_audio(path, _Encoder(), seek_fraction=fraction)[0]

    assert fraction == 0.75
    assert calls[0]["max_seconds"] == 60
    assert calls[0]["start_seconds"] == 195
    assert fragment["duration"] == 300
    assert fragment["media_experience"]["mode"] == "listening"
    assert fragment["media_experience"]["revisit_policy"]["allowed"] is True
    assert media_seek_fraction("revisit", {"read_count": 1}) != fraction


def test_self_read_language_benchmark_scores_each_policy_component():
    v1, v2 = benchmark_module("self_read_language")
    assert v2.accuracy > v1.accuracy
    assert set(v2.component_scores) == {
        "cadence", "decoder_bound", "discovery", "media_agency", "output_bridge",
        "resource_bound", "revisit", "spoken_written", "sung_language",
        "video_policy", "visual_practice", "watching",
    }
    assert all(row["correct"] == row["total"] for row in v2.component_scores.values())
