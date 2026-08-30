import hashlib
import json

from public_music_library import MANIFEST_NAME, SCHEMA, admitted_track, validate_track_record
from raw_file_manager import annotate_fragment_source


def _record(data):
    return {
        "license_id": "CC0-1.0",
        "source_url": "https://example.test/track",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "creator": "Example Artist",
        "recording_rights": "CC0 recording",
        "composition_rights": "CC0 composition",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def test_public_music_admission_benchmark_v1_folder_trust_vs_v2_record_gate(tmp_path):
    audio = b"bounded fixture audio"
    track = tmp_path / "artist" / "track.wav"
    track.parent.mkdir()
    track.write_bytes(audio)
    manifest = {"schema": SCHEMA, "tracks": {"artist/track.wav": _record(audio)}}
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    decision = admitted_track(tmp_path, "artist/track.wav")

    assert decision["admitted"] is True
    assert decision["learning_scope"]["self_voice_identity"] is False


def test_public_music_fails_closed_without_rights_or_matching_hash(tmp_path):
    track = tmp_path / "track.wav"
    track.write_bytes(b"audio")
    missing = _record(b"audio")
    missing["recording_rights"] = ""
    assert validate_track_record(missing, track)["reason"] == "missing_provenance"
    wrong = _record(b"different")
    assert validate_track_record(wrong, track)["reason"] == "hash_mismatch"


def test_public_vocals_never_become_self_voice_references(tmp_path):
    fragment = {"modality": "audio", "tags": [], "source_context": {}}
    annotate_fragment_source(fragment, "public_music", "artist/vocals.wav", tmp_path)

    assert "self_voice" not in fragment["tags"]
    assert fragment["source_context"]["voice_learning_policy"]["learn_words"] is True
    assert fragment["source_context"]["voice_learning_policy"]["self_voice_identity"] is False
    assert fragment["language_learning"]["supports_voice_identity"] is False
    assert fragment["language_learning"]["voice_identity_authority"] == "external_reference_only"
    assert fragment["learning_blocked"] is True
