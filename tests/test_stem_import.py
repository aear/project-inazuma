import io
import json
from pathlib import Path
import wave
import zipfile

import pytest
import stem_import as stem_module

from stem_import import (
    StemImportCancelled,
    StemImportError,
    import_stem_zip,
    import_wav_stems,
    inspect_pcm_wav,
)


def _wav_bytes(*, frames=120, channels=1, rate=8_000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(b"\x00\x00" * frames * channels)
    return buffer.getvalue()


def _write_wav(path: Path, **kwargs) -> Path:
    path.write_bytes(_wav_bytes(**kwargs))
    return path


def test_zip_import_copies_aligned_stems_and_retains_language_context(tmp_path):
    source = tmp_path / "Godhunter's Lullaby Stems.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("stems/0 Lead Vocals.wav", _wav_bytes())
        archive.writestr("stems/1 Drums.wav", _wav_bytes(channels=2))
        archive.writestr(
            "lyrics and style.txt",
            "Lyrics: follow the thread.\nStyle: slow battle hymn.",
        )
        archive.writestr("README.txt", "Export notes only.")
    source_bytes = source.read_bytes()

    result = import_stem_zip(source, tmp_path / "studio", collection_name="Godhunter's Lullaby")

    assert result.collection_name == "Godhunter's Lullaby"
    assert [stem.role for stem in result.stems] == ["vocals", "drums"]
    assert all(stem.path.is_file() for stem in result.stems)
    assert all(stem.path.parent == result.collection_dir for stem in result.stems)
    assert len(result.companions) == 2
    companions = {companion.name: companion for companion in result.companions}
    assert companions["lyrics and style"].path.read_text() == (
        "Lyrics: follow the thread.\nStyle: slow battle hymn."
    )
    assert companions["lyrics and style"].kind == "lyrics_style_context"
    assert companions["README"].kind == "music_context"
    assert source.read_bytes() == source_bytes

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["kind"] == "ina_music_stem_collection"
    assert manifest["source"] == {"kind": "zip", "name": source.name}
    manifest_kinds = {item["name"]: item["kind"] for item in manifest["companions"]}
    assert manifest_kinds == {
        "lyrics and style": "lyrics_style_context",
        "README": "music_context",
    }
    assert str(tmp_path) not in result.manifest_path.read_text()


def test_selected_wav_import_is_atomic_and_records_pcm_metadata(tmp_path):
    vocals = _write_wav(tmp_path / "Vocals.wav", rate=16_000)
    guitar = _write_wav(tmp_path / "Guitar.wav", channels=2)

    result = import_wav_stems(
        [vocals, guitar],
        tmp_path / "studio",
        collection_name="Practice",
    )

    assert [stem.role for stem in result.stems] == ["vocals", "guitar"]
    assert result.stems[0].wav.sample_rate == 16_000
    assert result.stems[1].wav.channels == 2
    assert not any(path.name.startswith(".stem_import_") for path in (tmp_path / "studio").iterdir())


@pytest.mark.parametrize(
    "member_name",
    ["../escape.wav", "/absolute.wav", r"..\escape.wav", "C:/drive.wav"],
)
def test_zip_import_rejects_unsafe_member_paths_without_partial_collection(
    tmp_path, member_name
):
    source = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(member_name, _wav_bytes())
    destination = tmp_path / "studio"

    with pytest.raises(StemImportError, match="relative|unsafe"):
        import_stem_zip(source, destination)

    assert not (tmp_path / "escape.wav").exists()
    assert destination.exists()
    assert list(destination.iterdir()) == []


def test_zip_import_honors_project_remaining_stem_slots(tmp_path):
    source = tmp_path / "two.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("Vocals.wav", _wav_bytes())
        archive.writestr("Drums.wav", _wav_bytes())

    with pytest.raises(StemImportError, match="at most 1"):
        import_stem_zip(source, tmp_path / "studio", maximum_stems=1)

    assert list((tmp_path / "studio").iterdir()) == []


def test_zip_import_rejects_malformed_wav_and_cleans_stage(tmp_path):
    source = tmp_path / "bad.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("Vocals.wav", b"not a wave")
    destination = tmp_path / "studio"

    with pytest.raises(StemImportError, match="PCM WAV"):
        import_stem_zip(source, destination)

    assert list(destination.iterdir()) == []


def test_cancelled_import_does_not_publish_a_collection(tmp_path):
    source = _write_wav(tmp_path / "Vocals.wav")
    destination = tmp_path / "studio"

    with pytest.raises(StemImportCancelled):
        import_wav_stems(
            [source],
            destination,
            collection_name="Cancelled",
            cancelled=lambda: True,
        )

    assert not destination.exists()


def test_pcm_inspection_rejects_truncated_frames(tmp_path):
    path = _write_wav(tmp_path / "truncated.wav", frames=50)
    data = path.read_bytes()
    path.write_bytes(data[:-8])

    with pytest.raises(StemImportError, match="incomplete|read PCM"):
        inspect_pcm_wav(path)


def test_selected_import_rejects_invalid_remaining_slot_type(tmp_path):
    source = _write_wav(tmp_path / "Vocals.wav")

    with pytest.raises(StemImportError, match="integer"):
        import_wav_stems(
            [source],
            tmp_path / "studio",
            collection_name="Bad limit",
            maximum_stems=True,
        )


@pytest.mark.parametrize(
    "maximum_stems",
    [0, -1, stem_module.MAX_AUDIO_STEMS + 1],
)
def test_importers_reject_stem_limits_outside_engine_range(
    tmp_path,
    maximum_stems,
):
    wav_source = _write_wav(tmp_path / "Vocals.wav")
    zip_source = tmp_path / "stems.zip"
    with zipfile.ZipFile(zip_source, "w") as archive:
        archive.writestr("Vocals.wav", _wav_bytes())

    with pytest.raises(StemImportError, match="between 1"):
        import_wav_stems(
            [wav_source],
            tmp_path / "wav_destination",
            collection_name="Bad limit",
            maximum_stems=maximum_stems,
        )
    with pytest.raises(StemImportError, match="between 1"):
        import_stem_zip(
            zip_source,
            tmp_path / "zip_destination",
            maximum_stems=maximum_stems,
        )

    assert not (tmp_path / "wav_destination").exists()
    assert not (tmp_path / "zip_destination").exists()


def test_collection_budget_matches_self_read_and_fits_godhunter_bundle():
    godhunter_expanded_bytes = 261_704_492

    assert stem_module.MAX_STEM_TOTAL_BYTES == 256 * 1024 * 1024
    assert godhunter_expanded_bytes <= stem_module.MAX_STEM_TOTAL_BYTES


def test_selected_import_rejects_duplicate_canonical_source(tmp_path):
    source = _write_wav(tmp_path / "Vocals.wav")
    alias = tmp_path / "Vocals alias.wav"
    try:
        alias.symlink_to(source)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(StemImportError, match="selected more than once"):
        import_wav_stems(
            [source, alias],
            tmp_path / "studio",
            collection_name="Duplicate",
        )

    assert not (tmp_path / "studio").exists()


def test_missing_selected_sources_raise_import_errors(tmp_path):
    with pytest.raises(StemImportError, match="Could not read stem source"):
        import_wav_stems(
            [tmp_path / "missing.wav"],
            tmp_path / "wav_destination",
            collection_name="Missing",
        )
    with pytest.raises(StemImportError, match="Could not read stem ZIP"):
        import_stem_zip(
            tmp_path / "missing.zip",
            tmp_path / "zip_destination",
        )


def test_selected_source_disappearance_during_open_is_wrapped_and_cleaned(
    monkeypatch,
    tmp_path,
):
    source = _write_wav(tmp_path / "Vocals.wav")
    resolved_source = source.resolve()
    destination = tmp_path / "studio"
    real_open = Path.open

    def disappearing_open(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == resolved_source and mode == "rb":
            raise FileNotFoundError("source disappeared")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", disappearing_open)

    with pytest.raises(StemImportError, match="Could not copy stem source"):
        import_wav_stems(
            [source],
            destination,
            collection_name="Disappeared",
        )

    assert destination.exists()
    assert list(destination.iterdir()) == []


def test_zip_source_disappearance_during_open_is_wrapped_and_cleaned(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "stems.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("Vocals.wav", _wav_bytes())
    destination = tmp_path / "studio"

    def disappearing_zip(*_args, **_kwargs):
        raise FileNotFoundError("ZIP disappeared")

    monkeypatch.setattr(stem_module.zipfile, "ZipFile", disappearing_zip)

    with pytest.raises(StemImportError, match="Could not read stem ZIP"):
        import_stem_zip(source, destination)

    assert destination.exists()
    assert list(destination.iterdir()) == []


def test_cleanup_failure_does_not_mask_original_import_error(monkeypatch, tmp_path):
    source = _write_wav(tmp_path / "Vocals.wav")
    destination = tmp_path / "studio"
    real_rmtree = stem_module.shutil.rmtree

    def fail_copy(*_args, **_kwargs):
        raise StemImportError("original copy failure")

    def remove_then_fail(path):
        real_rmtree(path)
        raise OSError("cleanup failure")

    monkeypatch.setattr(stem_module, "_copy_stream", fail_copy)
    monkeypatch.setattr(stem_module.shutil, "rmtree", remove_then_fail)

    with pytest.raises(StemImportError, match="original copy failure"):
        import_wav_stems(
            [source],
            destination,
            collection_name="Cleanup",
        )

    assert destination.exists()
    assert list(destination.iterdir()) == []


def test_containment_failure_cleans_stage_created_before_assignment(
    monkeypatch,
    tmp_path,
):
    source = _write_wav(tmp_path / "Vocals.wav")
    destination = tmp_path / "studio"
    removed = []
    real_rmtree = stem_module.shutil.rmtree

    def track_removal(path):
        removed.append(Path(path))
        real_rmtree(path)

    monkeypatch.setattr(stem_module, "_is_within", lambda *_args: False)
    monkeypatch.setattr(stem_module.shutil, "rmtree", track_removal)

    with pytest.raises(StemImportError, match="staging folder escaped"):
        import_wav_stems(
            [source],
            destination,
            collection_name="Containment",
        )

    assert len(removed) == 1
    assert removed[0].name.startswith(".stem_import_")
    assert destination.exists()
    assert list(destination.iterdir()) == []
