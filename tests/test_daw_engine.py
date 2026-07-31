import json
from pathlib import Path

import numpy as np
import pytest

import daw_engine as daw
from daw_engine import (
    PROJECT_SCHEMA_VERSION,
    DawProject,
    DawValidationError,
    InstrumentTrack,
    Step,
    VocalClip,
    export_project_wav,
    load_project,
    midi_note_to_frequency,
    normalize_audio,
    read_wav,
    render_instrument_track,
    render_project,
    save_project,
    synthesize_note,
    write_wav,
)


def _short_project(**changes):
    values = {
        "name": "Engine test",
        "sample_rate": 8_000,
        "bpm": 120,
        "beats_per_bar": 2,
        "steps_per_beat": 2,
        "bars": 1,
        "master_gain": 1.0,
        "seed": 17,
    }
    values.update(changes)
    return DawProject(**values)


def test_note_frequency_and_supported_waveforms_are_bounded():
    assert midi_note_to_frequency(69) == pytest.approx(440.0)

    for waveform in ("sine", "square", "triangle", "saw", "noise"):
        audio = synthesize_note(
            69,
            0.1,
            8_000,
            waveform=waveform,
            attack_seconds=0.005,
            release_seconds=0.01,
            seed=123,
        )
        assert audio.dtype == np.float32
        assert audio.shape == (800,)
        assert float(np.max(np.abs(audio))) <= 1.0
        assert audio[0] == pytest.approx(0.0)
        assert audio[-1] == pytest.approx(0.0)


def test_noise_synthesis_is_deterministic_for_a_seed():
    first = synthesize_note(60, 0.08, 8_000, waveform="noise", seed=9)
    second = synthesize_note(60, 0.08, 8_000, waveform="noise", seed=9)
    different = synthesize_note(60, 0.08, 8_000, waveform="noise", seed=10)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_step_timing_and_track_tail_follow_project_tempo():
    project = _short_project()
    track = InstrumentTrack(
        waveform="sine",
        attack_seconds=0,
        release_seconds=0,
        gain=1,
        steps=[Step(position=2, note=69, velocity=1, duration_steps=1)],
    )
    project.tracks.append(track)

    rendered = render_instrument_track(track, project)

    # At 120 BPM with two steps/beat, position 2 starts after half a second.
    assert rendered.shape == (8_000,)
    assert np.count_nonzero(rendered[:4_000]) == 0
    assert np.max(np.abs(rendered[4_000:6_000])) > 0.9
    assert np.count_nonzero(rendered[6_000:]) == 0


def test_project_mix_sums_overlapping_tracks_before_optional_normalization():
    shared_step = Step(position=0, note=69, velocity=0.2, duration_steps=1)
    project = _short_project(
        tracks=[
            InstrumentTrack(
                waveform="sine",
                gain=1,
                attack_seconds=0,
                release_seconds=0,
                steps=[shared_step],
            ),
            InstrumentTrack(
                waveform="sine",
                gain=1,
                attack_seconds=0,
                release_seconds=0,
                steps=[Step.from_dict(shared_step.to_dict())],
            ),
        ]
    )

    first = render_instrument_track(project.tracks[0], project, track_index=0)
    second = render_instrument_track(project.tracks[1], project, track_index=1)
    mixed = render_project(project, normalize=False)

    np.testing.assert_allclose(mixed, first + second, atol=1e-7)
    assert float(np.max(np.abs(mixed))) == pytest.approx(0.4, abs=0.002)


def test_vocal_clip_is_resampled_and_placed_at_beat_offset(tmp_path: Path):
    source_rate = 4_000
    source_audio = np.full(1_000, 0.25, dtype=np.float32)
    vocal_path = write_wav(tmp_path / "voice.wav", source_audio, source_rate)
    project = _short_project(
        bpm=60,
        beats_per_bar=1,
        steps_per_beat=4,
        vocal_clips=[VocalClip(path=str(vocal_path), offset_beats=0.5, gain=0.5)],
    )

    rendered = render_project(project, normalize=False)

    assert rendered.shape == (8_000,)
    assert np.count_nonzero(rendered[:4_000]) == 0
    assert rendered[4_100] == pytest.approx(0.125, abs=1e-4)
    assert rendered[5_999] == pytest.approx(0.125, abs=1e-4)
    assert np.count_nonzero(rendered[6_000:]) == 0


def test_vocal_clip_can_extend_project_render(tmp_path: Path):
    vocal_path = write_wav(
        tmp_path / "long_voice.wav",
        np.full(4_000, 0.1, dtype=np.float32),
        8_000,
    )
    project = _short_project(
        bpm=120,
        beats_per_bar=1,
        steps_per_beat=2,
        vocal_clips=[VocalClip(path=str(vocal_path), offset_beats=0.4)],
    )

    rendered = render_project(project, normalize=False)

    # Base duration is 0.5s; a 0.5s clip beginning at 0.2s reaches 0.7s.
    assert rendered.shape == (5_600,)
    assert rendered[-2] == pytest.approx(0.1, abs=1e-4)


def test_pcm_wav_read_write_and_project_export(tmp_path: Path):
    stereo = np.column_stack(
        [np.linspace(-0.5, 0.5, 100), np.linspace(0.5, -0.5, 100)]
    ).astype(np.float32)
    wav_path = write_wav(tmp_path / "nested" / "stereo.wav", stereo, 8_000)

    mono, sample_rate = read_wav(wav_path)

    assert sample_rate == 8_000
    assert mono.shape == (100,)
    assert np.max(np.abs(mono)) < 1e-4

    project = _short_project(
        tracks=[InstrumentTrack(steps=[Step(position=0, note=60)])]
    )
    export_path = export_project_wav(project, tmp_path / "render.wav")
    exported, exported_rate = read_wav(export_path)
    assert exported_rate == project.sample_rate
    assert exported.shape == (8_000,)
    assert np.max(np.abs(exported)) > 0


def test_project_json_round_trip_preserves_model(tmp_path: Path):
    project = _short_project(
        tracks=[
            InstrumentTrack(
                name="Soft saw",
                waveform="saw",
                gain=0.4,
                steps=[Step(position=1, note=62, velocity=0.7, duration_steps=1.5)],
            )
        ],
        vocal_clips=[
            VocalClip(path="recordings/verse.wav", offset_beats=1.25, gain=0.6)
        ],
    )
    path = save_project(project, tmp_path / "projects" / "song.ina-daw.json")

    loaded = load_project(path)

    assert loaded == project
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == PROJECT_SCHEMA_VERSION
    assert not path.with_name(path.name + ".tmp").exists()


def test_invalid_project_data_is_rejected_early():
    with pytest.raises(DawValidationError, match="unsupported waveform"):
        InstrumentTrack(waveform="laser")
    with pytest.raises(DawValidationError, match="outside"):
        _short_project(
            tracks=[InstrumentTrack(steps=[Step(position=4, note=60)])]
        )
    with pytest.raises(DawValidationError, match="schema version"):
        DawProject.from_dict({"schema_version": 999})


def test_render_boundary_revalidates_mutations_and_coerces_appended_mappings():
    project = _short_project()
    project.tracks.append(
        {
            "waveform": "sine",
            "attack_seconds": 0,
            "release_seconds": 0,
            "steps": [{"position": 0, "note": 60}],
        }
    )

    rendered = render_project(project, normalize=False)

    assert np.max(np.abs(rendered)) > 0
    assert isinstance(project.tracks[0], InstrumentTrack)

    project.bpm = 0
    with pytest.raises(DawValidationError, match="BPM"):
        render_project(project)

    invalid_step_project = _short_project(
        tracks=[InstrumentTrack(steps=[Step(position=0)])]
    )
    invalid_step_project.tracks[0].steps[0].position = -1
    with pytest.raises(DawValidationError, match="step position"):
        render_project(invalid_step_project)


def test_render_budget_rejects_base_note_tail_and_vocal_offset_before_allocation():
    with pytest.raises(DawValidationError, match="sample count"):
        _short_project(bpm=1e-300)

    with pytest.raises(DawValidationError, match="render budget"):
        _short_project(
            tracks=[
                InstrumentTrack(
                    steps=[Step(position=0, duration_steps=20_000)]
                )
            ]
        )

    with pytest.raises(DawValidationError, match="render budget"):
        _short_project(
            bpm=60,
            vocal_clips=[VocalClip(path="voice.wav", offset_beats=10_000)],
        )


def test_non_finite_audio_and_mutated_project_values_are_rejected(tmp_path: Path):
    with pytest.raises(DawValidationError, match="finite samples"):
        normalize_audio([0.0, np.nan])
    with pytest.raises(DawValidationError, match="finite samples"):
        write_wav(tmp_path / "nan.wav", [np.inf], 8_000)

    project = _short_project()
    project.master_gain = np.nan
    destination = tmp_path / "invalid.json"
    with pytest.raises(DawValidationError, match="master gain"):
        save_project(project, destination)
    assert not destination.exists()


def test_relative_vocal_paths_use_explicit_project_directory(tmp_path: Path):
    project_dir = tmp_path / "portable_project"
    relative_voice = Path("vocal_assets") / "relative_voice.wav"
    write_wav(
        project_dir / relative_voice,
        np.full(80, 0.2, dtype=np.float32),
        8_000,
    )
    project_path = save_project(
        _short_project(vocal_clips=[VocalClip(path=str(relative_voice))]),
        project_dir / "song.ina-daw.json",
    )
    loaded = load_project(project_path)

    with pytest.raises(FileNotFoundError):
        render_project(loaded, normalize=False)

    rendered = render_project(
        loaded, normalize=False, base_path=project_path.parent
    )
    assert rendered[10] == pytest.approx(0.2, abs=1e-4)


def test_export_accepts_project_mapping(tmp_path: Path):
    project = _short_project(
        tracks=[InstrumentTrack(steps=[Step(position=0, note=60)])]
    )
    destination = export_project_wav(project.to_dict(), tmp_path / "mapping.wav")

    exported, sample_rate = read_wav(destination)

    assert sample_rate == project.sample_rate
    assert exported.shape == (8_000,)
    assert np.max(np.abs(exported)) > 0


def test_project_save_uses_unique_temp_and_cleans_failed_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "song.json"
    legacy_temp = destination.with_name(destination.name + ".tmp")
    legacy_temp.write_text("keep me", encoding="utf-8")

    save_project(_short_project(), destination)

    assert legacy_temp.read_text(encoding="utf-8") == "keep me"

    failed_destination = tmp_path / "failed.json"

    def fail_replace(source, target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(daw.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        save_project(_short_project(), failed_destination)
    assert not list(tmp_path.glob(f".{failed_destination.name}.*.tmp"))


def test_oversized_wav_header_is_rejected_before_reading_frames(monkeypatch):
    class OversizedWav:
        read_called = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def getcomptype(self):
            return "NONE"

        def getnchannels(self):
            return 1

        def getsampwidth(self):
            return 2

        def getframerate(self):
            return 8_000

        def getnframes(self):
            return daw.MAX_RENDER_SAMPLES + 1

        def readframes(self, frame_count):
            self.read_called = True
            raise AssertionError("oversized WAV must be rejected before reading")

    oversized = OversizedWav()
    monkeypatch.setattr(daw.wave, "open", lambda *args, **kwargs: oversized)

    with pytest.raises(DawValidationError, match="render budget"):
        read_wav("oversized.wav")
    assert oversized.read_called is False


def test_synthesis_above_shared_cap_is_rejected_before_waveform_allocation(monkeypatch):
    allocation_called = False

    def fail_waveform_allocation(*args, **kwargs):
        nonlocal allocation_called
        allocation_called = True
        raise AssertionError("waveform allocation must not run above the cap")

    monkeypatch.setattr(daw, "_synthesize_waveform", fail_waveform_allocation)
    duration = (daw.MAX_RENDER_SAMPLES + 1) / 8_000

    with pytest.raises(DawValidationError, match="render budget"):
        synthesize_note(60, duration, 8_000)

    assert allocation_called is False
    assert daw.MAX_RENDER_SAMPLES == 16_000_000
    assert "MAX_RENDER_SAMPLES" in daw.__all__


def test_oversized_project_json_is_rejected_before_open_or_parse(tmp_path, monkeypatch):
    project_path = tmp_path / "oversized.ina-daw.json"
    with project_path.open("wb") as handle:
        handle.seek(daw.MAX_PROJECT_JSON_BYTES)
        handle.write(b"x")

    def fail_open(*args, **kwargs):
        raise AssertionError("oversized project JSON must be rejected before reading")

    def fail_parse(*args, **kwargs):
        raise AssertionError("oversized project JSON must be rejected before parsing")

    monkeypatch.setattr(Path, "open", fail_open)
    monkeypatch.setattr(daw.json, "loads", fail_parse)

    with pytest.raises(DawValidationError, match="input budget"):
        load_project(project_path)
    assert "MAX_PROJECT_JSON_BYTES" in daw.__all__
