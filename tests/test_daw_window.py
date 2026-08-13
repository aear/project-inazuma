import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tkinter")

import daw_window as daw_module
from daw_engine import AudioStem, VocalClip
from daw_window import (
    BoundedExecutor,
    DawWindow,
    MAX_RENDER_SAMPLES,
    MicrophoneRecorder,
    StudioInstanceLock,
    TransportGate,
    _bool_value,
    _strict_int,
    configured_device_index,
    STEP_COUNT,
    create_default_project,
    daw_control_api_payload,
    path_is_within,
    render_local_symbolic_vocal,
    safe_filename_stem,
    studio_paths,
    validate_child_identifier,
)


def test_studio_paths_and_names_stay_bounded(tmp_path):
    paths = studio_paths("Ina", base_path=tmp_path)

    assert paths.projects == tmp_path / "Ina" / "memory" / "music_studio" / "projects"
    assert paths.stems == tmp_path / "Ina" / "memory" / "music_studio" / "stems"
    assert path_is_within(paths.recordings / "take.wav", paths.root)
    assert not path_is_within(tmp_path / "outside.wav", paths.root)
    assert safe_filename_stem("  Soft orbit / take #1  ") == "Soft_orbit_take_1"

def test_studio_paths_reject_symlinked_managed_stem_folder(tmp_path):
    paths = studio_paths("Ina", base_path=tmp_path)
    paths.root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        paths.stems.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link"):
        paths.ensure()

    assert list(outside.iterdir()) == []




def test_default_project_is_a_multi_track_sixteen_step_loop():
    project = create_default_project()

    assert project.total_steps == STEP_COUNT
    assert len(project.tracks) >= 3
    assert all(step.position < STEP_COUNT for track in project.tracks for step in track.steps)


class _FakeInputStream:
    def __init__(self, *, callback, **_kwargs):
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True
        self.callback(
            np.array([[0.1], [0.2], [-0.1]], dtype=np.float32),
            3,
            None,
            None,
        )

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _FakeSoundDevice:
    def __init__(self):
        self.stream = None

    def InputStream(self, **kwargs):
        self.stream = _FakeInputStream(**kwargs)
        return self.stream


def test_microphone_recorder_collects_frames_and_closes_stream():
    fake_sd = _FakeSoundDevice()
    recorder = MicrophoneRecorder(sample_rate=8_000, sounddevice_module=fake_sd)

    recorder.start()
    audio = recorder.stop()

    np.testing.assert_allclose(audio, np.array([0.1, 0.2, -0.1], dtype=np.float32))
    assert fake_sd.stream.started is True
    assert fake_sd.stream.stopped is True
    assert fake_sd.stream.closed is True
    assert recorder.recording is False


def test_microphone_recorder_reports_missing_optional_device():
    recorder = MicrophoneRecorder(sounddevice_module=None)

    with pytest.raises(RuntimeError, match="sounddevice"):
        recorder.start()


def test_local_symbolic_vocal_requests_file_only_render(monkeypatch, tmp_path):
    calls = []
    fake_module = types.ModuleType("language_processing")

    def fake_generate(text, **kwargs):
        calls.append((text, kwargs))
        kwargs["record_path"].write_bytes(b"local-symbol-placeholder")
        return {"symbols": ["sym_soft"], "unknown": []}

    fake_module.generate_symbolic_reply_from_text = fake_generate
    monkeypatch.setitem(sys.modules, "language_processing", fake_module)
    destination = tmp_path / "voice.wav"

    payload = render_local_symbolic_vocal(
        "soft orbit",
        child="Ina",
        output_path=destination,
    )

    assert payload["symbols"] == ["sym_soft"]
    assert destination.exists()
    assert calls[0][0] == "soft orbit"
    assert calls[0][1]["playback"] is False
    assert calls[0][1]["record_path"] == destination
    assert calls[0][1]["record_format"] == "wav"


def test_microphone_recorder_caps_capture_memory_and_duration():
    fake_sd = _FakeSoundDevice()
    recorder = MicrophoneRecorder(
        sample_rate=1_000,
        sounddevice_module=fake_sd,
        max_seconds=0.002,
    )

    recorder.start()
    audio = recorder.stop()

    np.testing.assert_allclose(audio, np.array([0.1, 0.2], dtype=np.float32))
    assert recorder.limit_reached is True


def test_command_value_parsers_reject_ambiguous_or_out_of_range_values():
    assert _bool_value("false", "enabled") is False
    assert _bool_value("on", "enabled") is True
    assert _strict_int("15", "step", 0, 15) == 15

    with pytest.raises(ValueError, match="boolean"):
        _bool_value("perhaps", "enabled")
    with pytest.raises(ValueError, match="at most 15"):
        _strict_int(16, "step", 0, 15)
    with pytest.raises(ValueError, match="integer"):
        _strict_int(1.5, "step", 0, 15)


def test_configured_audio_device_uses_first_valid_alias():
    config = {"output_headset_index": "bad", "ouput_headset_index": "11"}

    assert configured_device_index(config, "output_headset_index", "ouput_headset_index") == 11
    assert configured_device_index({"mic_headset_index": -1}, "mic_headset_index") is None


def test_audio_launch_status_is_concise_while_full_warning_stays_in_state():
    window = object.__new__(DawWindow)
    window.output_device_resolution = SimpleNamespace(
        warning="backend detail " * 100,
        available=True,
        source="system_default",
    )
    window.input_device_resolution = SimpleNamespace(
        warning=None,
        available=True,
        source="system_default",
    )

    status = DawWindow._audio_startup_status(window)

    assert status == (
        "Configured output unavailable; using the system default. "
        "Config unchanged."
    )
    assert len(status) < 100


def test_audio_header_label_bounds_override_and_device_name_text():
    resolution = SimpleNamespace(
        available=True,
        device="override-" + ("x" * 100),
        name="device-" + ("y" * 100),
    )

    label = DawWindow._audio_resolution_label(resolution)

    assert label.startswith("Output: override-")
    assert len(label) < 90
    assert label.count("...") == 2

@pytest.mark.parametrize("child", ["", ".", "..", "../escape", "Ina/escape", r"Ina\\escape", "/tmp/escape"])
def test_studio_paths_reject_unsafe_child_identifiers(tmp_path, child):
    with pytest.raises(ValueError, match="child identifier"):
        studio_paths(child, base_path=tmp_path)


def test_studio_paths_reject_existing_symlink_escape(tmp_path):
    base = tmp_path / "AI_Children"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    try:
        (base / "Ina").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="stay inside"):
        studio_paths("Ina", base_path=base)


def test_validate_child_identifier_accepts_normal_child_names():
    assert validate_child_identifier(" Inazuma_Yagami ") == "Inazuma_Yagami"


def test_child_scoped_studio_lock_excludes_duplicate_and_releases(tmp_path):
    path = tmp_path / "Ina" / "memory" / "music_studio" / "daw_window.lock"
    first = StudioInstanceLock(path)
    second = StudioInstanceLock(path)
    try:
        assert first.acquire() is True
        assert second.acquire() is False
        first.release()
        assert second.acquire() is True
    finally:
        first.release()
        second.release()


def test_transport_gate_skips_stale_work_and_invalidates_current_work():
    gate = TransportGate()
    stale_generation = gate.advance()
    current_generation = gate.advance()
    calls = []

    ran, value = gate.run_if_current(stale_generation, lambda: calls.append("stale"))
    assert (ran, value) == (False, None)
    ran, value = gate.run_if_current(current_generation, lambda: calls.append("current") or 7)
    assert (ran, value) == (True, 7)
    gate.invalidate()
    ran, value = gate.run_if_current(current_generation, lambda: calls.append("late"))

    assert (ran, value) == (False, None)
    assert calls == ["current"]


def test_bounded_executor_rejects_work_beyond_running_and_queue_cap():
    executor = BoundedExecutor(max_workers=1, max_jobs=2)
    started = threading.Event()
    release = threading.Event()

    def blocking_job():
        started.set()
        assert release.wait(2.0)
        return "first"

    try:
        first = executor.submit(blocking_job)
        assert first is not None
        assert started.wait(2.0)
        second = executor.submit(lambda: "second")
        assert second is not None
        assert executor.submit(lambda: "overflow") is None
        release.set()
        assert first.result(timeout=2.0) == "first"
        assert second.result(timeout=2.0) == "second"
    finally:
        release.set()
        executor.shutdown()


def test_microphone_recorder_freezes_take_rate_and_obeys_engine_frame_cap():
    fake_sd = _FakeSoundDevice()
    recorder = MicrophoneRecorder(
        sample_rate=192_000,
        sounddevice_module=fake_sd,
        max_seconds=300.0,
    )

    assert recorder.max_frames == MAX_RENDER_SAMPLES
    recorder.start()
    recorder.sample_rate = 8_000

    assert recorder.recording_sample_rate == 192_000
    recorder.stop()
    assert recorder.recording_sample_rate == 192_000


class _BlockingConstructionSoundDevice:
    def __init__(self):
        self.constructor_entered = threading.Event()
        self.release_constructor = threading.Event()
        self.stream = None

    def InputStream(self, **kwargs):
        self.constructor_entered.set()
        assert self.release_constructor.wait(2.0)
        self.stream = _FakeInputStream(**kwargs)
        return self.stream


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Button:
    def __init__(self):
        self.state = None

    def configure(self, *, state):
        self.state = state


class _PlaybackDevice:
    def __init__(self):
        self.stop_calls = 0
        self.play_calls = []

    def stop(self):
        self.stop_calls += 1

    def play(self, audio, **options):
        self.play_calls.append((audio, options))


class _StereoOnlyPlaybackDevice(_PlaybackDevice):
    def __init__(self, *, max_output_channels=2, settings_error=None):
        super().__init__()
        self.max_output_channels = max_output_channels
        self.settings_error = settings_error
        self.query_calls = []
        self.settings_calls = []

    def query_devices(self, device, kind):
        self.query_calls.append((device, kind))
        return {"name": "Test output", "max_output_channels": self.max_output_channels}

    def check_output_settings(self, **settings):
        self.settings_calls.append(settings)
        if self.settings_error is not None:
            raise self.settings_error


def _bare_transport_window(monkeypatch):
    device = _PlaybackDevice()
    monkeypatch.setattr(daw_module, "_sounddevice", device)
    window = object.__new__(DawWindow)
    window._transport_gate = TransportGate()
    window._playing = True
    window.output_device = None
    window._set_status = lambda _text: None
    window._publish_workspace = lambda _event, **_extra: {}
    window._show_error = lambda _title, _error: None
    return window, device


def test_play_audio_expands_mono_for_stereo_output(monkeypatch):
    device = _StereoOnlyPlaybackDevice()
    monkeypatch.setattr(daw_module, "_sounddevice", device)
    window = object.__new__(DawWindow)
    window.output_device = 11
    mono = np.array([0.25, -0.5, 0.75], dtype=np.float32)

    DawWindow._play_audio(window, mono, 44_100)

    assert device.query_calls == [(11, "output")]
    assert device.settings_calls == [
        {"samplerate": 44_100, "channels": 2, "dtype": "float32", "device": 11}
    ]
    played, options = device.play_calls[0]
    assert played.shape == (3, 2)
    np.testing.assert_array_equal(played[:, 0], mono)
    np.testing.assert_array_equal(played[:, 1], mono)
    assert options == {"samplerate": 44_100, "device": 11}


def test_play_audio_uses_resolved_system_default_without_stale_index(monkeypatch):
    device = _StereoOnlyPlaybackDevice()
    monkeypatch.setattr(daw_module, "_sounddevice", device)
    window = object.__new__(DawWindow)
    window.output_device = None
    mono = np.array([0.25, -0.5], dtype=np.float32)

    DawWindow._play_audio(window, mono, 44_100)

    assert device.query_calls == [(None, "output")]
    assert device.settings_calls == [
        {"samplerate": 44_100, "channels": 2, "dtype": "float32"}
    ]
    played, options = device.play_calls[0]
    assert played.shape == (2, 2)
    assert options == {"samplerate": 44_100}


def test_play_audio_rejects_input_only_configured_device(monkeypatch):
    device = _StereoOnlyPlaybackDevice(max_output_channels=0)
    monkeypatch.setattr(daw_module, "_sounddevice", device)
    window = object.__new__(DawWindow)
    window.output_device = 11

    with pytest.raises(RuntimeError, match="Configured output device 11 has no output channels"):
        DawWindow._play_audio(window, np.zeros(4, dtype=np.float32), 44_100)

    assert device.play_calls == []


def test_play_audio_reports_unsupported_device_settings(monkeypatch):
    device = _StereoOnlyPlaybackDevice(settings_error=ValueError("unsupported rate"))
    monkeypatch.setattr(daw_module, "_sounddevice", device)
    window = object.__new__(DawWindow)
    window.output_device = 11

    with pytest.raises(RuntimeError, match="cannot play 2-channel audio at 44100 Hz"):
        DawWindow._play_audio(window, np.zeros(4, dtype=np.float32), 44_100)

    assert device.play_calls == []


def test_microphone_close_during_construction_prevents_stream_start():
    fake_sd = _BlockingConstructionSoundDevice()
    recorder = MicrophoneRecorder(sample_rate=8_000, sounddevice_module=fake_sd)
    errors = []

    def open_microphone():
        try:
            recorder.start()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=open_microphone)
    worker.start()
    assert fake_sd.constructor_entered.wait(2.0)

    recorder.close()
    fake_sd.release_constructor.set()
    worker.join(2.0)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert "cancelled" in str(errors[0]).lower()
    assert fake_sd.stream.started is False
    assert fake_sd.stream.closed is True
    assert recorder.recording is False
    with pytest.raises(RuntimeError, match="closed"):
        recorder.start()


def test_executor_keeps_child_lock_until_running_job_drains(tmp_path):
    lock_path = tmp_path / "Ina" / "memory" / "music_studio" / "daw_window.lock"
    owner = StudioInstanceLock(lock_path)
    contender = StudioInstanceLock(lock_path)
    executor = BoundedExecutor(max_workers=1, max_jobs=1)
    started = threading.Event()
    release_job = threading.Event()
    drained = threading.Event()

    def blocking_job():
        started.set()
        assert release_job.wait(2.0)

    def release_owner():
        owner.release()
        drained.set()

    try:
        assert owner.acquire() is True
        future = executor.submit(blocking_job)
        assert future is not None
        assert started.wait(2.0)

        executor.shutdown(on_drained=release_owner)

        assert contender.acquire() is False
        release_job.set()
        assert drained.wait(2.0)
        assert contender.acquire() is True
    finally:
        release_job.set()
        owner.release()
        contender.release()
        executor.shutdown()



def test_control_api_exposes_mix_only_stem_commands_and_manual_import():
    payload = daw_control_api_payload()
    actions = {item["action"] for item in payload["commands"]}

    assert payload["version"] == 2
    assert {"set_stem", "preview_stem"} <= actions
    assert "import_stem" not in actions
    assert payload["stem_import"]["mode"] == "manual_only"
    assert payload["limits"]["audio_stems_max"] >= 1


def _committed_stem_result(tmp_path):
    paths = studio_paths("Ina", base_path=tmp_path).ensure()
    collection = paths.stems / "Godhunter_Lullaby"
    collection.mkdir()
    stem_path = collection / "01_Lead_Vocals.wav"
    stem_path.write_bytes(b"RIFF")
    manifest_path = collection / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    result = SimpleNamespace(
        collection_dir=collection,
        collection_name="Godhunter's Lullaby",
        stems=(
            SimpleNamespace(
                path=stem_path,
                name="Lead Vocals",
                role="vocals",
            ),
        ),
        companions=(),
        manifest_path=manifest_path,
    )
    return paths, result


def test_committed_stem_collection_attaches_only_after_full_validation(tmp_path):
    paths, result = _committed_stem_result(tmp_path)
    window = object.__new__(DawWindow)
    window.paths = paths
    window.project = create_default_project()
    window._known_stem_collections = {}
    events = []
    window._refresh_stem_list = lambda: events.append("refresh")
    window._set_status = lambda text: events.append(text)
    window._publish_workspace = lambda event, **extra: events.append((event, extra))

    DawWindow._add_stem_import_result(window, result)

    assert len(window.project.audio_stems) == 1
    assert window.project.audio_stems[0].path == (
        "stems/Godhunter_Lullaby/01_Lead_Vocals.wav"
    )
    assert window._known_stem_collections["Godhunter's Lullaby"] == (
        "stems/Godhunter_Lullaby/manifest.json"
    )
    assert events[-1][0] == "stems_imported"


def test_committed_stem_collection_remains_library_only_when_project_is_full(tmp_path):
    paths, result = _committed_stem_result(tmp_path)
    window = object.__new__(DawWindow)
    window.paths = paths
    window.project = create_default_project()
    window.project.audio_stems = [
        AudioStem(
            path=f"stems/existing/{index:02d}.wav",
            name=f"Existing {index}",
            collection="Existing",
        )
        for index in range(daw_module.MAX_AUDIO_STEMS)
    ]
    window.project.validate()
    window._known_stem_collections = {}
    before = list(window.project.audio_stems)

    with pytest.raises(daw_module.StemCollectionAttachmentError, match="no longer fits"):
        DawWindow._add_stem_import_result(window, result)

    assert window.project.audio_stems == before
    assert "Godhunter's Lullaby" in window._known_stem_collections


def test_vocal_completion_cannot_overfill_shared_placed_wav_limit(tmp_path):
    paths = studio_paths("Ina", base_path=tmp_path).ensure()
    take = paths.recordings / "late_take.wav"
    take.write_bytes(b"RIFF")
    window = object.__new__(DawWindow)
    window.paths = paths
    window.project = create_default_project()
    window.project.audio_stems = [
        AudioStem(
            path=f"stems/existing/{index:02d}.wav",
            name=f"Existing {index}",
        )
        for index in range(daw_module.MAX_AUDIO_STEMS)
    ]
    vocal_count = daw_module.MAX_PLACED_WAVS - len(window.project.audio_stems)
    window.project.vocal_clips = [
        VocalClip(
            path=f"recordings/existing_{index:03d}.wav",
            name=f"Existing take {index}",
        )
        for index in range(vocal_count)
    ]
    window.project.validate()
    before = list(window.project.vocal_clips)
    window._refresh_vocal_list = lambda: None
    window._publish_workspace = lambda _event, **_extra: {}

    with pytest.raises(ValueError, match="placed WAV"):
        DawWindow._add_vocal_path(window, take, 0.0, "Late take")

    assert window.project.vocal_clips == before


def test_stem_import_waits_for_pending_project_load():
    window = object.__new__(DawWindow)
    window._stem_import_pending = False
    window._project_load_pending = True
    messages = []
    window._set_status = messages.append


def test_set_stem_updates_bounded_mix_fields_without_tk():
    window = object.__new__(DawWindow)
    window.project = create_default_project()
    window.project.audio_stems = [
        AudioStem(
            path="stems/Godhunter/01_Vocals.wav",
            name="Lead Vocals",
            collection="Godhunter's Lullaby",
        )
    ]
    events = []
    window._refresh_stem_list = lambda: events.append("refresh")
    window._publish_workspace = lambda event, **extra: events.append((event, extra))

    payload = DawWindow.set_stem(
        window,
        0,
        {"gain": 0.75, "offset_beats": 0.5, "role": "vocals", "solo": True},
    )

    assert payload["gain"] == pytest.approx(0.75)
    assert payload["offset_beats"] == pytest.approx(0.5)
    assert payload["role"] == "vocals"
    assert payload["solo"] is True
    assert payload["collection"] == "Godhunter's Lullaby"
    assert events[0] == "refresh"
    with pytest.raises(ValueError, match="stem gain"):
        DawWindow.set_stem(window, 0, {"gain": 4.1})


def test_selected_stem_cannot_start_after_stop(monkeypatch, tmp_path):
    window, device = _bare_transport_window(monkeypatch)
    window.project = SimpleNamespace(
        audio_stems=[
            SimpleNamespace(
                path="stems/Godhunter/01_Vocals.wav",
                gain=0.8,
                name="Lead Vocals",
            )
        ]
    )
    window.paths = SimpleNamespace(root=tmp_path, stems=tmp_path / "stems")
    pending = {}

    def capture(_label, operation, on_success=None, on_error=None):
        pending.update(operation=operation, success=on_success, error=on_error)
        return True

    window._background = capture
    monkeypatch.setattr(
        daw_module,
        "read_wav",
        lambda _path: pytest.fail("stale stem job should not read or play audio"),
    )

    assert DawWindow._preview_stem_index(window, 0) is True
    assert device.stop_calls == 1

    DawWindow.stop_playback(window)
    sample_count, cancelled = pending["operation"]()

    assert (sample_count, cancelled) == (0, True)
    assert device.play_calls == []

def test_note_preview_cannot_start_after_stop(monkeypatch):
    window, device = _bare_transport_window(monkeypatch)
    window.project = create_default_project()
    window.preview_waveform_var = _Value("sine")
    pending = {}

    def capture(_label, operation, on_success=None, on_error=None):
        pending.update(operation=operation, success=on_success, error=on_error)
        return True

    window._background = capture

    assert DawWindow.preview_note(window, 60) is True
    assert device.stop_calls == 1

    DawWindow.stop_playback(window)
    sample_count, cancelled = pending["operation"]()

    assert (sample_count, cancelled) == (0, True)
    assert device.play_calls == []


def test_selected_vocal_cannot_start_after_stop(monkeypatch, tmp_path):
    window, device = _bare_transport_window(monkeypatch)
    window.project = SimpleNamespace(
        vocal_clips=[SimpleNamespace(path="recordings/take.wav", gain=0.8, name="Take")]
    )
    window.paths = SimpleNamespace(root=tmp_path)
    window._selected_vocal_index = lambda: 0
    pending = {}

    def capture(_label, operation, on_success=None, on_error=None):
        pending.update(operation=operation, success=on_success, error=on_error)
        return True

    window._background = capture
    monkeypatch.setattr(
        daw_module,
        "read_wav",
        lambda _path: pytest.fail("stale vocal job should not read or play audio"),
    )

    assert DawWindow.play_selected_vocal(window) is True
    assert device.stop_calls == 1

    DawWindow.stop_playback(window)
    sample_count, cancelled = pending["operation"]()

    assert (sample_count, cancelled) == (0, True)
    assert device.play_calls == []


def test_project_play_failure_stops_replaced_audio(monkeypatch, tmp_path):
    window, device = _bare_transport_window(monkeypatch)
    window._project_snapshot = create_default_project
    window.loop_var = _Value(False)
    window._render_lock = threading.Lock()
    window.paths = SimpleNamespace(root=tmp_path)
    pending = {}

    def capture(_label, operation, on_success=None, on_error=None):
        pending.update(operation=operation, success=on_success, error=on_error)
        return True

    window._background = capture

    assert DawWindow.play_project(window) is True
    assert device.stop_calls == 1
    pending["error"](RuntimeError("device failed"))

    assert device.stop_calls == 2
    assert window._playing is False


def test_project_play_rejection_stops_replaced_audio(monkeypatch, tmp_path):
    window, device = _bare_transport_window(monkeypatch)
    window._project_snapshot = create_default_project
    window.loop_var = _Value(False)
    window._render_lock = threading.Lock()
    window.paths = SimpleNamespace(root=tmp_path)
    window._background = lambda *_args, **_kwargs: False

    assert DawWindow.play_project(window) is False

    assert device.stop_calls == 2
    assert window._playing is False


def test_rejected_microphone_stop_keeps_active_take_stoppable(tmp_path):
    window = object.__new__(DawWindow)
    window.recorder = SimpleNamespace(
        recording=True,
        recording_sample_rate=44_100,
        max_capture_seconds=30.0,
    )
    window.vocal_offset_var = _Value(0.0)
    window.paths = SimpleNamespace(recordings=tmp_path)
    window.start_record_button = _Button()
    window.stop_record_button = _Button()
    window._shutdown_event = threading.Event()
    window._set_status = lambda _text: None
    window._show_error = lambda _title, _error: None

    def reject(_label, _operation, _on_success=None, on_error=None):
        on_error(RuntimeError("busy"))
        return False

    window._background = reject

    assert DawWindow.stop_recording(window) is False

    assert window.start_record_button.state == "disabled"
    assert window.stop_record_button.state == "normal"


def test_api_reports_background_rejection_instead_of_scheduled():
    window = object.__new__(DawWindow)
    window.preview_waveform_var = _Value("sine")
    window.preview_note = lambda _note: False

    result = DawWindow._process_api_command(
        window,
        {"id": "preview-1", "action": "preview_note", "note": 60},
    )

    assert result["status"] == "error"
    assert "not scheduled" in result["error"]


def test_save_and_close_waits_for_success_callback(tmp_path):
    window = object.__new__(DawWindow)
    snapshot = SimpleNamespace(name="Quiet Orbit")
    callbacks = []
    closed = []
    window.project_path = None
    window.paths = SimpleNamespace(projects=tmp_path)
    window._project_snapshot = lambda: snapshot
    window._show_error = lambda *_args: None

    def schedule(_snapshot, _path, *, after_save=None):
        callbacks.append(after_save)
        return True

    window._save_snapshot = schedule
    window.close_window = lambda: closed.append(True)

    path = window.save_and_close()

    assert path == tmp_path / "Quiet_Orbit.ina-daw.json"
    assert closed == []
    callbacks[0](path)
    assert closed == [True]


def test_close_api_saves_by_default_and_can_explicitly_discard(tmp_path):
    window = object.__new__(DawWindow)
    window.save_and_close = lambda filename=None: tmp_path / "saved.ina-daw.json"
    window.after = lambda _delay, callback: callback()
    closed = []
    window.close_window = lambda: closed.append(True)

    safe = window._process_api_command({"id": "safe", "action": "close"})
    assert safe["closing_after_save"] is True
    assert safe["closed"] is False
    assert closed == []

    discard = window._process_api_command({"id": "discard", "action": "close", "save": False})
    assert discard["closed"] is True


def test_control_api_advertises_exact_dispatcher_without_autonomous_microphone():
    payload = daw_control_api_payload()
    advertised = {
        name
        for command in payload["commands"]
        for name in [command["action"], *command["aliases"]]
    }

    assert advertised == {
        "inspect",
        "state",
        "snapshot",
        "set_step",
        "set_track",
        "set_stem",
        "preview_stem",
        "preview_note",
        "generate_vocal",
        "play",
        "stop",
        "save",
        "export",
        "save_close",
        "save_and_close",
        "close",
        "done",
        "finish",
    }
    assert payload["state_keys"]["queue"] == "daw_command_queue"
    assert payload["state_keys"]["last_result"] == "daw_last_command_result"
    assert payload["limits"]["queue_max_pending"] == 32
    assert payload["limits"]["queue_max_per_poll"] == 4
    assert payload["offline"]["network_allowed"] is False
    assert payload["microphone"]["mode"] == "manual_only"
    assert payload["microphone"]["commands"] == []
    assert not advertised & {"record", "start_recording", "stop_recording", "microphone"}

    play_examples = [
        example for example in payload["examples"] if example.get("action") == "play"
    ]
    assert play_examples == [{"action": "play", "loop": False}]

    payload["commands"][0]["aliases"].append("mutated")
    fresh_actions = {
        name
        for command in daw_control_api_payload()["commands"]
        for name in [command["action"], *command["aliases"]]
    }
    assert "mutated" not in fresh_actions


def test_cli_child_is_forwarded_immutably_to_window(monkeypatch):
    received = []

    class FakeWindow:
        _closing = True

        def mainloop(self):
            return None

    def make_window(*, child=None):
        received.append(child)
        return FakeWindow()

    monkeypatch.setattr(daw_module, "DawWindow", make_window)

    daw_module.main(["--child", "Ina_At_Request_Time"])

    assert received == ["Ina_At_Request_Time"]
    assert daw_module._parse_args([]).child is None

    with pytest.raises(SystemExit):
        daw_module._parse_args(["--child", "../Other"])
