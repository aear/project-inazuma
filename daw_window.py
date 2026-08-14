"""Standalone Tkinter music studio for Ina.

The window is intentionally a thin controller around :mod:`daw_engine`: the
engine owns project validation and offline rendering, while this module owns
human/Ina interaction, optional audio devices, and bounded workspace state.
No network service is used by this studio.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Optional
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from ina_desktop.client import launch_environment
from audio_device_resolution import AudioDeviceResolution, resolve_audio_device
from daw_engine import (
    AudioStem,
    DawProject,
    InstrumentTrack,
    MAX_AUDIO_STEMS,
    MAX_PLACED_WAVS,
    MAX_RENDER_SAMPLES,
    Step,
    SUPPORTED_WAVEFORMS,
    VocalClip,
    export_project_wav,
    load_project,
    read_wav,
    render_project,
    save_project,
    synthesize_note,
    write_wav,
)
from runtime_state import drain_inastate_queue, get_inastate, load_config, update_inastate
from music_delivery import ensure_opus_sidecar
from creative_experience import choose_next
from stem_import import (
    StemImportResult,
    import_stem_zip,
    import_wav_stems,
)

try:  # Audio devices are optional; offline rendering still works without them.
    import sounddevice as _sounddevice
except Exception:  # pragma: no cover - depends on host audio setup
    _sounddevice = None


try:  # POSIX child-scoped single-instance locking.
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None


STEP_COUNT = 16
DAW_COMMAND_QUEUE_KEY = "daw_command_queue"
DAW_LAST_COMMAND_RESULT_KEY = "daw_last_command_result"
DAW_WINDOW_OPEN_KEY = "daw_window_open"
DAW_WORKSPACE_STATE_KEY = "daw_workspace_state"
DAW_API_POLL_MS = 350
DAW_API_MAX_COMMANDS = 4
DAW_API_QUEUE_LIMIT = 32
DAW_BACKGROUND_WORKERS = 2
DAW_BACKGROUND_JOB_LIMIT = 4
WAVEFORMS = tuple(sorted(SUPPORTED_WAVEFORMS))

_NOTE_NAMES = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")
KEYBOARD_NOTES = tuple(range(60, 73))


def daw_control_api_payload() -> dict[str, Any]:
    """Describe Ina's bounded, offline DAW command surface without reading state."""
    return {
        "version": 2,
        "state_keys": {
            "queue": DAW_COMMAND_QUEUE_KEY,
            "last_result": DAW_LAST_COMMAND_RESULT_KEY,
            "workspace": DAW_WORKSPACE_STATE_KEY,
            "window_open": DAW_WINDOW_OPEN_KEY,
        },
        "commands": [
            {
                "action": "inspect",
                "aliases": ["state", "snapshot"],
                "arguments": {},
            },
            {
                "action": "experience_choice",
                "aliases": [],
                "arguments": {"choice": "keep, revise, revisit, or stop", "reflection": "optional text"},
            },
            {
                "action": "set_step",
                "aliases": [],
                "arguments": {
                    "track": "index or exact track name",
                    "position": f"integer 0..{STEP_COUNT - 1}",
                    "enabled": "optional boolean",
                    "note": "optional MIDI integer 0..127",
                },
            },
            {
                "action": "set_track",
                "aliases": [],
                "arguments": {
                    "track": "index or exact track name",
                    "editable": [
                        "name",
                        "waveform",
                        "note",
                        "gain",
                        "muted",
                        "attack_seconds",
                        "release_seconds",
                    ],
                },
            },
            {
                "action": "set_stem",
                "aliases": [],
                "arguments": {
                    "stem": "index or exact stem name",
                    "editable": [
                        "name",
                        "role",
                        "gain",
                        "offset_beats",
                        "muted",
                        "solo",
                    ],
                },
            },
            {
                "action": "preview_stem",
                "aliases": [],
                "arguments": {"stem": "index or exact stem name"},
            },
            {
                "action": "preview_note",
                "aliases": [],
                "arguments": {
                    "note": "MIDI integer 0..127",
                    "waveform": f"optional: {', '.join(WAVEFORMS)}",
                },
            },
            {
                "action": "generate_vocal",
                "aliases": [],
                "arguments": {
                    "prompt": "required local symbolic prompt",
                    "offset_beats": "optional number 0..64",
                },
            },
            {
                "action": "play",
                "aliases": [],
                "arguments": {"loop": "optional boolean"},
            },
            {"action": "stop", "aliases": [], "arguments": {}},
            {
                "action": "save",
                "aliases": [],
                "arguments": {"filename": "optional studio-local filename stem"},
            },
            {
                "action": "save_close",
                "aliases": ["save_and_close"],
                "arguments": {"filename": "optional studio-local filename stem"},
            },
            {
                "action": "export",
                "aliases": [],
                "arguments": {"filename": "optional studio-local filename stem"},
            },
            {
                "action": "close",
                "aliases": ["done", "finish"],
                "arguments": {"save": "optional boolean; defaults true", "filename": "optional save name"},
            },
        ],
        "limits": {
            "step_count": STEP_COUNT,
            "note_min": 0,
            "note_max": 127,
            "vocal_offset_beats_max": 64.0,
            "stem_offset_beats_max": 64.0,
            "stem_gain_max": 4.0,
            "audio_stems_max": MAX_AUDIO_STEMS,
            "placed_wavs_max": MAX_PLACED_WAVS,
            "symbolic_prompt_max_characters": 1000,
            "queue_max_pending": DAW_API_QUEUE_LIMIT,
            "queue_max_per_poll": DAW_API_MAX_COMMANDS,
            "poll_interval_ms": DAW_API_POLL_MS,
            "background_max_active_or_queued": DAW_BACKGROUND_JOB_LIMIT,
        },
        "offline": {
            "network_allowed": False,
            "note": (
                "Instrument rendering and symbolic vocal generation are local/offline; "
                "this control API does not call Suno or another network service."
            ),
        },
        "microphone": {
            "mode": "manual_only",
            "commands": [],
            "note": "Microphone start/stop remains a person-operated studio control.",
        },
        "stem_import": {
            "mode": "manual_only",
            "commands": [],
            "note": (
                "A person chooses external WAV/ZIP files. Ina may inspect, mix, "
                "preview, save, and export stems already imported into the studio."
            ),
        },
        "examples": [
            {"action": "inspect"},
            {"action": "set_step", "track": 0, "position": 0, "enabled": True, "note": 60},
            {"action": "preview_note", "note": 60, "waveform": "sine"},
            {"action": "set_stem", "stem": 0, "gain": 0.8, "solo": True},
            {"action": "preview_stem", "stem": 0},
            {"action": "generate_vocal", "prompt": "soft orbit", "offset_beats": 0.0},
            {"action": "play", "loop": False},
            {"action": "stop"},
            {"action": "save", "filename": "ina_first_loop"},
            {"action": "export", "filename": "ina_first_loop"},
            {"action": "save_close", "filename": "ina_first_loop"},
        ],
    }


class StudioAlreadyRunningError(RuntimeError):
    """Raised when this child's studio already owns the command queue."""


class StemCollectionAttachmentError(ValueError):
    """Raised when a committed local collection cannot join the current project."""


class StudioInstanceLock:
    """Process lock that keeps one DAW queue consumer per child."""

    def __init__(self, path: Path | str, *, lock_module: Any = fcntl) -> None:
        self.path = Path(path)
        self._lock_module = lock_module
        self._handle = None
        self._fallback_path = self.path.with_name(f"{self.path.name}.exclusive")
        self._fallback_created = False
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> bool:
        if self._held:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = None
        if self._lock_module is None:
            try:
                descriptor = os.open(
                    self._fallback_path,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                    0o600,
                )
            except FileExistsError:
                return False
            except OSError:
                return False
            handle = os.fdopen(descriptor, "w+", encoding="utf-8")
            self._fallback_created = True
        else:
            try:
                handle = self.path.open("a+", encoding="utf-8")
                self._lock_module.flock(
                    handle.fileno(),
                    self._lock_module.LOCK_EX | self._lock_module.LOCK_NB,
                )
            except (BlockingIOError, OSError):
                if handle is not None:
                    try:
                        handle.close()
                    except OSError:
                        pass
                return False
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle
        self._held = True
        atexit.register(self.release)
        return True

    def release(self) -> None:
        if not self._held:
            return
        handle, self._handle = self._handle, None
        self._held = False
        if handle is not None:
            if self._lock_module is not None and not self._fallback_created:
                try:
                    self._lock_module.flock(handle.fileno(), self._lock_module.LOCK_UN)
                except OSError:
                    pass
            try:
                handle.close()
            except OSError:
                pass
        if self._fallback_created:
            self._fallback_created = False
            try:
                self._fallback_path.unlink(missing_ok=True)
            except OSError:
                pass


class TransportGate:
    """Atomically invalidates stale play jobs before device playback starts."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._generation = 0

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def advance(self) -> int:
        with self._lock:
            self._generation += 1
            return self._generation

    def is_current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation

    def run_if_current(self, generation: int, operation: Callable[[], Any]) -> tuple[bool, Any]:
        with self._lock:
            if generation != self._generation:
                return False, None
            return True, operation()

    def invalidate(self, operation: Optional[Callable[[], Any]] = None) -> tuple[int, Any]:
        with self._lock:
            self._generation += 1
            result = operation() if operation is not None else None
            return self._generation, result

    def invalidate_if_current(
        self,
        generation: int,
        operation: Optional[Callable[[], Any]] = None,
    ) -> tuple[bool, int, Any]:
        """Invalidate a failed generation without disturbing newer playback."""
        with self._lock:
            if generation != self._generation:
                return False, self._generation, None
            self._generation += 1
            result = operation() if operation is not None else None
            return True, self._generation, result


class BoundedExecutor:
    """Small executor with a hard bound across running and queued work."""

    def __init__(self, *, max_workers: int = 2, max_jobs: int = 4) -> None:
        if max_workers < 1 or max_jobs < max_workers:
            raise ValueError("max_jobs must be at least max_workers and both must be positive")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ina_daw_worker")
        self._slots = threading.BoundedSemaphore(max_jobs)
        self._state_lock = threading.Lock()
        self._closed = False
        self._futures: set[Any] = set()
        self._drain_callbacks: list[Callable[[], None]] = []

    def submit(self, operation: Callable[[], Any]):
        with self._state_lock:
            if self._closed or not self._slots.acquire(blocking=False):
                return None
            try:
                future = self._executor.submit(operation)
            except BaseException:
                self._slots.release()
                raise
            self._futures.add(future)
        future.add_done_callback(self._finished)
        return future

    def _finished(self, future: Any) -> None:
        self._slots.release()
        callbacks: list[Callable[[], None]] = []
        with self._state_lock:
            self._futures.discard(future)
            if self._closed and not self._futures:
                callbacks, self._drain_callbacks = self._drain_callbacks, []
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

    def shutdown(self, *, on_drained: Optional[Callable[[], None]] = None) -> None:
        callbacks: list[Callable[[], None]] = []
        with self._state_lock:
            if on_drained is not None:
                self._drain_callbacks.append(on_drained)
            self._closed = True
            if not self._futures:
                callbacks, self._drain_callbacks = self._drain_callbacks, []
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        finally:
            for callback in callbacks:
                try:
                    callback()
                except Exception:
                    pass


@dataclass(frozen=True)
class StudioPaths:
    root: Path
    projects: Path
    recordings: Path
    renders: Path
    stems: Path

    def ensure(self) -> "StudioPaths":
        self.root.mkdir(parents=True, exist_ok=True)
        root = self.root.resolve()
        for path in (self.projects, self.recordings, self.renders, self.stems):
            if path.is_symlink():
                raise ValueError(f"Music studio managed folder cannot be a symbolic link: {path.name}")
            path.mkdir(parents=True, exist_ok=True)
            try:
                path.resolve().relative_to(root)
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Music studio managed folder must stay inside the studio: {path.name}"
                ) from exc
            if not path.is_dir():
                raise ValueError(f"Music studio managed path is not a folder: {path.name}")
        return self


def validate_child_identifier(child: Any) -> str:
    if not isinstance(child, str):
        raise ValueError("child identifier must be a string")
    identifier = child.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", identifier):
        raise ValueError("child identifier may contain only letters, numbers, underscores, and hyphens")
    return identifier


def studio_paths(child: str, *, base_path: Path | str = Path("AI_Children")) -> StudioPaths:
    identifier = validate_child_identifier(child)
    base = Path(base_path).expanduser()
    root = base / identifier / "memory" / "music_studio"
    try:
        root.resolve().relative_to(base.resolve())
    except (OSError, ValueError) as exc:
        raise ValueError("music studio path must stay inside AI_Children") from exc
    return StudioPaths(
        root=root,
        projects=root / "projects",
        recordings=root / "recordings",
        renders=root / "renders",
        stems=root / "stems",
    )


def safe_filename_stem(value: str, default: str = "untitled") -> str:
    """Return a small filesystem-safe stem without changing the display name."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")[:80]
    return text or default


def path_is_within(path: Path | str, parent: Path | str) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(Path(parent).expanduser().resolve())
        return True
    except (OSError, ValueError):
        return False


def midi_note_label(note: int) -> str:
    number = max(0, min(127, int(note)))
    return f"{_NOTE_NAMES[number % 12]}{(number // 12) - 1}"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _float_value(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _int_value(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _bool_value(value: Any, name: str) -> bool:
    """Parse command booleans without Python's surprising bool-of-string behavior."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{name} must be a boolean")


def _strict_int(
    value: Any,
    name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Parse an API integer and reject malformed or out-of-range values."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        number = int(value.strip())
    elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer():
        number = int(value)
    else:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return number


def _strict_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return number


def configured_device_index(config: dict[str, Any], *keys: str) -> Optional[int]:
    """Return the first usable non-negative audio device index in config."""
    for key in keys:
        raw = config.get(key)
        if raw in (None, ""):
            continue
        try:
            return _strict_int(raw, key, 0)
        except ValueError:
            continue
    return None


def create_default_project() -> DawProject:
    """Create a quiet, editable one-bar starter groove."""
    return DawProject(
        name="Ina's first loop",
        bpm=108.0,
        beats_per_bar=4,
        steps_per_beat=4,
        bars=1,
        seed=7,
        tracks=[
            InstrumentTrack(
                name="Pulse",
                waveform="noise",
                gain=0.42,
                release_seconds=0.025,
                steps=[Step(position=pos, note=36, velocity=0.85, duration_steps=0.45) for pos in (0, 4, 8, 12)],
            ),
            InstrumentTrack(
                name="Answer",
                waveform="triangle",
                gain=0.35,
                steps=[Step(position=pos, note=55, velocity=0.72) for pos in (4, 12)],
            ),
            InstrumentTrack(
                name="Bass",
                waveform="square",
                gain=0.24,
                steps=[Step(position=pos, note=43, velocity=0.68) for pos in (0, 3, 8, 11)],
            ),
            InstrumentTrack(
                name="Light",
                waveform="sine",
                gain=0.28,
                steps=[Step(position=pos, note=67, velocity=0.64) for pos in (2, 6, 10, 14)],
            ),
        ],
    )


class MicrophoneRecorder:
    """Small sounddevice InputStream owner with bounded in-memory capture."""

    def __init__(
        self,
        *,
        sample_rate: int = 44_100,
        channels: int = 1,
        device: Any = None,
        sounddevice_module: Any = _sounddevice,
        max_seconds: float = 300.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("Microphone sample rate and channel count must be positive.")
        self.max_seconds = _strict_float(max_seconds, "maximum recording duration", 0.001, 300.0)
        self.device = device
        self._sounddevice = sounddevice_module
        self._stream = None
        self._opening_stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._captured_frames = 0
        self._max_frames = self._frame_budget(self.sample_rate)
        self._limit_reached = False
        self._active_sample_rate: Optional[int] = None
        self._last_sample_rate = self.sample_rate
        self._generation = 0
        self._starting = False
        self._accepting_frames = False
        self._closed = False

    def _frame_budget(self, sample_rate: int) -> int:
        requested = max(1, round(int(sample_rate) * self.max_seconds))
        return min(MAX_RENDER_SAMPLES, requested)

    @property
    def available(self) -> bool:
        return self._sounddevice is not None

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._stream is not None

    @property
    def recording_sample_rate(self) -> int:
        with self._lock:
            return self._active_sample_rate or self._last_sample_rate

    @property
    def max_frames(self) -> int:
        with self._lock:
            return self._max_frames

    @property
    def max_capture_seconds(self) -> float:
        rate = self.recording_sample_rate
        return self.max_frames / rate

    @property
    def limit_reached(self) -> bool:
        with self._lock:
            return self._limit_reached

    def _callback(self, indata, _frames, _time_info, _status) -> None:
        chunk = np.asarray(indata, dtype=np.float32).copy()
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        with self._lock:
            if not self._accepting_frames:
                return
            remaining = self._max_frames - self._captured_frames
            if remaining <= 0:
                self._limit_reached = True
                return
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
                self._limit_reached = True
            if not len(chunk):
                return
            self._chunks.append(chunk)
            self._captured_frames += len(chunk)
            if self._captured_frames >= self._max_frames:
                self._limit_reached = True

    @staticmethod
    def _close_stream(stream: Any) -> None:
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def start(self) -> None:
        if self._sounddevice is None:
            raise RuntimeError("Microphone recording needs the optional sounddevice package.")
        sample_rate = int(self.sample_rate)
        if sample_rate <= 0:
            raise ValueError("Microphone sample rate must be positive.")
        with self._lifecycle_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("The microphone recorder is closed.")
                if self._stream is not None or self._starting:
                    raise RuntimeError("A microphone recording is already active or opening.")
                self._generation += 1
                generation = self._generation
                self._starting = True
                self._accepting_frames = True
                self._chunks = []
                self._captured_frames = 0
                self._max_frames = self._frame_budget(sample_rate)
                self._limit_reached = False
                self._active_sample_rate = sample_rate
                self._last_sample_rate = sample_rate

        stream = None
        try:
            stream = self._sounddevice.InputStream(
                samplerate=sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                callback=self._callback,
            )
            with self._lifecycle_lock:
                with self._lock:
                    cancelled = (
                        self._closed
                        or generation != self._generation
                        or not self._starting
                    )
                    if not cancelled:
                        self._opening_stream = stream
                if cancelled:
                    raise RuntimeError("Microphone start was cancelled.")

                # Close/abort takes this same lifecycle lock, so it either
                # revokes the start above or waits to close the published stream.
                stream.start()
                with self._lock:
                    cancelled = (
                        self._closed
                        or generation != self._generation
                        or not self._starting
                        or self._opening_stream is not stream
                    )
                    if self._opening_stream is stream:
                        self._opening_stream = None
                    if not cancelled:
                        self._stream = stream
                        self._starting = False
                if cancelled:
                    raise RuntimeError("Microphone start was cancelled.")
        except Exception:
            with self._lifecycle_lock:
                if stream is not None:
                    self._close_stream(stream)
                with self._lock:
                    if self._opening_stream is stream:
                        self._opening_stream = None
                    if generation == self._generation:
                        self._starting = False
                        self._accepting_frames = False
                        self._active_sample_rate = None
            raise

    def stop(self) -> np.ndarray:
        with self._lifecycle_lock:
            with self._lock:
                stream, self._stream = self._stream, None
                if stream is None:
                    raise RuntimeError("No microphone recording is active.")
                self._generation += 1
                self._accepting_frames = False
            stop_error = None
            try:
                stream.stop()
            except Exception as exc:
                stop_error = exc
            try:
                stream.close()
            except Exception as exc:
                stop_error = stop_error or exc
            with self._lock:
                chunks = list(self._chunks)
                self._chunks = []
                self._captured_frames = 0
                self._active_sample_rate = None
        if not chunks:
            if stop_error is not None:
                raise RuntimeError(f"The microphone stopped without usable audio: {stop_error}") from stop_error
            raise RuntimeError("The microphone returned no audio frames.")
        audio = np.concatenate(chunks, axis=0)
        return audio[:, 0] if audio.shape[1] == 1 else audio

    def _cancel(self, *, permanent: bool) -> None:
        with self._lifecycle_lock:
            with self._lock:
                if permanent:
                    self._closed = True
                self._generation += 1
                self._starting = False
                self._accepting_frames = False
                stream, self._stream = self._stream, None
                opening_stream, self._opening_stream = self._opening_stream, None
                self._chunks = []
                self._captured_frames = 0
                self._limit_reached = False
                self._active_sample_rate = None
            if stream is not None:
                self._close_stream(stream)
            if opening_stream is not None and opening_stream is not stream:
                self._close_stream(opening_stream)

    def abort(self) -> None:
        self._cancel(permanent=False)

    def close(self) -> None:
        """Permanently fence starts and close any active or opening stream."""
        self._cancel(permanent=True)


def render_local_symbolic_vocal(
    prompt: str,
    *,
    child: str,
    output_path: Path | str,
    max_symbols: int = 8,
) -> dict[str, Any]:
    """Render known local sound symbols for ``prompt`` without device playback."""
    text = str(prompt or "").strip()
    if not text:
        raise ValueError("Enter a short prompt for Ina's local symbol vocabulary.")
    destination = Path(output_path)
    from language_processing import generate_symbolic_reply_from_text

    payload = generate_symbolic_reply_from_text(
        text,
        child=child,
        max_symbols=max_symbols,
        context={"source": "daw_window", "tags": ["music_studio", "local_symbolic_vocal"]},
        playback=False,
        record_path=destination,
        record_format="wav",
    )
    if not payload or not payload.get("symbols"):
        raise RuntimeError("No known local sound symbols matched that prompt yet.")
    if not destination.exists():
        raise RuntimeError("The local symbol renderer did not create a WAV file.")
    return payload


class DawWindow(tk.Tk):
    def __init__(self, child: Optional[str] = None) -> None:
        config = load_config()
        resolved_child = validate_child_identifier(
            child
            if child is not None
            else (config.get("current_child", "Inazuma_Yagami") or "Inazuma_Yagami")
        )
        paths = studio_paths(resolved_child).ensure()
        super().__init__()
        self.child = resolved_child
        self.paths = paths
        self._instance_lock = StudioInstanceLock(self.paths.root / "daw_window.lock")
        if not self._instance_lock.acquire():
            try:
                self.destroy()
            except tk.TclError:
                pass
            raise StudioAlreadyRunningError(f"Ina Music Studio is already open for {self.child}.")

        self.project = create_default_project()
        self.input_device_resolution = resolve_audio_device(
            config,
            label="mic_headset",
            role="input",
            index_keys=("mic_headset_index",),
            name_keys=("mic_headset_name",),
            sounddevice_module=_sounddevice,
            sample_rate=self.project.sample_rate,
            channels=1,
        )
        self.output_device_resolution = resolve_audio_device(
            config,
            label="output_headset",
            role="output",
            index_keys=(
                "output_headset_index",
                "ouput_headset_index",
                "output_TV_index",
                "ouput_TV_index",
            ),
            name_keys=(
                "output_headset_name",
                "ouput_headset_name",
                "output_TV_name",
                "ouput_TV_name",
            ),
            sounddevice_module=_sounddevice,
            sample_rate=self.project.sample_rate,
            channels=2,
        )
        self.input_device = (
            self.input_device_resolution.device
            if self.input_device_resolution.available
            else None
        )
        self.output_device = (
            self.output_device_resolution.device
            if self.output_device_resolution.available
            else None
        )
        self.project_path: Optional[Path] = None
        self.last_render_path: Optional[Path] = None
        self.track_rows: list[dict[str, Any]] = []
        self.recorder = MicrophoneRecorder(sample_rate=self.project.sample_rate, device=self.input_device)
        self._jobs = BoundedExecutor(
            max_workers=DAW_BACKGROUND_WORKERS,
            max_jobs=DAW_BACKGROUND_JOB_LIMIT,
        )
        self._render_lock = threading.Lock()
        self._transport_gate = TransportGate()
        self._shutdown_event = threading.Event()
        self._api_poll_active = True
        self._playing = False
        self._recording_pending = False
        self._stem_import_pending = False
        self._project_load_pending = False
        self._known_stem_collections: dict[str, str] = {}
        self._closing = False
        self.title(f"Ina Music Studio — {self.child}")
        self.geometry("1320x820")
        self.minsize(1050, 690)
        self.protocol("WM_DELETE_WINDOW", self.close_window)

        self.project_name_var = tk.StringVar(value=self.project.name)
        self.bpm_var = tk.DoubleVar(value=self.project.bpm)
        self.loop_var = tk.BooleanVar(value=True)
        self.preview_waveform_var = tk.StringVar(value="sine")
        self.vocal_offset_var = tk.DoubleVar(value=0.0)
        self.symbolic_prompt_var = tk.StringVar(value="")
        self.stem_gain_var = tk.DoubleVar(value=1.0)
        self.stem_offset_var = tk.DoubleVar(value=0.0)
        self.stem_role_var = tk.StringVar(value="other")
        self.audio_device_var = tk.StringVar(
            value=self._audio_resolution_label(self.output_device_resolution)
        )
        self.status_var = tk.StringVar(value=self._audio_startup_status())

        self._configure_style()
        self._build_ui()
        self._rebuild_track_grid()
        self._refresh_vocal_list()
        self._refresh_stem_list()
        self._publish_workspace("opened")
        self._set_window_open(True)
        self.after(DAW_API_POLL_MS, self._poll_api_commands)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("StudioTitle.TLabel", font=("Helvetica", 18, "bold"))
        style.configure("StudioNote.TLabel", foreground="#555f70")
        style.configure("StudioAccent.TButton", font=("Helvetica", 10, "bold"))

    @staticmethod
    def _audio_resolution_label(resolution: AudioDeviceResolution) -> str:
        if not resolution.available:
            return "Output unavailable · offline render works"
        device_label = (
            "System default"
            if resolution.device is None
            else str(resolution.device)
        )
        if len(device_label) > 28:
            device_label = device_label[:25] + "..."
        name = str(resolution.name or "").strip()
        if len(name) > 44:
            name = name[:41] + "..."
        suffix = (
            f" · {name}"
            if name and name.casefold() != device_label.casefold()
            else ""
        )
        return f"Output: {device_label}{suffix}"

    def _audio_startup_status(self) -> str:
        output = self.output_device_resolution
        microphone = self.input_device_resolution
        if output.warning:
            if not output.available:
                return "Live output unavailable; offline render/export still work. Config unchanged."
            if output.source == "configured_name":
                return "Saved output index changed; matched the output by name. Config unchanged."
            return "Configured output unavailable; using the system default. Config unchanged."
        if microphone.warning:
            if not microphone.available:
                return "Microphone unavailable; playback and offline work remain available."
            if microphone.source == "configured_name":
                return "Saved mic index changed; matched the microphone by name. Config unchanged."
            return "Configured microphone unavailable; using the system default. Config unchanged."
        return "Ready — audio checked; local/offline studio; no Suno or network generation."

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Ina Music Studio", style="StudioTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="A small local sequencer, stem mixer, and symbolic vocal sketchpad",
            style="StudioNote.TLabel",
        ).grid(row=0, column=1, sticky="w", padx=14)
        ttk.Label(
            header,
            textvariable=self.audio_device_var,
            style="StudioNote.TLabel",
        ).grid(row=0, column=2, sticky="e", padx=(14, 0))

        toolbar = ttk.LabelFrame(outer, text="Project and transport", padding=8)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(toolbar, textvariable=self.project_name_var, width=30).grid(row=0, column=1, sticky="ew", padx=(5, 12))
        ttk.Label(toolbar, text="Tempo").grid(row=0, column=2)
        ttk.Spinbox(toolbar, from_=40, to=240, increment=1, textvariable=self.bpm_var, width=7).grid(row=0, column=3, padx=5)
        ttk.Label(toolbar, text="BPM").grid(row=0, column=4, padx=(0, 10))
        ttk.Checkbutton(toolbar, text="Loop playback", variable=self.loop_var).grid(row=0, column=5, padx=(0, 12))
        ttk.Button(toolbar, text="Render", command=self.render_to_library).grid(row=0, column=6, padx=3)
        ttk.Button(toolbar, text="Play", command=self.play_project, style="StudioAccent.TButton").grid(row=0, column=7, padx=3)
        ttk.Button(toolbar, text="Stop", command=self.stop_playback).grid(row=0, column=8, padx=3)
        ttk.Separator(toolbar, orient=tk.VERTICAL).grid(row=0, column=9, sticky="ns", padx=7)
        ttk.Button(toolbar, text="Save project", command=self.save_project_dialog).grid(row=0, column=10, padx=3)
        ttk.Button(toolbar, text="Load project", command=self.load_project_dialog).grid(row=0, column=11, padx=3)
        ttk.Button(toolbar, text="Export WAV", command=self.export_wav_dialog).grid(row=0, column=12, padx=3)
        ttk.Button(toolbar, text="Save & close", command=self.save_and_close).grid(row=0, column=13, padx=3)

        notebook = ttk.Notebook(outer)
        notebook.grid(row=2, column=0, sticky="nsew")
        sequence_tab = ttk.Frame(notebook, padding=10)
        stem_tab = ttk.Frame(notebook, padding=10)
        vocal_tab = ttk.Frame(notebook, padding=10)
        notebook.add(sequence_tab, text="Step sequencer")
        notebook.add(stem_tab, text="Stems")
        notebook.add(vocal_tab, text="Vocals")

        sequence_tab.columnconfigure(0, weight=1)
        sequence_tab.rowconfigure(0, weight=1)
        track_canvas = tk.Canvas(sequence_tab, highlightthickness=0)
        track_scroll = ttk.Scrollbar(sequence_tab, orient=tk.VERTICAL, command=track_canvas.yview)
        self.track_grid = ttk.Frame(track_canvas)
        self.track_grid.bind(
            "<Configure>",
            lambda _event: track_canvas.configure(scrollregion=track_canvas.bbox("all")),
        )
        track_canvas.create_window((0, 0), window=self.track_grid, anchor="nw")
        track_canvas.configure(yscrollcommand=track_scroll.set)
        track_canvas.grid(row=0, column=0, sticky="nsew")
        track_scroll.grid(row=0, column=1, sticky="ns")

        keyboard = ttk.LabelFrame(sequence_tab, text="Note preview keyboard", padding=8)
        keyboard.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(keyboard, text="Waveform").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Combobox(
            keyboard,
            textvariable=self.preview_waveform_var,
            values=WAVEFORMS,
            state="readonly",
            width=9,
        ).pack(side=tk.LEFT, padx=(0, 12))
        for note in KEYBOARD_NOTES:
            accidental = "♯" in midi_note_label(note)
            tk.Button(
                keyboard,
                text=midi_note_label(note),
                width=4,
                bg="#30343b" if accidental else "#f4f4f1",
                fg="white" if accidental else "#20242b",
                activebackground="#7d5fff",
                command=lambda midi=note: self.preview_note(midi),
            ).pack(side=tk.LEFT, padx=1, pady=2)

        self._build_stem_tab(stem_tab)
        self._build_vocal_tab(vocal_tab)

        status = ttk.Frame(outer)
        status.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_var, style="StudioNote.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status, text=str(self.paths.root), style="StudioNote.TLabel").grid(row=0, column=1, sticky="e")

    def _build_vocal_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        record_box = ttk.LabelFrame(parent, text="Mic recording", padding=8)
        record_box.grid(row=0, column=0, sticky="ew")
        ttk.Label(record_box, text="Place at beat").grid(row=0, column=0, padx=(0, 5))
        ttk.Spinbox(
            record_box,
            from_=0,
            to=64,
            increment=0.25,
            textvariable=self.vocal_offset_var,
            width=8,
        ).grid(row=0, column=1, padx=(0, 10))
        self.start_record_button = ttk.Button(record_box, text="Start mic", command=self.start_recording)
        self.start_record_button.grid(row=0, column=2, padx=3)
        self.stop_record_button = ttk.Button(record_box, text="Stop and add", command=self.stop_recording, state=tk.DISABLED)
        self.stop_record_button.grid(row=0, column=3, padx=3)
        ttk.Label(
            record_box,
            text="sounddevice is optional; takes are capped by five minutes and the engine render limit.",
            style="StudioNote.TLabel",
        ).grid(row=0, column=4, sticky="w", padx=12)

        symbolic_box = ttk.LabelFrame(parent, text="Local symbolic vocal", padding=8)
        symbolic_box.grid(row=1, column=0, sticky="ew", pady=8)
        symbolic_box.columnconfigure(1, weight=1)
        ttk.Label(symbolic_box, text="Prompt").grid(row=0, column=0, sticky="w")
        prompt_entry = ttk.Entry(symbolic_box, textvariable=self.symbolic_prompt_var)
        prompt_entry.grid(row=0, column=1, sticky="ew", padx=6)
        prompt_entry.bind("<Return>", lambda _event: self.generate_symbolic_vocal())
        ttk.Button(
            symbolic_box,
            text="Generate local symbols (not Suno)",
            command=self.generate_symbolic_vocal,
        ).grid(row=0, column=2, padx=3)
        ttk.Label(
            symbolic_box,
            text="This renders Ina's existing sound-symbol vocabulary locally. It is not singing, Suno, or an online model.",
            style="StudioNote.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        clips_box = ttk.LabelFrame(parent, text="Vocal clips in this arrangement", padding=8)
        clips_box.grid(row=2, column=0, sticky="nsew")
        clips_box.columnconfigure(0, weight=1)
        clips_box.rowconfigure(0, weight=1)
        self.vocal_tree = ttk.Treeview(
            clips_box,
            columns=("offset", "gain", "source"),
            show="tree headings",
            selectmode="browse",
        )
        self.vocal_tree.heading("#0", text="Clip")
        self.vocal_tree.heading("offset", text="Beat offset")
        self.vocal_tree.heading("gain", text="Gain")
        self.vocal_tree.heading("source", text="WAV")
        self.vocal_tree.column("#0", width=240)
        self.vocal_tree.column("offset", width=90, anchor=tk.CENTER)
        self.vocal_tree.column("gain", width=80, anchor=tk.CENTER)
        self.vocal_tree.column("source", width=520)
        self.vocal_tree.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(clips_box, orient=tk.VERTICAL, command=self.vocal_tree.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.vocal_tree.configure(yscrollcommand=scroll.set)
        controls = ttk.Frame(clips_box)
        controls.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Button(controls, text="Play selected clip", command=self.play_selected_vocal).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls, text="Remove from project", command=self.remove_selected_vocal).pack(side=tk.LEFT)
        ttk.Label(
            controls,
            text="Removing a clip keeps its WAV in the studio recordings folder.",
            style="StudioNote.TLabel",
        ).pack(side=tk.LEFT, padx=12)

    def _build_stem_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        import_box = ttk.LabelFrame(parent, text="Bring in a separated stem collection", padding=8)
        import_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(
            import_box,
            text="Import WAV stem(s)…",
            command=self.import_wav_stems_dialog,
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            import_box,
            text="Import stem ZIP…",
            command=self.import_stem_zip_dialog,
            style="StudioAccent.TButton",
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(
            import_box,
            text=(
                "Imports are copied locally and bounded. ZIP TXT files are retained as "
                "lyrics/style or general music context; the source ZIP is unchanged."
            ),
            style="StudioNote.TLabel",
        ).pack(side=tk.LEFT)

        stems_box = ttk.LabelFrame(parent, text="Stem mixer", padding=8)
        stems_box.grid(row=1, column=0, sticky="nsew")
        stems_box.columnconfigure(0, weight=1)
        stems_box.rowconfigure(0, weight=1)
        self.stem_tree = ttk.Treeview(
            stems_box,
            columns=("role", "offset", "gain", "state", "source"),
            show="tree headings",
            selectmode="browse",
        )
        self.stem_tree.heading("#0", text="Stem")
        self.stem_tree.heading("role", text="Role")
        self.stem_tree.heading("offset", text="Beat offset")
        self.stem_tree.heading("gain", text="Gain")
        self.stem_tree.heading("state", text="Mix state")
        self.stem_tree.heading("source", text="Studio WAV")
        self.stem_tree.column("#0", width=210)
        self.stem_tree.column("role", width=90)
        self.stem_tree.column("offset", width=90, anchor=tk.CENTER)
        self.stem_tree.column("gain", width=70, anchor=tk.CENTER)
        self.stem_tree.column("state", width=90, anchor=tk.CENTER)
        self.stem_tree.column("source", width=480)
        self.stem_tree.grid(row=0, column=0, sticky="nsew")
        stem_scroll = ttk.Scrollbar(stems_box, orient=tk.VERTICAL, command=self.stem_tree.yview)
        stem_scroll.grid(row=0, column=1, sticky="ns")
        self.stem_tree.configure(yscrollcommand=stem_scroll.set)
        self.stem_tree.bind("<<TreeviewSelect>>", self._load_selected_stem_controls)

        controls = ttk.Frame(stems_box)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(controls, text="Play selected", command=self.play_selected_stem).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(controls, text="Gain").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls,
            from_=0,
            to=4,
            increment=0.05,
            textvariable=self.stem_gain_var,
            width=6,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(controls, text="Beat").pack(side=tk.LEFT)
        ttk.Spinbox(
            controls,
            from_=0,
            to=64,
            increment=0.25,
            textvariable=self.stem_offset_var,
            width=7,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(controls, text="Role").pack(side=tk.LEFT)
        ttk.Combobox(
            controls,
            textvariable=self.stem_role_var,
            values=("vocals", "drums", "bass", "guitar", "keys", "other"),
            width=9,
        ).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(controls, text="Apply mix", command=self.apply_selected_stem_mix).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls, text="Mute / unmute", command=self.toggle_selected_stem_mute).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls, text="Solo / un-solo", command=self.toggle_selected_stem_solo).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls, text="Remove", command=self.remove_selected_stem).pack(side=tk.LEFT)

        ttk.Label(
            stems_box,
            text=(
                "All imported stems begin aligned at beat 0. Gain, mute, solo, and offset "
                "affect playback and WAV export; removing keeps the local collection."
            ),
            style="StudioNote.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(7, 0))

    def _rebuild_track_grid(self) -> None:
        for child in self.track_grid.winfo_children():
            child.destroy()
        self.track_rows = []
        headers = ("Track", "Wave", "MIDI note", "Volume")
        for column, label in enumerate(headers):
            ttk.Label(self.track_grid, text=label, font=("Helvetica", 9, "bold")).grid(
                row=0, column=column, sticky="w", padx=3, pady=(0, 5)
            )
        for step in range(STEP_COUNT):
            ttk.Label(
                self.track_grid,
                text=str(step + 1),
                foreground="#5d6572" if step % 4 else "#7d5fff",
            ).grid(row=0, column=step + 4, padx=1, pady=(0, 5))

        for index, track in enumerate(self.project.tracks):
            row_no = index + 1
            name_var = tk.StringVar(value=track.name)
            waveform_var = tk.StringVar(value=track.waveform)
            note_value = track.steps[0].note if track.steps else 60
            note_var = tk.IntVar(value=note_value)
            gain_var = tk.DoubleVar(value=track.gain)
            ttk.Entry(self.track_grid, textvariable=name_var, width=15).grid(row=row_no, column=0, sticky="ew", padx=3, pady=4)
            ttk.Combobox(
                self.track_grid,
                textvariable=waveform_var,
                values=WAVEFORMS,
                state="readonly",
                width=9,
            ).grid(row=row_no, column=1, padx=3)
            note_box = ttk.Spinbox(self.track_grid, from_=24, to=96, textvariable=note_var, width=7)
            note_box.grid(row=row_no, column=2, padx=3)
            gain_scale = ttk.Scale(self.track_grid, from_=0.0, to=1.0, variable=gain_var, length=100)
            gain_scale.grid(row=row_no, column=3, padx=5)
            row = {
                "name": name_var,
                "waveform": waveform_var,
                "note": note_var,
                "gain": gain_var,
                "buttons": [],
            }
            self.track_rows.append(row)
            for position in range(STEP_COUNT):
                button = tk.Button(
                    self.track_grid,
                    width=2,
                    height=1,
                    bd=1,
                    command=lambda ti=index, pos=position: self.toggle_step(ti, pos),
                )
                button.grid(row=row_no, column=position + 4, padx=1, pady=3)
                row["buttons"].append(button)
                self._refresh_step_button(index, position)

    def _track_step(self, track_index: int, position: int) -> Optional[Step]:
        for step in self.project.tracks[track_index].steps:
            if step.position == position:
                return step
        return None

    def _refresh_step_button(self, track_index: int, position: int) -> None:
        if track_index >= len(self.track_rows):
            return
        button = self.track_rows[track_index]["buttons"][position]
        active = self._track_step(track_index, position) is not None
        button.configure(
            text="●" if active else "·",
            relief=tk.SUNKEN if active else tk.RAISED,
            bg="#7d5fff" if active else ("#e7e3ff" if position % 4 == 0 else "#f1f2f4"),
            fg="white" if active else "#687080",
            activebackground="#6c4fe5",
        )

    def _commit_controls(self) -> None:
        self.project.name = self.project_name_var.get().strip() or "Untitled"
        self.project.bpm = _float_value(self.bpm_var.get(), self.project.bpm, 40.0, 240.0)
        self.bpm_var.set(self.project.bpm)
        for index, track in enumerate(self.project.tracks):
            if index >= len(self.track_rows):
                break
            row = self.track_rows[index]
            track.name = row["name"].get().strip() or f"Track {index + 1}"
            waveform = row["waveform"].get().strip().lower()
            track.waveform = waveform if waveform in SUPPORTED_WAVEFORMS else "sine"
            track.gain = _float_value(row["gain"].get(), track.gain, 0.0, 1.0)
            note = _int_value(row["note"].get(), 60, 0, 127)
            row["note"].set(note)
            for step in track.steps:
                step.note = note
        self.project.validate()

    def _track_pitch(self, track_index: int) -> int:
        track = self.project.tracks[track_index]
        if track_index < len(self.track_rows):
            return _int_value(self.track_rows[track_index]["note"].get(), 60, 0, 127)
        return track.steps[0].note if track.steps else 60

    def _track_payload(self, track_index: int) -> dict[str, Any]:
        track = self.project.tracks[track_index]
        return {
            "index": track_index,
            "name": track.name,
            "waveform": track.waveform,
            "gain": round(track.gain, 3),
            "note": self._track_pitch(track_index),
            "muted": track.muted,
            "active_steps": sorted({step.position for step in track.steps if step.position < STEP_COUNT}),
        }

    def toggle_step(
        self,
        track_index: int,
        position: int,
        enabled: Optional[bool] = None,
        note: Optional[int] = None,
    ) -> bool:
        track_index = _strict_int(track_index, "track", 0, len(self.project.tracks) - 1)
        position = _strict_int(position, "step", 0, STEP_COUNT - 1)
        track = self.project.tracks[track_index]
        existing = [step for step in track.steps if step.position == position]
        should_enable = not bool(existing) if enabled is None else _bool_value(enabled, "enabled")

        if note is not None:
            pitch = _strict_int(note, "note", 0, 127)
            if track_index < len(self.track_rows):
                self.track_rows[track_index]["note"].set(pitch)
            for step in track.steps:
                step.note = pitch
        else:
            pitch = self._track_pitch(track_index)

        if should_enable:
            if existing:
                for step in existing:
                    step.note = pitch
            else:
                track.steps.append(Step(position=position, note=pitch))
                track.steps.sort(key=lambda item: item.position)
        elif existing:
            track.steps = [step for step in track.steps if step.position != position]

        self._refresh_step_button(track_index, position)
        self._publish_workspace("set_step", track=track_index, step=position, enabled=should_enable)
        return should_enable

    def set_track(self, track_index: int, updates: dict[str, Any]) -> dict[str, Any]:
        track_index = _strict_int(track_index, "track", 0, len(self.project.tracks) - 1)
        supported = {"name", "waveform", "note", "gain", "muted", "attack_seconds", "release_seconds"}
        unknown = set(updates) - supported
        if unknown:
            raise ValueError(f"unsupported track fields: {', '.join(sorted(unknown))}")
        if not updates:
            raise ValueError("set_track needs at least one editable field")

        track = self.project.tracks[track_index]
        row = self.track_rows[track_index] if track_index < len(self.track_rows) else None
        if "name" in updates:
            name = str(updates["name"] or "").strip()
            if not name:
                raise ValueError("track name must not be empty")
            track.name = name[:80]
            if row is not None:
                row["name"].set(track.name)
        if "waveform" in updates:
            waveform = str(updates["waveform"] or "").strip().lower()
            if waveform not in SUPPORTED_WAVEFORMS:
                raise ValueError(f"waveform must be one of: {', '.join(WAVEFORMS)}")
            track.waveform = waveform
            if row is not None:
                row["waveform"].set(waveform)
        if "note" in updates:
            note = _strict_int(updates["note"], "note", 0, 127)
            for step in track.steps:
                step.note = note
            if row is not None:
                row["note"].set(note)
        if "gain" in updates:
            track.gain = _strict_float(updates["gain"], "gain", 0.0, 1.0)
            if row is not None:
                row["gain"].set(track.gain)
        if "muted" in updates:
            track.muted = _bool_value(updates["muted"], "muted")
        if "attack_seconds" in updates:
            track.attack_seconds = _strict_float(updates["attack_seconds"], "attack_seconds", 0.0, 5.0)
        if "release_seconds" in updates:
            track.release_seconds = _strict_float(updates["release_seconds"], "release_seconds", 0.0, 5.0)
        self.project.validate()
        self._publish_workspace("set_track", track=track_index)
        return self._track_payload(track_index)

    def _play_audio(self, audio: np.ndarray, sample_rate: int, *, loop: bool = False) -> None:
        if _sounddevice is None:
            raise RuntimeError("Playback needs the optional sounddevice package.")
        output = np.asarray(audio, dtype=np.float32)
        if output.ndim == 1:
            channels = 1
        elif output.ndim == 2 and output.shape[1] > 0:
            channels = output.shape[1]
        else:
            raise ValueError("Playback audio must be a mono vector or a sample-by-channel matrix.")

        query_devices = getattr(_sounddevice, "query_devices", None)
        if callable(query_devices):
            device_label = (
                f"configured output device {self.output_device}"
                if self.output_device is not None
                else "the default output device"
            )
            try:
                device_info = query_devices(self.output_device, "output")
                max_output_channels = int(device_info.get("max_output_channels", 0))
            except Exception as exc:
                raise RuntimeError(f"Could not use {device_label}: {exc}") from exc
            if max_output_channels < 1:
                raise RuntimeError(f"{device_label.capitalize()} has no output channels.")
            if channels > max_output_channels:
                raise RuntimeError(
                    f"{device_label.capitalize()} supports {max_output_channels} output "
                    f"channel(s), but the audio has {channels}."
                )
            # Some ALSA/PipeWire stereo endpoints reject a one-channel PortAudio
            # stream even though the source material itself is mono.
            if channels == 1 and max_output_channels >= 2:
                mono = output if output.ndim == 1 else output[:, 0]
                output = np.repeat(mono[:, np.newaxis], 2, axis=1)
                channels = 2

        check_output_settings = getattr(_sounddevice, "check_output_settings", None)
        if callable(check_output_settings):
            settings: dict[str, Any] = {
                "samplerate": sample_rate,
                "channels": channels,
                "dtype": output.dtype.name,
            }
            if self.output_device is not None:
                settings["device"] = self.output_device
            try:
                check_output_settings(**settings)
            except Exception as exc:
                device_label = (
                    f"configured output device {self.output_device}"
                    if self.output_device is not None
                    else "default output device"
                )
                raise RuntimeError(
                    f"The {device_label} cannot play {channels}-channel audio at "
                    f"{sample_rate} Hz: {exc}"
                ) from exc

        options: dict[str, Any] = {"samplerate": sample_rate}
        if loop:
            options["loop"] = True
        if self.output_device is not None:
            options["device"] = self.output_device
        _sounddevice.play(output, **options)

    def _stop_audio_device(self) -> Optional[str]:
        if _sounddevice is None:
            return None
        try:
            _sounddevice.stop()
        except Exception as exc:
            return str(exc)
        return None

    def _begin_transport(self) -> int:
        generation, _warning = self._transport_gate.invalidate(self._stop_audio_device)
        self._playing = False
        return generation

    def _cancel_transport(self, generation: int) -> Optional[str]:
        cancelled, _next_generation, warning = self._transport_gate.invalidate_if_current(
            generation,
            self._stop_audio_device,
        )
        if cancelled:
            self._playing = False
            return warning
        return None

    def preview_note(self, note: int) -> bool:
        note = _strict_int(note, "note", 0, 127)
        waveform = self.preview_waveform_var.get().strip().lower()
        if waveform not in SUPPORTED_WAVEFORMS:
            raise ValueError(f"waveform must be one of: {', '.join(WAVEFORMS)}")
        sample_rate = self.project.sample_rate
        generation = self._begin_transport()

        def work() -> tuple[int, bool]:
            if not self._transport_gate.is_current(generation):
                return 0, True
            audio = synthesize_note(
                note,
                0.35,
                sample_rate,
                waveform=waveform,
                velocity=0.55,
                release_seconds=0.08,
                seed=note,
            )
            played, _value = self._transport_gate.run_if_current(
                generation,
                lambda: self._play_audio(audio, sample_rate),
            )
            return len(audio), not played

        def done(result: tuple[int, bool]) -> None:
            _sample_count, cancelled = result
            if cancelled or not self._transport_gate.is_current(generation):
                return
            self._set_status(f"Preview: {midi_note_label(note)} · {waveform}")

        def failed(exc: BaseException) -> None:
            if self._transport_gate.is_current(generation):
                self._cancel_transport(generation)
                self._show_error("Could not preview note", exc)

        accepted = self._background(
            f"Previewing {midi_note_label(note)}…",
            work,
            done,
            failed,
        )
        if not accepted:
            self._cancel_transport(generation)
        return accepted

    def _project_snapshot(self) -> DawProject:
        self._commit_controls()
        return DawProject.from_dict(self.project.to_dict())

    def render_to_library(self) -> None:
        try:
            snapshot = self._project_snapshot()
        except Exception as exc:
            self._show_error("Cannot render project", exc)
            return
        destination = self.paths.renders / f"{safe_filename_stem(snapshot.name)}_{_utc_timestamp()}.wav"

        def work() -> tuple[Path, dict]:
            with self._render_lock:
                path = export_project_wav(snapshot, destination, base_path=self.paths.root)
                return path, ensure_opus_sidecar(path)

        def done(result: tuple[Path, dict]) -> None:
            path, opus_sidecar = result
            self.last_render_path = path
            self._set_status(f"Rendered WAV: {path.name}")
            self._publish_workspace("rendered", opus_sidecar=opus_sidecar)

        self._background("Rendering arrangement…", work, done)

    def play_project(self) -> bool:
        generation = self._begin_transport()
        try:
            snapshot = self._project_snapshot()
            loop = bool(self.loop_var.get())
        except Exception as exc:
            self._cancel_transport(generation)
            self._show_error("Cannot play project", exc)
            return False

        def work() -> tuple[int, bool]:
            if not self._transport_gate.is_current(generation):
                return 0, True
            with self._render_lock:
                if not self._transport_gate.is_current(generation):
                    return 0, True
                audio = render_project(snapshot, base_path=self.paths.root)
            played, _value = self._transport_gate.run_if_current(
                generation,
                lambda: self._play_audio(audio, snapshot.sample_rate, loop=loop),
            )
            if not played:
                return len(audio), True
            if not loop:
                wait_for_audio = getattr(_sounddevice, "wait", None)
                if callable(wait_for_audio):
                    wait_for_audio()
            return len(audio), False

        def done(result: tuple[int, bool]) -> None:
            sample_count, cancelled = result
            if cancelled or not self._transport_gate.is_current(generation):
                return
            if loop:
                self._playing = True
                self._set_status(f"Playing {snapshot.name} · looping · {sample_count:,} samples")
                self._publish_workspace("play")
            else:
                self._playing = False
                self._set_status(f"Playback finished: {snapshot.name} · {sample_count:,} samples")
                self._publish_workspace("playback_finished")

        def failed(exc: BaseException) -> None:
            if self._transport_gate.is_current(generation):
                self._cancel_transport(generation)
                self._show_error("Could not play project", exc)

        accepted = self._background("Rendering for playback…", work, done, failed)
        if accepted:
            self._playing = True
            self._publish_workspace("play_scheduled")
        else:
            self._cancel_transport(generation)
        return accepted

    def stop_playback(self) -> None:
        _generation, warning = self._transport_gate.invalidate(self._stop_audio_device)
        self._playing = False
        if warning:
            self._set_status(f"Playback marked stopped; device warning: {warning}")
            self._publish_workspace("stop", warning=warning)
        else:
            self._set_status("Playback stopped.")
            self._publish_workspace("stop")

    def _safe_dialog_path(self, value: str, folder: Path, suffix: str) -> Optional[Path]:
        if not value:
            return None
        path = Path(value).expanduser()
        if path.suffix.lower() != suffix.lower():
            path = path.with_suffix(suffix)
        if not path_is_within(path, folder):
            messagebox.showerror("Studio folder only", f"Choose a file inside:\n{folder}", parent=self)
            return None
        return path

    def save_project_dialog(self) -> None:
        try:
            snapshot = self._project_snapshot()
        except Exception as exc:
            self._show_error("Cannot save project", exc)
            return
        initial = self.project_path.name if self.project_path else f"{safe_filename_stem(snapshot.name)}.ina-daw.json"
        selected = filedialog.asksaveasfilename(
            parent=self,
            title="Save Ina music project",
            initialdir=self.paths.projects,
            initialfile=initial,
            defaultextension=".json",
            filetypes=[("Ina DAW project", "*.json")],
        )
        path = self._safe_dialog_path(selected, self.paths.projects, ".json")
        if path is not None:
            self._save_snapshot(snapshot, path)

    def _save_snapshot(
        self,
        snapshot: DawProject,
        path: Path,
        *,
        after_save: Optional[Callable[[Path], None]] = None,
    ) -> bool:
        def done(saved: Path) -> None:
            self.project_path = saved
            self._set_status(f"Project saved: {saved.name}")
            self._publish_workspace("saved")
            if after_save is not None:
                after_save(saved)

        return self._background("Saving project…", lambda: save_project(snapshot, path), done)

    def save_and_close(self, filename: Any = None) -> Optional[Path]:
        """Save to the studio project folder, then close only after success."""
        try:
            snapshot = self._project_snapshot()
        except Exception as exc:
            self._show_error("Cannot save project", exc)
            return None
        if filename not in (None, ""):
            name = safe_filename_stem(filename) + ".ina-daw.json"
            path = self.paths.projects / name
        elif self.project_path is not None and path_is_within(self.project_path, self.paths.projects):
            path = self.project_path
        else:
            path = self.paths.projects / (
                safe_filename_stem(snapshot.name) + ".ina-daw.json"
            )
        accepted = self._save_snapshot(snapshot, path, after_save=lambda _saved: self.close_window())
        return path if accepted else None

    def load_project_dialog(self) -> None:
        if self._stem_import_pending:
            self._set_status("Wait for the stem import to finish before loading another project.")
            return
        if self._project_load_pending:
            self._set_status("A project load is already in progress.")
            return
        if self.recorder.recording or self._recording_pending:
            self._set_status("Stop the microphone take before loading another project.")
            try:
                messagebox.showwarning(
                    "Microphone recording active",
                    "Stop and add or cancel the current microphone take before loading a project.",
                    parent=self,
                )
            except tk.TclError:
                pass
            return
        selected = filedialog.askopenfilename(
            parent=self,
            title="Load Ina music project",
            initialdir=self.paths.projects,
            filetypes=[("Ina DAW project", "*.json")],
        )
        path = self._safe_dialog_path(selected, self.paths.projects, ".json")
        if path is None:
            return

        self._project_load_pending = True
        self._publish_workspace("project_load_scheduled")

        def work() -> DawProject:
            loaded = load_project(path)
            if loaded.total_steps != STEP_COUNT:
                raise ValueError("This first studio window currently supports 16-step projects.")
            placed_groups = (
                ("Vocal clip", loaded.vocal_clips, self.paths.root, False),
                ("Stem", loaded.audio_stems, self.paths.stems, True),
            )
            for label, clips, allowed_root, require_regular_wav in placed_groups:
                for clip in clips:
                    resolved = Path(clip.path)
                    if not resolved.is_absolute():
                        resolved = self.paths.root / resolved
                    if not path_is_within(resolved, allowed_root):
                        raise ValueError(
                            f"{label} lies outside its studio folder: {clip.path}"
                        )
                    if require_regular_wav and (
                        resolved.is_symlink()
                        or not resolved.is_file()
                        or resolved.suffix.casefold() != ".wav"
                    ):
                        raise ValueError(f"{label} is not a regular local WAV file: {clip.path}")
            return loaded

        def done(loaded: DawProject) -> None:
            self._project_load_pending = False
            if self.recorder.recording or self._recording_pending:
                self._set_status("Project load skipped because a microphone take started.")
                self._publish_workspace("project_load_skipped")
                return
            self.project = loaded
            self.project_path = path
            self.project_name_var.set(loaded.name)
            self.bpm_var.set(loaded.bpm)
            self.recorder.sample_rate = loaded.sample_rate
            self._rebuild_track_grid()
            self._refresh_vocal_list()
            self._refresh_stem_list()
            for stem in loaded.audio_stems:
                if stem.collection:
                    self._known_stem_collections.setdefault(stem.collection, "")
            self._set_status(f"Project loaded: {path.name}")
            self._publish_workspace("loaded")

        def failed(exc: BaseException) -> None:
            self._project_load_pending = False
            self._show_error("Could not load project", exc)

        accepted = self._background("Loading project…", work, done, failed)
        if not accepted:
            self._project_load_pending = False

    def export_wav_dialog(self) -> None:
        try:
            snapshot = self._project_snapshot()
        except Exception as exc:
            self._show_error("Cannot export project", exc)
            return
        selected = filedialog.asksaveasfilename(
            parent=self,
            title="Export arrangement WAV",
            initialdir=self.paths.renders,
            initialfile=f"{safe_filename_stem(snapshot.name)}.wav",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
        )
        path = self._safe_dialog_path(selected, self.paths.renders, ".wav")
        if path is not None:
            self._export_snapshot(snapshot, path)

    def _export_snapshot(self, snapshot: DawProject, path: Path) -> bool:
        def work() -> tuple[Path, dict]:
            with self._render_lock:
                exported = export_project_wav(snapshot, path, base_path=self.paths.root)
                return exported, ensure_opus_sidecar(exported)

        def done(result: tuple[Path, dict]) -> None:
            exported, opus_sidecar = result
            self.last_render_path = exported
            self._set_status(f"Exported WAV: {exported.name}")
            self._publish_workspace("exported", opus_sidecar=opus_sidecar)

        return self._background("Exporting WAV…", work, done)

    def _set_recording_controls(self, *, active: bool) -> None:
        self.start_record_button.configure(state=tk.DISABLED if active else tk.NORMAL)
        self.stop_record_button.configure(state=tk.NORMAL if active else tk.DISABLED)

    def start_recording(self) -> bool:
        if self.recorder.recording or self._recording_pending:
            self._set_status("A microphone take is already active or opening.")
            return False
        self._recording_pending = True
        self.start_record_button.configure(state=tk.DISABLED)
        self.stop_record_button.configure(state=tk.DISABLED)

        def done(_value: None) -> None:
            self._recording_pending = False
            active = self.recorder.recording
            self._set_recording_controls(active=active)
            if not active:
                self._set_status("Microphone opening was cancelled.")
                self._publish_workspace("recording_cancelled")
                return
            seconds = self.recorder.max_capture_seconds
            self._set_status(
                f"Microphone recording at {self.recorder.recording_sample_rate:,} Hz "
                f"(capture cap {seconds:.1f}s)… press ‘Stop and add’ when ready."
            )
            self._publish_workspace("recording_started")

        def failed(exc: BaseException) -> None:
            self._recording_pending = False
            self._set_recording_controls(active=self.recorder.recording)
            self._show_error("Microphone unavailable", exc)

        accepted = self._background("Opening microphone…", self.recorder.start, done, failed)
        if not accepted:
            self._recording_pending = False
            self._set_recording_controls(active=self.recorder.recording)
        return accepted

    def stop_recording(self) -> bool:
        if not self.recorder.recording:
            self._set_status("No microphone recording is active.")
            return False
        offset = _float_value(self.vocal_offset_var.get(), 0.0, 0.0, 64.0)
        destination = self.paths.recordings / f"mic_{_utc_timestamp()}.wav"
        take_sample_rate = self.recorder.recording_sample_rate
        capture_limit_seconds = self.recorder.max_capture_seconds
        self.stop_record_button.configure(state=tk.DISABLED)

        def work() -> tuple[Path, bool]:
            audio = self.recorder.stop()
            capped = self.recorder.limit_reached
            if self._shutdown_event.is_set():
                raise RuntimeError("Music studio closed before the microphone take was saved.")
            return write_wav(destination, audio, take_sample_rate), capped

        def done(result: tuple[Path, bool]) -> None:
            path, capped = result
            self._set_recording_controls(active=False)
            self._add_vocal_path(path, offset, f"Mic take {len(self.project.vocal_clips) + 1}")
            suffix = f" (capture limit {capture_limit_seconds:.1f}s reached)" if capped else ""
            self._set_status(
                f"Mic take added at beat {offset:g}: {path.name} · {take_sample_rate:,} Hz{suffix}"
            )

        def failed(exc: BaseException) -> None:
            self._set_recording_controls(active=self.recorder.recording)
            self._show_error("Could not finish recording", exc)

        accepted = self._background("Saving microphone take…", work, done, failed)
        if not accepted:
            self._set_recording_controls(active=self.recorder.recording)
        return accepted

    def generate_symbolic_vocal(self) -> None:
        try:
            self._schedule_symbolic_vocal(
                self.symbolic_prompt_var.get(),
                _float_value(self.vocal_offset_var.get(), 0.0, 0.0, 64.0),
            )
        except Exception as exc:
            self._show_error("Cannot generate local symbols", exc)

    def _schedule_symbolic_vocal(self, prompt: Any, offset: Any) -> tuple[Path, bool]:
        text = str(prompt or "").strip()
        if not text:
            raise ValueError("Enter a short prompt for Ina's local symbol vocabulary.")
        if len(text) > 1000:
            raise ValueError("symbolic vocal prompt must be at most 1000 characters")
        beat_offset = _strict_float(offset, "offset", 0.0, 64.0)
        destination = self.paths.recordings / f"local_symbols_{_utc_timestamp()}.wav"

        def work() -> tuple[Path, dict[str, Any]]:
            payload = render_local_symbolic_vocal(
                text,
                child=self.child,
                output_path=destination,
            )
            return destination, payload

        def done(result: tuple[Path, dict[str, Any]]) -> None:
            path, payload = result
            symbols = payload.get("symbols") or []
            self._add_vocal_path(path, beat_offset, f"Local symbols: {text[:38]}")
            self._set_status(
                f"Added {len(symbols)} local sound symbol(s) at beat {beat_offset:g}. No network call was made."
            )

        accepted = self._background("Rendering local sound symbols…", work, done)
        return destination, accepted

    def _add_vocal_path(self, path: Path, offset: float, name: str) -> None:
        if not path_is_within(path, self.paths.root):
            raise ValueError("Vocal WAV must stay inside the music studio folder.")
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_file() or resolved.suffix.casefold() != ".wav":
            raise ValueError("Vocal clip must be a regular local WAV file.")
        relative = resolved.relative_to(self.paths.root.resolve()).as_posix()
        clip = VocalClip(path=relative, offset_beats=offset, gain=1.0, name=name)

        # Validate a detached candidate first so a late stem/vocal completion
        # cannot leave the live project above its shared placed-WAV ceiling.
        candidate = DawProject.from_dict(self.project.to_dict())
        candidate.vocal_clips.append(clip)
        candidate.validate()
        self.project.vocal_clips = candidate.vocal_clips
        self._refresh_vocal_list()
        self._publish_workspace(
            "vocal_added",
            vocal_path=relative,
        )

    def _refresh_vocal_list(self) -> None:
        for item in self.vocal_tree.get_children():
            self.vocal_tree.delete(item)
        for index, clip in enumerate(self.project.vocal_clips):
            self.vocal_tree.insert(
                "",
                tk.END,
                iid=str(index),
                text=clip.name,
                values=(f"{clip.offset_beats:g}", f"{clip.gain:.2f}", clip.path),
            )

    def _selected_vocal_index(self) -> Optional[int]:
        selection = self.vocal_tree.selection()
        if not selection:
            self._set_status("Select a vocal clip first.")
            return None
        try:
            return int(selection[0])
        except (TypeError, ValueError):
            return None

    def play_selected_vocal(self) -> bool:
        index = self._selected_vocal_index()
        if index is None or not 0 <= index < len(self.project.vocal_clips):
            return False
        clip = self.project.vocal_clips[index]
        path = Path(clip.path)
        if not path.is_absolute():
            path = self.paths.root / path
        generation = self._begin_transport()

        def work() -> tuple[int, bool]:
            if not self._transport_gate.is_current(generation):
                return 0, True
            if _sounddevice is None:
                raise RuntimeError("Playback needs the optional sounddevice package.")
            audio, rate = read_wav(path)
            played, _value = self._transport_gate.run_if_current(
                generation,
                lambda: self._play_audio(audio * clip.gain, rate),
            )
            return len(audio), not played

        def done(result: tuple[int, bool]) -> None:
            _sample_count, cancelled = result
            if cancelled or not self._transport_gate.is_current(generation):
                return
            self._set_status(f"Playing vocal clip: {clip.name}")

        def failed(exc: BaseException) -> None:
            if self._transport_gate.is_current(generation):
                self._cancel_transport(generation)
                self._show_error("Could not play vocal clip", exc)

        accepted = self._background(f"Loading {clip.name}…", work, done, failed)
        if not accepted:
            self._cancel_transport(generation)
        return accepted

    def remove_selected_vocal(self) -> None:
        index = self._selected_vocal_index()
        if index is None or not 0 <= index < len(self.project.vocal_clips):
            return
        clip = self.project.vocal_clips.pop(index)
        self._refresh_vocal_list()
        self._set_status(f"Removed {clip.name} from the arrangement; its WAV was kept.")
        self._publish_workspace("vocal_removed")

    def _remaining_stem_slots(self) -> int:
        return max(
            0,
            min(
                MAX_AUDIO_STEMS - len(self.project.audio_stems),
                MAX_PLACED_WAVS
                - len(self.project.vocal_clips)
                - len(self.project.audio_stems),
            ),
        )

    def import_wav_stems_dialog(self) -> None:
        selected = filedialog.askopenfilenames(
            parent=self,
            title="Import aligned WAV stems",
            filetypes=[("PCM WAV stems", "*.wav")],
        )
        if not selected:
            return
        paths = tuple(Path(value).expanduser() for value in selected)
        collection = self.project_name_var.get().strip() or "Stem collection"
        slots = self._remaining_stem_slots()
        self._schedule_stem_import(
            "Importing selected WAV stems…",
            lambda: import_wav_stems(
                paths,
                self.paths.stems,
                collection_name=collection,
                maximum_stems=slots,
                cancelled=self._shutdown_event.is_set,
            ),
        )

    def import_stem_zip_dialog(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self,
            title="Import a WAV stem bundle ZIP",
            filetypes=[("Stem ZIP", "*.zip")],
        )
        if not selected:
            return
        source = Path(selected).expanduser()
        slots = self._remaining_stem_slots()
        self._schedule_stem_import(
            f"Importing {source.name}…",
            lambda: import_stem_zip(
                source,
                self.paths.stems,
                collection_name=source.stem,
                maximum_stems=slots,
                cancelled=self._shutdown_event.is_set,
            ),
        )

    def _schedule_stem_import(
        self,
        label: str,
        operation: Callable[[], StemImportResult],
    ) -> bool:
        if self._stem_import_pending:
            self._set_status("A stem import is already in progress.")
            return False
        if self._project_load_pending:
            self._set_status("Wait for the project load to finish before importing stems.")
            return False
        if self._remaining_stem_slots() <= 0:
            self._show_error(
                "Cannot import stems",
                ValueError("This project has reached its placed-WAV stem limit."),
            )
            return False
        self._stem_import_pending = True
        self._publish_workspace("stem_import_scheduled")

        def done(result: StemImportResult) -> None:
            self._stem_import_pending = False
            try:
                self._add_stem_import_result(result)
            except StemCollectionAttachmentError as exc:
                manifest = self._known_stem_collections.get(result.collection_name)
                self._set_status(
                    f"Imported {result.collection_name} to the local stem library, "
                    "but it could not be attached to the current project."
                )
                self._publish_workspace(
                    "stem_collection_imported_unattached",
                    stem_collection=result.collection_name,
                    imported_stems=len(result.stems),
                    stem_manifest=manifest,
                    attachment_error=str(exc),
                )
            except Exception as exc:
                self._show_error("Could not add imported stems", exc)

        def failed(exc: BaseException) -> None:
            self._stem_import_pending = False
            self._show_error("Could not import stem collection", exc)

        accepted = self._background(label, operation, done, failed)
        if not accepted:
            self._stem_import_pending = False
            self._publish_workspace("stem_import_rejected")
        return accepted

    def _add_stem_import_result(self, result: StemImportResult) -> None:
        root = self.paths.root.resolve()
        stems_root = self.paths.stems.resolve()
        collection = Path(result.collection_dir)
        if (
            collection.is_symlink()
            or not collection.is_dir()
            or not path_is_within(collection, stems_root)
        ):
            raise ValueError("Imported stem collection escaped the studio folder.")
        collection = collection.resolve()

        manifest_path = Path(result.manifest_path)
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or manifest_path.suffix.casefold() != ".json"
            or not path_is_within(manifest_path, collection)
        ):
            raise ValueError("Imported stem manifest is not a regular file in its collection.")
        manifest = manifest_path.resolve().relative_to(root).as_posix()

        for companion in result.companions:
            companion_path = Path(companion.path)
            if (
                companion_path.is_symlink()
                or not companion_path.is_file()
                or companion_path.suffix.casefold() != ".txt"
                or not path_is_within(companion_path, collection)
            ):
                raise ValueError(
                    f"Imported music-context file is invalid: {companion.name}"
                )

        self._known_stem_collections[result.collection_name] = manifest
        added: list[AudioStem] = []
        for imported in result.stems:
            imported_path = Path(imported.path)
            if (
                imported_path.is_symlink()
                or not imported_path.is_file()
                or imported_path.suffix.casefold() != ".wav"
                or not path_is_within(imported_path, collection)
            ):
                raise ValueError(f"Imported stem escaped the studio folder: {imported.name}")
            relative = imported_path.resolve().relative_to(root).as_posix()
            try:
                added.append(
                    AudioStem(
                        path=relative,
                        name=imported.name,
                        role=imported.role,
                        collection=result.collection_name,
                        offset_beats=0.0,
                        gain=1.0,
                    )
                )
            except ValueError as exc:
                raise StemCollectionAttachmentError(str(exc)) from exc

        if len(added) > self._remaining_stem_slots():
            raise StemCollectionAttachmentError(
                "The imported collection no longer fits in this project."
            )

        candidate = DawProject.from_dict(self.project.to_dict())
        candidate.audio_stems.extend(added)
        try:
            candidate.validate()
        except ValueError as exc:
            raise StemCollectionAttachmentError(str(exc)) from exc
        self.project.audio_stems = candidate.audio_stems
        self._refresh_stem_list()
        context_note = (
            f" Retained {len(result.companions)} music-context TXT file(s)."
            if result.companions
            else ""
        )
        self._set_status(
            f"Imported {len(added)} aligned stem(s) for {result.collection_name}."
            f"{context_note}"
        )
        self._publish_workspace(
            "stems_imported",
            stem_collection=result.collection_name,
            imported_stems=len(added),
            retained_context_files=len(result.companions),
            stem_manifest=manifest,
        )

    def _stem_payload(self, index: int) -> dict[str, Any]:
        stem = self.project.audio_stems[index]
        return {
            "index": index,
            "name": stem.name,
            "role": stem.role,
            "collection": stem.collection,
            "path": stem.path,
            "offset_beats": stem.offset_beats,
            "gain": stem.gain,
            "muted": stem.muted,
            "solo": stem.solo,
        }

    def _refresh_stem_list(self) -> None:
        selected = self.stem_tree.selection()
        selected_iid = selected[0] if selected else None
        for item in self.stem_tree.get_children():
            self.stem_tree.delete(item)
        for index, stem in enumerate(self.project.audio_stems):
            states = []
            if stem.muted:
                states.append("muted")
            if stem.solo:
                states.append("solo")
            self.stem_tree.insert(
                "",
                tk.END,
                iid=str(index),
                text=stem.name,
                values=(
                    stem.role,
                    f"{stem.offset_beats:g}",
                    f"{stem.gain:.2f}",
                    " · ".join(states) or "active",
                    stem.path,
                ),
            )
        if selected_iid in self.stem_tree.get_children():
            self.stem_tree.selection_set(selected_iid)

    def _selected_stem_index(self, *, notify: bool = True) -> Optional[int]:
        selection = self.stem_tree.selection()
        if not selection:
            if notify:
                self._set_status("Select a stem first.")
            return None
        try:
            index = int(selection[0])
        except (TypeError, ValueError):
            return None
        return index if 0 <= index < len(self.project.audio_stems) else None

    def _load_selected_stem_controls(self, _event: Any = None) -> None:
        index = self._selected_stem_index(notify=False)
        if index is None:
            return
        stem = self.project.audio_stems[index]
        self.stem_gain_var.set(stem.gain)
        self.stem_offset_var.set(stem.offset_beats)
        self.stem_role_var.set(stem.role)

    def set_stem(self, index: int, updates: dict[str, Any]) -> dict[str, Any]:
        index = _strict_int(index, "stem", 0, len(self.project.audio_stems) - 1)
        supported = {"name", "role", "gain", "offset_beats", "muted", "solo"}
        unknown = set(updates) - supported
        if unknown:
            raise ValueError(f"unsupported stem fields: {', '.join(sorted(unknown))}")
        if not updates:
            raise ValueError("set_stem needs at least one editable field")
        values = self.project.audio_stems[index].to_dict()
        if "name" in updates:
            name = str(updates["name"] or "").strip()
            if not name:
                raise ValueError("stem name must not be empty")
            values["name"] = name[:120]
        if "role" in updates:
            role = str(updates["role"] or "").strip()
            if not role:
                raise ValueError("stem role must not be empty")
            values["role"] = role[:80]
        if "gain" in updates:
            values["gain"] = _strict_float(updates["gain"], "stem gain", 0.0, 4.0)
        if "offset_beats" in updates:
            values["offset_beats"] = _strict_float(
                updates["offset_beats"], "stem beat offset", 0.0, 64.0
            )
        if "muted" in updates:
            values["muted"] = _bool_value(updates["muted"], "stem muted")
        if "solo" in updates:
            values["solo"] = _bool_value(updates["solo"], "stem solo")
        self.project.audio_stems[index] = AudioStem.from_dict(values)
        self.project.validate()
        self._refresh_stem_list()
        self._publish_workspace("set_stem", stem=index)
        return self._stem_payload(index)

    def apply_selected_stem_mix(self) -> None:
        index = self._selected_stem_index()
        if index is None:
            return
        try:
            payload = self.set_stem(
                index,
                {
                    "gain": self.stem_gain_var.get(),
                    "offset_beats": self.stem_offset_var.get(),
                    "role": self.stem_role_var.get(),
                },
            )
            self._set_status(f"Updated stem mix: {payload['name']}")
        except Exception as exc:
            self._show_error("Could not update stem mix", exc)

    def toggle_selected_stem_mute(self) -> None:
        index = self._selected_stem_index()
        if index is None:
            return
        stem = self.project.audio_stems[index]
        payload = self.set_stem(index, {"muted": not stem.muted})
        self._set_status(f"{payload['name']}: {'muted' if payload['muted'] else 'active'}")

    def toggle_selected_stem_solo(self) -> None:
        index = self._selected_stem_index()
        if index is None:
            return
        stem = self.project.audio_stems[index]
        payload = self.set_stem(index, {"solo": not stem.solo})
        self._set_status(f"{payload['name']}: {'solo' if payload['solo'] else 'normal mix'}")

    def _preview_stem_index(self, index: int) -> bool:
        index = _strict_int(index, "stem", 0, len(self.project.audio_stems) - 1)
        stem = self.project.audio_stems[index]
        path = Path(stem.path)
        if not path.is_absolute():
            path = self.paths.root / path
        if not path_is_within(path, self.paths.stems):
            raise ValueError("Stem WAV must stay inside the studio stem folder.")
        generation = self._begin_transport()

        def work() -> tuple[int, bool]:
            if not self._transport_gate.is_current(generation):
                return 0, True
            if _sounddevice is None:
                raise RuntimeError("Playback needs the optional sounddevice package.")
            audio, rate = read_wav(path)
            played, _value = self._transport_gate.run_if_current(
                generation,
                lambda: self._play_audio(audio * stem.gain, rate),
            )
            return len(audio), not played

        def done(result: tuple[int, bool]) -> None:
            _sample_count, cancelled = result
            if cancelled or not self._transport_gate.is_current(generation):
                return
            self._set_status(f"Playing stem: {stem.name}")

        def failed(exc: BaseException) -> None:
            if self._transport_gate.is_current(generation):
                self._cancel_transport(generation)
                self._show_error("Could not play stem", exc)

        accepted = self._background(f"Loading {stem.name}…", work, done, failed)
        if not accepted:
            self._cancel_transport(generation)
        return accepted

    def play_selected_stem(self) -> bool:
        index = self._selected_stem_index()
        return False if index is None else self._preview_stem_index(index)

    def remove_selected_stem(self) -> None:
        index = self._selected_stem_index()
        if index is None:
            return
        stem = self.project.audio_stems.pop(index)
        self._refresh_stem_list()
        self._set_status(
            f"Removed {stem.name} from the arrangement; its local collection was kept."
        )
        self._publish_workspace("stem_removed", stem=index)

    def _background(
        self,
        label: str,
        operation: Callable[[], Any],
        on_success: Optional[Callable[[Any], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> bool:
        if self._closing:
            return False
        self._set_status(label)

        def run() -> None:
            try:
                result = operation()
            except BaseException as exc:
                self._after_safe(lambda error=exc: (on_error or self._background_error)(error))
            else:
                if on_success is not None:
                    def deliver(value: Any = result) -> None:
                        try:
                            on_success(value)
                        except BaseException as exc:
                            (on_error or self._background_error)(exc)

                    self._after_safe(deliver)

        try:
            future = self._jobs.submit(run)
        except BaseException as exc:
            (on_error or self._background_error)(exc)
            return False
        if future is None:
            error = RuntimeError(
                f"Music studio is busy (maximum {DAW_BACKGROUND_JOB_LIMIT} active or queued tasks)."
            )
            (on_error or self._background_error)(error)
            return False
        return True

    def _after_safe(self, callback: Callable[[], None]) -> None:
        if self._closing:
            return
        try:
            self.after(0, callback)
        except tk.TclError:
            pass

    def _background_error(self, exc: BaseException) -> None:
        self._show_error("Music studio task failed", exc)

    def _show_error(self, title: str, error: BaseException) -> None:
        self._set_status(f"{title}: {error}")
        try:
            messagebox.showerror(title, str(error), parent=self)
        except tk.TclError:
            pass
        self._publish_workspace("error", error=str(error))

    def _set_status(self, text: str) -> None:
        self.status_var.set(str(text))

    def _workspace_payload(self, event: str, **extra: Any) -> dict[str, Any]:
        try:
            self._commit_controls()
        except Exception:
            pass
        known_collections = dict(getattr(self, "_known_stem_collections", {}))
        for stem in self.project.audio_stems:
            if stem.collection:
                known_collections.setdefault(stem.collection, "")
        stem_library = sorted(
            known_collections.items(),
            key=lambda item: item[0].casefold(),
        )[:128]
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "window_open": not self._closing,
            "child": self.child,
            "project": {
                "name": self.project.name,
                "path": str(self.project_path) if self.project_path else None,
                "bpm": self.project.bpm,
                "loop": bool(self.loop_var.get()),
                "steps": STEP_COUNT,
                "playing": self._playing,
                "recording": self.recorder.recording,
                "recording_pending": self._recording_pending,
                "stem_import_pending": self._stem_import_pending,
                "project_load_pending": self._project_load_pending,
                "recording_sample_rate": self.recorder.recording_sample_rate,
                "recording_limit_seconds": round(self.recorder.max_capture_seconds, 3),
                "recording_limit_reached": self.recorder.limit_reached,
                "last_render": str(self.last_render_path) if self.last_render_path else None,
            },
            "audio_devices": {
                "input": self.input_device,
                "output": self.output_device,
                "launch_check": {
                    "input": self.input_device_resolution.to_payload(),
                    "output": self.output_device_resolution.to_payload(),
                },
            },
            "control_api": daw_control_api_payload(),
            "creative_experience": get_inastate("creative_experience_daw", child=self.child) or {},
            "tracks": [self._track_payload(index) for index in range(len(self.project.tracks))],
            "vocal_clips": [
                {
                    "index": index,
                    "name": clip.name,
                    "path": clip.path,
                    "offset_beats": clip.offset_beats,
                    "gain": clip.gain,
                    "muted": clip.muted,
                }
                for index, clip in enumerate(self.project.vocal_clips)
            ],
            "audio_stems": [
                self._stem_payload(index)
                for index in range(len(self.project.audio_stems))
            ],
            "stem_collections": [name for name, _manifest in stem_library],
            "stem_library": [
                {
                    "collection": name,
                    "manifest": manifest or None,
                }
                for name, manifest in stem_library
            ],
        }
        payload.update(extra)
        return payload

    def _publish_workspace(self, event: str, **extra: Any) -> dict[str, Any]:
        payload = self._workspace_payload(event, **extra)
        try:
            update_inastate(DAW_WORKSPACE_STATE_KEY, payload, child=self.child)
        except Exception:
            pass
        return payload

    def _set_window_open(self, value: bool) -> None:
        try:
            update_inastate(DAW_WINDOW_OPEN_KEY, bool(value), child=self.child)
        except Exception:
            pass

    def _poll_api_commands(self) -> None:
        if not self._api_poll_active or self._closing:
            return
        try:
            self._process_api_queue()
        finally:
            if self._api_poll_active and not self._closing:
                self.after(DAW_API_POLL_MS, self._poll_api_commands)

    def _process_api_queue(self) -> None:
        drained = drain_inastate_queue(
            DAW_COMMAND_QUEUE_KEY,
            batch_limit=DAW_API_MAX_COMMANDS,
            queue_limit=DAW_API_QUEUE_LIMIT,
            child=self.child,
        )
        batch = drained["batch"]
        invalid = bool(drained["invalid"])
        if not batch and not invalid:
            return
        if invalid:
            results = [{"status": "error", "error": "daw_command_queue must be an object or list"}]
            processed = 0
        else:
            results = [self._process_api_command(command) for command in batch]
            processed = len(results)
        update_inastate(
            DAW_LAST_COMMAND_RESULT_KEY,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processed": processed,
                "remaining": drained["remaining"],
                "dropped": drained["dropped"],
                "results": results,
            },
            child=self.child,
        )

    def _process_api_command(self, command: Any) -> dict[str, Any]:
        if not isinstance(command, dict):
            return {"status": "error", "error": "command must be an object"}
        action = str(command.get("action") or command.get("type") or "").strip().lower()
        command_id = str(command.get("id") or f"daw_cmd_{_utc_timestamp()}")
        try:
            if action in {"inspect", "state", "snapshot"}:
                result = {"status": "ok", "workspace": self._publish_workspace("inspect")}
            elif action == "experience_choice":
                current = get_inastate("creative_experience_daw", child=self.child) or {}
                if not isinstance(current, dict) or not current.get("cycle_id"):
                    raise RuntimeError("no active DAW experience cycle")
                updated = choose_next(current, command.get("choice"), reflection=command.get("reflection", ""))
                update_inastate("creative_experience_daw", updated, child=self.child)
                result = {"status": "ok", "creative_experience": updated}
            elif action == "set_step":
                track_index = self._resolve_command_track(command.get("track", command.get("track_index", 0)))
                position = _strict_int(command.get("position", command.get("step", 0)), "step", 0, STEP_COUNT - 1)
                enabled_raw = command.get("enabled")
                enabled = None if enabled_raw is None else _bool_value(enabled_raw, "enabled")
                note = None if "note" not in command else _strict_int(command["note"], "note", 0, 127)
                active = self.toggle_step(track_index, position, enabled=enabled, note=note)
                result = {
                    "status": "ok",
                    "track": track_index,
                    "step": position,
                    "enabled": active,
                    "note": self._track_pitch(track_index),
                }
            elif action == "set_track":
                track_index = self._resolve_command_track(command.get("track", command.get("track_index", 0)))
                editable = {"name", "waveform", "note", "gain", "muted", "attack_seconds", "release_seconds"}
                updates = {key: command[key] for key in editable if key in command}
                result = {"status": "ok", "track": self.set_track(track_index, updates)}
            elif action == "set_stem":
                stem_index = self._resolve_command_stem(
                    command.get("stem", command.get("stem_index", 0))
                )
                editable = {"name", "role", "gain", "offset_beats", "muted", "solo"}
                updates = {key: command[key] for key in editable if key in command}
                result = {"status": "ok", "stem": self.set_stem(stem_index, updates)}
            elif action == "preview_stem":
                stem_index = self._resolve_command_stem(
                    command.get("stem", command.get("stem_index", 0))
                )
                if not self._preview_stem_index(stem_index):
                    raise RuntimeError("stem preview was not scheduled")
                result = {"status": "scheduled", "stem": stem_index}
            elif action == "preview_note":
                if "note" not in command:
                    raise ValueError("preview_note needs a note")
                note = _strict_int(command["note"], "note", 0, 127)
                if "waveform" in command:
                    waveform = str(command["waveform"] or "").strip().lower()
                    if waveform not in SUPPORTED_WAVEFORMS:
                        raise ValueError(f"waveform must be one of: {', '.join(WAVEFORMS)}")
                    self.preview_waveform_var.set(waveform)
                accepted = self.preview_note(note)
                if not accepted:
                    raise RuntimeError("note preview was not scheduled")
                result = {"status": "scheduled", "note": note, "waveform": self.preview_waveform_var.get()}
            elif action == "generate_vocal":
                prompt = command.get("prompt", command.get("text", ""))
                offset = command.get("offset", command.get("offset_beats", 0.0))
                path, accepted = self._schedule_symbolic_vocal(prompt, offset)
                if not accepted:
                    raise RuntimeError("local symbolic vocal was not scheduled")
                result = {"status": "scheduled", "path": str(path), "kind": "local_symbolic_vocal"}
            elif action == "play":
                if "loop" in command:
                    self.loop_var.set(_bool_value(command["loop"], "loop"))
                if not self.play_project():
                    raise RuntimeError("project playback was not scheduled")
                result = {"status": "scheduled", "playing": True, "loop": bool(self.loop_var.get())}
            elif action == "stop":
                self.stop_playback()
                result = {"status": "ok", "playing": False}
            elif action == "save":
                snapshot = self._project_snapshot()
                filename = safe_filename_stem(command.get("filename") or snapshot.name) + ".ina-daw.json"
                path = self.paths.projects / filename
                if not self._save_snapshot(snapshot, path):
                    raise RuntimeError("project save was not scheduled")
                result = {"status": "scheduled", "path": str(path)}
            elif action in {"save_close", "save_and_close"}:
                path = self.save_and_close(command.get("filename"))
                if path is None:
                    raise RuntimeError("project save-and-close was not scheduled")
                result = {
                    "status": "scheduled", "path": str(path),
                    "saved": False, "closed": False, "closing_after_save": True,
                }
            elif action == "export":
                snapshot = self._project_snapshot()
                filename = safe_filename_stem(command.get("filename") or snapshot.name) + ".wav"
                path = self.paths.renders / filename
                if not self._export_snapshot(snapshot, path):
                    raise RuntimeError("WAV export was not scheduled")
                result = {"status": "scheduled", "path": str(path)}
            elif action in {"close", "done", "finish"}:
                save_before_close = _bool_value(command.get("save", True), "save")
                if save_before_close:
                    path = self.save_and_close(command.get("filename"))
                    if path is None:
                        raise RuntimeError("project save-and-close was not scheduled")
                    result = {
                        "status": "scheduled", "path": str(path),
                        "saved": False, "closed": False, "closing_after_save": True,
                    }
                else:
                    self.after(50, self.close_window)
                    result = {"status": "scheduled", "saved": False, "closed": True}
            else:
                result = {"status": "error", "error": f"unknown action: {action or 'missing'}"}
        except Exception as exc:
            result = {"status": "error", "error": str(exc)}
        result.update(
            {
                "id": command_id,
                "action": action or None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return result

    def _resolve_command_stem(self, value: Any) -> int:
        if not self.project.audio_stems:
            raise ValueError("The project has no imported stems.")
        if isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value.strip()):
            wanted = value.strip().casefold()
            for index, stem in enumerate(self.project.audio_stems):
                if stem.name.casefold() == wanted:
                    return index
            raise ValueError(f"Stem not found: {value}")
        return _strict_int(value, "stem", 0, len(self.project.audio_stems) - 1)

    def _resolve_command_track(self, value: Any) -> int:
        if not self.project.tracks:
            raise ValueError("The project has no instrument tracks.")
        if isinstance(value, str) and not re.fullmatch(r"[+-]?\d+", value.strip()):
            wanted = value.strip().casefold()
            for index, track in enumerate(self.project.tracks):
                if track.name.casefold() == wanted:
                    return index
            raise ValueError(f"Track not found: {value}")
        return _strict_int(value, "track", 0, len(self.project.tracks) - 1)

    def close_window(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._api_poll_active = False
        self._recording_pending = False
        self._stem_import_pending = False
        self._project_load_pending = False
        self._shutdown_event.set()
        self.recorder.close()
        self._transport_gate.invalidate(self._stop_audio_device)
        self._playing = False
        self._set_window_open(False)
        try:
            update_inastate(
                DAW_WORKSPACE_STATE_KEY,
                self._workspace_payload("closed", window_open=False),
                child=self.child,
            )
        except Exception:
            pass
        try:
            self._jobs.shutdown(on_drained=self._instance_lock.release)
        except Exception:
            # The executor's drain callback still owns lock release for running work.
            pass
        try:
            self.destroy()
        except tk.TclError:
            pass


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open Ina's local music studio.")
    parser.add_argument(
        "--child",
        type=validate_child_identifier,
        help="Explicit child identity to keep a scheduled studio launch immutable.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = load_config()
    workspace_child = str(args.child or cfg.get("current_child") or "Inazuma_Yagami")
    os.environ.update(launch_environment(workspace_child))
    try:
        window = DawWindow(child=args.child)
    except StudioAlreadyRunningError as exc:
        print(f"[Music Studio] {exc}")
        return
    try:
        window.mainloop()
    finally:
        if not window._closing:
            window.close_window()


if __name__ == "__main__":
    main()
