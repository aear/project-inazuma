"""Pure offline audio engine for Ina's small music studio.

The module deliberately has no GUI or audio-device dependencies.  Projects are
rendered to mono ``float32`` arrays and can be saved as ordinary PCM WAV files,
which keeps the future window, automation API, and tests independent from live
hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import wave

import numpy as np


PROJECT_SCHEMA_VERSION = 1
SUPPORTED_WAVEFORMS = frozenset({"sine", "square", "triangle", "saw", "noise"})

# These limits are deliberately generous for a small interactive studio while
# still putting a hard ceiling on malformed project files and accidental UI
# values. The render limit is an absolute mono-frame budget (about six minutes
# at 44.1 kHz or five and a half at 48 kHz). Synthesis and resampling use
# bounded scratch chunks so the cap also bounds transient working memory.
MAX_SAMPLE_RATE = 192_000
MAX_BPM = 1_000.0
MAX_BEATS_PER_BAR = 64
MAX_STEPS_PER_BEAT = 64
MAX_BARS = 4_096
MAX_TRACKS = 128
MAX_STEPS = 100_000
MAX_VOCAL_CLIPS = 256
MAX_PROJECT_JSON_BYTES = 32 * 1024 * 1024
MAX_GAIN = 16.0
MAX_ENVELOPE_SECONDS = 60.0
MAX_DURATION_STEPS = 65_536.0
MAX_OFFSET_BEATS = 100_000.0
MAX_RENDER_SAMPLES = 16_000_000
AUDIO_SCRATCH_CHUNK_SAMPLES = 262_144
MAX_WAV_CHANNELS = 32
MAX_WAV_DECODE_SAMPLES = MAX_RENDER_SAMPLES * 2


class DawValidationError(ValueError):
    """Raised when a project contains values the renderer cannot interpret."""


def _require_int(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DawValidationError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise DawValidationError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise DawValidationError(f"{name} must be at most {maximum}")
    return result


def _require_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool):
        raise DawValidationError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DawValidationError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise DawValidationError(f"{name} must be finite")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            relation = "at least" if minimum_inclusive else "greater than"
            raise DawValidationError(f"{name} must be {relation} {minimum}")
    if maximum is not None and result > maximum:
        raise DawValidationError(f"{name} must be at most {maximum}")
    return result


def _require_render_sample_count(value: int, name: str) -> int:
    count = _require_int(value, name, minimum=0)
    if count > MAX_RENDER_SAMPLES:
        raise DawValidationError(
            f"{name} exceeds the {MAX_RENDER_SAMPLES:,}-sample render budget"
        )
    return count


def _samples_for_seconds(
    seconds: Any,
    sample_rate: Any,
    name: str,
    *,
    minimum_one: bool = False,
) -> int:
    duration = _require_float(seconds, name, minimum=0.0)
    rate = _require_int(
        sample_rate, "sample rate", minimum=1, maximum=MAX_SAMPLE_RATE
    )
    exact_samples = duration * rate
    if not math.isfinite(exact_samples):
        raise DawValidationError(f"{name} produces a non-finite sample count")
    sample_count = round(exact_samples)
    if minimum_one:
        sample_count = max(1, sample_count)
    return _require_render_sample_count(sample_count, f"{name} sample count")


def _resampled_sample_count(
    source_size: int, source_sample_rate: int, target_sample_rate: int
) -> int:
    size = _require_int(source_size, "source sample count", minimum=0)
    source_rate = _require_int(
        source_sample_rate,
        "source sample rate",
        minimum=1,
        maximum=MAX_SAMPLE_RATE,
    )
    target_rate = _require_int(
        target_sample_rate,
        "target sample rate",
        minimum=1,
        maximum=MAX_SAMPLE_RATE,
    )
    if size == 0:
        return 0
    return _require_render_sample_count(
        max(1, round(size * target_rate / source_rate)),
        "resampled audio length",
    )


def _as_finite_float32_vector(
    audio: Sequence[float] | np.ndarray, name: str
) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            output = np.asarray(audio, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DawValidationError(f"{name} must contain numeric samples") from exc
    _require_render_sample_count(output.size, f"{name} length")
    if not np.all(np.isfinite(output)):
        raise DawValidationError(f"{name} must contain only finite samples")
    return output


@dataclass(eq=True)
class Step:
    """One note event at a zero-based sequencer position."""

    position: int
    note: int = 60
    velocity: float = 0.8
    duration_steps: float = 0.9

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.position = _require_int(self.position, "step position", minimum=0)
        self.note = _require_int(self.note, "MIDI note", minimum=0, maximum=127)
        self.velocity = _require_float(
            self.velocity, "step velocity", minimum=0.0, maximum=1.0
        )
        self.duration_steps = _require_float(
            self.duration_steps,
            "step duration",
            minimum=0.0,
            maximum=MAX_DURATION_STEPS,
            minimum_inclusive=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "note": self.note,
            "velocity": self.velocity,
            "duration_steps": self.duration_steps,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Step":
        if not isinstance(value, Mapping):
            raise DawValidationError("step must be an object")
        return cls(
            position=value.get("position"),
            note=value.get("note", 60),
            velocity=value.get("velocity", 0.8),
            duration_steps=value.get("duration_steps", 0.9),
        )


@dataclass(eq=True)
class InstrumentTrack:
    """A sparse monophonic-or-polyphonic step track.

    Multiple :class:`Step` objects may share a position, allowing chords without
    adding a second data model.
    """

    name: str = "Instrument"
    waveform: str = "sine"
    steps: list[Step] = field(default_factory=list)
    gain: float = 0.7
    attack_seconds: float = 0.008
    release_seconds: float = 0.04
    muted: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.name = str(self.name or "Instrument")
        self.waveform = str(self.waveform).strip().lower()
        if self.waveform not in SUPPORTED_WAVEFORMS:
            choices = ", ".join(sorted(SUPPORTED_WAVEFORMS))
            raise DawValidationError(
                f"unsupported waveform {self.waveform!r}; expected one of {choices}"
            )
        self.gain = _require_float(
            self.gain, "track gain", minimum=0.0, maximum=MAX_GAIN
        )
        self.attack_seconds = _require_float(
            self.attack_seconds,
            "track attack",
            minimum=0.0,
            maximum=MAX_ENVELOPE_SECONDS,
        )
        self.release_seconds = _require_float(
            self.release_seconds,
            "track release",
            minimum=0.0,
            maximum=MAX_ENVELOPE_SECONDS,
        )
        if not isinstance(self.muted, bool):
            raise DawValidationError("track muted must be a boolean")
        if not isinstance(self.steps, list):
            raise DawValidationError("track steps must be a list")
        if len(self.steps) > MAX_STEPS:
            raise DawValidationError(f"track steps must contain at most {MAX_STEPS:,} events")
        validated_steps: list[Step] = []
        for item in self.steps:
            step = item if isinstance(item, Step) else Step.from_dict(item)
            step.validate()
            validated_steps.append(step)
        self.steps = validated_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "waveform": self.waveform,
            "gain": self.gain,
            "attack_seconds": self.attack_seconds,
            "release_seconds": self.release_seconds,
            "muted": self.muted,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InstrumentTrack":
        if not isinstance(value, Mapping):
            raise DawValidationError("instrument track must be an object")
        return cls(
            name=value.get("name", "Instrument"),
            waveform=value.get("waveform", "sine"),
            gain=value.get("gain", 0.7),
            attack_seconds=value.get("attack_seconds", 0.008),
            release_seconds=value.get("release_seconds", 0.04),
            muted=value.get("muted", False),
            steps=value.get("steps", []),
        )


@dataclass(eq=True)
class VocalClip:
    """A WAV clip placed at an offset measured in musical beats."""

    path: str
    offset_beats: float = 0.0
    gain: float = 1.0
    muted: bool = False
    name: str = "Vocal"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.path, (str, os.PathLike)) or not str(self.path):
            raise DawValidationError("vocal clip path must not be empty")
        self.path = os.fspath(self.path)
        if not isinstance(self.path, str):
            raise DawValidationError("vocal clip path must be text")
        self.offset_beats = _require_float(
            self.offset_beats,
            "vocal offset",
            minimum=0.0,
            maximum=MAX_OFFSET_BEATS,
        )
        self.gain = _require_float(
            self.gain, "vocal gain", minimum=0.0, maximum=MAX_GAIN
        )
        if not isinstance(self.muted, bool):
            raise DawValidationError("vocal muted must be a boolean")
        self.name = str(self.name or "Vocal")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "offset_beats": self.offset_beats,
            "gain": self.gain,
            "muted": self.muted,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VocalClip":
        if not isinstance(value, Mapping):
            raise DawValidationError("vocal clip must be an object")
        return cls(
            name=value.get("name", "Vocal"),
            path=value.get("path"),
            offset_beats=value.get("offset_beats", 0.0),
            gain=value.get("gain", 1.0),
            muted=value.get("muted", False),
        )


@dataclass(eq=True)
class DawProject:
    """Serializable arrangement consumed by the offline renderer."""

    name: str = "Untitled"
    sample_rate: int = 44_100
    bpm: float = 120.0
    beats_per_bar: int = 4
    steps_per_beat: int = 4
    bars: int = 1
    tracks: list[InstrumentTrack] = field(default_factory=list)
    vocal_clips: list[VocalClip] = field(default_factory=list)
    master_gain: float = 0.85
    seed: int = 0

    def __post_init__(self) -> None:
        self.validate()

    @property
    def total_beats(self) -> int:
        return self.bars * self.beats_per_bar

    @property
    def total_steps(self) -> int:
        return self.total_beats * self.steps_per_beat

    @property
    def step_seconds(self) -> float:
        return 60.0 / (self.bpm * self.steps_per_beat)

    @property
    def base_duration_seconds(self) -> float:
        return self.total_beats * 60.0 / self.bpm

    def validate(self) -> None:
        self.name = str(self.name or "Untitled")
        self.sample_rate = _require_int(
            self.sample_rate,
            "sample rate",
            minimum=1,
            maximum=MAX_SAMPLE_RATE,
        )
        self.bpm = _require_float(
            self.bpm,
            "BPM",
            minimum=0.0,
            maximum=MAX_BPM,
            minimum_inclusive=False,
        )
        self.beats_per_bar = _require_int(
            self.beats_per_bar,
            "beats per bar",
            minimum=1,
            maximum=MAX_BEATS_PER_BAR,
        )
        self.steps_per_beat = _require_int(
            self.steps_per_beat,
            "steps per beat",
            minimum=1,
            maximum=MAX_STEPS_PER_BEAT,
        )
        self.bars = _require_int(
            self.bars, "bars", minimum=1, maximum=MAX_BARS
        )
        self.master_gain = _require_float(
            self.master_gain, "master gain", minimum=0.0, maximum=MAX_GAIN
        )
        self.seed = _require_int(
            self.seed, "project seed", minimum=0, maximum=(1 << 128) - 1
        )
        if not isinstance(self.tracks, list):
            raise DawValidationError("project tracks must be a list")
        if len(self.tracks) > MAX_TRACKS:
            raise DawValidationError(f"project tracks must contain at most {MAX_TRACKS} tracks")
        if not isinstance(self.vocal_clips, list):
            raise DawValidationError("project vocal clips must be a list")
        if len(self.vocal_clips) > MAX_VOCAL_CLIPS:
            raise DawValidationError(
                f"project vocal clips must contain at most {MAX_VOCAL_CLIPS} clips"
            )

        validated_tracks: list[InstrumentTrack] = []
        step_count = 0
        for item in self.tracks:
            track = (
                item
                if isinstance(item, InstrumentTrack)
                else InstrumentTrack.from_dict(item)
            )
            track.validate()
            step_count += len(track.steps)
            if step_count > MAX_STEPS:
                raise DawValidationError(
                    f"project must contain at most {MAX_STEPS:,} note events"
                )
            validated_tracks.append(track)
        self.tracks = validated_tracks

        validated_clips: list[VocalClip] = []
        for item in self.vocal_clips:
            clip = item if isinstance(item, VocalClip) else VocalClip.from_dict(item)
            clip.validate()
            validated_clips.append(clip)
        self.vocal_clips = validated_clips

        for track in self.tracks:
            for step in track.steps:
                if step.position >= self.total_steps:
                    raise DawValidationError(
                        f"step position {step.position} lies outside the "
                        f"{self.total_steps}-step project"
                    )
        _validate_project_timeline(self)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "name": self.name,
            "sample_rate": self.sample_rate,
            "bpm": self.bpm,
            "beats_per_bar": self.beats_per_bar,
            "steps_per_beat": self.steps_per_beat,
            "bars": self.bars,
            "master_gain": self.master_gain,
            "seed": self.seed,
            "tracks": [track.to_dict() for track in self.tracks],
            "vocal_clips": [clip.to_dict() for clip in self.vocal_clips],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DawProject":
        if not isinstance(value, Mapping):
            raise DawValidationError("project must be an object")
        schema_version = value.get("schema_version", PROJECT_SCHEMA_VERSION)
        if schema_version != PROJECT_SCHEMA_VERSION:
            raise DawValidationError(
                f"unsupported project schema version {schema_version!r}"
            )
        return cls(
            name=value.get("name", "Untitled"),
            sample_rate=value.get("sample_rate", 44_100),
            bpm=value.get("bpm", 120.0),
            beats_per_bar=value.get("beats_per_bar", 4),
            steps_per_beat=value.get("steps_per_beat", 4),
            bars=value.get("bars", 1),
            master_gain=value.get("master_gain", 0.85),
            seed=value.get("seed", 0),
            tracks=value.get("tracks", []),
            vocal_clips=value.get("vocal_clips", []),
        )


# A short name is convenient in small UI/controller modules.
Project = DawProject


def _validated_project(project: DawProject | Mapping[str, Any]) -> DawProject:
    if isinstance(project, DawProject):
        project.validate()
        return project
    if isinstance(project, Mapping):
        return DawProject.from_dict(project)
    raise DawValidationError("project must be an object")


def _step_sample_bounds(step: Step, project: DawProject) -> tuple[int, int, int]:
    start_seconds = step.position * project.step_seconds
    start = _samples_for_seconds(
        start_seconds, project.sample_rate, "note start"
    )
    duration_seconds = step.duration_steps * project.step_seconds
    duration_samples = _samples_for_seconds(
        duration_seconds, project.sample_rate, "note duration", minimum_one=True
    )
    end = _require_render_sample_count(start + duration_samples, "note end")
    return start, duration_samples, end


def _vocal_offset_sample(clip: VocalClip, project: DawProject) -> int:
    offset_seconds = clip.offset_beats * 60.0 / project.bpm
    return _samples_for_seconds(
        offset_seconds, project.sample_rate, "vocal offset"
    )


def _validate_project_timeline(project: DawProject) -> None:
    _samples_for_seconds(
        project.base_duration_seconds,
        project.sample_rate,
        "project duration",
        minimum_one=True,
    )
    for track in project.tracks:
        for step in track.steps:
            _step_sample_bounds(step, project)
    for clip in project.vocal_clips:
        _vocal_offset_sample(clip, project)


def midi_note_to_frequency(note: int, *, tuning_hz: float = 440.0) -> float:
    """Convert a MIDI note number to equal-tempered frequency."""

    midi_note = _require_int(note, "MIDI note", minimum=0, maximum=127)
    tuning = _require_float(
        tuning_hz,
        "tuning frequency",
        minimum=0.0,
        maximum=20_000.0,
        minimum_inclusive=False,
    )
    return tuning * (2.0 ** ((midi_note - 69) / 12.0))


def _chunk_ranges(sample_count: int):
    for start in range(0, sample_count, AUDIO_SCRATCH_CHUNK_SAMPLES):
        yield start, min(sample_count, start + AUDIO_SCRATCH_CHUNK_SAMPLES)


def _amplitude_envelope(
    sample_count: int,
    sample_rate: int,
    attack_seconds: float,
    release_seconds: float,
) -> np.ndarray:
    envelope = np.ones(sample_count, dtype=np.float32)
    if sample_count == 0:
        return envelope

    attack_samples = min(
        sample_count,
        _samples_for_seconds(attack_seconds, sample_rate, "attack"),
    )
    release_samples = min(
        sample_count,
        _samples_for_seconds(release_seconds, sample_rate, "release"),
    )
    total_edge_samples = attack_samples + release_samples
    if total_edge_samples > sample_count:
        scale = sample_count / total_edge_samples
        attack_samples = int(math.floor(attack_samples * scale))
        release_samples = sample_count - attack_samples

    if attack_samples:
        attack_denominator = np.float32(attack_samples)
        for start, end in _chunk_ranges(attack_samples):
            ramp = np.arange(start, end, dtype=np.float32)
            ramp /= attack_denominator
            envelope[start:end] = ramp
    if release_samples:
        if release_samples == 1:
            envelope[-1] = 0.0
        else:
            release_start = sample_count - release_samples
            release_denominator = np.float32(release_samples - 1)
            for start, end in _chunk_ranges(release_samples):
                ramp = np.arange(start, end, dtype=np.float32)
                ramp /= release_denominator
                ramp *= -1.0
                ramp += 1.0
                target_start = release_start + start
                target_end = release_start + end
                envelope[target_start:target_end] *= ramp
    return envelope


def _synthesize_waveform(
    sample_count: int,
    frequency: float,
    sample_rate: int,
    waveform: str,
    seed: int,
) -> np.ndarray:
    signal = np.empty(sample_count, dtype=np.float32)
    rng = np.random.default_rng(seed) if waveform == "noise" else None
    angular_step = 2.0 * np.pi * frequency / sample_rate
    cycle_step = frequency / sample_rate

    for start, end in _chunk_ranges(sample_count):
        if rng is not None:
            values = rng.uniform(-1.0, 1.0, end - start)
        else:
            values = np.arange(start, end, dtype=np.float64)
            if waveform == "saw":
                values *= cycle_step
                values += 0.5
                np.remainder(values, 1.0, out=values)
                values -= 0.5
                values *= 2.0
            else:
                values *= angular_step
                np.remainder(values, 2.0 * np.pi, out=values)
                np.sin(values, out=values)
                if waveform == "square":
                    values = np.where(values >= 0.0, 1.0, -1.0)
                elif waveform == "triangle":
                    np.arcsin(values, out=values)
                    values *= 2.0 / np.pi
        signal[start:end] = values
    return signal


def synthesize_note(
    note: int,
    duration_seconds: float,
    sample_rate: int,
    *,
    waveform: str = "sine",
    velocity: float = 1.0,
    attack_seconds: float = 0.008,
    release_seconds: float = 0.04,
    seed: int = 0,
) -> np.ndarray:
    """Render one deterministic note as a mono ``float32`` array."""

    frequency = midi_note_to_frequency(note)
    duration = _require_float(
        duration_seconds, "note duration", minimum=0.0, minimum_inclusive=False
    )
    rate = _require_int(
        sample_rate, "sample rate", minimum=1, maximum=MAX_SAMPLE_RATE
    )
    level = _require_float(velocity, "velocity", minimum=0.0, maximum=1.0)
    attack = _require_float(
        attack_seconds, "attack", minimum=0.0, maximum=MAX_ENVELOPE_SECONDS
    )
    release = _require_float(
        release_seconds, "release", minimum=0.0, maximum=MAX_ENVELOPE_SECONDS
    )
    shape = str(waveform).strip().lower()
    if shape not in SUPPORTED_WAVEFORMS:
        raise DawValidationError(f"unsupported waveform {shape!r}")
    random_seed = _require_int(
        seed, "synthesis seed", minimum=0, maximum=(1 << 128) - 1
    )

    sample_count = _samples_for_seconds(
        duration, rate, "note duration", minimum_one=True
    )
    signal = _synthesize_waveform(
        sample_count, frequency, rate, shape, random_seed
    )
    envelope = _amplitude_envelope(sample_count, rate, attack, release)
    signal *= envelope
    signal *= np.float32(level)
    return signal


def _stable_note_seed(project_seed: int, track_index: int, step: Step) -> int:
    payload = f"{project_seed}:{track_index}:{step.position}:{step.note}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "little")


def _base_sample_count(project: DawProject) -> int:
    return _samples_for_seconds(
        project.base_duration_seconds,
        project.sample_rate,
        "project duration",
        minimum_one=True,
    )


def render_instrument_track(
    track: InstrumentTrack,
    project: DawProject,
    *,
    track_index: int = 0,
) -> np.ndarray:
    """Render one instrument track without normalization or master gain."""

    project = _validated_project(project)
    index = _require_int(
        track_index, "track index", minimum=0, maximum=MAX_TRACKS - 1
    )
    if isinstance(track, InstrumentTrack):
        track.validate()
    else:
        track = InstrumentTrack.from_dict(track)
    for step in track.steps:
        if step.position >= project.total_steps:
            raise DawValidationError(
                f"step position {step.position} lies outside the "
                f"{project.total_steps}-step project"
            )
    if track.muted:
        return np.zeros(_base_sample_count(project), dtype=np.float32)

    output_samples = _base_sample_count(project)
    for step in track.steps:
        _, _, end = _step_sample_bounds(step, project)
        output_samples = max(output_samples, end)

    rendered = np.zeros(output_samples, dtype=np.float32)
    for step in track.steps:
        start, _, _ = _step_sample_bounds(step, project)
        duration = step.duration_steps * project.step_seconds
        audio = synthesize_note(
            step.note,
            duration,
            project.sample_rate,
            waveform=track.waveform,
            velocity=step.velocity * min(track.gain, 1.0),
            attack_seconds=track.attack_seconds,
            release_seconds=track.release_seconds,
            seed=_stable_note_seed(project.seed, index, step),
        )
        if track.gain > 1.0:
            audio = audio * track.gain
        rendered[start : start + len(audio)] += audio
    return _as_finite_float32_vector(rendered, "rendered instrument track")


def mix_audio(buffers: Iterable[Sequence[float] | np.ndarray]) -> np.ndarray:
    """Sum mono buffers from time zero, extending to the longest buffer."""

    arrays = [
        _as_finite_float32_vector(buffer, "audio buffer") for buffer in buffers
    ]
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    output_samples = _require_render_sample_count(
        max(len(buffer) for buffer in arrays), "mixed audio length"
    )
    mixed = np.zeros(output_samples, dtype=np.float32)
    for buffer in arrays:
        with np.errstate(over="ignore", invalid="ignore"):
            mixed[: len(buffer)] += buffer
        _as_finite_float32_vector(mixed, "mixed audio")
    return mixed


def resample_audio(
    audio: Sequence[float] | np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """Linearly resample a mono buffer without optional third-party packages."""

    source_rate = _require_int(
        source_sample_rate,
        "source sample rate",
        minimum=1,
        maximum=MAX_SAMPLE_RATE,
    )
    target_rate = _require_int(
        target_sample_rate,
        "target sample rate",
        minimum=1,
        maximum=MAX_SAMPLE_RATE,
    )
    source = _as_finite_float32_vector(audio, "source audio")
    if source_rate == target_rate or source.size == 0:
        return source.copy()
    target_length = _resampled_sample_count(
        source.size, source_rate, target_rate
    )
    if source.size == 1:
        return np.full(target_length, source[0], dtype=np.float32)
    output = np.empty(target_length, dtype=np.float32)
    position_scale = source_rate / target_rate
    final_source_index = source.size - 1
    for start, end in _chunk_ranges(target_length):
        positions = np.arange(start, end, dtype=np.float64)
        positions *= position_scale
        np.minimum(positions, final_source_index, out=positions)
        left_indices = positions.astype(np.int64)
        right_indices = np.minimum(left_indices + 1, final_source_index)
        fractions = positions - left_indices
        left_values = source[left_indices]
        right_values = source[right_indices]
        output[start:end] = left_values + (right_values - left_values) * fractions
    return _as_finite_float32_vector(output, "resampled audio")


def normalize_audio(
    audio: Sequence[float] | np.ndarray, *, peak: float = 0.98
) -> np.ndarray:
    """Reduce a buffer only when its absolute peak exceeds ``peak``."""

    target_peak = _require_float(
        peak, "normalization peak", minimum=0.0, maximum=1.0, minimum_inclusive=False
    )
    output = _as_finite_float32_vector(audio, "audio to normalize").copy()
    if output.size == 0:
        return output
    current_peak = 0.0
    for start, end in _chunk_ranges(output.size):
        chunk_peak = float(np.max(np.abs(output[start:end])))
        current_peak = max(current_peak, chunk_peak)
    if current_peak > target_peak:
        output *= target_peak / current_peak
    return _as_finite_float32_vector(output, "normalized audio")


def _resolve_clip_path(path: str, base_path: str | os.PathLike[str] | None) -> Path:
    clip_path = Path(path).expanduser()
    if not clip_path.is_absolute() and base_path is not None:
        clip_path = Path(base_path).expanduser() / clip_path
    return clip_path


def _extend_audio(audio: np.ndarray, output_samples: int) -> np.ndarray:
    required = _require_render_sample_count(output_samples, "rendered audio length")
    if required <= len(audio):
        return audio
    extended = np.zeros(required, dtype=np.float32)
    extended[: len(audio)] = audio
    return extended


def render_project(
    project: DawProject,
    *,
    include_vocals: bool = True,
    normalize: bool = True,
    peak: float = 0.98,
    base_path: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Render a project, extending it to include clip and note tails.

    Relative vocal paths are resolved against ``base_path``. A caller that
    loaded a project file should pass that project's parent directory.
    """

    project = _validated_project(project)
    rendered = np.zeros(_base_sample_count(project), dtype=np.float32)
    for track_index, track in enumerate(project.tracks):
        track_audio = render_instrument_track(
            track, project, track_index=track_index
        )
        rendered = _extend_audio(rendered, len(track_audio))
        with np.errstate(over="ignore", invalid="ignore"):
            rendered[: len(track_audio)] += track_audio
        _as_finite_float32_vector(rendered, "project mix")

    if include_vocals:
        for clip in project.vocal_clips:
            if clip.muted:
                continue
            offset_samples = _vocal_offset_sample(clip, project)
            clip_audio, clip_rate = _read_wav_for_render(
                _resolve_clip_path(clip.path, base_path),
                target_sample_rate=project.sample_rate,
                maximum_target_frames=MAX_RENDER_SAMPLES - offset_samples,
            )
            clip_audio = resample_audio(
                clip_audio, clip_rate, project.sample_rate
            )
            if clip_audio.size == 0:
                continue
            clip_end = _require_render_sample_count(
                offset_samples + len(clip_audio), "vocal clip end"
            )
            rendered = _extend_audio(rendered, clip_end)
            with np.errstate(over="ignore", invalid="ignore"):
                rendered[offset_samples:clip_end] += clip_audio * clip.gain
            _as_finite_float32_vector(rendered, "project mix")

    with np.errstate(over="ignore", invalid="ignore"):
        rendered *= project.master_gain
    _as_finite_float32_vector(rendered, "master output")
    if normalize:
        rendered = normalize_audio(rendered, peak=peak)
    return rendered.astype(np.float32, copy=False)


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8_388_608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")


def _read_wav_for_render(
    path: str | os.PathLike[str],
    *,
    target_sample_rate: int | None = None,
    maximum_target_frames: int | None = None,
) -> tuple[np.ndarray, int]:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError(f"compressed WAV is not supported: {wav_path}")
        channels = _require_int(
            source.getnchannels(),
            "WAV channel count",
            minimum=1,
            maximum=MAX_WAV_CHANNELS,
        )
        sample_width = _require_int(
            source.getsampwidth(), "WAV sample width", minimum=1, maximum=4
        )
        sample_rate = _require_int(
            source.getframerate(),
            "WAV sample rate",
            minimum=1,
            maximum=MAX_SAMPLE_RATE,
        )
        frame_count = _require_render_sample_count(
            source.getnframes(), "WAV frame count"
        )
        decoded_sample_count = frame_count * channels
        if decoded_sample_count > MAX_WAV_DECODE_SAMPLES:
            raise DawValidationError(
                "WAV channel data exceeds the decode-sample budget"
            )
        if target_sample_rate is not None or maximum_target_frames is not None:
            if target_sample_rate is None or maximum_target_frames is None:
                raise DawValidationError("incomplete WAV render limits")
            maximum = _require_render_sample_count(
                maximum_target_frames, "available vocal render length"
            )
            target_frames = _resampled_sample_count(
                frame_count, sample_rate, target_sample_rate
            )
            if target_frames > maximum:
                raise DawValidationError(
                    "vocal clip exceeds the remaining render-sample budget"
                )
        expected_bytes = frame_count * channels * sample_width
        frames = source.readframes(frame_count)

    if len(frames) != expected_bytes:
        raise ValueError(f"WAV frame data is incomplete: {wav_path}")

    decoded = _decode_pcm(frames, sample_width)
    if decoded.size % channels:
        raise ValueError(f"WAV frame data is incomplete: {wav_path}")
    if channels > 1:
        decoded = decoded.reshape(-1, channels).mean(axis=1)
    return _as_finite_float32_vector(decoded, "decoded WAV audio"), sample_rate


def read_wav(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    """Read a bounded uncompressed PCM WAV as mono float audio plus its rate."""

    return _read_wav_for_render(path)


def write_wav(
    path: str | os.PathLike[str],
    audio: Sequence[float] | np.ndarray,
    sample_rate: int,
) -> Path:
    """Write mono or channel-last audio as signed 16-bit PCM WAV."""

    rate = _require_int(
        sample_rate, "sample rate", minimum=1, maximum=MAX_SAMPLE_RATE
    )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            output = np.asarray(audio, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DawValidationError("audio must contain numeric samples") from exc
    if output.ndim == 1:
        channels = 1
    elif output.ndim == 2 and output.shape[1] > 0:
        channels = output.shape[1]
    else:
        raise ValueError("audio must be a mono vector or a channel-last matrix")
    _require_int(
        channels, "audio channel count", minimum=1, maximum=MAX_WAV_CHANNELS
    )
    _require_render_sample_count(output.shape[0], "WAV output frame count")
    if output.size > MAX_WAV_DECODE_SAMPLES:
        raise DawValidationError("audio exceeds the channel-sample budget")
    if not np.all(np.isfinite(output)):
        raise DawValidationError("audio must contain only finite samples")
    pcm = np.rint(np.clip(output, -1.0, 1.0) * 32767.0).astype("<i2")
    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(pcm.tobytes())
    return wav_path


def export_project_wav(
    project: DawProject,
    path: str | os.PathLike[str],
    *,
    normalize: bool = True,
    peak: float = 0.98,
    base_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Render and save a project as a PCM WAV file."""

    project = _validated_project(project)
    rendered = render_project(
        project, normalize=normalize, peak=peak, base_path=base_path
    )
    return write_wav(path, rendered, project.sample_rate)


def save_project(project: DawProject, path: str | os.PathLike[str]) -> Path:
    """Atomically save a project as readable, versioned JSON."""

    project = _validated_project(project)
    payload = json.dumps(
        project.to_dict(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def load_project(path: str | os.PathLike[str]) -> DawProject:
    """Load and validate a project JSON file."""

    source = Path(path)
    if source.stat().st_size > MAX_PROJECT_JSON_BYTES:
        raise DawValidationError(
            f"project JSON exceeds the {MAX_PROJECT_JSON_BYTES:,}-byte input budget"
        )
    with source.open("rb") as handle:
        raw = handle.read(MAX_PROJECT_JSON_BYTES + 1)
    if len(raw) > MAX_PROJECT_JSON_BYTES:
        raise DawValidationError(
            f"project JSON exceeds the {MAX_PROJECT_JSON_BYTES:,}-byte input budget"
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise DawValidationError("project JSON must be UTF-8 text") from exc
    except json.JSONDecodeError as exc:
        raise DawValidationError(f"invalid project JSON: {exc}") from exc
    return DawProject.from_dict(payload)


__all__ = [
    "PROJECT_SCHEMA_VERSION",
    "SUPPORTED_WAVEFORMS",
    "MAX_RENDER_SAMPLES",
    "MAX_PROJECT_JSON_BYTES",
    "DawValidationError",
    "Step",
    "InstrumentTrack",
    "VocalClip",
    "DawProject",
    "Project",
    "midi_note_to_frequency",
    "synthesize_note",
    "render_instrument_track",
    "mix_audio",
    "resample_audio",
    "normalize_audio",
    "render_project",
    "read_wav",
    "write_wav",
    "export_project_wav",
    "save_project",
    "load_project",
]
