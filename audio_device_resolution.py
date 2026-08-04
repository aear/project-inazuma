"""Dependency-light, event-driven audio device resolution helpers.

The project stores both stable-ish names and transient PortAudio indices.  A
device index can change role after a reboot or hot-plug, so callers should
resolve and probe it at the human-triggered moment they need audio rather than
assuming that a non-negative integer is still valid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Optional, Sequence


_DEFAULT_NAMES = frozenset({"default", "system default"})


@dataclass(frozen=True)
class AudioDeviceResolution:
    """Inspectable result of one bounded input/output device check."""

    role: str
    label: str
    requested: dict[str, Any]
    device: Any
    name: Optional[str]
    source: str
    available: bool
    channels: int
    sample_rate: int
    warning: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "label": self.label,
            "requested": dict(self.requested),
            "device": self.device,
            "name": self.name,
            "source": self.source,
            "available": self.available,
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "warning": self.warning,
        }


def _safe_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _normalise_name(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _is_default_name(value: Any) -> bool:
    return _normalise_name(value) in _DEFAULT_NAMES


def _first_config_text(config: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        text = _safe_text(config.get(key))
        if text is not None:
            return text
    return None


def _configured_indices(config: Mapping[str, Any], keys: Sequence[str]) -> list[int]:
    indices: list[int] = []
    for key in keys:
        raw = config.get(key)
        if raw in (None, "") or isinstance(raw, bool):
            continue
        try:
            numeric = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
            continue
        index = int(numeric)
        if index not in indices:
            indices.append(index)
    return indices


def _configured_candidates(
    config: Mapping[str, Any],
    index_keys: Sequence[str],
    name_keys: Sequence[str],
) -> list[tuple[int, Optional[str]]]:
    """Pair each canonical/legacy index with its corresponding saved name."""
    candidates: list[tuple[int, Optional[str]]] = []
    for position, index_key in enumerate(index_keys):
        parsed = _configured_indices(config, (index_key,))
        if not parsed:
            continue
        expected_name = (
            _safe_text(config.get(name_keys[position]))
            if position < len(name_keys)
            else None
        )
        candidate = (parsed[0], expected_name)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _configured_names(config: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    names: list[str] = []
    for key in keys:
        name = _safe_text(config.get(key))
        if name and not _is_default_name(name) and name not in names:
            names.append(name)
    return names


def _device_channels(info: Any, role: str) -> int:
    key = f"max_{role}_channels"
    try:
        value = info.get(key, 0)
    except AttributeError:
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _device_name(info: Any, device: Any) -> str:
    try:
        name = _safe_text(info.get("name"))
    except AttributeError:
        name = None
    if name:
        return name
    return "System default" if device is None else f"Device {device}"


def _name_matches(expected: str, actual: str) -> bool:
    expected_norm = _normalise_name(expected)
    actual_norm = _normalise_name(actual)
    if not expected_norm or not actual_norm:
        return False
    return expected_norm == actual_norm


def _short_error(error: BaseException) -> str:
    text = " ".join(str(error).split()) or error.__class__.__name__
    return text[:240]


def _probe_device(
    sounddevice_module: Any,
    device: Any,
    *,
    role: str,
    sample_rate: int,
    requested_channels: int,
) -> tuple[str, int]:
    query_devices = getattr(sounddevice_module, "query_devices", None)
    if not callable(query_devices):
        raise RuntimeError("sounddevice cannot query devices")
    info = query_devices(device, role)
    maximum = _device_channels(info, role)
    if maximum < 1:
        raise RuntimeError(f"not a usable {role} device")
    channels = min(maximum, requested_channels)

    check_settings = getattr(sounddevice_module, f"check_{role}_settings", None)
    if callable(check_settings):
        settings: dict[str, Any] = {
            "samplerate": sample_rate,
            "channels": channels,
            "dtype": "float32",
        }
        if device is not None:
            settings["device"] = device
        check_settings(**settings)
    return _device_name(info, device), channels


def _matching_name_indices(
    sounddevice_module: Any,
    expected_name: str,
    *,
    role: str,
) -> list[int]:
    query_devices = getattr(sounddevice_module, "query_devices", None)
    if not callable(query_devices):
        return []
    try:
        devices = list(query_devices())
    except Exception:
        return []

    expected_norm = _normalise_name(expected_name)
    exact: list[int] = []
    partial: list[int] = []
    for index, info in enumerate(devices):
        if _device_channels(info, role) < 1:
            continue
        actual_norm = _normalise_name(_device_name(info, index))
        if actual_norm == expected_norm:
            exact.append(index)
        elif expected_norm and expected_norm in actual_norm:
            partial.append(index)
    if len(exact) == 1:
        return exact
    if not exact and len(partial) == 1:
        return partial
    return []


def resolve_audio_device(
    config: Mapping[str, Any],
    *,
    label: str,
    role: str,
    index_keys: Sequence[str],
    name_keys: Sequence[str],
    sounddevice_module: Any,
    sample_rate: int,
    channels: int,
) -> AudioDeviceResolution:
    """Resolve one configured role without opening a stream or rewriting config.

    Resolution order is explicit default policy, override/index, a unique stored
    name match, then the role-specific system default.  Every accepted candidate
    is probed for role, channel and sample-rate support.
    """
    if role not in {"input", "output"}:
        raise ValueError("audio device role must be 'input' or 'output'")
    try:
        sample_rate_value = int(sample_rate)
        channels_value = int(channels)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("audio sample rate and channel count must be positive integers") from exc
    if (
        isinstance(sample_rate, bool)
        or isinstance(channels, bool)
        or sample_rate_value <= 0
        or channels_value <= 0
    ):
        raise ValueError("audio sample rate and channel count must be positive integers")
    sample_rate = sample_rate_value
    channels = channels_value

    overrides = config.get("audio_device_overrides")
    overrides = overrides if isinstance(overrides, Mapping) else {}
    raw_override = overrides.get(label)
    if isinstance(raw_override, str):
        override_candidate: Any = raw_override.strip() or None
    elif (
        isinstance(raw_override, int)
        and not isinstance(raw_override, bool)
        and raw_override >= 0
    ):
        override_candidate = raw_override
    else:
        override_candidate = None
    configured_name = _first_config_text(config, name_keys)
    candidates = _configured_candidates(config, index_keys, name_keys)
    configured_names = _configured_names(config, name_keys)
    indices = _configured_indices(config, index_keys)
    requested = {
        "override": override_candidate,
        "name": configured_name,
        "indices": indices,
        "candidates": [
            {"index": index, "name": name}
            for index, name in candidates
        ],
    }

    if sounddevice_module is None:
        return AudioDeviceResolution(
            role=role,
            label=label,
            requested=requested,
            device=None,
            name=None,
            source="unavailable",
            available=False,
            channels=0,
            sample_rate=sample_rate,
            warning="sounddevice is not installed; offline render and export remain available.",
        )

    override_present = override_candidate not in (None, "")
    explicit_default = (
        _is_default_name(override_candidate)
        or (
            not override_present
            and (
                _is_default_name(configured_name)
                or (configured_name is None and not candidates)
            )
        )
    )
    failures: list[str] = []

    def successful(
        device: Any,
        source: str,
        *,
        expected_name: Optional[str] = None,
    ) -> Optional[AudioDeviceResolution]:
        try:
            name, resolved_channels = _probe_device(
                sounddevice_module,
                device,
                role=role,
                sample_rate=sample_rate,
                requested_channels=channels,
            )
            if expected_name and not _name_matches(expected_name, name):
                raise RuntimeError(
                    f"saved name '{expected_name}' now identifies '{name}'"
                )
        except Exception as exc:
            candidate = "system default" if device is None else repr(device)
            failures.append(f"{candidate}: {_short_error(exc)}")
            return None
        return AudioDeviceResolution(
            role=role,
            label=label,
            requested=requested,
            device=device,
            name=name,
            source=source,
            available=True,
            channels=resolved_channels,
            sample_rate=sample_rate,
        )

    if explicit_default:
        resolution = successful(None, "system_default")
        if resolution is not None:
            return resolution
    else:
        if override_present:
            resolution = successful(override_candidate, "override")
            if resolution is not None:
                return resolution

        for index, expected_name in candidates:
            resolution = successful(
                index,
                "configured_index",
                expected_name=expected_name,
            )
            if resolution is not None:
                return resolution

        for candidate_name in configured_names:
            matches = _matching_name_indices(
                sounddevice_module,
                candidate_name,
                role=role,
            )
            if len(matches) == 1:
                resolution = successful(matches[0], "configured_name")
                if resolution is not None:
                    warning = (
                        f"The saved {role} selection moved; matched '{resolution.name}' "
                        "by name for this session."
                    )
                    return replace(resolution, warning=warning)

        resolution = successful(None, "system_default")
        if resolution is not None:
            reason = failures[0] if failures else "the configured device was unavailable"
            warning = (
                f"The configured {role} device could not be used ({reason}); using "
                f"the system default '{resolution.name}' for this session."
            )
            return replace(resolution, warning=warning)

    reason = "; ".join(failures[:3]) or "no role-compatible default is configured"
    return AudioDeviceResolution(
        role=role,
        label=label,
        requested=requested,
        device=None,
        name=None,
        source="unavailable",
        available=False,
        channels=0,
        sample_rate=sample_rate,
        warning=(
            f"No usable {role} device was found ({reason}); offline render and export "
            "remain available."
        ),
    )
