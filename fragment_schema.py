# fragment_schema.py
"""
Core schema and helpers for Ina's memory fragments.

This module defines:
  - Modality: which sense / channel the fragment came from
  - FragmentMetadata: common metadata shared across all fragments
  - make_timestamp(): UTC ISO8601 timestamp helper
  - helper constructors for audio/vision metadata

Validation rules live in fragment_validator.py – this file is definition only.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


class Modality(str, Enum):
    """High-level modality of a fragment."""

    AUDIO = "audio"
    VISION = "vision"
    # Future: TEXT = "text", PROPRIOCEPTION = "proprio", etc.


class AttentionState(str, Enum):
    """How much focus Ina had on this input at capture time."""

    FOCUSED = "focused"        # inside attention window / primary device
    PERIPHERAL = "peripheral"  # seen/heard, but not primary focus
    IGNORED = "ignored"        # captured incidentally, low relevance


@dataclass
class FragmentMetadata:
    """
    Shared metadata for all fragments, regardless of content.

    This is intentionally generic; modality-specific payloads
    live alongside this in the fragment dict under their own keys,
    e.g. "audio_features", "raw_audio_path", "image_focus", etc.
    """

    modality: Modality
    source: str                      # e.g. "headset_mic", "webcam", "screen"
    device_name: Optional[str]       # OS/driver name, e.g. "beyerdynamic", "USB PHY"

    timestamp_start: str             # ISO8601 UTC
    timestamp_end: str               # ISO8601 UTC

    attention_state: AttentionState  # focused / peripheral / ignored

    # Pointer into emotion_engine's log, if available at capture/fragment time.
    emotion_snapshot_id: Optional[str] = None

    # Profile name from precision_evolution (e.g. "hi_fidelity", "low_power").
    precision_profile: Optional[str] = None

    # Free-form flags to annotate behaviour or context, e.g.:
    # ["self_voice"], ["music"], ["reading"], ["dreamstate"], ["boredom_explore"]
    flags: List[str] = field(default_factory=list)

    # For future extension without breaking callers.
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a plain dict for JSON/storage."""
        data = asdict(self)
        # Enums should be stored as their value strings
        data["modality"] = self.modality.value
        data["attention_state"] = self.attention_state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FragmentMetadata":
        """
        Reconstruct FragmentMetadata from a dict.

        This is mainly for loaders (memory_graph, training, etc.).
        """
        return cls(
            modality=Modality(data["modality"]),
            source=data["source"],
            device_name=data.get("device_name"),
            timestamp_start=data["timestamp_start"],
            timestamp_end=data["timestamp_end"],
            attention_state=AttentionState(data["attention_state"]),
            emotion_snapshot_id=data.get("emotion_snapshot_id"),
            precision_profile=data.get("precision_profile"),
            flags=list(data.get("flags", [])),
            extra=dict(data.get("extra", {})),
        )


# Type alias for the full fragment object other modules pass around.
# Convention:
#   fragment = {
#       "metadata": FragmentMetadata().to_dict(),
#       ... modality-specific payload ...
#   }
Fragment = Dict[str, Any]


def make_timestamp() -> str:
    """Return a UTC ISO8601 timestamp with 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Convenience helpers – not required, but keep callers clean and consistent.

def make_audio_metadata(
    source: str,
    device_name: Optional[str],
    attention_state: AttentionState,
    emotion_snapshot_id: Optional[str] = None,
    precision_profile: Optional[str] = None,
    flags: Optional[List[str]] = None,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> FragmentMetadata:
    """Helper for creating audio metadata with sane defaults."""
    ts_start = timestamp_start or make_timestamp()
    ts_end = timestamp_end or ts_start
    return FragmentMetadata(
        modality=Modality.AUDIO,
        source=source,
        device_name=device_name,
        timestamp_start=ts_start,
        timestamp_end=ts_end,
        attention_state=attention_state,
        emotion_snapshot_id=emotion_snapshot_id,
        precision_profile=precision_profile,
        flags=list(flags or []),
        extra=dict(extra or {}),
    )


def make_vision_metadata(
    source: str,
    device_name: Optional[str],
    attention_state: AttentionState,
    emotion_snapshot_id: Optional[str] = None,
    precision_profile: Optional[str] = None,
    flags: Optional[List[str]] = None,
    timestamp_start: Optional[str] = None,
    timestamp_end: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> FragmentMetadata:
    """Helper for creating vision metadata with sane defaults."""
    ts_start = timestamp_start or make_timestamp()
    ts_end = timestamp_end or ts_start
    return FragmentMetadata(
        modality=Modality.VISION,
        source=source,
        device_name=device_name,
        timestamp_start=ts_start,
        timestamp_end=ts_end,
        attention_state=attention_state,
        emotion_snapshot_id=emotion_snapshot_id,
        precision_profile=precision_profile,
        flags=list(flags or []),
        extra=dict(extra or {}),
    )
