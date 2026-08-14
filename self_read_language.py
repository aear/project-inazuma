"""Inspectable language-learning roles for self-read music and channel media."""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


VIDEO_ESSAY_THRESHOLD_SECONDS = 600.0
_VOCAL_RE = re.compile(r"(?:^|[\s_.-])(lead[\s_.-]*)?(vocal(?:s)?|vox|voice|a[\s_.-]*cappella)(?:$|[\s_.-])", re.I)
_WRITTEN_RE = re.compile(r"(?:lyric|transcript|subtitle|caption|script)", re.I)
_TRAILING_CONTEXT_RE = re.compile(r"(?:[\s_.-]+(?:lyrics?|transcript|subtitles?|captions?|script|style|prompt|context))+$", re.I)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")[:160]


_REVISIT_FRACTIONS = (0.5, 0.1, 0.75, 0.3, 0.9)


def media_seek_fraction(read_reason: str, prior: Mapping[str, Any] | None = None) -> float:
    """Choose one bounded viewing/listening point; revisits move elsewhere."""
    prior = prior if isinstance(prior, Mapping) else {}
    navigation = prior.get("media_navigation")
    navigation = navigation if isinstance(navigation, Mapping) else {}
    requested = navigation.get("requested_seek_fraction")
    if requested is not None:
        try:
            return round(max(0.0, min(1.0, float(requested))), 4)
        except (TypeError, ValueError):
            pass
    try:
        read_count = max(0, int(prior.get("read_count") or 0))
    except (TypeError, ValueError):
        read_count = 0
    index = read_count if str(read_reason) == "revisit" else 0
    return _REVISIT_FRACTIONS[index % len(_REVISIT_FRACTIONS)]


def attach_media_experience(
    fragment: dict[str, Any], *, media_kind: str, duration_seconds: Any,
    seek_fraction: float, observed_start: float, observed_end: float,
) -> dict[str, Any]:
    try:
        duration = max(0.0, float(duration_seconds or 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    fraction = round(max(0.0, min(1.0, float(seek_fraction))), 4)
    fragment["media_experience"] = {
        "schema": "ina.media_experience/V2",
        "mode": "watching" if media_kind == "video" else "listening",
        "duration_seconds": round(duration, 3),
        "observed_spans": [{
            "start_seconds": round(max(0.0, float(observed_start)), 3),
            "end_seconds": round(max(0.0, float(observed_end)), 3),
        }],
        "seek_fraction": fraction,
        "controls": {
            "can_seek": True, "seek_seconds_parameter": "seek_seconds",
            "can_revisit": True, "can_skip": True,
        },
        "revisit_policy": {
            "allowed": True, "alternate_points": list(_REVISIT_FRACTIONS),
            "repetition_is_new_evidence_only_when_span_changes": True,
        },
    }
    return fragment


def video_language_kind(duration_seconds: Any) -> str:
    """Use strict >10 minute routing; unknown duration remains unclassified."""
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        return "unclassified_channel_video"
    if duration <= 0:
        return "unclassified_channel_video"
    return "video_essay" if duration > VIDEO_ESSAY_THRESHOLD_SECONDS else "channel_video"


def _alignment_keys(relative_label: str, context: Mapping[str, Any]) -> list[str]:
    normalized = str(relative_label or "").replace("\\", "/").strip("/")
    archive_member = str(context.get("archive_member_path") or "")
    collection = str(context.get("stem_collection_relative_path") or "")
    if archive_member:
        container = str(PurePosixPath(normalized).with_suffix(""))
        return [key for key in (_slug(container),) if key]
    if collection:
        return [key for key in (_slug(collection),) if key]

    path = PurePosixPath(normalized)
    stem = _TRAILING_CONTEXT_RE.sub("", path.stem).strip(" ._-") or path.stem
    asset_key = _slug(str(path.parent / stem))
    collection_key = "" if path.parent.as_posix() == "." else _slug(path.parent.as_posix())
    return list(dict.fromkeys(key for key in (asset_key, collection_key) if key))[:2]


def annotate_music_language_evidence(fragment: dict[str, Any], relative_label: str) -> dict[str, Any]:
    """Attach descriptive evidence policy without asserting a transcription."""
    tags = fragment.setdefault("tags", [])
    context = fragment.setdefault("source_context", {})
    modality = str(fragment.get("modality") or "").casefold()
    label = str(context.get("stem_label") or context.get("archive_member_path") or relative_label)
    evidence: dict[str, Any] = {
        "schema": "ina.self_read_language/V2",
        "alignment_keys": _alignment_keys(relative_label, context),
        "token_alignment_claimed": False,
    }

    def add(*values: str) -> None:
        for value in values:
            if value and value not in tags:
                tags.append(value)

    if modality == "audio":
        is_stem = "music_stem" in tags
        is_vocal = bool(_VOCAL_RE.search(" " + label + " "))
        if is_stem and is_vocal:
            add("vocal_stem", "sung_language_audio", "pronunciation_evidence", "cadence_evidence")
            evidence.update({
                "role": "isolated_vocal_stem", "language_mode": "sung",
                "acoustic_clarity": "high", "supports_pronunciation": True,
                "supports_cadence": True, "supports_written_alignment": True,
            })
        elif is_stem:
            add("instrumental_stem", "language_acoustic_contrast")
            evidence.update({
                "role": "instrumental_contrast", "language_mode": None,
                "acoustic_clarity": "not_language", "supports_pronunciation": False,
                "supports_cadence": False, "supports_written_alignment": False,
            })
        else:
            add("mixed_music_audio", "sung_language_candidate")
            evidence.update({
                "role": "mixed_song", "language_mode": "sung",
                "acoustic_clarity": "mixed", "supports_pronunciation": True,
                "supports_cadence": True, "supports_written_alignment": True,
            })
    elif modality == "text":
        written_kind = "lyrics" if "lyric" in label.casefold() else "spoken_script" if _WRITTEN_RE.search(label) else "music_context"
        add("written_language_reference", "language_alignment_target")
        if written_kind == "lyrics": add("written_lyrics")
        if written_kind == "spoken_script": add("spoken_language_script")
        evidence.update({
            "role": written_kind, "language_mode": "written",
            "supports_pronunciation": False, "supports_cadence": False,
            "supports_written_alignment": written_kind != "music_context",
        })
    elif modality == "video":
        kind = video_language_kind(fragment.get("duration"))
        if kind == "video_essay":
            add("video_essay", "spoken_language_audio", "written_language_alignment", "cadence_excluded")
            evidence.update({
                "role": "video_essay", "language_mode": "spoken",
                "supports_pronunciation": True, "supports_cadence": False,
                "supports_written_alignment": True,
                "cadence_exclusion_reason": "video_essay_delivery_is_not_a_cadence_model",
            })
        elif kind == "channel_video":
            add("channel_video", "multimodal_language_evidence")
            evidence.update({
                "role": "channel_video", "language_mode": "mixed_or_spoken",
                "supports_pronunciation": True, "supports_cadence": True,
                "supports_written_alignment": True,
            })
        else:
            add("channel_video", "duration_unclassified", "cadence_excluded")
            evidence.update({
                "role": kind, "language_mode": "unknown",
                "supports_pronunciation": False, "supports_cadence": False,
                "supports_written_alignment": True,
                "cadence_exclusion_reason": "duration_unavailable",
            })
    elif modality == "image":
        role = "album_cover" if re.search(r"(?:cover|artwork|album|front|folder)", label, re.I) else "channel_artwork"
        add(role, "drawing_reference", "visual_composition_reference")
        for alignment_key in evidence["alignment_keys"]:
            add(f"language_alignment:{alignment_key}")
        fragment["visual_learning"] = {
            "schema": "ina.self_read_visual/V2",
            "role": role,
            "practice_use": "drawing",
            "study_dimensions": ["composition", "colour", "shape_language", "typography"],
            "alignment_keys": evidence["alignment_keys"],
            "copying_required": False,
            "revisit_allowed": True,
        }
        context.setdefault("language_alignment_keys", evidence["alignment_keys"])
        return fragment
    else:
        return fragment

    for alignment_key in evidence["alignment_keys"]:
        add(f"language_alignment:{alignment_key}")
    fragment["language_learning"] = evidence
    context.setdefault("language_alignment_keys", evidence["alignment_keys"])
    return fragment


__all__ = [
    "VIDEO_ESSAY_THRESHOLD_SECONDS", "annotate_music_language_evidence",
    "attach_media_experience", "media_seek_fraction", "video_language_kind",
]
