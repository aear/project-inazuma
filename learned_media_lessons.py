"""Bounded bridge from self-read media evidence to creative/language outputs."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from io_utils import atomic_write_json, load_json_dict

SCHEMA = "ina.learned_media_lessons/V2"
MAX_LESSONS_PER_DOMAIN = 64
MAX_GUIDANCE_ITEMS = 4


def _path(child: str, base_path: Path | None = None) -> Path:
    root = Path(base_path) if base_path is not None else Path("AI_Children")
    return root / str(child) / "memory" / "learned_media_lessons.json"


def _bounded(values: Any, limit: int) -> list[str]:
    result = []
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in result: result.append(text[:160])
        if len(result) >= limit: break
    return result


def _lesson(fragment: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    language = fragment.get("language_learning")
    visual = fragment.get("visual_learning")
    media = fragment.get("media_experience")
    if isinstance(visual, Mapping):
        domain, evidence = "visual", visual
    elif isinstance(language, Mapping):
        role = str(language.get("role") or "")
        domain = "written" if role in {"lyrics", "spoken_script", "music_context"} else "language"
        evidence = language
    else:
        return None
    source_context = fragment.get("source_context")
    source_context = source_context if isinstance(source_context, Mapping) else {}
    lesson = {
        "id": str(fragment.get("id") or fragment.get("source") or "")[:200],
        "source": str(fragment.get("source") or source_context.get("relative_path") or "")[:500],
        "role": evidence.get("role"),
        "alignment_keys": _bounded(evidence.get("alignment_keys"), 2),
        "tags": _bounded(fragment.get("tags"), 12),
        "supports_pronunciation": bool(evidence.get("supports_pronunciation")),
        "supports_cadence": bool(evidence.get("supports_cadence")),
        "supports_written_alignment": bool(evidence.get("supports_written_alignment")),
        "symbols": _bounded(fragment.get("symbols"), 48),
        "proto_words": _bounded(fragment.get("proto_words"), 24),
        "study_dimensions": _bounded(evidence.get("study_dimensions"), 8),
        "text_excerpt": str(fragment.get("text") or "")[:320],
        "media_mode": media.get("mode") if isinstance(media, Mapping) else None,
        "observed_spans": list(media.get("observed_spans") or ())[:2] if isinstance(media, Mapping) else [],
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
    return domain, lesson


def record_media_lesson(child: str, fragment: Mapping[str, Any], *, base_path: Path | None = None) -> bool:
    prepared = _lesson(fragment)
    if not prepared: return False
    domain, lesson = prepared
    path = _path(child, base_path)
    state = load_json_dict(path)
    domains = state.get("domains") if isinstance(state.get("domains"), dict) else {}
    rows = [dict(row) for row in domains.get(domain, ()) if isinstance(row, Mapping)]
    identity = (lesson["id"], lesson["role"], tuple(lesson["alignment_keys"]))
    rows = [row for row in rows if (row.get("id"), row.get("role"), tuple(row.get("alignment_keys") or ())) != identity]
    rows.append(lesson)
    domains[domain] = rows[-MAX_LESSONS_PER_DOMAIN:]
    atomic_write_json(path, {"schema": SCHEMA, "domains": domains, "updated_at": lesson["observed_at"]})
    return True


def load_output_guidance(child: str, consumer: str, *, base_path: Path | None = None) -> dict[str, Any]:
    state = load_json_dict(_path(child, base_path))
    domains = state.get("domains") if isinstance(state.get("domains"), dict) else {}
    consumer = str(consumer or "text").casefold()
    if consumer == "drawing": keys = ("visual",)
    elif consumer == "daw": keys = ("language", "written")
    elif consumer == "speech": keys = ("language",)
    else: keys = ("written", "language")
    selected = []
    for key in keys:
        for row in reversed(list(domains.get(key) or ())):
            if not isinstance(row, Mapping): continue
            if consumer == "daw" and row.get("role") == "video_essay": continue
            selected.append(dict(row))
            if len(selected) >= MAX_GUIDANCE_ITEMS: break
        if len(selected) >= MAX_GUIDANCE_ITEMS: break
    return {"schema": SCHEMA, "consumer": consumer, "lessons": selected, "bounded": True}


__all__ = ["load_output_guidance", "record_media_lesson"]
