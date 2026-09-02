"""Output-neutral expression intent, realisation, and reaction provenance.

The core describes what Ina intends to express.  Medium-specific realisers own
how that intent becomes text, native symbols, voice, gesture, or future media.
Observed reactions remain witnesses; they are never silently converted into a
reward or treated as proof that the intended meaning was communicated.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol
import uuid

from io_utils import file_lock, flush_for_durability


INTENT_SCHEMA = "ina.expression_intent/V1"
REALISATION_SCHEMA = "ina.expression_realisation/V1"
REACTION_SCHEMA = "ina.expression_reaction/V1"
INTERPRETATION_SCHEMA = "ina.expression_reaction_interpretation/V1"
MAX_RECORD_BYTES = 64 * 1024
MEDIA = frozenset({"text", "native_symbol", "voice", "gesture", "music"})
_FORBIDDEN_INTENT_KEYS = frozenset({
    "text", "punctuation", "emoji", "phoneme", "pitch", "voice",
    "animation", "pose", "avatar", "discord_formatting", "rendered",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _unit(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be 0..1")
    return round(number, 6)


def _references(values: Iterable[Any] | None, limit: int = 32) -> list[str]:
    result = []
    for value in values or ():
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item[:500])
        if len(result) >= limit:
            break
    return result


def _bounded(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError(f"expression record exceeds {MAX_RECORD_BYTES} bytes")
    return result


def create_expression_intent(
    purpose: str, *, semantic_references: Iterable[Any] | None = None,
    meaning_references: Iterable[Any] | None = None,
    concept_references: Iterable[Any] | None = None,
    affect_references: Iterable[Any] | None = None,
    body_references: Iterable[Any] | None = None,
    audience_references: Iterable[Any] | None = None,
    dimensions: Mapping[str, Any] | None = None,
    uncertainty: Mapping[str, Any] | None = None,
    allowed_media: Iterable[str] | None = None,
    provenance: Iterable[Any] | None = None,
) -> dict[str, Any]:
    purpose = str(purpose or "").strip()
    if not purpose or len(purpose) > 500:
        raise ValueError("purpose must be 1..500 characters")
    dimensions = dict(dimensions or {})
    forbidden = _FORBIDDEN_INTENT_KEYS & {str(key).casefold() for key in dimensions}
    if forbidden:
        raise ValueError("medium-specific dimensions belong to a realiser: " + ", ".join(sorted(forbidden)))
    normalized_dimensions = {str(key)[:80]: _unit(value, str(key)) for key, value in dimensions.items()}
    media = _references(allowed_media or MEDIA, len(MEDIA))
    unknown_media = set(media) - MEDIA
    if unknown_media:
        raise ValueError("unknown expression media: " + ", ".join(sorted(unknown_media)))
    return _bounded({
        "schema": INTENT_SCHEMA, "intent_id": _identifier("expression_intent"),
        "purpose": purpose,
        "meaning_references": _references(meaning_references),
        "semantic_references": _references(semantic_references),
        "concept_references": _references(concept_references),
        "affect_references": _references(affect_references),
        "body_references": _references(body_references),
        "audience_references": _references(audience_references),
        "dimensions": normalized_dimensions,
        "uncertainty": dict(uncertainty or {}),
        "allowed_media": media,
        "provenance": _references(provenance, 64),
        "created_at": _now(),
    })


def create_realisation(intent: Mapping[str, Any], *, medium: str,
                       content: Mapping[str, Any], conventions: Iterable[Any] | None = None,
                       provenance: Iterable[Any] | None = None,
                       realiser: str, version: str = "V1") -> dict[str, Any]:
    if intent.get("schema") != INTENT_SCHEMA or not intent.get("intent_id"):
        raise ValueError("a valid expression intent is required")
    medium = str(medium or "").strip()
    if medium not in MEDIA:
        raise ValueError("unknown expression medium")
    if medium not in set(intent.get("allowed_media") or ()):
        raise PermissionError(f"intent does not allow {medium} realisation")
    return _bounded({
        "schema": REALISATION_SCHEMA, "realisation_id": _identifier("expression_realisation"),
        "intent_id": str(intent["intent_id"]), "medium": medium,
        "realiser": str(realiser)[:160], "realiser_version": str(version)[:40],
        "content": dict(content), "conventions": _references(conventions),
        "provenance": _references(provenance, 64), "created_at": _now(),
    })


def create_reaction_observation(realisation_id: str, observation: Mapping[str, Any], *,
                                source: str, provenance: Iterable[Any] | None = None,
                                causal_confidence: float | None = None) -> dict[str, Any]:
    if "reward" in observation:
        raise ValueError("observed reactions are witnesses, not rewards")
    confidence = None if causal_confidence is None else _unit(causal_confidence, "causal_confidence")
    return _bounded({
        "schema": REACTION_SCHEMA, "reaction_id": _identifier("expression_reaction"),
        "realisation_id": str(realisation_id), "observation": dict(observation),
        "source": str(source)[:160], "causal_confidence": confidence,
        "provenance": _references(provenance, 64), "observed_at": _now(),
    })


def create_reaction_interpretation(reaction_id: str, interpretations: Iterable[Mapping[str, Any]], *,
                                   provenance: Iterable[Any] | None = None) -> dict[str, Any]:
    candidates = []
    for raw in list(interpretations)[:8]:
        candidate = dict(raw)
        candidate["confidence"] = _unit(candidate.get("confidence", 0.0), "confidence")
        candidates.append(candidate)
    return _bounded({
        "schema": INTERPRETATION_SCHEMA,
        "interpretation_id": _identifier("expression_interpretation"),
        "reaction_id": str(reaction_id), "candidates": candidates,
        "provenance": _references(provenance, 64), "created_at": _now(),
    })


class ExpressionRealiser(Protocol):
    name: str
    version: str
    medium: str

    def realise(self, intent: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]: ...


class TextRealiser:
    name = "expression.text"
    version = "V1"
    medium = "text"

    def realise(self, intent: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        return create_realisation(
            intent, medium=self.medium, content=dict(kwargs.get("content") or {}),
            conventions=kwargs.get("conventions"), provenance=kwargs.get("provenance"),
            realiser=self.name, version=self.version,
        )


class NativeSymbolRealiser:
    name = "expression.native_symbol"
    version = "V1"
    medium = "native_symbol"

    def realise(self, intent: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        return create_realisation(
            intent, medium=self.medium, content=dict(kwargs.get("content") or {}),
            conventions=kwargs.get("conventions"), provenance=kwargs.get("provenance"),
            realiser=self.name, version=self.version,
        )


class ExpressionTraceStore:
    """Append-only expression chain; source observations remain immutable."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        payload = _bounded(record)
        if payload.get("schema") not in {
            INTENT_SCHEMA, REALISATION_SCHEMA, REACTION_SCHEMA, INTERPRETATION_SCHEMA,
        }:
            raise ValueError("unknown expression record schema")
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with file_lock(self.path.with_suffix(self.path.suffix + ".lock")):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                flush_for_durability(handle, self.path)


__all__ = [
    "INTENT_SCHEMA", "REALISATION_SCHEMA", "REACTION_SCHEMA", "INTERPRETATION_SCHEMA",
    "ExpressionRealiser", "ExpressionTraceStore", "TextRealiser", "NativeSymbolRealiser",
    "create_expression_intent",
    "create_realisation", "create_reaction_observation", "create_reaction_interpretation",
]
