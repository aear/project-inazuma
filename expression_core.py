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
from typing import Any, Callable, Iterable, Mapping, Protocol
import uuid

from io_utils import file_lock, flush_for_durability


INTENT_SCHEMA = "ina.expression_intent/V1"
REALISATION_SCHEMA = "ina.expression_realisation/V1"
REACTION_SCHEMA = "ina.expression_reaction/V1"
INTERPRETATION_SCHEMA = "ina.expression_reaction_interpretation/V1"
REQUEST_SCHEMA = "ina.requested_effect/V1"
AFFORDANCE_SCHEMA = "ina.expression_affordance/V1"
SELECTION_SCHEMA = "ina.expression_affordance_selection/V1"
MAX_RECORD_BYTES = 64 * 1024
MEDIA = frozenset({"text", "native_symbol", "voice", "gesture", "music"})
FULFILMENT = frozenset({"direct", "approximation", "representation_only"})
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


def create_requested_effect(
    intent: Mapping[str, Any], *, effects: Iterable[Mapping[str, Any]],
    constraints: Mapping[str, Any] | None = None,
    provenance: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Describe what an expression should cause, without prescribing a medium.

    Interpretation belongs upstream.  This record can represent informing,
    demonstrating, evoking a sensation, coordinating an action, social play,
    or combinations of those effects.  Medium-specific properties stay in a
    candidate affordance rather than leaking into the expression intent.
    """
    if intent.get("schema") != INTENT_SCHEMA or not intent.get("intent_id"):
        raise ValueError("a valid expression intent is required")
    normalized = []
    for raw in list(effects)[:16]:
        effect = dict(raw)
        kind = str(effect.get("kind") or "").strip()
        target = str(effect.get("target") or "").strip()
        if not kind or not target:
            raise ValueError("each requested effect requires kind and target")
        normalized.append({
            "kind": kind[:80], "target": target[:500],
            "importance": _unit(effect.get("importance", 1.0), "importance"),
        })
    if not normalized:
        raise ValueError("at least one requested effect is required")
    return _bounded({
        "schema": REQUEST_SCHEMA, "request_id": _identifier("requested_effect"),
        "intent_id": str(intent["intent_id"]), "effects": normalized,
        "constraints": dict(constraints or {}),
        "provenance": _references(provenance, 64), "created_at": _now(),
    })


def create_expression_affordance(
    request: Mapping[str, Any], *, medium: str, action: Mapping[str, Any],
    assessments: Mapping[str, Any], witnesses: Mapping[str, Iterable[Any]],
    fulfilment: str = "direct", available: bool = True,
    provenance: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Offer one capability-owned way to pursue a requested effect."""
    if request.get("schema") != REQUEST_SCHEMA or not request.get("request_id"):
        raise ValueError("a valid requested effect is required")
    medium = str(medium or "").strip()
    if not medium or len(medium) > 80 or any(character.isspace() for character in medium):
        raise ValueError("affordance medium must be a compact capability name")
    fulfilment = str(fulfilment or "").strip()
    if fulfilment not in FULFILMENT:
        raise ValueError("unknown fulfilment class")
    scores = {str(key)[:80]: _unit(value, str(key)) for key, value in assessments.items()}
    required = {"effect_fit", "capability", "willingness"}
    if not required.issubset(scores):
        raise ValueError("assessments require effect_fit, capability, and willingness")
    evidence = {str(key)[:80]: _references(values, 16) for key, values in witnesses.items()}
    missing = [key for key in required if not evidence.get(key)]
    if missing:
        raise ValueError("each required assessment needs a witness: " + ", ".join(sorted(missing)))
    return _bounded({
        "schema": AFFORDANCE_SCHEMA, "affordance_id": _identifier("expression_affordance"),
        "request_id": str(request["request_id"]), "medium": medium,
        "action": dict(action), "assessments": scores, "witnesses": evidence,
        "fulfilment": fulfilment, "available": bool(available),
        "provenance": _references(provenance, 64), "created_at": _now(),
    })


def select_expression_affordance(
    request: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]], *,
    minimum_signal: float = 0.5, ambiguity_margin: float = 0.05,
    ambiguity_resolver: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select an inspectable, corroborated affordance or explicitly abstain."""
    if request.get("schema") != REQUEST_SCHEMA or not request.get("request_id"):
        raise ValueError("a valid requested effect is required")
    threshold = _unit(minimum_signal, "minimum_signal")
    considered = []
    fulfilment_rank = {"direct": 2, "approximation": 1, "representation_only": 0}
    for raw in list(candidates)[:32]:
        candidate = dict(raw)
        if candidate.get("schema") != AFFORDANCE_SCHEMA:
            raise ValueError("all candidates must be expression affordances")
        if candidate.get("request_id") != request.get("request_id"):
            raise ValueError("candidate belongs to a different requested effect")
        scores = dict(candidate.get("assessments") or {})
        evidence = dict(candidate.get("witnesses") or {})
        required_signals = ["effect_fit", "capability", "willingness"]
        if request.get("constraints"):
            required_signals.append("constraint_fit")
        origins = {item for key in required_signals for item in evidence.get(key, ())}
        eligible = (
            bool(candidate.get("available")) and len(origins) >= 2
            and all(float(scores.get(key, 0.0)) >= threshold for key in required_signals)
        )
        bottleneck = min((float(scores.get(key, 0.0)) for key in required_signals), default=0.0)
        mean = sum(float(scores.get(key, 0.0)) for key in required_signals) / len(required_signals)
        considered.append({
            "candidate": candidate, "eligible": eligible,
            "required_signals": required_signals, "independent_witnesses": sorted(origins),
            "bottleneck": round(bottleneck, 6), "mean": round(mean, 6),
        })
    eligible = [item for item in considered if item["eligible"]]
    eligible.sort(key=lambda item: (
        item["candidate"].get("fulfilment") == "representation_only",
        -item["bottleneck"], -item["mean"],
        -fulfilment_rank.get(str(item["candidate"].get("fulfilment")), -1),
        str(item["candidate"].get("affordance_id")),
    ))
    resolution = None
    if eligible and ambiguity_resolver is not None:
        margin = _unit(ambiguity_margin, "ambiguity_margin")
        leading = eligible[0]
        leading_class = leading["candidate"].get("fulfilment") == "representation_only"
        ambiguous = [item for item in eligible if (
            (item["candidate"].get("fulfilment") == "representation_only") == leading_class
            and leading["bottleneck"] - item["bottleneck"] <= margin
        )]
        if len(ambiguous) > 1:
            resolved = dict(ambiguity_resolver([{
                "id": item["candidate"].get("affordance_id"),
                "activation": max(0.0, item["bottleneck"] * 0.7 + item["mean"] * 0.3),
            } for item in ambiguous], context=str(request.get("request_id"))))
            selected_id = resolved.get("selected_id")
            match = next((item for item in ambiguous
                          if item["candidate"].get("affordance_id") == selected_id), None)
            if match is not None:
                eligible.remove(match)
                eligible.insert(0, match)
                resolution = {
                    "resolver": "candidate_superposition",
                    "candidate_ids": [item["candidate"].get("affordance_id") for item in ambiguous],
                    "selected_id": selected_id,
                    "origins": list(resolved.get("origins") or ())[:8],
                }
    chosen = eligible[0]["candidate"] if eligible else None
    return _bounded({
        "schema": SELECTION_SCHEMA, "selection_id": _identifier("affordance_selection"),
        "request_id": str(request["request_id"]),
        "selected_affordance_id": chosen.get("affordance_id") if chosen else None,
        "selected_medium": chosen.get("medium") if chosen else None,
        "fulfils_request": bool(chosen and chosen.get("fulfilment") != "representation_only"),
        "status": "selected" if chosen else "abstained",
        "ambiguity_resolution": resolution,
        "viable_affordance_ids": [
            item["candidate"].get("affordance_id") for item in considered if item["eligible"]
        ],
        "created_at": _now(),
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
            REQUEST_SCHEMA, AFFORDANCE_SCHEMA, SELECTION_SCHEMA,
        }:
            raise ValueError("unknown expression record schema")
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        with file_lock(self.path.with_suffix(self.path.suffix + ".lock")):
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                flush_for_durability(handle, self.path)


__all__ = [
    "INTENT_SCHEMA", "REALISATION_SCHEMA", "REACTION_SCHEMA", "INTERPRETATION_SCHEMA",
    "REQUEST_SCHEMA", "AFFORDANCE_SCHEMA", "SELECTION_SCHEMA",
    "ExpressionRealiser", "ExpressionTraceStore", "TextRealiser", "NativeSymbolRealiser",
    "create_expression_intent",
    "create_requested_effect", "create_expression_affordance", "select_expression_affordance",
    "create_realisation", "create_reaction_observation", "create_reaction_interpretation",
]
