"""Evidence-gated hooks for promoting whole utterances into quick-reach memory."""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

RETENTION_TAGS = frozenset({"remember", "quote", "keep_words", "explicit_retention"})
RELATIONAL_TAGS = frozenset({"relationship", "intimacy", "trust", "milestone", "identity"})
REPAIR_TAGS = frozenset({"message_edit", "correction", "repair", "misunderstanding"})
NOVELTY_TAGS = frozenset({"novel", "first", "discovery"})


def _unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, min(1.0, number if math.isfinite(number) else 0.0))


def evaluate_utterance_memory_hook(
    utterance: Mapping[str, Any], event: Mapping[str, Any], *, threshold: float = 0.55,
    rephrasing_candidates: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Decide whether a source utterance merits a sentence-level projection.

    The event remains owned by experiential memory either way.  No single
    resonance, novelty, tag, or lexical cue can promote the sentence alone.
    """
    text = " ".join(str(utterance.get("utterance") or "").split())[:1000]
    tags = {str(value).casefold() for value in event.get("situation_tags") or ()}
    internal = event.get("internal_state") if isinstance(event.get("internal_state"), Mapping) else {}
    entity_links = utterance.get("entity_links") if isinstance(utterance.get("entity_links"), list) else []
    signals = []

    def add(name: str, value: float, origin: str) -> None:
        if value > 0:
            signals.append({"signal": name, "value": round(_unit(value), 6), "origin": origin})

    add("explicit_retention", 1.0 if tags & RETENTION_TAGS else 0.0, "event_tags")
    add("relational_significance", .8 if tags & RELATIONAL_TAGS else 0.0, "social_context")
    add("repair_value", .8 if tags & REPAIR_TAGS or any(
        isinstance(link, Mapping) and link.get("type") == "discord_message_edit"
        for link in entity_links
    ) else 0.0, "communication_history")
    add("novelty", max(
        _unit(internal.get("novelty")), .75 if tags & NOVELTY_TAGS else 0.0,
    ), "novelty_evaluator")
    add("resonance", max(
        _unit(internal.get("communicative_resonance")),
        _unit(internal.get("emotional_resonance")),
    ), "affect_resonance")
    origins = {row["origin"] for row in signals if row["value"] >= .5}
    score = sum(row["value"] for row in signals) / max(2, len(signals))
    retained = bool(text and len(origins) >= 2 and score >= max(0.0, min(1.0, threshold)))
    projection_text = text if retained else None
    surface_kind = "source" if retained else None
    rephrasing = {"status": "not_considered", "candidate_count": 0}
    if retained and rephrasing_candidates is not None:
        source_meanings = {
            str(value) for value in utterance.get("meaning_references") or () if str(value)
        }
        candidates = []
        for raw in tuple(rephrasing_candidates)[:8]:
            if not isinstance(raw, Mapping):
                continue
            candidate_text = " ".join(str(raw.get("text") or "").split())[:1000]
            meanings = {str(value) for value in raw.get("meaning_references") or () if str(value)}
            provenance = {str(value) for value in raw.get("provenance") or () if str(value)}
            purpose = str(raw.get("purpose") or "").strip().lower()
            eligible = (
                bool(candidate_text) and bool(source_meanings) and meanings == source_meanings
                and _unit(raw.get("confidence")) >= .7 and len(provenance) >= 2
                and purpose in {"clarity", "compression"}
                and candidate_text != text
            )
            candidates.append({
                "text": candidate_text, "meaning_references": sorted(meanings),
                "confidence": _unit(raw.get("confidence")), "purpose": purpose,
                "provenance": sorted(provenance), "eligible": eligible,
            })
        eligible = sorted(
            (row for row in candidates if row["eligible"]),
            key=lambda row: (-row["confidence"], len(row["text"]), row["text"]),
        )
        if eligible:
            projection_text = eligible[0]["text"]
            surface_kind = "rephrased"
            rephrasing = {
                "status": "selected", "candidate_count": len(candidates),
                "purpose": eligible[0]["purpose"],
                "meaning_references": eligible[0]["meaning_references"],
                "provenance": eligible[0]["provenance"],
            }
        else:
            rephrasing = {"status": "source_retained", "candidate_count": len(candidates)}
    return {
        "schema": "ina.utterance_memory_hook/V1",
        "decision": "retain_projection" if retained else "do_not_promote",
        "retained": retained,
        "score": round(score, 6),
        "signals": signals,
        "independent_origins": len(origins),
        "text": projection_text,
        "surface_kind": surface_kind,
        "rephrasing": rephrasing,
        "source_event_id": str(event.get("id") or "")[:160],
        "source_unchanged": True,
    }
