"""Bounded, medium-neutral hypotheses about what Ina may convey.

Witnesses must offer communicative structure explicitly.  Affect, urge, and
audience state may support or modulate a candidate, but cannot invent its
proposition.  Empty or conflicting evidence is a normal abstention.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Any, Iterable, Mapping
import uuid

from semantic_event import build_native_intent, build_semantic_event


MEANING_SET_SCHEMA = "ina.communicative_meaning_set/V1"
MEANING_INTERPRETATION_SCHEMA = "ina.meaning_interpretation/V1"
CONVERSATION_EXAMPLES_SCHEMA = "ina.conversation_meaning_examples/V1"
MAX_RECORD_BYTES = 64 * 1024
MAX_CANDIDATES = 8
MAX_WITNESSES = 32
STANCE_DIMENSIONS = frozenset({
    "commitment", "directness", "urgency", "warmth", "playfulness",
    "vulnerability", "formality",
})
DISCLOSURE_STATES = frozenset({"shareable", "private", "withhold", "unknown"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _unit(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return round(max(0.0, min(1.0, number)), 6)


def _refs(values: Iterable[Any] | None, limit: int = 32) -> list[str]:
    result: list[str] = []
    for value in values or ():
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item[:500])
        if len(result) >= limit:
            break
    return result


def _bounded(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    if len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError(f"communicative meaning record exceeds {MAX_RECORD_BYTES} bytes")
    return result


def _stance(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(key): _unit(value)
        for key, value in raw.items()
        if str(key) in STANCE_DIMENSIONS
    }


def _candidate_key(witness: Mapping[str, Any]) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    act = str(witness.get("communicative_act") or "").strip()[:120]
    propositions = tuple(_refs(witness.get("proposition_references"), 16))
    audience = tuple(_refs(witness.get("audience_references"), 8))
    if not act or not propositions:
        return None
    return act, propositions, audience


def _redact_surface(value: Any) -> Any:
    """Remove verbatim linguistic surfaces while retaining semantic structure."""
    if isinstance(value, Mapping):
        return {
            str(key): _redact_surface(item)
            for key, item in value.items()
            if str(key) not in {
                "source_text", "surface_text", "agent_surface", "surface",
                "lexical_realizations",
            }
        }
    if isinstance(value, list):
        return [_redact_surface(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_surface(item) for item in value]
    return value


def interpret_communicative_meaning(
    witnesses: Iterable[Mapping[str, Any]], *, context_id: str = "",
    minimum_support: float = 0.25, ambiguity_margin: float = 0.08,
    max_candidates: int = MAX_CANDIDATES,
) -> dict[str, Any]:
    """Compose explicit structural witnesses into uncertain meaning candidates."""
    rows = [dict(item) for item in tuple(witnesses)[:MAX_WITNESSES] if isinstance(item, Mapping)]
    grouped: dict[tuple[str, tuple[str, ...], tuple[str, ...]], list[dict[str, Any]]] = {}
    omitted = []
    for index, row in enumerate(rows):
        key = _candidate_key(row)
        if key is None:
            omitted.append({
                "witness": str(row.get("witness_id") or index)[:160],
                "reason": "no_explicit_act_or_proposition",
            })
            continue
        grouped.setdefault(key, []).append(row)

    candidates = []
    for (act, propositions, audience), evidence_rows in grouped.items():
        support = []
        contradictions = []
        stance_totals: dict[str, float] = {}
        stance_weights: dict[str, float] = {}
        disclosure_votes: dict[str, float] = {}
        provenance: list[str] = []
        signed_total = 0.0
        absolute_total = 0.0
        for index, row in enumerate(evidence_rows):
            confidence = _unit(row.get("confidence", 1.0), 1.0)
            relevance = _unit(row.get("relevance", 1.0), 1.0)
            try:
                direction = float(row.get("direction", 1.0))
            except (TypeError, ValueError):
                direction = 0.0
            direction = max(-1.0, min(1.0, direction)) if math.isfinite(direction) else 0.0
            contribution = round(confidence * relevance * direction, 6)
            evidence = {
                "witness_id": str(row.get("witness_id") or f"witness:{index}")[:160],
                "role": str(row.get("role") or "evidence")[:80],
                "contribution": contribution,
                "provenance": _refs(row.get("provenance"), 8),
            }
            (support if contribution >= 0.0 else contradictions).append(evidence)
            signed_total += contribution
            absolute_total += abs(contribution)
            for name, value in _stance(row.get("stance")).items():
                weight = abs(contribution)
                stance_totals[name] = stance_totals.get(name, 0.0) + value * weight
                stance_weights[name] = stance_weights.get(name, 0.0) + weight
            disclosure = str(row.get("disclosure") or "unknown").strip().lower()
            if disclosure not in DISCLOSURE_STATES:
                disclosure = "unknown"
            disclosure_votes[disclosure] = disclosure_votes.get(disclosure, 0.0) + abs(contribution)
            for item in _refs(row.get("provenance"), 16):
                if item not in provenance:
                    provenance.append(item)
        normalized_support = max(0.0, signed_total) / max(1.0, absolute_total)
        confidence = _unit(normalized_support)
        if confidence < max(0.0, float(minimum_support)):
            omitted.append({"communicative_act": act, "reason": "minimum_support_not_met"})
            continue
        disclosure = max(disclosure_votes, key=lambda name: (disclosure_votes[name], name))
        candidate_stance = {
            name: round(total / stance_weights[name], 6)
            for name, total in stance_totals.items()
            if stance_weights.get(name, 0.0) > 0.0
        }
        candidates.append({
            "candidate_id": _id("meaning_candidate"),
            "communicative_act": act,
            "proposition_references": list(propositions),
            "stance": candidate_stance,
            "audience_references": list(audience),
            "disclosure": disclosure,
            "support": support[:16],
            "contradictions": contradictions[:16],
            "confidence": confidence,
            "specificity": _unit(len(propositions) / 8.0),
            "provenance": provenance[:32],
        })

    candidates.sort(key=lambda row: (-row["confidence"], row["candidate_id"]))
    candidates = candidates[:max(1, min(MAX_CANDIDATES, int(max_candidates)))]
    ambiguous = len(candidates) > 1 and (
        candidates[0]["confidence"] - candidates[1]["confidence"]
        <= max(0.0, float(ambiguity_margin))
    )
    private_only = bool(candidates) and all(
        row["disclosure"] in {"private", "withhold"} for row in candidates
    )
    if not candidates:
        abstention = {"active": True, "reason": "no_supported_meaning"}
    elif private_only:
        abstention = {"active": True, "reason": "meaning_not_shareable"}
    elif ambiguous:
        abstention = {"active": True, "reason": "meaning_ambiguous"}
    else:
        abstention = {"active": False, "reason": None}
    interpretation_id = _id("meaning_interpretation")
    interpretation = _bounded({
        "schema": MEANING_INTERPRETATION_SCHEMA,
        "interpretation_id": interpretation_id,
        "context_id": str(context_id or "")[:500],
        "witnesses_considered": len(rows),
        "candidate_groups": len(grouped),
        "rejections": omitted[:32],
        "bounds": {"max_witnesses": MAX_WITNESSES, "max_candidates": MAX_CANDIDATES},
        "created_at": _now(),
    })
    return _bounded({
        "schema": MEANING_SET_SCHEMA,
        "meaning_set_id": _id("meaning_set"),
        "context_id": str(context_id or "")[:500],
        "candidates": candidates,
        "unresolved_tensions": omitted[:32],
        "abstention": abstention,
        "interpreter": "communicative_meaning.compose",
        "interpreter_version": "V1",
        "interpretation_id": interpretation_id,
        "provenance": list(dict.fromkeys(
            item for row in candidates for item in row.get("provenance", [])
        ))[:64],
        "interpretation": interpretation,
        "created_at": _now(),
    })


def build_expression_cognition_event(meaning_set: Mapping[str, Any]) -> dict[str, Any]:
    """Project bounded meaning evidence into cognition signals for expression.

    This projection references meaning witnesses only.  It performs no recall
    and does not turn affect or conversational pressure into a proposition.
    """
    if meaning_set.get("schema") != MEANING_SET_SCHEMA:
        raise ValueError("a valid communicative meaning set is required")
    candidates = [
        dict(item) for item in list(meaning_set.get("candidates") or ())[:MAX_CANDIDATES]
        if isinstance(item, Mapping)
    ]
    abstention = meaning_set.get("abstention")
    abstention = abstention if isinstance(abstention, Mapping) else {}
    leading = candidates[0] if candidates else {}
    support_refs = [
        str(row.get("witness_id"))[:256]
        for row in list(leading.get("support") or ())[:16]
        if isinstance(row, Mapping) and row.get("witness_id")
    ]
    contradiction_refs = [
        str(row.get("witness_id"))[:256]
        for candidate in candidates
        for row in list(candidate.get("contradictions") or ())[:16]
        if isinstance(row, Mapping) and row.get("witness_id")
    ][:16]
    confidence = _unit(leading.get("confidence", 0.0)) if leading else 0.0
    ambiguous = str(abstention.get("reason") or "") == "meaning_ambiguous"
    evidence: dict[str, list[str]] = {}
    if support_refs:
        evidence["social"] = support_refs
    if contradiction_refs:
        evidence["contradiction"] = contradiction_refs
    return {
        "signals": {
            "uncertainty": 1.0 if abstention.get("active") else round(1.0 - confidence, 6),
            "contradiction": .8 if contradiction_refs else (.6 if ambiguous else 0.0),
            "social": 1.0,
            "affect": max((max((item.get("stance") or {}).values(), default=0.0) for item in candidates), default=0.0),
        },
        "candidate_answer": leading.get("candidate_id") if leading and not abstention.get("active") else None,
        "required_evidence": ["social"],
        "evidence": evidence,
        "meaning_set_reference": str(meaning_set.get("meaning_set_id") or "")[:256],
    }
def build_conversation_examples(
    turns: Iterable[Mapping[str, Any]], *, context_id: str = "",
    include_surface: bool = False, max_turns: int = 12,
) -> dict[str, Any]:
    """Create bounded semantic/native examples without assuming feedback is reward."""
    examples = []
    for index, raw in enumerate(tuple(turns)[-max(1, min(32, int(max_turns))):]):
        if not isinstance(raw, Mapping):
            continue
        text = str(raw.get("content", raw.get("text", "")) or "")[:4096]
        if not text.strip():
            continue
        event = raw.get("semantic_event")
        if not isinstance(event, Mapping):
            event = build_semantic_event(text, raw.get("discourse"))
        native_intent = raw.get("native_intent")
        if not isinstance(native_intent, Mapping):
            native_intent = build_native_intent(dict(event))
        semantic_record = dict(event) if include_surface else _redact_surface(event)
        native_record = dict(native_intent) if include_surface else _redact_surface(native_intent)
        example = {
            "example_id": _id("conversation_example"),
            "turn_index": index,
            "speaker_reference": str(raw.get("speaker_id") or raw.get("author_id") or raw.get("author_name") or "unknown")[:160],
            "semantic_event": semantic_record,
            "native_intent": native_record,
            "context_tags": _refs(raw.get("tags"), 16),
            "provenance": _refs(raw.get("provenance") or [raw.get("event_id"), raw.get("message_id")], 16),
            "reaction_observation": None,
        }
        if include_surface:
            example["surface_text"] = text
        examples.append(example)
    return _bounded({
        "schema": CONVERSATION_EXAMPLES_SCHEMA,
        "context_id": str(context_id or "")[:500],
        "examples": examples,
        "surface_included": bool(include_surface),
        "feedback_role": "witness_not_reward",
        "created_at": _now(),
    })


__all__ = [
    "MEANING_SET_SCHEMA", "MEANING_INTERPRETATION_SCHEMA", "CONVERSATION_EXAMPLES_SCHEMA",
    "interpret_communicative_meaning", "build_conversation_examples",
    "build_expression_cognition_event",
]
