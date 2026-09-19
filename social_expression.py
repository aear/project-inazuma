"""Bounded social-context hypotheses and communicative repair proposals.

These records describe evidence about possible listener understanding.  They do
not claim access to another mind and never send, rewrite, or reward expression.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from typing import Any, Iterable, Mapping
import uuid

LISTENER_MODEL_SCHEMA = "ina.listener_hypothesis_set/V1"
REPAIR_SCHEMA = "ina.communicative_repair_assessment/V1"
INTELLIGIBILITY_SCHEMA = "ina.mutual_intelligibility_assessment/V1"
MAX_WITNESSES = 32
MAX_HYPOTHESES = 12
MAX_RECORD_BYTES = 64 * 1024
LISTENER_STATES = frozenset({
    "may_know", "may_not_know", "may_infer", "may_misunderstand", "unknown",
})
BRIDGE_MODES = frozenset({"native_only", "native_with_english_bridge", "clarify", "abstain"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return round(max(0.0, min(1.0, number if math.isfinite(number) else 0.0)), 6)


def _refs(values: Iterable[Any] | None, limit: int = 32) -> list[str]:
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
    if len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()) > MAX_RECORD_BYTES:
        raise ValueError("social expression record exceeds 65536 bytes")
    return result


def build_listener_hypotheses(
    audience_reference: str, witnesses: Iterable[Mapping[str, Any]],
    *, context_id: str = "",
) -> dict[str, Any]:
    """Build a federation of non-authoritative common-ground hypotheses."""
    audience = str(audience_reference or "").strip()
    if not audience:
        raise ValueError("audience_reference is required")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    rejected = []
    for index, raw in enumerate(tuple(witnesses)[:MAX_WITNESSES]):
        if not isinstance(raw, Mapping):
            continue
        proposition = str(raw.get("proposition_reference") or "").strip()[:500]
        state = str(raw.get("state") or "unknown").strip().lower()
        witness_id = str(raw.get("witness_id") or "").strip()[:160]
        if not proposition or not witness_id or state not in LISTENER_STATES:
            rejected.append({"witness": witness_id or str(index), "reason": "invalid_hypothesis_witness"})
            continue
        row = {
            "witness_id": witness_id,
            "confidence": _unit(raw.get("confidence")),
            "provenance": _refs(raw.get("provenance"), 8),
            "observed_at": str(raw.get("observed_at") or "")[:80],
        }
        grouped.setdefault((proposition, state), []).append(row)
    hypotheses = []
    for (proposition, state), support in grouped.items():
        origins = {ref for row in support for ref in row["provenance"]}
        hypotheses.append({
            "hypothesis_id": _id("listener_hypothesis"),
            "proposition_reference": proposition,
            "state": state,
            "confidence": round(sum(row["confidence"] for row in support) / len(support), 6),
            "support": support[:16],
            "independent_origins": len(origins),
            "authoritative": False,
        })
    hypotheses.sort(key=lambda row: (-row["confidence"], row["proposition_reference"], row["state"]))
    return _bounded({
        "schema": LISTENER_MODEL_SCHEMA,
        "listener_model_id": _id("listener_model"),
        "audience_reference": audience[:500],
        "context_id": str(context_id or "")[:500],
        "hypotheses": hypotheses[:MAX_HYPOTHESES],
        "unresolved_witnesses": rejected[:32],
        "epistemic_status": "hypotheses_not_mind_reading",
        "created_at": _now(),
    })


def assess_communicative_repair(
    intended_meaning_references: Iterable[Any],
    received_candidates: Iterable[Mapping[str, Any]],
    *, realisation_id: str, minimum_confidence: float = 0.55,
) -> dict[str, Any]:
    """Compare intended and possibly received meanings without auto-repairing."""
    intended = set(_refs(intended_meaning_references, 16))
    if not intended:
        raise ValueError("intended meaning references are required")
    candidates = []
    for raw in tuple(received_candidates)[:8]:
        if not isinstance(raw, Mapping):
            continue
        candidates.append({
            "meaning_references": _refs(raw.get("meaning_references"), 16),
            "confidence": _unit(raw.get("confidence")),
            "witness_references": _refs(raw.get("witness_references"), 16),
            "provenance": _refs(raw.get("provenance"), 16),
        })
    candidates.sort(key=lambda row: -row["confidence"])
    leading = candidates[0] if candidates else None
    received = set(leading["meaning_references"]) if leading else set()
    origins = set((leading or {}).get("witness_references") or ()) | set(
        (leading or {}).get("provenance") or ()
    )
    mismatch = bool(leading and received != intended)
    corroborated = len(origins) >= 2
    confident = bool(leading and leading["confidence"] >= _unit(minimum_confidence))
    if mismatch and confident and corroborated:
        status = "repair_candidate"
        actions = ["clarify_intended_meaning", "invite_listener_correction", "rephrase_if_chosen"]
    elif mismatch:
        status = "observe_or_clarify"
        actions = ["seek_more_uptake_evidence"]
    elif leading:
        status = "no_repair_indicated"
        actions = []
    else:
        status = "uptake_unknown"
        actions = ["allow_unresolved"]
    return _bounded({
        "schema": REPAIR_SCHEMA,
        "repair_assessment_id": _id("repair_assessment"),
        "realisation_id": str(realisation_id)[:500],
        "intended_meaning_references": sorted(intended),
        "received_candidates": candidates,
        "leading_received_meaning_references": sorted(received),
        "meaning_mismatch": mismatch,
        "independent_evidence_origins": len(origins),
        "status": status,
        "proposed_actions": actions,
        "automatic_expression": False,
        "created_at": _now(),
    })


def assess_mutual_intelligibility(
    intent: Mapping[str, Any],
    realisation_assessments: Iterable[Mapping[str, Any]],
    *,
    listener_model: Mapping[str, Any],
    bridge_willingness: float,
    willingness_witnesses: Iterable[Any],
    minimum_fidelity: float = 0.7,
    minimum_recoverability: float = 0.7,
    minimum_uncertainty_preservation: float = 0.6,
) -> dict[str, Any]:
    """Choose an optional listener bridge without ranking a language as thought.

    Each candidate retains separate fidelity, listener-recoverability, and
    uncertainty-preservation dimensions.  A usable candidate needs two
    independent evidence origins for every dimension; several scores derived
    from one model therefore remain one witness.  The result is advisory and
    never emits, rewrites, or rewards an expression.
    """
    if intent.get("schema") != "ina.expression_intent/V1" or not intent.get("intent_id"):
        raise ValueError("a valid expression intent is required")
    if listener_model.get("schema") != LISTENER_MODEL_SCHEMA:
        raise ValueError("a valid listener hypothesis set is required")
    audience = set(_refs(intent.get("audience_references"), 16))
    listener = str(listener_model.get("audience_reference") or "").strip()
    if audience and listener not in audience:
        raise ValueError("listener model does not match the expression audience")

    thresholds = {
        "fidelity": _unit(minimum_fidelity),
        "recoverability": _unit(minimum_recoverability),
        "uncertainty_preservation": _unit(minimum_uncertainty_preservation),
    }
    candidates = []
    for raw in tuple(realisation_assessments)[:8]:
        if not isinstance(raw, Mapping):
            continue
        medium = str(raw.get("medium") or "").strip()
        language = str(raw.get("language") or "").strip().casefold()
        scores = {key: _unit(raw.get(key)) for key in thresholds}
        supplied = raw.get("witnesses") if isinstance(raw.get("witnesses"), Mapping) else {}
        evidence = {key: _refs(supplied.get(key), 16) for key in thresholds}
        corroborated = {key: len(set(evidence[key])) >= 2 for key in thresholds}
        faithful = all(corroborated.values()) and all(
            scores[key] >= thresholds[key]
            for key in ("fidelity", "uncertainty_preservation")
        )
        candidates.append({
            "realisation_id": str(raw.get("realisation_id") or "")[:500],
            "medium": medium[:80],
            "language": language[:40],
            "scores": scores,
            "witnesses": evidence,
            "corroborated_dimensions": corroborated,
            "faithful": faithful,
        })

    native = next((row for row in candidates if row["medium"] == "native_symbol"), None)
    english = next((row for row in candidates
                    if row["medium"] == "text" and row["language"] == "english"), None)
    english_listener_origins = {
        origin
        for hypothesis in listener_model.get("hypotheses") or ()
        if isinstance(hypothesis, Mapping)
        and str(hypothesis.get("proposition_reference") or "").casefold()
        in {"language:english", "language:en"}
        and hypothesis.get("state") in {"may_know", "may_infer"}
        and _unit(hypothesis.get("confidence")) >= 0.5
        for support in hypothesis.get("support") or ()
        if isinstance(support, Mapping)
        for origin in support.get("provenance") or ()
    }
    willingness = _unit(bridge_willingness)
    willingness_origins = _refs(willingness_witnesses, 16)
    willing = willingness >= 0.5 and len(set(willingness_origins)) >= 2

    native_faithful = bool(native and native["faithful"])
    native_understandable = bool(
        native_faithful and native["scores"]["recoverability"] >= thresholds["recoverability"]
    )
    english_usable = bool(
        english and english["faithful"]
        and english["scores"]["recoverability"] >= thresholds["recoverability"]
        and len(english_listener_origins) >= 2
    )
    if native_understandable:
        mode = "native_only"
        reason = "native_is_faithful_and_listener_recoverable"
    elif native_faithful and english_usable and willing:
        mode = "native_with_english_bridge"
        reason = "native_is_faithful_but_listener_recoverability_needs_a_voluntary_bridge"
    elif native_faithful and not willing:
        mode = "native_only"
        reason = "bridge_not_voluntarily_chosen"
    elif native_faithful:
        mode = "clarify"
        reason = "no_corroborated_faithful_english_bridge"
    else:
        mode = "abstain"
        reason = "native_fidelity_not_corroborated"
    assert mode in BRIDGE_MODES

    return _bounded({
        "schema": INTELLIGIBILITY_SCHEMA,
        "assessment_id": _id("mutual_intelligibility"),
        "intent_id": str(intent["intent_id"]),
        "listener_model_id": str(listener_model.get("listener_model_id") or "")[:500],
        "audience_reference": listener[:500],
        "dimensions": ["native_fidelity", "listener_recoverability", "uncertainty_preservation"],
        "thresholds": thresholds,
        "candidates": candidates,
        "english_listener_evidence_origins": sorted(english_listener_origins)[:16],
        "bridge_willingness": willingness,
        "willingness_witnesses": willingness_origins,
        "willingness_corroborated": len(set(willingness_origins)) >= 2,
        "mode": mode,
        "reason": reason,
        "automatic_expression": False,
        "english_is_internal_representation": False,
        "engagement_is_understanding_evidence": False,
        "created_at": _now(),
    })


__all__ = [
    "LISTENER_MODEL_SCHEMA", "REPAIR_SCHEMA", "INTELLIGIBILITY_SCHEMA", "BRIDGE_MODES",
    "build_listener_hypotheses", "assess_communicative_repair",
    "assess_mutual_intelligibility",
]
