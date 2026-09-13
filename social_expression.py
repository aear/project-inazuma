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
MAX_WITNESSES = 32
MAX_HYPOTHESES = 12
MAX_RECORD_BYTES = 64 * 1024
LISTENER_STATES = frozenset({
    "may_know", "may_not_know", "may_infer", "may_misunderstand", "unknown",
})


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
