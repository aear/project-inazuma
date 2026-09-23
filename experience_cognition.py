"""Bounded cognitive coordination for Experience Learning Model events.

This module adapts useful conditional-computation mechanisms without treating
experience as tokens or turning derived state into memory.  It is deliberately
pure: callers provide a bounded event snapshot and receive an inspectable plan.
It does not retrieve, persist, train, schedule, or continue an Experience Cycle.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


SCHEMA = "ina.experience_cognition/V1"
MAX_ROUTES = 4
MAX_COGNITIVE_STEPS = 5
HORIZONS = ("immediate", "near", "later")
SIGNALS = (
    "novelty", "contradiction", "uncertainty", "stakes", "affect",
    "social", "temporal", "causal", "sensory", "identity",
)
EVIDENCE_DIMENSIONS = ("causal", "temporal", "sensory", "social", "affective", "contradiction")

# These are relevance lenses, not authorities.  No route is always activated.
ROUTE_WEIGHTS: dict[str, dict[str, float]] = {
    "prediction": {"novelty": .30, "uncertainty": .25, "causal": .25, "stakes": .20},
    "hindsight": {"contradiction": .45, "causal": .25, "uncertainty": .20, "stakes": .10},
    "emotion": {"affect": .50, "stakes": .25, "social": .15, "novelty": .10},
    "continuity": {"temporal": .35, "identity": .35, "contradiction": .20, "causal": .10},
    "communication": {"social": .45, "uncertainty": .25, "affect": .15, "stakes": .15},
    "world_model": {"sensory": .35, "causal": .30, "novelty": .20, "temporal": .15},
}

LENS_SIGNALS: dict[str, tuple[str, ...]] = {
    "causal": ("causal", "contradiction", "uncertainty"),
    "temporal": ("temporal", "novelty", "causal"),
    "affective": ("affect", "stakes", "uncertainty"),
    "social": ("social", "affect", "uncertainty"),
    "contradiction": ("contradiction", "uncertainty", "stakes"),
}


def _unit(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return max(0.0, min(1.0, number))


def bounded_signals(event: Mapping[str, Any]) -> dict[str, float]:
    """Extract only declared normalized coordination signals from an event."""
    raw = event.get("signals") if isinstance(event.get("signals"), Mapping) else event
    return {name: _unit(raw.get(name, 0.0)) for name in SIGNALS}


def route_experience(
    event: Mapping[str, Any], *, max_routes: int = MAX_ROUTES,
    minimum_score: float = .20,
) -> dict[str, Any]:
    """Select a sparse set of relevant subsystems while preserving rejections."""
    signals = bounded_signals(event)
    limit = max(0, min(MAX_ROUTES, int(max_routes)))
    threshold = _unit(minimum_score)
    rows = []
    for route, weights in ROUTE_WEIGHTS.items():
        contributions = {
            signal: round(signals[signal] * weight, 6)
            for signal, weight in weights.items()
            if signals[signal] > 0.0
        }
        rows.append({
            "route": route,
            "score": round(sum(contributions.values()), 6),
            "contributions": contributions,
        })
    ranked = sorted(rows, key=lambda row: (-row["score"], row["route"]))
    eligible = [row for row in ranked if row["score"] >= threshold]
    selected = eligible[:limit]
    selected_names = {row["route"] for row in selected}
    rejected = []
    for row in ranked:
        if row["route"] in selected_names:
            continue
        rejected.append({
            **row,
            "reason": "below_threshold" if row["score"] < threshold else "capacity_limit",
        })
    return {
        "selected": selected,
        "rejected": rejected,
        "abstained": not selected,
        "capacity": limit,
        "minimum_score": threshold,
        "signals": signals,
    }


def inspect_attention_lenses(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Inspect the same event through independent, non-voting relation lenses."""
    signals = bounded_signals(event)
    evidence = event.get("evidence") if isinstance(event.get("evidence"), Mapping) else {}
    rows = []
    for lens, names in LENS_SIGNALS.items():
        values = {name: signals[name] for name in names}
        supplied = evidence.get(lens)
        references = []
        if isinstance(supplied, (list, tuple)):
            references = [str(item)[:256] for item in supplied[:8] if str(item)]
        elif supplied:
            references = [str(supplied)[:256]]
        rows.append({
            "lens": lens,
            "salience": round(max(values.values(), default=0.0), 6),
            "signals": values,
            "evidence_references": references,
            "available": bool(references) or any(value > 0.0 for value in values.values()),
        })
    return rows


def gate_transient_state(
    candidates: Iterable[Mapping[str, Any]], *, propagate_threshold: float = .58,
    hold_threshold: float = .30, limit: int = 16,
) -> dict[str, Any]:
    """Classify derived state without mutating or deleting source experience."""
    propagate_at = _unit(propagate_threshold)
    hold_at = min(propagate_at, _unit(hold_threshold))
    rows = []
    for candidate in list(candidates)[:max(0, min(64, int(limit)))]:
        if not isinstance(candidate, Mapping):
            continue
        factors = {
            "relevance": _unit(candidate.get("relevance")),
            "confidence": _unit(candidate.get("confidence")),
            "salience": _unit(candidate.get("salience")),
            "novelty": _unit(candidate.get("novelty")),
        }
        corroboration = max(0, min(4, int(candidate.get("independent_witnesses", 0) or 0)))
        score = (
            .30 * factors["relevance"] + .25 * factors["confidence"]
            + .25 * factors["salience"] + .10 * factors["novelty"]
            + .10 * min(1.0, corroboration / 2.0)
        )
        source_references = [
            str(item)[:256] for item in list(candidate.get("source_references") or ())[:8]
            if str(item)
        ]
        action = "propagate" if score >= propagate_at else ("hold" if score >= hold_at else "decay")
        if action == "propagate" and not source_references:
            action = "hold"
        rows.append({
            "state_id": str(candidate.get("state_id") or "")[:256],
            "action": action,
            "score": round(score, 6),
            "factors": factors,
            "independent_witnesses": corroboration,
            "source_references": source_references,
            "hold_reason": "missing_source_reference" if action == "hold" and not source_references else None,
            "source_experience_unchanged": True,
        })
    return {
        "candidates": rows,
        "thresholds": {"propagate": propagate_at, "hold": hold_at},
        "mutates_memory": False,
    }


def adaptive_computation_plan(
    event: Mapping[str, Any], *, max_steps: int = MAX_COGNITIVE_STEPS,
) -> dict[str, Any]:
    """Allocate a finite number of cognitive passes from independent triggers."""
    signals = bounded_signals(event)
    ceiling = max(1, min(MAX_COGNITIVE_STEPS, int(max_steps)))
    triggers = [
        ("contradiction", signals["contradiction"], .45),
        ("uncertainty", signals["uncertainty"], .55),
        ("stakes", signals["stakes"], .65),
        ("novelty", signals["novelty"], .70),
    ]
    reasons = [name for name, value, threshold in triggers if value >= threshold]
    requested = min(ceiling, 1 + len(reasons))
    return {
        "steps": requested,
        "ceiling": ceiling,
        "reasons": reasons,
        "may_stop_early": True,
        "stop_conditions": ["resolved", "evidence_exhausted", "uncertainty_retained", "budget_exhausted"],
        "autonomous_continuation": False,
    }


def assess_uncertainty(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return a valid unknown/uncertain/known state and identify missing evidence.

    Missing evidence is descriptive.  Suggested observations are optional and
    do not authorize retrieval, sensing, another cognitive pass, or speech.
    """
    signals = bounded_signals(event)
    evidence = event.get("evidence") if isinstance(event.get("evidence"), Mapping) else {}
    available = sorted(
        dimension for dimension in EVIDENCE_DIMENSIONS
        if evidence.get(dimension)
    )
    required_raw = event.get("required_evidence")
    required = [
        str(item) for item in list(required_raw or ())[:len(EVIDENCE_DIMENSIONS)]
        if str(item) in EVIDENCE_DIMENSIONS
    ] if isinstance(required_raw, (list, tuple)) else []
    if not required:
        required = [
            dimension for dimension in EVIDENCE_DIMENSIONS
            if signals.get(dimension if dimension != "affective" else "affect", 0.0) > 0.0
        ]
    missing = sorted(set(required) - set(available))
    conflict = signals["contradiction"]
    uncertainty = signals["uncertainty"]
    candidate_answer = event.get("candidate_answer")
    if not available or candidate_answer is None or (missing and uncertainty >= .75):
        status = "unknown"
        confidence = 0.0
    elif conflict >= .55 or uncertainty >= .45 or missing:
        status = "uncertain"
        confidence = max(0.0, min(.74, 1.0 - max(conflict, uncertainty)))
    else:
        status = "known"
        confidence = max(.75, 1.0 - max(conflict, uncertainty))
    suggestions = {
        "causal": "observe whether the proposed cause precedes and changes the outcome",
        "temporal": "obtain an ordered observation or timestamp",
        "sensory": "obtain a direct observation from the relevant modality",
        "social": "ask the relevant participant or observe their response",
        "affective": "obtain a current affective-state witness",
        "contradiction": "seek an independent witness able to distinguish the alternatives",
    }
    return {
        "status": status,
        "answer": None if status == "unknown" else candidate_answer,
        "confidence": round(confidence, 6),
        "available_evidence": available,
        "missing_evidence": missing,
        "conflict_retained": conflict >= .55,
        "optional_observations": [suggestions[item] for item in missing],
        "unknown_reasons": [
            reason for reason, present in (
                ("no_evidence", not available),
                ("no_candidate_answer", candidate_answer is None),
                ("required_evidence_missing", bool(missing)),
            ) if present
        ] if status == "unknown" else [],
        "may_say_unknown": True,
        "continuation_required": False,
    }


def build_multi_horizon_predictions(
    candidates: Iterable[Mapping[str, Any]], *, limit_per_horizon: int = 4,
) -> dict[str, Any]:
    """Retain supplied future hypotheses by horizon, including disconfirmation."""
    limit = max(1, min(8, int(limit_per_horizon)))
    grouped = {horizon: [] for horizon in HORIZONS}
    rejected = []
    for candidate in list(candidates)[:64]:
        if not isinstance(candidate, Mapping):
            continue
        horizon = str(candidate.get("horizon") or "").lower()
        description = str(candidate.get("prediction") or "")[:1000]
        sources = [str(item)[:256] for item in list(candidate.get("source_references") or ())[:8] if str(item)]
        disconfirm = [str(item)[:512] for item in list(candidate.get("disconfirming_observations") or ())[:8] if str(item)]
        if horizon not in grouped or not description or not sources or not disconfirm:
            rejected.append({
                "prediction": description, "horizon": horizon,
                "reason": "requires_known_horizon_prediction_sources_and_disconfirmation",
            })
            continue
        row = {
            "prediction": description,
            "confidence": _unit(candidate.get("confidence", .5)),
            "source_references": sources,
            "disconfirming_observations": disconfirm,
        }
        if len(grouped[horizon]) < limit:
            grouped[horizon].append(row)
        else:
            rejected.append({"prediction": description, "horizon": horizon, "reason": "capacity_limit"})
    return {
        "horizons": grouped,
        "rejected": rejected,
        "prediction_is_hypothesis": True,
        "writes_memory": False,
    }


def plan_experience_cognition(
    event: Mapping[str, Any], *,
    transient_candidates: Iterable[Mapping[str, Any]] = (),
    prediction_candidates: Iterable[Mapping[str, Any]] = (),
    max_routes: int = MAX_ROUTES,
    max_steps: int = MAX_COGNITIVE_STEPS,
) -> dict[str, Any]:
    """Compose the mechanisms without collapsing their independent evidence."""
    return {
        "schema": SCHEMA,
        "routing": route_experience(event, max_routes=max_routes),
        "attention_lenses": inspect_attention_lenses(event),
        "transient_state": gate_transient_state(transient_candidates),
        "computation": adaptive_computation_plan(event, max_steps=max_steps),
        "epistemic_state": assess_uncertainty(event),
        "predictions": build_multi_horizon_predictions(prediction_candidates),
        "memory_boundary": {
            "reads_fragment_store": False,
            "writes_memory": False,
            "trains_parameters": False,
        },
    }


__all__ = [
    "HORIZONS", "MAX_COGNITIVE_STEPS", "MAX_ROUTES", "ROUTE_WEIGHTS", "SCHEMA",
    "adaptive_computation_plan", "bounded_signals", "build_multi_horizon_predictions",
    "assess_uncertainty", "gate_transient_state", "inspect_attention_lenses", "plan_experience_cognition",
    "route_experience",
]
