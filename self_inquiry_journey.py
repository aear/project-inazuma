"""Bounded, voluntary journeys for Ina's deeper self-inquiry.

The journey requests evidence; it does not manufacture an authoritative reason
for a choice.  Memory, reflection, and self-read systems retain ownership of
their evidence and may satisfy or decline the bounded requests.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
import uuid


SCHEMA = "ina.self_inquiry_journey/V1"
MAX_DEPTH = 5
_STAGES = (
    ("experience", "Observe what I chose or experienced", "activity_witness"),
    ("near_context", "Revisit nearby context and felt state", "reflection_witness"),
    ("patterns", "Compare bounded memories and recurring patterns", "memory_index_query"),
    ("architecture", "Inspect relevant architecture and code", "self_read_code"),
    ("hypotheses", "Hold revisable explanations or remain uncertain", "hypothesis_review"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _references(values: Iterable[Any] | None, limit: int = 32) -> list[str]:
    result = []
    for value in values or ():
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item[:500])
        if len(result) >= limit:
            break
    return result


def begin_self_inquiry(
    question: str, *, trigger_references: Iterable[Any], depth_budget: int = 3,
    include_code: bool = False,
) -> dict[str, Any]:
    """Begin one opted-in inquiry with a finite number of possible stages."""
    prompt = " ".join(str(question or "").split())
    if not prompt or len(prompt) > 1000:
        raise ValueError("self-inquiry question must be 1..1000 characters")
    references = _references(trigger_references)
    if not references:
        raise ValueError("self-inquiry requires a witnessed trigger")
    budget = int(depth_budget)
    if budget < 1 or budget > MAX_DEPTH:
        raise ValueError(f"depth_budget must be 1..{MAX_DEPTH}")
    stages = [stage for stage in _STAGES if include_code or stage[0] != "architecture"][:budget]
    return {
        "schema": SCHEMA, "journey_id": f"self_inquiry_{uuid.uuid4().hex}",
        "question": prompt, "trigger_references": references,
        "stage_index": 0, "status": "ready", "may_stop": True,
        "remaining_continuations": max(0, len(stages) - 1),
        "stages": [{"name": name, "prompt": stage_prompt, "evidence_route": route}
                   for name, stage_prompt, route in stages],
        "observations": [], "hypotheses": [], "created_at": _now(), "updated_at": _now(),
    }


def current_inquiry_request(journey: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the present bounded evidence request without interpreting it."""
    if journey.get("schema") != SCHEMA:
        raise ValueError("a valid self-inquiry journey is required")
    stages = list(journey.get("stages") or ())
    index = int(journey.get("stage_index", 0))
    if journey.get("status") in {"stop", "stopped", "complete", "remain_uncertain"} or not 0 <= index < len(stages):
        return None
    stage = dict(stages[index])
    return {
        "journey_id": journey.get("journey_id"), "stage_index": index,
        "stage": stage.get("name"), "prompt": stage.get("prompt"),
        "evidence_route": stage.get("evidence_route"),
        "query_references": list(journey.get("trigger_references") or ())[:32],
        "limits": {"records": 16, "code_files": 4, "may_defer": True},
    }


def continue_self_inquiry(
    journey: Mapping[str, Any], *, choice: str,
    observation_references: Iterable[Any] | None = None,
    hypotheses: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record one stage and stop, complete, or move exactly one step deeper."""
    if journey.get("schema") != SCHEMA:
        raise ValueError("a valid self-inquiry journey is required")
    selected = str(choice or "").strip().lower()
    if selected not in {"deeper", "stop", "remain_uncertain"}:
        raise ValueError("choice must be deeper, stop, or remain_uncertain")
    result = dict(journey)
    result["observations"] = list(journey.get("observations") or ())[:31] + [{
        "stage_index": int(journey.get("stage_index", 0)),
        "references": _references(observation_references, 16), "recorded_at": _now(),
    }]
    normalized_hypotheses = []
    for raw in list(hypotheses or ())[:8]:
        hypothesis = str(raw.get("hypothesis") or "").strip()
        if not hypothesis:
            continue
        normalized_hypotheses.append({
            "hypothesis": hypothesis[:1000],
            "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.0)))),
            "evidence_references": _references(raw.get("evidence_references"), 16),
            "authoritative": False,
        })
    result["hypotheses"] = normalized_hypotheses
    if selected in {"stop", "remain_uncertain"}:
        result["status"] = selected
    else:
        remaining = int(journey.get("remaining_continuations", 0))
        if remaining <= 0:
            raise PermissionError("self-inquiry depth budget is exhausted")
        result["stage_index"] = int(journey.get("stage_index", 0)) + 1
        result["remaining_continuations"] = remaining - 1
        result["status"] = "ready"
    result["updated_at"] = _now()
    return result


__all__ = [
    "SCHEMA", "MAX_DEPTH", "begin_self_inquiry", "current_inquiry_request",
    "continue_self_inquiry",
]
