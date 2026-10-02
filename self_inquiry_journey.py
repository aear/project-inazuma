"""Bounded, voluntary journeys for Ina's deeper self-inquiry.

The journey requests evidence; it does not manufacture an authoritative reason
for a choice.  Memory, reflection, and self-read systems retain ownership of
their evidence and may satisfy or decline the bounded requests.
"""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import islice
from copy import deepcopy
import math
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
    for value in islice(values or (), limit):
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
        **({"countercheck": stage['countercheck']} if 'countercheck' in stage else {}),
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
    if current_inquiry_request(journey) is None:
        raise PermissionError("self-inquiry is terminal; begin a new explicitly chosen inquiry")
    if selected not in {"deeper", "stop", "remain_uncertain"}:
        raise ValueError("choice must be deeper, stop, or remain_uncertain")
    result = dict(journey)
    result["observations"] = list(journey.get("observations") or ())[:31] + [{
        "stage_index": int(journey.get("stage_index", 0)),
        "references": _references(observation_references, 16), "recorded_at": _now(),
    }]
    normalized_hypotheses = []
    for raw in islice(hypotheses or (), 8):
        hypothesis = str(raw.get("hypothesis") or "").strip()
        if not hypothesis:
            continue
        confidence = float(raw.get("confidence", 0.0))
        if not math.isfinite(confidence):
            raise ValueError("hypothesis confidence must be finite")
        normalized_hypotheses.append({
            "hypothesis": hypothesis[:1000],
            "confidence": max(0.0, min(1.0, confidence)),
            "evidence_references": _references(raw.get("evidence_references"), 16),
            "counterevidence_references": _references(raw.get("counterevidence_references"), 16),
            "authoritative": False,
        })
    # Retain earlier candidates when evidence is revised, including disagreements.
    result["hypothesis_history"] = deepcopy(list(journey.get("hypothesis_history") or ())[-4:])
    if hypotheses is not None:
        result["hypothesis_history"].append({
            "stage_index": journey.get("stage_index", 0),
            "previous": deepcopy(list(journey.get("hypotheses") or ())),
            "replacement": deepcopy(normalized_hypotheses),
            "observation_references": list(result["observations"][-1]["references"]),
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


def begin_intuition_inquiry(hunch: str, *, trigger_references: Iterable[Any],
                           question: str, countercheck: str, depth_budget: int = 1) -> dict[str, Any]:
    """Create a voluntary investigation of a hunch, not a truth certificate.

    The hunch needs no explanation. Investigation requires a question and a way
    to look for error. Requests neither execute actions nor retrieve memory.
    """
    for text in (hunch, countercheck):
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise ValueError('hunch and countercheck must be 1..1000 characters')
    journey = begin_self_inquiry(question, trigger_references=trigger_references,
                                 depth_budget=depth_budget)
    journey['intuition'] = {
        'schema': 'ina.intuition_inquiry/V1', 'hunch': hunch,
        'countercheck': countercheck, 'truth_status': 'unresolved',
        'explanation_required': False, 'confidence_is_calibrated': False,
        'automatic_execution': False, 'grants_authority': False,
    }
    journey['stages'][0]['prompt'] = question
    journey['stages'][0]['countercheck'] = countercheck
    return journey


__all__ = [
    "SCHEMA", "MAX_DEPTH", "begin_self_inquiry", "current_inquiry_request",
    "continue_self_inquiry",
    "begin_intuition_inquiry",
]
