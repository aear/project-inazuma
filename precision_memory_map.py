"""Experience memory for precision/cost/outcome relationships.

The precision memory map is intentionally advisory. It records what
precision was used in a context, what it cost, and how it turned out, then
returns suggestions from nearby past contexts without changing precision
settings itself.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_CHILD = "Inazuma_Yagami"
LOG_FILENAME = "precision_memory_map.jsonl"
OUTCOME_STATUSES = {"stable", "overload", "efficient", "stalled"}
SUCCESS_STATUSES = {"stable", "efficient"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _optional_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _clamp01(value: Any, default: float = 0.0) -> float:
    result = _safe_float(value, default)
    return max(0.0, min(1.0, result))


def _load_config() -> Dict[str, Any]:
    path = Path("config.json")
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_child(child: Optional[str] = None) -> str:
    if child:
        return str(child)
    return str(_load_config().get("current_child") or DEFAULT_CHILD)


def precision_memory_log_path(
    child: Optional[str] = None,
    *,
    base_path: Optional[Path] = None,
) -> Path:
    root = Path(base_path) if base_path is not None else Path("AI_Children")
    return root / _resolve_child(child) / "memory" / LOG_FILENAME


def _coerce_module_list(value: Any) -> List[str]:
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = []
    modules: List[str] = []
    seen = set()
    for item in values:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        modules.append(name)
        seen.add(name)
    return modules


def _numeric_dict(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, float] = {}
    for key, raw in value.items():
        number = _optional_float(raw)
        if number is None:
            continue
        result[str(key)] = number
    return result


def _normalize_context(context: Dict[str, Any]) -> Dict[str, Any]:
    source = context if isinstance(context, dict) else {}
    emotion_state = source.get("emotion_state")
    if isinstance(emotion_state, dict) and isinstance(emotion_state.get("values"), dict):
        emotion_state = emotion_state.get("values")
    return {
        "active_modules": _coerce_module_list(source.get("active_modules", [])),
        "queue_pressure": _safe_float(source.get("queue_pressure"), 0.0),
        "emotion_state": _numeric_dict(emotion_state),
        "energy": _safe_float(source.get("energy"), 0.0),
    }


def _normalize_cost(cost: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    source = cost if isinstance(cost, dict) else {}
    return {
        "cpu": _numeric_dict(source.get("cpu")),
        "ram": _numeric_dict(source.get("ram")),
    }


def _normalize_precision(precision: Dict[str, Any]) -> Dict[str, Any]:
    source = precision if isinstance(precision, dict) else {"global": precision}
    if "global" in source:
        global_precision = _safe_float(source.get("global"), 0.0)
    elif "max_precision" in source:
        global_precision = _safe_float(source.get("max_precision"), 0.0)
    else:
        global_precision = 0.0
    per_module = source.get("per_module")
    if per_module is None:
        per_module = source.get("modules")
    return {
        "global": global_precision,
        "per_module": _numeric_dict(per_module),
    }


def _average(values: Iterable[float]) -> float:
    numbers = []
    for value in values:
        number = _optional_float(value)
        if number is not None:
            numbers.append(number)
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def calculate_regret(cost: Dict[str, Any], outcome: Dict[str, Any]) -> float:
    """Estimate regret from cost and completion quality.

    This mirrors the initial policy from the design brief:
    ``(cpu_avg * 0.6 + ram_avg * 0.4) * (1 - completion)``.
    """

    normalized_cost = _normalize_cost(cost)
    normalized_outcome = outcome if isinstance(outcome, dict) else {}
    cpu_avg = _average(normalized_cost.get("cpu", {}).values())
    ram_avg = _average(normalized_cost.get("ram", {}).values())
    completion = _clamp01(normalized_outcome.get("completion"), 0.0)
    return (cpu_avg * 0.6 + ram_avg * 0.4) * (1.0 - completion)


def _normalize_outcome(outcome: Dict[str, Any], cost: Dict[str, Any]) -> Dict[str, Any]:
    source = outcome if isinstance(outcome, dict) else {}
    status = str(source.get("status") or "stable").strip().lower()
    if status not in OUTCOME_STATUSES:
        status = "stable"
    regret = _optional_float(source.get("regret"))
    normalized = {
        "status": status,
        "duration": max(0.0, _safe_float(source.get("duration"), 0.0)),
        "completion": _clamp01(source.get("completion"), 0.0),
        "regret": regret if regret is not None else calculate_regret(cost, source),
    }
    return normalized


def log_event(
    context: Dict[str, Any],
    cost: Dict[str, Any],
    precision: Dict[str, Any],
    outcome: Dict[str, Any],
    *,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Append a precision decision event to memory and return the entry."""

    normalized_cost = _normalize_cost(cost)
    entry = {
        "timestamp": time.time(),
        "context": _normalize_context(context),
        "cost": normalized_cost,
        "precision": _normalize_precision(precision),
        "outcome": _normalize_outcome(outcome, normalized_cost),
    }
    path = precision_memory_log_path(child, base_path=base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _load_events(
    child: Optional[str] = None,
    *,
    base_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    path = precision_memory_log_path(child, base_path=base_path)
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def _module_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {str(item) for item in left if str(item)}
    right_set = {str(item) for item in right if str(item)}
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _emotion_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    keys = set(left.keys()) | set(right.keys())
    if not keys:
        return 1.0
    distance = 0.0
    for key in keys:
        distance += min(1.0, abs(_safe_float(left.get(key), 0.0) - _safe_float(right.get(key), 0.0)))
    return max(0.0, 1.0 - (distance / len(keys)))


def _queue_similarity(left: Any, right: Any) -> float:
    distance = min(1.0, abs(_safe_float(left, 0.0) - _safe_float(right, 0.0)))
    return max(0.0, 1.0 - distance)


def score_context_similarity(context: Dict[str, Any], past_context: Dict[str, Any]) -> float:
    """Score context closeness using modules, emotion, and queue pressure."""

    current = _normalize_context(context)
    past = _normalize_context(past_context)
    module_score = _module_similarity(current["active_modules"], past["active_modules"])
    emotion_score = _emotion_similarity(current["emotion_state"], past["emotion_state"])
    queue_score = _queue_similarity(current["queue_pressure"], past["queue_pressure"])
    return max(0.0, min(1.0, (module_score * 0.4) + (emotion_score * 0.35) + (queue_score * 0.25)))


def get_similar_contexts(
    context: Dict[str, Any],
    k: int = 5,
    *,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return the closest past precision events for a context."""

    try:
        limit = max(0, int(k))
    except (TypeError, ValueError):
        limit = 5
    if limit == 0:
        return []

    ranked: List[Dict[str, Any]] = []
    for event in _load_events(child, base_path=base_path):
        score = score_context_similarity(context, event.get("context") or {})
        ranked_event = dict(event)
        ranked_event["similarity"] = score
        ranked.append(ranked_event)
    ranked.sort(
        key=lambda item: (
            _safe_float(item.get("similarity"), 0.0),
            _safe_float(item.get("timestamp"), 0.0),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _success_weight(event: Dict[str, Any]) -> float:
    outcome = event.get("outcome") if isinstance(event.get("outcome"), dict) else {}
    status = str(outcome.get("status") or "").strip().lower()
    if status not in SUCCESS_STATUSES:
        return 0.0
    completion = _clamp01(outcome.get("completion"), 0.0)
    if completion <= 0.0:
        return 0.0
    regret = max(0.0, _safe_float(outcome.get("regret"), 0.0))
    status_bonus = 1.1 if status == "efficient" else 1.0
    return completion * status_bonus * (1.0 / (1.0 + regret))


def _weighted_average(values: Iterable[Tuple[float, float]]) -> Optional[float]:
    total_weight = 0.0
    total = 0.0
    for value, weight in values:
        if weight <= 0.0:
            continue
        total += value * weight
        total_weight += weight
    if total_weight <= 0.0:
        return None
    return total / total_weight


def suggest_precision(
    context: Dict[str, Any],
    *,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """Suggest precision from successful nearest-neighbor experiences."""

    similar = get_similar_contexts(context, k=k, child=child, base_path=base_path)
    weighted_events: List[Tuple[Dict[str, Any], float]] = []
    for event in similar:
        weight = _safe_float(event.get("similarity"), 0.0) * _success_weight(event)
        if weight > 0.0:
            weighted_events.append((event, weight))

    global_values: List[Tuple[float, float]] = []
    for event, weight in weighted_events:
        precision = event.get("precision") if isinstance(event.get("precision"), dict) else {}
        value = _optional_float(precision.get("global"))
        if value is not None:
            global_values.append((value, weight))

    global_precision = _weighted_average(global_values)
    current_context = _normalize_context(context)
    modules = set(current_context["active_modules"])
    if not modules:
        for event, _weight in weighted_events:
            precision = event.get("precision") if isinstance(event.get("precision"), dict) else {}
            per_module = precision.get("per_module") if isinstance(precision.get("per_module"), dict) else {}
            modules.update(str(key) for key in per_module)

    per_module: Dict[str, float] = {}
    for module in sorted(modules):
        module_values: List[Tuple[float, float]] = []
        for event, weight in weighted_events:
            precision = event.get("precision") if isinstance(event.get("precision"), dict) else {}
            module_map = precision.get("per_module") if isinstance(precision.get("per_module"), dict) else {}
            value = _optional_float(module_map.get(module))
            if value is not None:
                module_values.append((value, weight))
        average = _weighted_average(module_values)
        if average is not None:
            per_module[module] = average

    total_weight = sum(weight for _event, weight in weighted_events)
    confidence = min(1.0, total_weight / max(1, len(similar)))
    return {
        "global": global_precision,
        "per_module": per_module,
        "confidence": confidence,
        "similar_events": len(similar),
        "successful_events": len(weighted_events),
        "basis": "nearest_neighbors" if weighted_events else "insufficient_successful_memory",
    }


def compress_precision(precision: Dict[str, Any], *, digits: int = 4) -> Dict[str, Any]:
    """Return a rounded copy for efficiency comparisons, preserving raw logs."""

    normalized = _normalize_precision(precision)
    return {
        "global": round(normalized["global"], digits),
        "per_module": {
            module: round(value, digits)
            for module, value in normalized["per_module"].items()
        },
    }
