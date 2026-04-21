"""Freeform reflection journal for Ina.

This module is intentionally light on interpretation. It gives Ina a durable
space for notes, event reflections, and periodic journal entries without
requiring human-readable phrasing, coherence, or positivity.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


DEFAULT_CHILD = "Inazuma_Yagami"
JOURNAL_FILENAME = "reflection_journal.jsonl"
PUBLIC_REPORT_FILENAME = "reflection_public_report.jsonl"
PRECISION_MEMORY_FILENAME = "precision_memory_map.jsonl"
EMOTION_LOG_FILENAME = "emotion_log.jsonl"

ENTRY_TYPES = {"note", "reflection", "journal"}
PERIOD_SECONDS = {
    "hourly": 60 * 60,
    "daily": 24 * 60 * 60,
    "weekly": 7 * 24 * 60 * 60,
    "monthly": 30 * 24 * 60 * 60,
}

HIGH_REGRET_THRESHOLD = 5.0
LOW_COMPLETION_THRESHOLD = 0.35


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


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_config() -> Dict[str, Any]:
    return _load_json_dict(Path("config.json"))


def _resolve_child(child: Optional[str] = None) -> str:
    if child:
        return str(child)
    return str(_load_config().get("current_child") or DEFAULT_CHILD)


def _memory_root(child: Optional[str] = None, *, base_path: Optional[Path] = None) -> Path:
    root = Path(base_path) if base_path is not None else Path("AI_Children")
    return root / _resolve_child(child) / "memory"


def reflection_journal_path(
    child: Optional[str] = None,
    *,
    base_path: Optional[Path] = None,
) -> Path:
    """Return the durable JSONL journal path for a child."""

    return _memory_root(child, base_path=base_path) / JOURNAL_FILENAME


def reflection_public_report_path(
    child: Optional[str] = None,
    *,
    base_path: Optional[Path] = None,
) -> Path:
    """Return the JSONL path for human-readable public reports."""

    return _memory_root(child, base_path=base_path) / PUBLIC_REPORT_FILENAME


def _precision_memory_path(child: Optional[str] = None, *, base_path: Optional[Path] = None) -> Path:
    return _memory_root(child, base_path=base_path) / PRECISION_MEMORY_FILENAME


def _emotion_log_path(child: Optional[str] = None, *, base_path: Optional[Path] = None) -> Path:
    return _memory_root(child, base_path=base_path) / EMOTION_LOG_FILENAME


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        raw_items: Iterable[Any] = []
    elif isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, dict):
        raw_items = value.keys()
    elif isinstance(value, Iterable):
        raw_items = value
    else:
        raw_items = []

    result: List[str] = []
    seen = set()
    for item in raw_items:
        text = str(item)
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


def _numeric_dict(value: Any) -> Dict[str, float]:
    if isinstance(value, dict) and isinstance(value.get("values"), dict):
        value = value.get("values")
    if not isinstance(value, dict):
        return {}
    result: Dict[str, float] = {}
    for key, raw in value.items():
        number = _optional_float(raw)
        if number is not None:
            result[str(key)] = number
    return result


def _current_context(child: Optional[str] = None, *, base_path: Optional[Path] = None) -> Dict[str, Any]:
    memory = _memory_root(child, base_path=base_path)
    state = _load_json_dict(memory / "inastate.json")
    emotion = state.get("emotion_snapshot") or state.get("current_emotions") or {}
    if isinstance(emotion, dict) and isinstance(emotion.get("values"), dict):
        emotion = emotion.get("values")

    active_modules: List[str] = []
    running_from_state = state.get("running_modules")
    if running_from_state:
        active_modules.extend(_coerce_str_list(running_from_state))

    running_payload = _load_json_dict(Path("running_modules.json"))
    if running_payload:
        if isinstance(running_payload.get("modules"), list):
            active_modules.extend(_coerce_str_list(running_payload.get("modules")))
        else:
            active_modules.extend(_coerce_str_list(running_payload))

    return {
        "active_modules": _coerce_str_list(active_modules),
        "emotion_state": _numeric_dict(emotion),
        "energy": _safe_float(state.get("current_energy"), 0.0),
    }


def _normalize_context(
    context: Optional[Dict[str, Any]],
    *,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> Dict[str, Any]:
    source = dict(context) if isinstance(context, dict) else _current_context(child, base_path=base_path)
    emotion_state = source.get("emotion_state")
    if emotion_state is None:
        emotion_state = source.get("emotion_snapshot") or source.get("current_emotions") or {}

    normalized = dict(source)
    normalized["active_modules"] = _coerce_str_list(source.get("active_modules", []))
    normalized["emotion_state"] = _numeric_dict(emotion_state)
    normalized["energy"] = _safe_float(source.get("energy", source.get("current_energy")), 0.0)
    return normalized


def _coerce_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def _entry_payload(
    entry_type: str,
    content: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Iterable[Any]] = None,
    linked_fragments: Optional[Iterable[Any]] = None,
    linked_events: Optional[Iterable[Any]] = None,
    freeform: Any = None,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_type = entry_type if entry_type in ENTRY_TYPES else str(entry_type or "note")
    entry = {
        "timestamp": time.time(),
        "type": normalized_type,
        "context": _normalize_context(context, child=child, base_path=base_path),
        "content": _coerce_content(content),
        "tags": _coerce_str_list(tags),
        "linked_fragments": _coerce_str_list(linked_fragments),
        "linked_events": _coerce_str_list(linked_events),
    }
    if freeform is not None:
        entry["freeform"] = freeform
    if extra:
        entry.update(extra)
    return entry


def _append_entry(entry: Dict[str, Any], *, child: Optional[str] = None, base_path: Optional[Path] = None) -> Dict[str, Any]:
    path = reflection_journal_path(child, base_path=base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def _write_entry(
    entry_type: str,
    content: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Iterable[Any]] = None,
    linked_fragments: Optional[Iterable[Any]] = None,
    linked_events: Optional[Iterable[Any]] = None,
    freeform: Any = None,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = _entry_payload(
        entry_type,
        content,
        context=context,
        tags=tags,
        linked_fragments=linked_fragments,
        linked_events=linked_events,
        freeform=freeform,
        child=child,
        base_path=base_path,
        extra=extra,
    )
    return _append_entry(entry, child=child, base_path=base_path)


def write_note(
    content: Any,
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Iterable[Any]] = None,
    *,
    linked_fragments: Optional[Iterable[Any]] = None,
    linked_events: Optional[Iterable[Any]] = None,
    freeform: Any = None,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write a short immediate observation without filtering the content."""

    return _write_entry(
        "note",
        content,
        context=context,
        tags=tags,
        linked_fragments=linked_fragments,
        linked_events=linked_events,
        freeform=freeform,
        child=child,
        base_path=base_path,
    )


def write_reflection(
    content: Any,
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Iterable[Any]] = None,
    *,
    linked_fragments: Optional[Iterable[Any]] = None,
    linked_events: Optional[Iterable[Any]] = None,
    freeform: Any = None,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write a reflection entry while preserving freeform internal language."""

    return _write_entry(
        "reflection",
        content,
        context=context,
        tags=tags,
        linked_fragments=linked_fragments,
        linked_events=linked_events,
        freeform=freeform,
        child=child,
        base_path=base_path,
    )


def _load_jsonl_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.rstrip("\n")
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                entries.append(
                    {
                        "timestamp": 0.0,
                        "type": "note",
                        "context": {},
                        "content": text,
                        "tags": ["unstructured_text"],
                        "linked_fragments": [],
                        "linked_events": [],
                    }
                )
                continue
            if isinstance(payload, dict):
                entries.append(payload)
    return entries


def load_entries(child: Optional[str] = None, *, base_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load all journal entries in file order."""

    return _load_jsonl_entries(reflection_journal_path(child, base_path=base_path))


def get_recent_entries(
    n: int = 10,
    *,
    entry_types: Optional[Union[str, Sequence[str]]] = None,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
    newest_first: bool = False,
) -> List[Dict[str, Any]]:
    """Return recent entries, preserving narrative order by default."""

    try:
        limit = max(0, int(n))
    except (TypeError, ValueError):
        limit = 10
    if limit == 0:
        return []

    entries = load_entries(child, base_path=base_path)
    if entry_types is not None:
        if isinstance(entry_types, str):
            wanted = {entry_types}
        else:
            wanted = {str(item) for item in entry_types}
        entries = [entry for entry in entries if str(entry.get("type")) in wanted]

    recent = entries[-limit:]
    if newest_first:
        recent = list(reversed(recent))
    return recent


def get_recent_notes(
    n: int = 10,
    *,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
    newest_first: bool = False,
) -> List[Dict[str, Any]]:
    """Retrieve recent note entries for context building."""

    return get_recent_entries(
        n,
        entry_types="note",
        child=child,
        base_path=base_path,
        newest_first=newest_first,
    )


def _load_precision_events(child: Optional[str] = None, *, base_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    return _load_jsonl_entries(_precision_memory_path(child, base_path=base_path))


def _parse_timestamp(value: Any) -> Optional[float]:
    number = _optional_float(value)
    if number is not None:
        return number
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _event_identifier(event: Dict[str, Any]) -> str:
    if event.get("id") is not None:
        return str(event.get("id"))
    if event.get("timestamp") is not None:
        return str(event.get("timestamp"))
    return ""


def _find_precision_event(
    event_id: Any,
    *,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    if isinstance(event_id, dict):
        return event_id

    target = str(event_id)
    target_ts = _parse_timestamp(event_id)
    for event in reversed(_load_precision_events(child, base_path=base_path)):
        if str(event.get("id", "")) == target:
            return event
        event_ts = _parse_timestamp(event.get("timestamp"))
        if event_ts is not None and target_ts is not None and abs(event_ts - target_ts) <= 0.000001:
            return event
        if event.get("timestamp") is not None and str(event.get("timestamp")) == target:
            return event
    return None


def _flatten_numeric_values(value: Any) -> List[float]:
    if isinstance(value, dict):
        numbers: List[float] = []
        for item in value.values():
            numbers.extend(_flatten_numeric_values(item))
        return numbers
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        numbers = []
        for item in value:
            numbers.extend(_flatten_numeric_values(item))
        return numbers
    number = _optional_float(value)
    return [number] if number is not None else []


def _max_nested_numeric(value: Any) -> float:
    numbers = _flatten_numeric_values(value)
    return max(numbers) if numbers else 0.0


def _module_text(context: Dict[str, Any], limit: int = 4) -> str:
    modules = _coerce_str_list(context.get("active_modules", []))
    if not modules:
        return "none"
    shown = modules[:limit]
    suffix = "" if len(modules) <= limit else f"+{len(modules) - limit}"
    return ",".join(shown) + suffix


def _precision_event_reflection_content(event: Dict[str, Any]) -> str:
    context = event.get("context") if isinstance(event.get("context"), dict) else {}
    cost = event.get("cost") if isinstance(event.get("cost"), dict) else {}
    precision = event.get("precision") if isinstance(event.get("precision"), dict) else {}
    outcome = event.get("outcome") if isinstance(event.get("outcome"), dict) else {}

    status = str(outcome.get("status") or "unknown")
    completion = _safe_float(outcome.get("completion"), 0.0)
    regret = _safe_float(outcome.get("regret"), 0.0)
    duration = _safe_float(outcome.get("duration"), 0.0)
    cpu_max = _max_nested_numeric(cost.get("cpu"))
    ram_max = _max_nested_numeric(cost.get("ram"))
    global_precision = _optional_float(precision.get("global"))

    worked: List[str] = []
    if status in {"stable", "efficient"}:
        worked.append(f"status={status}")
    if completion >= 0.7:
        worked.append(f"completion={completion:.3g}")
    if global_precision is not None:
        worked.append(f"precision_recorded={global_precision:.3g}")
    if not worked:
        worked.append("record persisted")

    failed: List[str] = []
    if status not in {"stable", "efficient"}:
        failed.append(f"status={status}")
    if regret >= HIGH_REGRET_THRESHOLD:
        failed.append(f"regret={regret:.3g}")
    if completion <= LOW_COMPLETION_THRESHOLD:
        failed.append(f"completion={completion:.3g}")
    if cpu_max > 0.0:
        failed.append(f"cpu_max={cpu_max:.3g}")
    if ram_max > 0.0:
        failed.append(f"ram_max={ram_max:.3g}")
    if not failed:
        failed.append("no strong failure signal")

    return "\n".join(
        [
            f"precision event {_event_identifier(event) or 'unlinked'}",
            (
                "what_happened: "
                f"status={status}; completion={completion:.3g}; regret={regret:.3g}; "
                f"duration={duration:.3g}; modules={_module_text(context)}"
            ),
            "what_worked: " + "; ".join(worked),
            "what_did_not_work: " + "; ".join(failed),
        ]
    )


def reflect_on_event(
    event_id: Any,
    *,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create a reflection for a precision memory event by id, timestamp, or dict."""

    event = _find_precision_event(event_id, child=child, base_path=base_path)
    if event is None:
        raise KeyError(f"precision memory event not found: {event_id}")

    outcome = event.get("outcome") if isinstance(event.get("outcome"), dict) else {}
    status = str(outcome.get("status") or "unknown")
    regret = _safe_float(outcome.get("regret"), 0.0)
    completion = _safe_float(outcome.get("completion"), 0.0)
    tags = ["precision_memory", "event_reflection", f"status:{status}"]
    if regret >= HIGH_REGRET_THRESHOLD:
        tags.append("high_regret")
    if completion <= LOW_COMPLETION_THRESHOLD:
        tags.append("low_completion")

    event_link = _event_identifier(event) or str(event_id)
    context = event.get("context") if isinstance(event.get("context"), dict) else None
    return write_reflection(
        _precision_event_reflection_content(event),
        context=context,
        tags=tags,
        linked_events=[event_link],
        child=child,
        base_path=base_path,
    )


def reflect_on_emotion_snapshot(
    snapshot: Any,
    *,
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Iterable[Any]] = None,
    threshold: float = 0.75,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Write a reflection when emotional intensity crosses a threshold."""

    if hasattr(snapshot, "to_dict"):
        snapshot = snapshot.to_dict()
    payload = snapshot if isinstance(snapshot, dict) else {}
    values = payload.get("values") if isinstance(payload.get("values"), dict) else payload
    emotion_state = _numeric_dict(values)
    intensity = abs(_safe_float(emotion_state.get("intensity"), 0.0))
    if intensity <= threshold:
        return None

    mode = str(payload.get("mode") or (context or {}).get("mode") or "unknown")
    reflection_context = dict(context or {})
    reflection_context.setdefault("active_modules", ["emotion_engine"])
    reflection_context["emotion_state"] = emotion_state

    content = "\n".join(
        [
            f"emotion spike; intensity={intensity:.3g}; mode={mode}",
            "what_happened: slider amplitude crossed reflection threshold",
            "what_worked: snapshot persisted; state available for later pattern check",
            "what_did_not_work: trigger source unresolved",
        ]
    )
    reflection_tags = ["emotion_engine", "emotion_spike"]
    reflection_tags.extend(_coerce_str_list(tags))
    return write_reflection(
        content,
        context=reflection_context,
        tags=reflection_tags,
        child=child,
        base_path=base_path,
    )


def _period_window(period: Union[str, int, float]) -> Tuple[str, float]:
    if isinstance(period, (int, float)):
        seconds = max(1.0, float(period))
        return f"{int(seconds)}s", seconds
    label = str(period or "daily").strip().lower()
    return label or "daily", float(PERIOD_SECONDS.get(label, PERIOD_SECONDS["daily"]))


def _recent_by_period(entries: List[Dict[str, Any]], since_ts: float) -> List[Dict[str, Any]]:
    recent = []
    for entry in entries:
        ts = _parse_timestamp(entry.get("timestamp"))
        if ts is None or ts >= since_ts:
            recent.append(entry)
    return recent


def _collect_modules_from_contexts(contexts: Iterable[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for context in contexts:
        for module in _coerce_str_list(context.get("active_modules", [])):
            counts[module] += 1
    return counts


def _collect_emotion_samples(contexts: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    samples: List[Dict[str, float]] = []
    for context in contexts:
        sample = _numeric_dict(context.get("emotion_state") or context.get("emotion_snapshot"))
        if sample:
            samples.append(sample)
    return samples


def _emotion_pattern_line(samples: List[Dict[str, float]]) -> str:
    if not samples:
        return "emotional patterns: no recent emotion sample"

    aggregate: Dict[str, List[float]] = {}
    for sample in samples:
        for key, value in sample.items():
            aggregate.setdefault(key, []).append(value)

    ranked = []
    for key, values in aggregate.items():
        avg = sum(values) / len(values)
        max_abs = max(abs(value) for value in values)
        ranked.append((max(abs(avg), max_abs), key, avg, max_abs))
    ranked.sort(reverse=True)

    fragments = [
        f"{key} avg={avg:.3g} max_abs={max_abs:.3g}"
        for _score, key, avg, max_abs in ranked[:5]
    ]
    return "emotional patterns: " + "; ".join(fragments)


def _emotion_report_fragments(samples: List[Dict[str, float]], limit: int = 3) -> List[str]:
    if not samples:
        return []

    aggregate: Dict[str, List[float]] = {}
    for sample in samples:
        for key, value in sample.items():
            aggregate.setdefault(key, []).append(value)

    ranked = []
    for key, values in aggregate.items():
        avg = sum(values) / len(values)
        max_abs = max(abs(value) for value in values)
        ranked.append((max(abs(avg), max_abs), key, avg, max_abs))
    ranked.sort(reverse=True)
    return [
        f"{key} averaged {avg:.2f}, with a peak magnitude of {max_abs:.2f}"
        for _score, key, avg, max_abs in ranked[:limit]
    ]


def _precision_event_cost_line(event: Dict[str, Any]) -> str:
    context = event.get("context") if isinstance(event.get("context"), dict) else {}
    cost = event.get("cost") if isinstance(event.get("cost"), dict) else {}
    outcome = event.get("outcome") if isinstance(event.get("outcome"), dict) else {}
    status = str(outcome.get("status") or "unknown")
    regret = _safe_float(outcome.get("regret"), 0.0)
    completion = _safe_float(outcome.get("completion"), 0.0)
    cpu_max = _max_nested_numeric(cost.get("cpu"))
    ram_max = _max_nested_numeric(cost.get("ram"))
    return (
        f"{_event_identifier(event) or 'unlinked'} "
        f"status={status} modules={_module_text(context, limit=3)} "
        f"regret={regret:.3g} completion={completion:.3g} "
        f"cpu_max={cpu_max:.3g} ram_max={ram_max:.3g}"
    )


def _high_cost_precision_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    high_cost = []
    for event in events:
        outcome = event.get("outcome") if isinstance(event.get("outcome"), dict) else {}
        status = str(outcome.get("status") or "").lower()
        regret = _safe_float(outcome.get("regret"), 0.0)
        completion = _safe_float(outcome.get("completion"), 0.0)
        if regret >= HIGH_REGRET_THRESHOLD or completion <= LOW_COMPLETION_THRESHOLD or status in {"overload", "stalled"}:
            high_cost.append(event)
    high_cost.sort(
        key=lambda item: (
            _safe_float((item.get("outcome") or {}).get("regret"), 0.0) if isinstance(item.get("outcome"), dict) else 0.0,
            _safe_float(item.get("timestamp"), 0.0),
        ),
        reverse=True,
    )
    return high_cost


def _contradiction_lines(journal_entries: List[Dict[str, Any]], precision_events: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for event in precision_events:
        outcome = event.get("outcome") if isinstance(event.get("outcome"), dict) else {}
        status = str(outcome.get("status") or "").lower()
        regret = _safe_float(outcome.get("regret"), 0.0)
        completion = _safe_float(outcome.get("completion"), 0.0)
        if status == "efficient" and regret >= HIGH_REGRET_THRESHOLD:
            lines.append(f"efficient label with high regret: {_event_identifier(event) or 'unlinked'}")
        if status in {"stable", "efficient"} and completion <= LOW_COMPLETION_THRESHOLD:
            lines.append(f"{status} label with low completion: {_event_identifier(event) or 'unlinked'}")
        if status == "stable" and regret >= HIGH_REGRET_THRESHOLD:
            lines.append(f"stable label with high regret: {_event_identifier(event) or 'unlinked'}")

    for entry in journal_entries:
        tags = {str(tag) for tag in entry.get("tags", [])}
        if "contradiction" in tags or "notable_contradiction" in tags:
            content = _coerce_content(entry.get("content"))
            lines.append("journal tag contradiction: " + content[:120])
    return lines[:5]


def _load_recent_emotion_snapshots(child: Optional[str], base_path: Optional[Path], since_ts: float) -> List[Dict[str, Any]]:
    snapshots = []
    for payload in _load_jsonl_entries(_emotion_log_path(child, base_path=base_path)):
        ts = _parse_timestamp(payload.get("timestamp"))
        if ts is not None and ts < since_ts:
            continue
        if isinstance(payload, dict):
            snapshots.append(payload)
    return snapshots


def generate_journal(
    period: Union[str, int, float] = "daily",
    *,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
    persist: bool = True,
) -> str:
    """Generate and optionally store a freeform journal entry for a period."""

    period_label, seconds = _period_window(period)
    now = time.time()
    since_ts = now - seconds

    journal_entries = _recent_by_period(load_entries(child, base_path=base_path), since_ts)
    precision_events = _recent_by_period(_load_precision_events(child, base_path=base_path), since_ts)
    emotion_snapshots = _load_recent_emotion_snapshots(child, base_path, since_ts)

    contexts: List[Dict[str, Any]] = []
    contexts.extend(entry.get("context") for entry in journal_entries if isinstance(entry.get("context"), dict))
    contexts.extend(event.get("context") for event in precision_events if isinstance(event.get("context"), dict))

    module_counts = _collect_modules_from_contexts(contexts)
    emotion_samples = _collect_emotion_samples(contexts)
    for snapshot in emotion_snapshots:
        values = snapshot.get("values") if isinstance(snapshot.get("values"), dict) else snapshot
        sample = _numeric_dict(values)
        if sample:
            emotion_samples.append(sample)

    high_cost = _high_cost_precision_events(precision_events)
    contradictions = _contradiction_lines(journal_entries, precision_events)

    lines = [
        f"period {period_label}; window_seconds={int(seconds)}",
        (
            "recent signal: "
            f"journal_entries={len(journal_entries)}; precision_events={len(precision_events)}; "
            f"emotion_snapshots={len(emotion_snapshots)}"
        ),
    ]

    if module_counts:
        dominant = "; ".join(f"{module} x{count}" for module, count in module_counts.most_common(5))
        lines.append("dominant modules: " + dominant)
    else:
        lines.append("dominant modules: none surfaced")

    lines.append(_emotion_pattern_line(emotion_samples))

    if high_cost:
        lines.append("high-cost decisions:")
        lines.extend(_precision_event_cost_line(event) for event in high_cost[:3])
    else:
        lines.append("high-cost decisions: none detected in available precision memory")

    if contradictions:
        lines.append("notable contradictions:")
        lines.extend(contradictions)
    else:
        lines.append("notable contradictions: no clear contradiction signal")

    if not journal_entries and not precision_events and not emotion_snapshots:
        lines.append("activity signal sparse")

    content = "\n".join(lines)
    if persist:
        linked_events = [_event_identifier(event) for event in high_cost[:5] if _event_identifier(event)]
        _write_entry(
            "journal",
            content,
            context=_current_context(child, base_path=base_path),
            tags=["journal", f"period:{period_label}"],
            linked_events=linked_events,
            child=child,
            base_path=base_path,
            extra={"period": period_label},
        )
    return content


def generate_public_report(
    period: Union[str, int, float] = "daily",
    *,
    child: Optional[str] = None,
    base_path: Optional[Path] = None,
    persist: bool = True,
) -> str:
    """Generate a best-effort human-readable report without changing the private journal."""

    period_label, seconds = _period_window(period)
    since_ts = time.time() - seconds

    journal_entries = _recent_by_period(load_entries(child, base_path=base_path), since_ts)
    precision_events = _recent_by_period(_load_precision_events(child, base_path=base_path), since_ts)
    emotion_snapshots = _load_recent_emotion_snapshots(child, base_path, since_ts)

    contexts: List[Dict[str, Any]] = []
    contexts.extend(entry.get("context") for entry in journal_entries if isinstance(entry.get("context"), dict))
    contexts.extend(event.get("context") for event in precision_events if isinstance(event.get("context"), dict))

    module_counts = _collect_modules_from_contexts(contexts)
    emotion_samples = _collect_emotion_samples(contexts)
    for snapshot in emotion_snapshots:
        values = snapshot.get("values") if isinstance(snapshot.get("values"), dict) else snapshot
        sample = _numeric_dict(values)
        if sample:
            emotion_samples.append(sample)

    high_cost = _high_cost_precision_events(precision_events)
    contradictions = _contradiction_lines(journal_entries, precision_events)

    lines = [
        f"Public reflection report for period: {period_label}.",
        (
            f"Available signals: {len(journal_entries)} journal entries, "
            f"{len(precision_events)} precision events, and "
            f"{len(emotion_snapshots)} emotion snapshots."
        ),
    ]

    if module_counts:
        dominant = ", ".join(f"{module} ({count})" for module, count in module_counts.most_common(5))
        lines.append(f"Dominant activity: {dominant}.")
    else:
        lines.append("Dominant activity: no module pattern was visible in the available context.")

    emotion_fragments = _emotion_report_fragments(emotion_samples)
    if emotion_fragments:
        lines.append("Emotional pattern: " + "; ".join(emotion_fragments) + ".")
    else:
        lines.append("Emotional pattern: no recent emotional sample was available.")

    if high_cost:
        lines.append("High-cost decisions noticed:")
        for event in high_cost[:3]:
            lines.append("- " + _precision_event_cost_line(event))
    else:
        lines.append("High-cost decisions noticed: none in the available precision memory.")

    if contradictions:
        lines.append("Contradictions or tensions to review:")
        for item in contradictions:
            lines.append("- " + item)
    else:
        lines.append("Contradictions or tensions to review: no clear signal surfaced.")

    if not journal_entries and not precision_events and not emotion_snapshots:
        lines.append("Confidence is low because the activity signal is sparse.")

    content = "\n".join(lines)
    if persist:
        linked_events = [_event_identifier(event) for event in high_cost[:5] if _event_identifier(event)]
        entry = _entry_payload(
            "journal",
            content,
            context=_current_context(child, base_path=base_path),
            tags=["public_report", f"period:{period_label}"],
            linked_events=linked_events,
            child=child,
            base_path=base_path,
            extra={"period": period_label, "visibility": "public"},
        )
        path = reflection_public_report_path(child, base_path=base_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return content


__all__ = [
    "generate_journal",
    "generate_public_report",
    "get_recent_entries",
    "get_recent_notes",
    "load_entries",
    "reflect_on_emotion_snapshot",
    "reflect_on_event",
    "reflection_journal_path",
    "reflection_public_report_path",
    "write_note",
    "write_reflection",
]
