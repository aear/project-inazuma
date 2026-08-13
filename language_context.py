"""Bounded, descriptive context for language candidate audits.

This module deliberately does not select meanings.  It records signals already
present when an expression is formed, then scores counterfactual candidates in
shadow mode so the established language path remains reversible and inspectable.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from discourse_context import build_discourse_context
from io_utils import file_lock


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_EVIDENCE_KINDS = (
    "human_feedback",
    "contextual_success",
    "self_consistency",
    "future_recall_match",
)
DEFAULT_POLICY = {
    "enabled": True,
    "shadow_only": True,
    "max_turns": 8,
    "max_turn_chars": 240,
    "max_memory_references": 4,
    "max_logic_entries": 3,
    "prediction_max_age_seconds": 1800.0,
    "prediction_min_confidence": 0.55,
    "prediction_min_clarity": 0.05,
    "reread_max_passes": 3,
}


def get_language_context_policy(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    raw = config.get("language_context_policy") if isinstance(config, Mapping) else None
    raw = raw if isinstance(raw, Mapping) else {}
    policy = dict(DEFAULT_POLICY)
    policy.update({key: raw[key] for key in policy if key in raw})
    policy["max_turns"] = max(0, min(12, int(policy["max_turns"])))
    policy["max_turn_chars"] = max(40, min(600, int(policy["max_turn_chars"])))
    policy["max_memory_references"] = max(0, min(8, int(policy["max_memory_references"])))
    policy["max_logic_entries"] = max(0, min(8, int(policy["max_logic_entries"])))
    policy["reread_max_passes"] = max(1, min(3, int(policy["reread_max_passes"])))
    return policy


def _words(value: Any) -> list[str]:
    if isinstance(value, str):
        return [word.lower() for word in _WORD_RE.findall(value)]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_words(item))
        return result
    return []


def _bounded_unique(values: Iterable[Any], limit: int = 16) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip().lower()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _age_seconds(timestamp: Any, now: datetime) -> Optional[float]:
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (now - parsed.astimezone(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compact_turns(context: Mapping[str, Any], policy: Mapping[str, Any]) -> list[dict]:
    scene = context.get("conversation_scene")
    scene = scene if isinstance(scene, Mapping) else {}
    turns = scene.get("turns") or context.get("conversation_context") or []
    result = []
    for raw in list(turns)[-int(policy["max_turns"]):]:
        if not isinstance(raw, Mapping):
            continue
        text = str(raw.get("text") or raw.get("content") or "")[: int(policy["max_turn_chars"])]
        if text:
            result.append({
                "speaker": str(raw.get("speaker") or raw.get("author_name") or "unknown")[:80],
                "direction": raw.get("direction"),
                "text": text,
                "source_id": raw.get("source_id") or raw.get("id"),
                "reply_to_id": raw.get("reply_to_id"),
            })
    return result


def _compact_memories(scene: Mapping[str, Any], limit: int) -> list[dict]:
    result = []
    for raw in list(scene.get("memory_references") or [])[:limit]:
        if not isinstance(raw, Mapping):
            continue
        result.append({
            "event_id": raw.get("event_id"),
            "cue": str(raw.get("cue") or "")[:80] or None,
            "summary": str(raw.get("summary") or "")[:240] or None,
            "tags": _bounded_unique(raw.get("tags") or [], 8),
        })
    return result


def _text_structure(text: str) -> dict:
    """Describe written form without turning it into separate language events."""
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    sentences = [
        part for part in re.split(r"(?<=[.!?])(?:\s+|$)|\n+", text) if part.strip()
    ]
    return {
        "character_count": len(text),
        "word_count": len(_words(text)),
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "line_count": len(text.splitlines()) if text else 0,
        "has_terminal_punctuation": bool(re.search(r"[.!?][\"')\]]*\s*$", text)),
    }


def _prediction_signal(raw: Any, policy: Mapping[str, Any], now: datetime) -> dict:
    prediction = raw if isinstance(raw, Mapping) else {}
    vector = prediction.get("predicted_vector")
    vector = vector if isinstance(vector, Mapping) else {}
    confidence = _number(vector.get("confidence"))
    clarity = _number(vector.get("clarity"))
    age = _age_seconds(prediction.get("timestamp"), now)
    fresh = age is not None and age <= float(policy["prediction_max_age_seconds"])
    eligible = bool(
        fresh
        and confidence >= float(policy["prediction_min_confidence"])
        and clarity >= float(policy["prediction_min_clarity"])
    )
    word = prediction.get("predicted_symbol_word")
    word = word if isinstance(word, Mapping) else {}
    return {
        "present": bool(prediction),
        "fresh": fresh,
        "eligible_for_shadow_score": eligible,
        "age_seconds": round(age, 3) if age is not None else None,
        "confidence": round(confidence, 4),
        "clarity": round(clarity, 4),
        "symbol_word_id": word.get("symbol_word_id"),
        "symbol": word.get("symbol"),
        "symbol_word_confidence": round(_number(word.get("confidence")), 4),
    }


def build_language_context_snapshot(
    context: Optional[Mapping[str, Any]],
    *,
    child: str,
    config: Optional[Mapping[str, Any]] = None,
    state_reader: Optional[Callable[..., Any]] = None,
    logic_reader: Optional[Callable[..., list[dict]]] = None,
    now: Optional[datetime] = None,
) -> dict:
    """Describe bounded language-relevant signals without choosing a mapping."""
    policy = get_language_context_policy(config)
    if not policy["enabled"]:
        return {"version": 1, "enabled": False, "shadow_only": True}
    context = context if isinstance(context, Mapping) else {}
    now = now or datetime.now(timezone.utc)
    if state_reader is None:
        from runtime_state import get_inastate
        state_reader = get_inastate
    scene = context.get("conversation_scene")
    scene = scene if isinstance(scene, Mapping) else {}
    signals = scene.get("signals")
    signals = signals if isinstance(signals, Mapping) else {}
    turns = _compact_turns(context, policy)
    memories = _compact_memories(scene, int(policy["max_memory_references"]))
    current_text = str(context.get("source_text") or "")
    current_words = _bounded_unique(_words(current_text), 64)
    topic_terms = _bounded_unique(scene.get("topic_terms") or [], 12)
    continuity = _bounded_unique(signals.get("continuity_terms") or [], 12)
    referents = _bounded_unique(
        [*topic_terms, *continuity, *(memory.get("cue") for memory in memories)], 16
    )
    supplied_discourse = scene.get("discourse")
    if not isinstance(supplied_discourse, Mapping):
        supplied_discourse = build_discourse_context(
            current_text, speaker=context.get("current_speaker") or "unknown",
            addressee={"id": child, "name": child, "is_self": True},
            self_identity={"id": child, "name": child, "is_self": True},
            current_subject=referents[0] if referents else None,
            mentioned_entities=scene.get("participants") or (),
        )
    supplied_state = context.get("language_state_signals")
    supplied_state = supplied_state if isinstance(supplied_state, Mapping) else {}

    def read_state(key: str, default: Any) -> Any:
        if key in supplied_state:
            return supplied_state.get(key, default)
        return state_reader(key, default, child=child)

    machine = read_state("machine_semantics", {})
    machine = machine if isinstance(machine, Mapping) else {}
    axes = machine.get("axes") if isinstance(machine.get("axes"), Mapping) else machine
    selected_axes = {}
    for key in ("signal_integrity", "temporal_coherence", "attention_value", "meaning_provenance", "novelty_safety", "io_bandwidth"):
        if key in axes:
            raw_axis = axes.get(key)
            axis_value = raw_axis.get("value") if isinstance(raw_axis, Mapping) else raw_axis
            selected_axes[key] = round(_number(axis_value), 4)
    affect = read_state("emotion_snapshot", {})
    affect = affect if isinstance(affect, Mapping) else {}
    affect_values = affect.get("values")
    affect_values = affect_values if isinstance(affect_values, Mapping) else affect
    compact_affect = {
        str(key): round(_number(value), 4)
        for key, value in list(affect_values.items())[:24]
        if isinstance(value, (int, float))
    }
    prediction = _prediction_signal(read_state("current_prediction", {}), policy, now)
    logic_signals = []
    if int(policy["max_logic_entries"]) and logic_reader is not False:
        if logic_reader is None:
            try:
                from logic_memory_store import recent_entries
                logic_reader = recent_entries
            except Exception:
                logic_reader = None
        if callable(logic_reader):
            try:
                entries = logic_reader(child, int(policy["max_logic_entries"]), config=config)
            except Exception:
                entries = []
            for entry in list(entries)[: int(policy["max_logic_entries"])]:
                if isinstance(entry, Mapping):
                    logic_signals.append({
                        "description": str(entry.get("description") or "")[:160],
                        "symbol_word_id": entry.get("symbol_word_id"),
                        "timestamp": entry.get("timestamp"),
                    })
    reply_ids = _bounded_unique(
        [turn.get("reply_to_id") for turn in turns if turn.get("reply_to_id")], 8
    )
    return {
        "version": 1,
        "enabled": True,
        "shadow_only": bool(policy["shadow_only"]),
        "captured_at": now.isoformat(),
        "message": {
            # This is the complete event text. Discord transport chunking occurs
            # later and must never masquerade as separate thoughts here.
            "text": current_text,
            "words": current_words,
            "channel": context.get("channel"),
            "written_structure": _text_structure(current_text),
        },
        "recent_scene": turns,
        "reply_ancestry": reply_ids,
        "candidate_referents": referents,
        "active_memory_references": memories,
        "social_context": {
            "current_speaker": dict(context.get("current_speaker") or {}),
            "participants": _bounded_unique(scene.get("participants") or [], 8),
            "is_dm": "dm" in _words(context.get("tags")),
            "is_high_trust": bool(context.get("is_high_trust")),
            "discourse": dict(supplied_discourse),
            "is_owner_friend": bool(context.get("is_owner_friend")),
        },
        "affective_state": compact_affect,
        "topic_continuity": {"topic_terms": topic_terms, "continuity_terms": continuity},
        "prediction": prediction,
        "machine_semantics": selected_axes,
        "logic_signals": logic_signals,
        "freshness": {
            "prediction_age_seconds": prediction.get("age_seconds"),
            "affect_age_seconds": _age_seconds(affect.get("timestamp"), now),
            "machine_semantics_age_seconds": _age_seconds(machine.get("updated_at"), now),
            "logic_age_seconds": [
                _age_seconds(item.get("timestamp"), now) for item in logic_signals
            ],
        },
        "ambiguity_set": [],
        "evidence_channels": {kind: [] for kind in _EVIDENCE_KINDS},
        "reread_policy": {
            "max_passes": int(policy["reread_max_passes"]),
            "trigger": "mapping_ambiguity",
            "may_support_expression_confidence": True,
            "mutates_mapping_confidence": False,
        },
        "provenance": {
            "message": "event_context",
            "scene": "conversation_scene",
            "memory": "accepted_bounded_references",
            "prediction": "inastate.current_prediction",
            "affect": "inastate.emotion_snapshot",
            "machine_semantics": "inastate.machine_semantics",
            "logic": "logic_memory_store.recent_entries",
        },
    }


def attach_ambiguity_set(snapshot: dict, ambiguity_set: Iterable[Mapping[str, Any]]) -> dict:
    result = dict(snapshot)
    result["ambiguity_set"] = [
        {
            "token": str(item.get("token") or "")[:64],
            "candidate_symbols": _bounded_unique(item.get("candidate_symbols") or [], 12),
        }
        for item in list(ambiguity_set)[:24]
        if isinstance(item, Mapping)
    ]
    return result


def _candidate_tags(candidate: Mapping[str, Any]) -> set[str]:
    values = []
    for key in ("tags", "sources", "contexts", "source", "channel", "adapter"):
        values.extend(_words(candidate.get(key)))
    return set(values)


def score_mapping_candidate(
    candidate: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    occurrence_context: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Return an explainable shadow score; never mutate candidate confidence."""
    tags = _candidate_tags(candidate)
    topic = snapshot.get("topic_continuity") if isinstance(snapshot.get("topic_continuity"), Mapping) else {}
    contextual_words = set(_words(topic.get("topic_terms"))) | set(_words(topic.get("continuity_terms")))
    contextual_words |= set(_words(snapshot.get("candidate_referents")))
    tag_matches = sorted(tags & contextual_words)
    memory_words = set()
    for memory in snapshot.get("active_memory_references") or []:
        if isinstance(memory, Mapping):
            memory_words.update(_words(memory.get("cue")))
            memory_words.update(_words(memory.get("tags")))
    memory_matches = sorted(tags & memory_words)
    occurrence_context = occurrence_context if isinstance(occurrence_context, Mapping) else {}
    neighbour_matches = sorted(tags & set(_words([
        occurrence_context.get("before"), occurrence_context.get("after")
    ])))
    prediction = snapshot.get("prediction") if isinstance(snapshot.get("prediction"), Mapping) else {}
    candidate_symbols = _bounded_unique([
        candidate.get("symbol"), candidate.get("symbol_id"), candidate.get("symbol_word_id"),
        candidate.get("symbol_word"), candidate.get("glyph"),
    ], 8)
    predicted = _bounded_unique([prediction.get("symbol"), prediction.get("symbol_word_id")], 4)
    prediction_match = bool(prediction.get("eligible_for_shadow_score") and set(candidate_symbols) & set(predicted))
    breakdown = {
        "topic_tag_overlap": len(tag_matches) * 3.0,
        "memory_tag_overlap": len(memory_matches) * 2.0,
        "written_neighbour_overlap": len(neighbour_matches) * 4.0,
        "prediction_identity_match": 4.0 if prediction_match else 0.0,
    }
    return {
        "score": round(sum(breakdown.values()), 4),
        "breakdown": breakdown,
        "matched_topic_tags": tag_matches,
        "matched_memory_tags": memory_matches,
        "matched_written_neighbours": neighbour_matches,
        "prediction_match": prediction_match,
    }


def audit_counterfactual_expression(
    tokens: list[str],
    current: list[Mapping[str, Any]],
    counterfactual: list[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
) -> dict:
    """Compare whole-expression coherence, not only local candidate scores."""
    def measure(selections: list[Mapping[str, Any]]) -> dict:
        symbols = [str(item.get("symbol") or "") for item in selections if item.get("symbol")]
        local = sum(_number(item.get("context_score")) for item in selections)
        repetition = max(0, len(symbols) - len(set(symbols))) * 2.0
        adjacent = 0.0
        for left, right in zip(selections, selections[1:]):
            adjacent += min(1.0, len(_candidate_tags(left.get("candidate") or {}) & _candidate_tags(right.get("candidate") or {})))
        coverage = len(selections) / max(1, len(tokens))
        total = local + adjacent + coverage - repetition
        return {
            "score": round(total, 4),
            "local_context": round(local, 4),
            "adjacent_consistency": round(adjacent, 4),
            "coverage": round(coverage, 4),
            "repetition_penalty": round(repetition, 4),
        }
    current_measure = measure(current)
    counter_measure = measure(counterfactual)
    delta = counter_measure["score"] - current_measure["score"]
    return {
        "current": current_measure,
        "counterfactual": counter_measure,
        "delta": round(delta, 4),
        "counterfactual_more_coherent": delta > 0.25,
        "changed_tokens": [
            tokens[index] for index in range(min(len(tokens), len(current), len(counterfactual)))
            if current[index].get("symbol") != counterfactual[index].get("symbol")
        ],
        "descriptive_only": True,
    }


def new_evidence_record(kind: str, payload: Mapping[str, Any]) -> dict:
    """Create a typed calibration observation without collapsing authority sources."""
    if kind not in _EVIDENCE_KINDS:
        raise ValueError(f"Unsupported language evidence kind: {kind!r}")
    return {
        "kind": kind,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "payload": dict(payload),
    }


def record_language_evidence(
    child: str,
    kind: str,
    payload: Mapping[str, Any],
    *,
    base_path: Optional[Path] = None,
) -> dict:
    """Append one explicitly typed observation; evidence kinds never collapse."""
    record = new_evidence_record(kind, payload)
    root = Path(base_path) if base_path is not None else Path("AI_Children")
    path = root / child / "memory" / "language_evidence.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(".lock")):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
    return {"status": "recorded", "kind": kind, "path": str(path)}


__all__ = [
    "attach_ambiguity_set",
    "audit_counterfactual_expression",
    "build_language_context_snapshot",
    "get_language_context_policy",
    "new_evidence_record",
    "record_language_evidence",
    "score_mapping_candidate",
]
