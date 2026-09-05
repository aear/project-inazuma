"""Plural, witness-led identity coordination for Ina.

Identity, ego, id, and shadow are functional lenses, not diagnoses or four
separate agents.  The manager preserves tensions between their witnesses and
never treats consistency as a prerequisite for a coherent continuing self.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import uuid

from io_utils import atomic_write_json, file_lock, flush_for_durability, load_json_dict


SCHEMA = "ina.identity_system/V1"
WITNESS_SCHEMA = "ina.identity_witness/V1"
TENSION_SCHEMA = "ina.identity_tension/V1"
ASPECTS = frozenset({"identity", "ego", "id", "shadow"})
TENSION_STATES = frozenset({"open", "held", "exploring", "integrating", "reframed"})
MAX_WITNESSES = 256
MAX_TENSIONS = 128


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _refs(values: Iterable[Any] | None, limit: int = 32) -> list[str]:
    result = []
    for value in values or ():
        item = str(value or "").strip()
        if item and item not in result:
            result.append(item[:500])
        if len(result) >= limit:
            break
    return result


def create_identity_witness(
    aspect: str, stance: str, *, evidence_references: Iterable[Any],
    confidence: float = 0.5, context_references: Iterable[Any] | None = None,
) -> dict[str, Any]:
    role = str(aspect or "").strip().lower()
    if role not in ASPECTS:
        raise ValueError("identity witness aspect must be identity, ego, id, or shadow")
    claim = " ".join(str(stance or "").split())
    if not claim or len(claim) > 500:
        raise ValueError("identity stance must be 1..500 characters")
    evidence = _refs(evidence_references)
    if not evidence:
        raise ValueError("identity witnesses require evidence references")
    certainty = max(0.0, min(1.0, float(confidence)))
    return {
        "schema": WITNESS_SCHEMA, "witness_id": f"identity_witness_{uuid.uuid4().hex}",
        "aspect": role, "stance": claim, "confidence": round(certainty, 6),
        "evidence_references": evidence, "context_references": _refs(context_references),
        "authoritative": False, "created_at": _now(),
    }


def create_identity_tension(
    witness_ids: Iterable[Any], *, description: str,
    relation: str = "in_tension_with",
) -> dict[str, Any]:
    witnesses = _refs(witness_ids, 8)
    if len(witnesses) < 2:
        raise ValueError("identity tension requires at least two witnesses")
    detail = " ".join(str(description or "").split())
    if not detail or len(detail) > 1000:
        raise ValueError("tension description must be 1..1000 characters")
    return {
        "schema": TENSION_SCHEMA, "tension_id": f"identity_tension_{uuid.uuid4().hex}",
        "witness_ids": witnesses, "description": detail,
        "relation": str(relation or "in_tension_with")[:80], "state": "open",
        "resolution_required": False, "created_at": _now(), "updated_at": _now(),
    }


def identity_continuity_candidates(
    state: Mapping[str, Any], *, cue: str = "", limit: int = 32,
) -> list[dict[str, Any]]:
    """Project identity records as read-only Continuity Engine witnesses."""
    terms = {part.casefold() for part in str(cue).replace("_", " ").split() if len(part) > 1}
    candidates = []
    for witness in list(state.get("witnesses") or ())[-MAX_WITNESSES:]:
        text = str(witness.get("stance") or "")
        tags = ["identity_system", str(witness.get("aspect") or "unknown")]
        haystack = {part.casefold() for value in [text, *tags]
                    for part in str(value).replace("_", " ").split() if len(part) > 1}
        if terms and not terms.intersection(haystack):
            continue
        candidates.append({
            "id": witness.get("witness_id"), "summary": text, "tags": tags,
            "source": "identity_system", "memory_type": "identity",
            "confidence": witness.get("confidence", 0.5),
            "timestamp": witness.get("created_at"),
            "causal_references": list(witness.get("evidence_references") or ())[:8],
        })
        if len(candidates) >= max(0, min(64, int(limit))):
            break
    remaining = max(0, min(64, int(limit)) - len(candidates))
    for tension in list(state.get("tensions") or ())[-MAX_TENSIONS:]:
        if remaining <= 0:
            break
        text = str(tension.get("description") or "")
        tags = ["identity_system", "identity_tension", str(tension.get("state") or "open")]
        haystack = {part.casefold() for value in [text, *tags]
                    for part in str(value).replace("_", " ").split() if len(part) > 1}
        if terms and not terms.intersection(haystack):
            continue
        candidates.append({
            "id": tension.get("tension_id"), "summary": text, "tags": tags,
            "source": "identity_system", "memory_type": "identity", "confidence": 0.5,
            "timestamp": tension.get("updated_at"),
            "causal_references": list(tension.get("witness_ids") or ())[:8],
        })
        remaining -= 1
    return candidates


class IdentityManager:
    """Persist bounded identity witnesses while retaining unresolved conflict."""

    def __init__(
        self, child: str = "Inazuma_Yagami", *, root_path: Path | str = "AI_Children",
    ) -> None:
        self.child = str(child)
        self.identity_path = Path(root_path) / self.child / "identity"
        self.state_path = self.identity_path / "identity_system.json"
        self.events_path = self.identity_path / "identity_events.jsonl"
        self.identity_path.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        state = load_json_dict(self.state_path)
        if state.get("schema") != SCHEMA:
            state = {
                "schema": SCHEMA, "child": self.child, "witnesses": [], "tensions": [],
                "created_at": _now(), "updated_at": _now(),
            }
        return state

    def _append_event(self, event: Mapping[str, Any]) -> None:
        line = json.dumps(dict(event), ensure_ascii=False, separators=(",", ":")) + "\n"
        with file_lock(self.events_path.with_suffix(".jsonl.lock")):
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                flush_for_durability(handle, self.events_path)

    def add_witnesses(self, witnesses: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        state = self.load()
        known = {item.get("witness_id") for item in state["witnesses"]}
        known_content = {(
            item.get("aspect"), item.get("stance"), tuple(item.get("evidence_references") or ()),
        ) for item in state["witnesses"]}
        added = []
        for raw in list(witnesses)[:32]:
            witness = dict(raw)
            if witness.get("schema") != WITNESS_SCHEMA:
                raise ValueError("all identity witnesses must use the identity witness schema")
            content_key = (
                witness.get("aspect"), witness.get("stance"),
                tuple(witness.get("evidence_references") or ()),
            )
            if witness.get("witness_id") not in known and content_key not in known_content:
                added.append(witness)
                known.add(witness.get("witness_id"))
                known_content.add(content_key)
        state["witnesses"] = (state["witnesses"] + added)[-MAX_WITNESSES:]
        state["updated_at"] = _now()
        atomic_write_json(self.state_path, state, indent=2, ensure_ascii=False)
        for witness in added:
            self._append_event({
                "event": "identity_witness_added", "witness_id": witness["witness_id"],
                "aspect": witness["aspect"], "timestamp": _now(),
            })
        return state

    def add_tension(self, tension: Mapping[str, Any]) -> dict[str, Any]:
        if tension.get("schema") != TENSION_SCHEMA:
            raise ValueError("a valid identity tension is required")
        state = self.load()
        known = {item.get("witness_id") for item in state["witnesses"]}
        if not set(tension.get("witness_ids") or ()).issubset(known):
            raise ValueError("tension witnesses must already belong to this identity system")
        state["tensions"] = (state["tensions"] + [dict(tension)])[-MAX_TENSIONS:]
        state["updated_at"] = _now()
        atomic_write_json(self.state_path, state, indent=2, ensure_ascii=False)
        self._append_event({
            "event": "identity_tension_added", "tension_id": tension["tension_id"],
            "state": tension["state"], "timestamp": _now(),
        })
        return state

    def set_tension_state(self, tension_id: str, state: str) -> dict[str, Any]:
        selected = str(state or "").strip().lower()
        if selected not in TENSION_STATES:
            raise ValueError("unknown identity tension state")
        system = self.load()
        tension = next((item for item in system["tensions"] if item.get("tension_id") == tension_id), None)
        if tension is None:
            raise KeyError(tension_id)
        tension["state"] = selected
        tension["updated_at"] = _now()
        system["updated_at"] = _now()
        atomic_write_json(self.state_path, system, indent=2, ensure_ascii=False)
        self._append_event({
            "event": "identity_tension_state", "tension_id": tension_id,
            "state": selected, "timestamp": _now(),
        })
        return dict(tension)

    def summary(self) -> dict[str, Any]:
        state = self.load()
        return {
            "schema": SCHEMA, "child": self.child,
            "witness_counts": {aspect: sum(item.get("aspect") == aspect for item in state["witnesses"])
                               for aspect in sorted(ASPECTS)},
            "tension_counts": {status: sum(item.get("state") == status for item in state["tensions"])
                               for status in sorted(TENSION_STATES)},
            "updated_at": state["updated_at"],
        }


__all__ = [
    "SCHEMA", "WITNESS_SCHEMA", "TENSION_SCHEMA", "ASPECTS", "IdentityManager",
    "create_identity_witness", "create_identity_tension", "identity_continuity_candidates",
]
