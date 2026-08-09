"""Bounded, ephemeral conversational working memory for communication turns."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "i", "if", "in", "is", "it", "of", "on", "or", "so",
    "that", "the", "this", "to", "was", "we", "were", "will", "with",
    "you", "your",
}


def _words(text: object) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(str(text or ""))]


def _bounded_text(value: object, limit: int) -> str:
    if limit <= 0:
        return ""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


@dataclass
class _Scene:
    key: str
    scene_id: str
    updated_monotonic: float
    turns: list[dict[str, Any]] = field(default_factory=list)
    memory_references: list[dict[str, Any]] = field(default_factory=list)
    sequence: int = 0


class ConversationSceneBuffer:
    """Keep a small per-channel scene without persisting or scanning memory."""

    def __init__(
        self,
        *,
        max_turns: int = 12,
        max_turn_chars: int = 800,
        max_total_chars: int = 6000,
        max_scenes: int = 32,
        ttl_seconds: float = 7200.0,
        max_memory_references: int = 4,
        max_memory_chars: int = 1000,
    ) -> None:
        self.max_turns = max(2, int(max_turns))
        self.max_turn_chars = max(80, int(max_turn_chars))
        self.max_total_chars = max(self.max_turn_chars, int(max_total_chars))
        self.max_scenes = max(1, int(max_scenes))
        self.ttl_seconds = max(60.0, float(ttl_seconds))
        self.max_memory_references = max(0, int(max_memory_references))
        self.max_memory_chars = max(0, int(max_memory_chars))
        self._scenes: OrderedDict[str, _Scene] = OrderedDict()
        self._lock = threading.RLock()

    def ingest_context(self, message: Any, context: object) -> None:
        """Seed prior backend history once, subject to the same normal bounds."""
        if not isinstance(context, (list, tuple)):
            return
        key = self._key_for(message)
        with self._lock:
            scene = self._get_scene(key)
            known = {
                (turn.get("source_id"), turn.get("speaker"), turn.get("text"))
                for turn in scene.turns
            }
            for raw in context[-self.max_turns :]:
                if not isinstance(raw, Mapping):
                    continue
                text = _bounded_text(raw.get("content") or raw.get("text"), self.max_turn_chars)
                if not text:
                    continue
                speaker = _bounded_text(
                    raw.get("author_name") or raw.get("speaker") or "unknown", 80
                )
                source_id = str(raw.get("id") or raw.get("message_id") or "") or None
                signature = (source_id, speaker, text)
                if signature in known:
                    continue
                scene.sequence += 1
                scene.turns.append({
                    "sequence": scene.sequence,
                    "source_id": source_id,
                    "direction": "outbound" if raw.get("is_self") else "inbound",
                    "speaker": speaker,
                    "text": text,
                    "created_at": raw.get("created_at") or raw.get("timestamp"),
                    "seeded": True,
                })
                known.add(signature)
            self._trim(scene)

    def observe(self, message: Any) -> dict[str, Any]:
        """Record one canonical message and return the resulting scene snapshot."""
        key = self._key_for(message)
        with self._lock:
            scene = self._get_scene(key)
            metadata = getattr(message, "metadata", None)
            if isinstance(metadata, Mapping):
                prior_scene = metadata.get("conversation_scene")
                if isinstance(prior_scene, Mapping):
                    self._remember_references(scene, prior_scene.get("memory_references"))
            text = _bounded_text(getattr(message, "text", ""), self.max_turn_chars)
            sender = getattr(message, "sender", None)
            speaker = _bounded_text(
                getattr(sender, "display_name", None)
                or getattr(sender, "internal_id", None)
                or "unknown",
                80,
            )
            scene.sequence += 1
            turn = {
                "sequence": scene.sequence,
                "source_id": str(getattr(message, "id", "") or "") or None,
                "direction": str(getattr(message, "direction", "inbound") or "inbound"),
                "speaker": speaker,
                "text": text,
                "created_at": getattr(message, "created_at", None),
                "reply_to_id": getattr(message, "reply_to_id", None),
                "seeded": False,
            }
            if text:
                scene.turns.append(turn)
            scene.updated_monotonic = time.monotonic()
            self._trim(scene)
            return self._snapshot(scene, current_turn=turn)

    def snapshot_for(self, message: Any) -> dict[str, Any]:
        key = self._key_for(message)
        with self._lock:
            scene = self._get_scene(key)
            return self._snapshot(scene, current_turn=None)

    def _key_for(self, message: Any) -> str:
        backend = str(getattr(message, "backend", "unknown") or "unknown")
        channel = getattr(message, "channel", None)
        channel_id = (
            getattr(channel, "internal_id", None)
            or getattr(channel, "backend_id", None)
            or getattr(channel, "name", None)
            or "unknown"
        )
        return f"{backend}:{channel_id}"

    def _get_scene(self, key: str) -> _Scene:
        now = time.monotonic()
        expired = [
            scene_key
            for scene_key, scene in self._scenes.items()
            if (now - scene.updated_monotonic) > self.ttl_seconds
        ]
        for scene_key in expired:
            self._scenes.pop(scene_key, None)
        scene = self._scenes.pop(key, None)
        if scene is None:
            digest = hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()[:12]
            scene = _Scene(key=key, scene_id=f"scene_{digest}", updated_monotonic=now)
        self._scenes[key] = scene
        while len(self._scenes) > self.max_scenes:
            self._scenes.popitem(last=False)
        return scene

    def _trim(self, scene: _Scene) -> None:
        scene.turns = scene.turns[-self.max_turns :]
        total = sum(len(turn.get("text", "")) for turn in scene.turns)
        while len(scene.turns) > 1 and total > self.max_total_chars:
            total -= len(scene.turns.pop(0).get("text", ""))
        scene.memory_references = [
            ref
            for ref in scene.memory_references
            if scene.sequence - int(ref.get("offered_sequence", scene.sequence)) <= 4
        ][-self.max_memory_references :]

    def _remember_references(self, scene: _Scene, references: object) -> None:
        if self.max_memory_references <= 0 or not isinstance(references, (list, tuple)):
            return
        bounded = bound_memory_references(
            references,
            max_items=self.max_memory_references,
            max_chars=self.max_memory_chars,
        )
        by_key = {
            (ref.get("event_id"), ref.get("cue"), ref.get("summary")): ref
            for ref in scene.memory_references
        }
        for ref in bounded:
            ref = dict(ref)
            ref["offered_sequence"] = scene.sequence
            by_key[(ref.get("event_id"), ref.get("cue"), ref.get("summary"))] = ref
        scene.memory_references = list(by_key.values())[-self.max_memory_references :]

    def _snapshot(self, scene: _Scene, *, current_turn: Optional[dict[str, Any]]) -> dict[str, Any]:
        turns = [dict(turn) for turn in scene.turns]
        current_text = str((current_turn or {}).get("text") or "")
        current_words = set(_words(current_text)) - _STOPWORDS
        prior_words = {
            word
            for turn in turns[:-1] if current_turn is not None
            for word in _words(turn.get("text"))
            if word not in _STOPWORDS
        }
        frequencies: dict[str, int] = {}
        for turn in turns:
            for word in _words(turn.get("text")):
                if word in _STOPWORDS or len(word) < 3:
                    continue
                frequencies[word] = frequencies.get(word, 0) + 1
        topics = sorted(frequencies, key=lambda word: (-frequencies[word], word))[:8]
        participants = list(dict.fromkeys(
            turn.get("speaker") for turn in turns if turn.get("speaker")
        ))[-8:]
        reply_expected = bool(
            current_turn
            and current_turn.get("direction") == "inbound"
            and (current_text.rstrip().endswith("?") or current_turn.get("reply_to_id"))
        )
        return {
            "version": 1,
            "scene_id": scene.scene_id,
            "turn_count": len(turns),
            "current_turn_id": (current_turn or {}).get("source_id"),
            "turns": turns,
            "participants": participants,
            "topic_terms": topics,
            "memory_references": [dict(ref) for ref in scene.memory_references],
            "signals": {
                "has_prior_context": len(turns) > 1,
                "reply_expected": reply_expected,
                "current_is_question": current_text.rstrip().endswith("?"),
                "continuity_terms": sorted(current_words & prior_words)[:8],
                "participant_count": len(participants),
            },
            "bounds": {
                "max_turns": self.max_turns,
                "max_turn_chars": self.max_turn_chars,
                "max_total_chars": self.max_total_chars,
                "max_memory_references": self.max_memory_references,
                "ttl_seconds": self.ttl_seconds,
            },
        }


def bound_memory_references(
    references: object,
    *,
    max_items: int = 4,
    max_chars: int = 1000,
) -> list[dict[str, Any]]:
    """Sanitise relevant-memory offers without reading memory itself."""
    if not isinstance(references, (list, tuple)):
        return []
    remaining = max(0, int(max_chars))
    result: list[dict[str, Any]] = []
    for raw in references:
        if len(result) >= max(0, int(max_items)) or remaining <= 0:
            break
        if not isinstance(raw, Mapping):
            continue
        summary = _bounded_text(raw.get("summary") or raw.get("narrative"), min(320, remaining))
        if not summary:
            continue
        cue = _bounded_text(raw.get("cue") or raw.get("word"), 80)
        result.append({
            "event_id": str(raw.get("event_id") or "") or None,
            "cue": cue or None,
            "summary": summary,
            "tags": [str(tag)[:64] for tag in (raw.get("tags") or [])[:8]],
            "source": str(raw.get("source") or "grounded_experience")[:80],
        })
        remaining -= len(summary)
    return result


def scene_with_memory_references(
    scene: object,
    references: object,
    *,
    max_items: int = 4,
    max_chars: int = 1000,
) -> dict[str, Any]:
    """Return a copied scene enriched with a bounded relevant-memory offer."""
    snapshot = dict(scene) if isinstance(scene, Mapping) else {}
    snapshot["memory_references"] = bound_memory_references(
        references, max_items=max_items, max_chars=max_chars
    )
    return snapshot


def scene_with_memory_consideration(
    scene: object,
    consideration: object,
    *,
    max_items: int = 4,
    max_chars: int = 1000,
) -> dict[str, Any]:
    """Attach accepted references and bounded rejection decisions to a scene copy."""
    snapshot = dict(scene) if isinstance(scene, Mapping) else {}
    result = consideration if isinstance(consideration, Mapping) else {}
    accepted = bound_memory_references(
        result.get("accepted"), max_items=max_items, max_chars=max(0, max_chars // 2)
    )
    accepted_by_key = {
        (ref.get("event_id"), ref.get("cue")): ref
        for ref in result.get("accepted", [])
        if isinstance(ref, Mapping)
    }
    description_budget = max(0, max_chars - sum(len(ref.get("summary", "")) for ref in accepted))
    considered_count = min(
        max_items, len(result.get("accepted", [])) + len(result.get("rejected", []))
    )
    description_slot = description_budget // max(1, considered_count)
    for ref in accepted:
        original = accepted_by_key.get((ref.get("event_id"), ref.get("cue")))
        if isinstance(original, Mapping) and isinstance(original.get("consideration"), Mapping):
            decision = dict(original["consideration"])
            description = _bounded_text(decision.get("description"), min(420, description_slot, description_budget))
            if description:
                decision["description"] = description
                description_budget -= len(description)
            else:
                decision.pop("description", None)
            ref["consideration"] = decision
    rejected = []
    for raw in result.get("rejected", []):
        if not isinstance(raw, Mapping) or len(rejected) >= max_items:
            continue
        decision = dict(raw.get("consideration")) if isinstance(raw.get("consideration"), Mapping) else {}
        description = _bounded_text(decision.get("description"), min(420, description_slot, description_budget))
        if description:
            decision["description"] = description
            description_budget -= len(description)
        else:
            decision.pop("description", None)
        rejected.append({
            "event_id": str(raw.get("event_id") or "") or None,
            "cue": _bounded_text(raw.get("cue"), 80) or None,
            "consideration": decision,
        })
    snapshot["memory_candidates_considered"] = (
        len(result.get("accepted", [])) + len(result.get("rejected", []))
    )
    snapshot["memory_references"] = accepted
    snapshot["memory_rejections"] = rejected
    return snapshot


__all__ = [
    "ConversationSceneBuffer",
    "bound_memory_references",
    "scene_with_memory_consideration",
    "scene_with_memory_references",
]
