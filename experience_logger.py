"""Experience-centered logging utilities for Ina.

This module introduces structured Event and Episode records that capture
multimodal context, allowing vocabulary to be grounded in lived
experiences instead of isolated fragments.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json


def _now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _generate_identifier(prefix: str) -> str:
    """Create a collision-resistant identifier for events/episodes."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{stamp}"


@dataclass
class EventRecord:
    """Structured description of a single experience sample."""

    id: str
    timestamp: str
    situation_tags: List[str] = field(default_factory=list)
    perceived_entities: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    outcome: Dict[str, Any] = field(default_factory=dict)
    internal_state: Dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    episode_id: Optional[str] = None
    word_usage: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventRecord":
        return cls(
            id=data.get("id", _generate_identifier("evt")),
            timestamp=data.get("timestamp", _now_iso()),
            situation_tags=list(data.get("situation_tags", [])),
            perceived_entities=list(data.get("perceived_entities", [])),
            actions=list(data.get("actions", [])),
            outcome=dict(data.get("outcome", {})),
            internal_state=dict(data.get("internal_state", {})),
            narrative=data.get("narrative", ""),
            episode_id=data.get("episode_id"),
            word_usage=list(data.get("word_usage", [])),
        )


@dataclass
class EpisodeRecord:
    """Container that binds events into a meaningful episode."""

    id: str
    start_time: str
    end_time: Optional[str] = None
    situation_tags: List[str] = field(default_factory=list)
    preconditions: Dict[str, Any] = field(default_factory=dict)
    intent: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)
    narrative: str = ""
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    feedback_hooks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeRecord":
        return cls(
            id=data.get("id", _generate_identifier("ep")),
            start_time=data.get("start_time", _now_iso()),
            end_time=data.get("end_time"),
            situation_tags=list(data.get("situation_tags", [])),
            preconditions=dict(data.get("preconditions", {})),
            intent=data.get("intent"),
            result=dict(data.get("result", {})),
            events=list(data.get("events", [])),
            narrative=data.get("narrative", ""),
            feedback=list(data.get("feedback", [])),
            feedback_hooks=list(data.get("feedback_hooks", [])),
        )


class ExperienceLogger:
    """Persist multimodal events and assemble them into episodes."""

    def __init__(self, child: str = "Inazuma_Yagami", base_path: Optional[Path] = None) -> None:
        self.child = child
        self._base_path = Path(base_path) if base_path else Path("AI_Children")
        self._root = self._base_path / child / "memory" / "experiences"
        self._events_dir = self._root / "events"
        self._episodes_dir = self._root / "episodes"
        self._root.mkdir(parents=True, exist_ok=True)
        self._events_dir.mkdir(parents=True, exist_ok=True)
        self._episodes_dir.mkdir(parents=True, exist_ok=True)
        self._active_episode: Optional[EpisodeRecord] = None

    # ------------------------------------------------------------------
    # Event management
    # ------------------------------------------------------------------
    def log_event(
        self,
        *,
        situation_tags: Optional[Iterable[str]] = None,
        perceived_entities: Optional[Iterable[Dict[str, Any]]] = None,
        actions: Optional[Iterable[Dict[str, Any]]] = None,
        outcome: Optional[Dict[str, Any]] = None,
        internal_state: Optional[Dict[str, Any]] = None,
        narrative: str = "",
        timestamp: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """Create and persist a new :class:`EventRecord`."""

        record = EventRecord(
            id=event_id or _generate_identifier("evt"),
            timestamp=timestamp or _now_iso(),
            situation_tags=list(situation_tags or []),
            perceived_entities=list(perceived_entities or []),
            actions=list(actions or []),
            outcome=dict(outcome or {}),
            internal_state=dict(internal_state or {}),
            narrative=narrative,
            episode_id=self._active_episode.id if self._active_episode else None,
        )
        self._save_event(record)
        if self._active_episode:
            self._active_episode.events.append(record.id)
            self._persist_active_episode()
        return record.id

    def attach_word_usage(
        self,
        event_id: str,
        *,
        speaker: str,
        utterance: str,
        words: Optional[Iterable[str]] = None,
        entity_links: Optional[Iterable[Dict[str, Any]]] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach a grounded word usage annotation to an event."""

        event = self._load_event(event_id)
        annotation = {
            "speaker": speaker,
            "utterance": utterance,
            "words": [w.lower() for w in (words or self._extract_words(utterance))],
            "entity_links": list(entity_links or []),
            "timestamp": timestamp or _now_iso(),
        }
        event.word_usage.append(annotation)
        self._save_event(event)
        return annotation

    def list_recent_events(self, limit: int = 5) -> List[EventRecord]:
        """Return the most recent *limit* events for quick inspection."""

        files = sorted(self._events_dir.glob("evt_*.json"), reverse=True)[:limit]
        return [self._load_event_from_path(path) for path in files]

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------
    def start_episode(
        self,
        *,
        situation_tags: Optional[Iterable[str]] = None,
        preconditions: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
    ) -> str:
        """Begin a new episode, closing any previously active one."""

        if self._active_episode:
            self.finish_episode()
        self._active_episode = EpisodeRecord(
            id=_generate_identifier("ep"),
            start_time=_now_iso(),
            situation_tags=list(situation_tags or []),
            preconditions=dict(preconditions or {}),
            intent=intent,
        )
        self._persist_active_episode()
        return self._active_episode.id

    def add_event_to_episode(self, event_id: str) -> None:
        """Associate an existing event with the active episode."""

        if not self._active_episode:
            raise RuntimeError("No active episode to attach the event to.")
        if event_id not in self._active_episode.events:
            self._active_episode.events.append(event_id)
            self._persist_active_episode()
        event = self._load_event(event_id)
        event.episode_id = self._active_episode.id
        self._save_event(event)

    def finish_episode(
        self,
        *,
        result: Optional[Dict[str, Any]] = None,
        feedback: Optional[Iterable[Dict[str, Any]]] = None,
        narrative: Optional[str] = None,
        feedback_hooks: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Optional[str]:
        """Close the currently active episode."""

        if not self._active_episode:
            return None
        self._active_episode.end_time = _now_iso()
        if result is not None:
            self._active_episode.result = dict(result)
        if feedback is not None:
            self._active_episode.feedback.extend(list(feedback))
        if narrative is not None:
            self._active_episode.narrative = narrative
        if feedback_hooks is not None:
            self._active_episode.feedback_hooks.extend(list(feedback_hooks))
        episode_id = self._active_episode.id
        self._persist_active_episode()
        self._active_episode = None
        return episode_id

    def narrate_episode(self, episode_id: str) -> str:
        """Generate a lightweight narration by stitching event narratives."""

        episode = self._load_episode(episode_id)
        event_narratives = []
        for event_id in episode.events:
            try:
                event = self._load_event(event_id)
                if event.narrative:
                    event_narratives.append(event.narrative)
                else:
                    event_narratives.append(
                        ", ".join(event.situation_tags) or "(no narrative)"
                    )
            except FileNotFoundError:
                continue
        narration = " -> ".join(event_narratives)
        episode.narrative = narration
        episode.feedback_hooks = self._ensure_feedback_hooks(episode)
        self._save_episode(episode)
        return narration

    def finish_and_narrate_episode(
        self,
        *,
        result: Optional[Dict[str, Any]] = None,
        feedback: Optional[Iterable[Dict[str, Any]]] = None,
        narrative: Optional[str] = None,
        feedback_hooks: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, str]]:
        """Finish the active episode and immediately narrate it.

        Returns a mapping containing the ``episode_id`` and generated
        ``narrative`` when an episode was active.
        """

        episode_id = self.finish_episode(
            result=result,
            feedback=feedback,
            narrative=narrative,
            feedback_hooks=feedback_hooks,
        )
        if episode_id is None:
            return None
        narration = self.narrate_episode(episode_id)
        return {"episode_id": episode_id, "narrative": narration}

    # ------------------------------------------------------------------
    # Cross-modal binding
    # ------------------------------------------------------------------
    def _pull_inastate(self, key: str) -> Any:
        """
        Best-effort accessor for inastate values without creating a hard dependency.
        """
        try:
            from model_manager import get_inastate  # type: ignore
        except Exception:
            return None
        try:
            return get_inastate(key)
        except Exception:
            return None

    @staticmethod
    def _emotion_values(emotion: Any) -> Dict[str, float]:
        if isinstance(emotion, dict):
            if isinstance(emotion.get("values"), dict):
                return emotion.get("values", {})
            return emotion
        return {}

    def bind_multimodal_event(
        self,
        *,
        cause_hint: str,
        audio: Optional[Dict[str, Any]] = None,
        vision: Optional[Dict[str, Any]] = None,
        symbols: Optional[Iterable[Dict[str, Any]]] = None,
        prediction: Optional[Dict[str, Any]] = None,
        emotion: Optional[Dict[str, Any]] = None,
        extra_tags: Optional[Iterable[str]] = None,
        narrative: Optional[str] = None,
        fragments: Optional[Iterable[str]] = None,
        start_episode_if_missing: bool = True,
        write_fragment: bool = True,
    ) -> Dict[str, Optional[str]]:
        """
        Bind audio + vision (+emotion/symbol/prediction) into one event-object.
        Returns a mapping with event_id and optional fragment_id.
        """
        if start_episode_if_missing and not self._active_episode:
            self.start_episode(
                situation_tags=["multimodal", "bound_event"],
                intent="Cross-modal binding stream",
            )

        emotion = emotion or self._pull_inastate("emotion_snapshot") or {}
        symbols = list(symbols) if symbols is not None else (self._pull_inastate("emotion_symbol_matches") or [])
        prediction = prediction or self._pull_inastate("current_prediction") or {}

        bound_modalities: List[str] = []
        entities: List[Dict[str, Any]] = []

        def _add_entity(kind: str, payload: Optional[Dict[str, Any]]) -> None:
            if not payload:
                return
            entry = {"type": kind}
            entry.update(payload)
            entities.append(entry)
            bound_modalities.append(kind)

        _add_entity("audio", audio)
        _add_entity("vision", vision)

        if symbols:
            entities.append({"type": "symbol_match", "candidates": symbols})
            bound_modalities.append("symbol")
        if prediction:
            entities.append({"type": "prediction", **prediction})
            bound_modalities.append("prediction")

        emo_values = self._emotion_values(emotion)
        internal_state = {
            "emotion_snapshot": emotion,
            "emotion_values": emo_values,
            "symbol_matches": symbols,
            "prediction": prediction,
            "bound_modalities": bound_modalities,
            "cause_hint": cause_hint,
        }

        tags = {"multimodal", "bound_event", "cross_modal", f"cause:{cause_hint}"}
        if extra_tags:
            tags.update(extra_tags)

        narrative_text = narrative or f"Bound {', '.join(bound_modalities) or 'signals'} around '{cause_hint}'."
        event_id = self.log_event(
            situation_tags=sorted(tags),
            perceived_entities=entities,
            internal_state=internal_state,
            narrative=narrative_text,
        )

        fragment_id: Optional[str] = None
        if write_fragment:
            try:
                from fragmentation_engine import make_fragment, store_fragment  # type: ignore

                payload = {
                    "event_id": event_id,
                    "bound_modalities": bound_modalities,
                    "cause_hint": cause_hint,
                    "constituent_fragments": list(fragments or []),
                    "entities": entities,
                }
                frag = make_fragment(
                    frag_type="experience",
                    source="cross_modal_binder",
                    summary=narrative_text,
                    tags=sorted(tags),
                    emotions=emo_values,
                    symbols=[s.get("symbol_word_id") for s in symbols if isinstance(s, dict) and s.get("symbol_word_id")],
                    payload=payload,
                    context={
                        "prediction": prediction,
                        "symbol_matches": symbols,
                        "emotion_snapshot": emotion,
                    },
                )
                stored_path = store_fragment(frag, reason="cross_modal_binding")
                fragment_id = frag.get("id")
                if stored_path:
                    internal_state["fragment_path"] = str(stored_path)
            except Exception:
                # Non-critical: binding still logged as an event.
                fragment_id = None

        return {"event_id": event_id, "fragment_id": fragment_id}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _event_path(self, event_id: str) -> Path:
        return self._events_dir / f"{event_id}.json"

    def _episode_path(self, episode_id: str) -> Path:
        return self._episodes_dir / f"{episode_id}.json"

    def _save_event(self, record: EventRecord) -> None:
        path = self._event_path(record.id)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record.to_dict(), fh, indent=2)

    def _save_episode(self, record: EpisodeRecord) -> None:
        path = self._episode_path(record.id)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record.to_dict(), fh, indent=2)

    def _load_event(self, event_id: str) -> EventRecord:
        return self._load_event_from_path(self._event_path(event_id))

    def _load_event_from_path(self, path: Path) -> EventRecord:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return EventRecord.from_dict(data)

    def _load_episode(self, episode_id: str) -> EpisodeRecord:
        path = self._episode_path(episode_id)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return EpisodeRecord.from_dict(data)

    def _persist_active_episode(self) -> None:
        if self._active_episode:
            self._save_episode(self._active_episode)

    @staticmethod
    def _extract_words(utterance: str) -> List[str]:
        cleaned = "".join(
            ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in utterance
        )
        return [token for token in cleaned.split() if token]

    @staticmethod
    def _ensure_feedback_hooks(episode: EpisodeRecord) -> List[Dict[str, Any]]:
        hooks = {hook.get("event_id"): hook for hook in episode.feedback_hooks if hook.get("event_id")}
        for event_id in episode.events:
            if event_id not in hooks:
                hooks[event_id] = {
                    "event_id": event_id,
                    "status": "pending",
                    "created_at": _now_iso(),
                    "notes": "",
                }
        return [hooks[event_id] for event_id in episode.events if event_id in hooks]


__all__ = [
    "EventRecord",
    "EpisodeRecord",
    "ExperienceLogger",
]
