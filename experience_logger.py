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
        self._save_episode(episode)
        return narration

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


__all__ = [
    "EventRecord",
    "EpisodeRecord",
    "ExperienceLogger",
]
