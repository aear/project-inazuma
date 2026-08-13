"""Bounded shared context for one cognitive task or cycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
from itertools import islice
import uuid


def _bounded(value: Any, *, depth: int = 0, max_items: int = 32, max_text: int = 4096) -> Any:
    if depth >= 3:
        return str(value)[:max_text]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:max_text]
    if isinstance(value, Mapping):
        return {str(key)[:128]: _bounded(item, depth=depth + 1, max_items=max_items, max_text=max_text)
                for key, item in islice(value.items(), max_items)}
    if isinstance(value, (list, tuple)):
        return tuple(_bounded(item, depth=depth + 1, max_items=max_items, max_text=max_text)
                     for item in value[:max_items])
    return str(value)[:max_text]


@dataclass(frozen=True)
class ContextReference:
    uri: str
    kind: str = "reference"
    provenance: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CognitiveContext:
    context_id: str
    created_at: str
    observations: tuple[Any, ...] = ()
    goals: tuple[Any, ...] = ()
    active_state: Mapping[str, Any] = field(default_factory=dict)
    discourse: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[str, ...] = ()
    references: tuple[ContextReference, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls, *, observations: Iterable[Any] = (), goals: Iterable[Any] = (),
        active_state: Mapping[str, Any] | None = None,
        discourse: Mapping[str, Any] | None = None, provenance: Iterable[str] = (),
        references: Iterable[ContextReference | Mapping[str, Any] | str] = (),
        metadata: Mapping[str, Any] | None = None, max_observations: int = 32,
        max_goals: int = 16, max_references: int = 64,
    ) -> "CognitiveContext":
        refs = []
        for item in islice(references, max(0, int(max_references))):
            if isinstance(item, ContextReference):
                refs.append(ContextReference(
                    uri=str(item.uri)[:4096], kind=str(item.kind)[:128],
                    provenance=str(item.provenance)[:512], metadata=_bounded(item.metadata),
                ))
            elif isinstance(item, Mapping):
                refs.append(ContextReference(
                    uri=str(item.get("uri") or item.get("path") or "")[:4096],
                    kind=str(item.get("kind") or "reference")[:128],
                    provenance=str(item.get("provenance") or "")[:512],
                    metadata=_bounded(item.get("metadata") or {}),
                ))
            else:
                refs.append(ContextReference(uri=str(item)[:4096]))
        return cls(
            context_id=uuid.uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(),
            observations=tuple(_bounded(item) for item in islice(observations, max(0, int(max_observations)))),
            goals=tuple(_bounded(item) for item in islice(goals, max(0, int(max_goals)))),
            active_state=_bounded(active_state or {}),
            discourse=_bounded(discourse or {}),
            provenance=tuple(str(item)[:512] for item in islice(provenance, 64)),
            references=tuple(refs), metadata=_bounded(metadata or {}),
        )

    def view(self, supported: Iterable[str]) -> "CognitiveContext":
        allowed = set(supported)
        if not allowed:
            return self
        return CognitiveContext(
            context_id=self.context_id, created_at=self.created_at,
            observations=self.observations if "observations" in allowed else (),
            goals=self.goals if "goals" in allowed else (),
            active_state=self.active_state if "active_state" in allowed else {},
            discourse=self.discourse if "discourse" in allowed else {},
            provenance=self.provenance,
            references=self.references if "references" in allowed else (),
            metadata=self.metadata if "metadata" in allowed else {},
        )
