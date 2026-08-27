"""Bounded thought formation and evidence-based decision composition.

Thoughts remain medium-neutral records.  Non-linguistic input is never forced
through words, while linguistic input crosses the existing semantic-event
boundary before it can contribute to a decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping
import math
import uuid

from cognition_runtime.cognitive_context import CognitiveContext, _bounded
from semantic_event import build_native_intent, build_semantic_event


def _unit(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return max(0.0, min(1.0, number))


@dataclass(frozen=True)
class Thought:
    thought_id: str
    mode: str
    content: Any
    confidence: float
    relevance: float
    provenance: tuple[str, ...] = ()
    context_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "ina.thought/V1", "thought_id": self.thought_id,
            "mode": self.mode, "content": self.content,
            "confidence": self.confidence, "relevance": self.relevance,
            "provenance": list(self.provenance), "context_id": self.context_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Decision:
    decision_id: str
    selected: str | None
    status: str
    scores: Mapping[str, float]
    evidence: tuple[Mapping[str, Any], ...]
    confidence: float
    modalities: tuple[str, ...]
    alternatives: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "ina.decision/V1", "decision_id": self.decision_id,
            "selected": self.selected, "status": self.status,
            "scores": dict(self.scores), "evidence": [dict(row) for row in self.evidence],
            "confidence": self.confidence, "modalities": list(self.modalities),
            "alternatives": list(self.alternatives), "reason": self.reason,
        }


class ThoughtProcessor:
    """Form two kinds of thought and combine either kind for one decision."""

    GUIDANCE_SOURCES = ("emotion", "instinct", "cognition", "memory")

    def process_non_linguistic(
        self, content: Any, *, context: CognitiveContext | None = None,
        confidence: float = 1.0, relevance: float = 1.0,
        provenance: Iterable[str] = (), metadata: Mapping[str, Any] | None = None,
    ) -> Thought:
        """Retain perceptions, vectors, affect, spatial state, or symbols as data."""
        return self._thought(
            "non_linguistic", _bounded(content), context=context,
            confidence=confidence, relevance=relevance,
            provenance=provenance, metadata=metadata,
        )

    def process_linguistic(
        self, text: str, *, context: CognitiveContext | None = None,
        discourse: Mapping[str, Any] | None = None, confidence: float = 1.0,
        relevance: float = 1.0, provenance: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Thought:
        """Represent language as semantic roles and constructions, not raw tokens alone."""
        active_discourse = discourse if discourse is not None else (context.discourse if context else {})
        source_text = str(text or "")[:4096]
        event = build_semantic_event(source_text, active_discourse)
        content = {
            "source_text": source_text,
            "semantic_event": event,
            "native_intent": build_native_intent(event),
        }
        return self._thought(
            "linguistic", content, context=context,
            confidence=confidence, relevance=relevance,
            provenance=provenance, metadata=metadata,
        )

    def decide(
        self, options: Iterable[str], thoughts: Iterable[Thought], *,
        evidence: Iterable[Mapping[str, Any]], minimum_score: float = 0.0,
        tie_margin: float = 0.05,
    ) -> Decision:
        """Choose once from explicit weighted evidence, or preserve uncertainty.

        Evidence rows name ``option`` and ``thought_id`` and may provide a signed
        ``weight`` from -1 to 1. A thought's confidence and relevance calibrate
        its contribution. No lexical or non-lexical modality gets an implicit
        preference.
        """
        names = tuple(dict.fromkeys(str(item)[:256] for item in options if str(item)))[:32]
        thought_index = {item.thought_id: item for item in tuple(thoughts)[:64]}
        scores = {name: 0.0 for name in names}
        used = []
        for row in tuple(evidence)[:128]:
            option = str(row.get("option") or "")[:256]
            thought = thought_index.get(str(row.get("thought_id") or ""))
            if option not in scores or thought is None:
                continue
            try:
                weight = float(row.get("weight", 1.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(weight):
                continue
            weight = max(-1.0, min(1.0, weight))
            contribution = weight * thought.confidence * thought.relevance
            scores[option] += contribution
            used.append({
                "option": option, "thought_id": thought.thought_id,
                "mode": thought.mode, "weight": weight,
                "contribution": round(contribution, 6),
                "reason": str(row.get("reason") or "")[:512],
            })
        ranked = sorted(names, key=lambda name: (-scores[name], names.index(name)))
        selected = ranked[0] if ranked else None
        runner_up = scores[ranked[1]] if len(ranked) > 1 else 0.0
        lead = scores[selected] - runner_up if selected is not None else 0.0
        if not used:
            selected, status, reason = None, "undecided", "no_applicable_evidence"
        elif scores[selected] < float(minimum_score):
            selected, status, reason = None, "undecided", "minimum_score_not_met"
        elif len(ranked) > 1 and lead <= max(0.0, float(tie_margin)):
            selected, status, reason = None, "undecided", "evidence_too_close"
        else:
            status, reason = "selected", "evidence_margin"
        scale = sum(abs(row["contribution"]) for row in used) or 1.0
        confidence = _unit(max(0.0, lead) / scale)
        modalities = tuple(sorted({row["mode"] for row in used}))
        return Decision(
            decision_id=uuid.uuid4().hex, selected=selected, status=status,
            scores={key: round(value, 6) for key, value in scores.items()},
            evidence=tuple(used), confidence=confidence, modalities=modalities,
            alternatives=tuple(ranked), reason=reason,
        )

    def guided_decision(
        self, options: Iterable[str], guidance: Mapping[str, Iterable[Any]], *,
        evidence: Iterable[Mapping[str, Any]], context: CognitiveContext | None = None,
        minimum_score: float = 0.0, tie_margin: float = 0.05,
    ) -> dict[str, Any]:
        """Compose emotion, instinct, cognition, and memory into one decision.

        Each guidance item may be a :class:`Thought`, a mapping containing
        ``content`` plus calibration/provenance fields, or a compact value.
        Evidence may identify a generated thought with ``thought_id`` or use
        ``source`` and optional ``index``. Sources influence only through
        explicit evidence weights; they have no hidden priority ordering.
        """
        unknown = sorted(set(str(key) for key in guidance) - set(self.GUIDANCE_SOURCES))
        if unknown:
            raise ValueError(f"unknown guidance sources: {', '.join(unknown)}")
        thoughts: list[Thought] = []
        by_source: dict[str, list[Thought]] = {source: [] for source in self.GUIDANCE_SOURCES}
        for source in self.GUIDANCE_SOURCES:
            for item in tuple(guidance.get(source) or ())[:16]:
                if isinstance(item, Thought):
                    thought = Thought(
                        thought_id=item.thought_id, mode=item.mode, content=item.content,
                        confidence=item.confidence, relevance=item.relevance,
                        provenance=tuple(dict.fromkeys((*item.provenance, f"guidance:{source}"))),
                        context_id=item.context_id or (context.context_id if context else ""),
                        metadata={**dict(item.metadata), "guidance_source": source},
                    )
                else:
                    row = item if isinstance(item, Mapping) else {"content": item}
                    content = row.get("content", row)
                    raw_provenance = row.get("provenance") or ()
                    item_provenance = (
                        (raw_provenance,) if isinstance(raw_provenance, str)
                        else tuple(raw_provenance)
                    )
                    common = {
                        "context": context,
                        "confidence": row.get("confidence", 1.0),
                        "relevance": row.get("relevance", 1.0),
                        "provenance": (*item_provenance, f"guidance:{source}"),
                        "metadata": {**dict(row.get("metadata") or {}), "guidance_source": source},
                    }
                    linguistic = bool(row.get("linguistic"))
                    if linguistic:
                        thought = self.process_linguistic(str(content or ""), **common)
                    else:
                        thought = self.process_non_linguistic(content, **common)
                thoughts.append(thought)
                by_source[source].append(thought)
        resolved_evidence = []
        for row in tuple(evidence)[:128]:
            resolved = dict(row)
            if not resolved.get("thought_id"):
                source = str(resolved.get("source") or "")
                try:
                    index = int(resolved.get("index", 0))
                    resolved["thought_id"] = by_source[source][index].thought_id
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
            resolved_evidence.append(resolved)
        decision = self.decide(
            options, thoughts, evidence=resolved_evidence,
            minimum_score=minimum_score, tie_margin=tie_margin,
        )
        return {
            "schema": "ina.guided_decision/V1",
            "context_id": context.context_id if context else "",
            "thoughts": [thought.as_dict() for thought in thoughts],
            "decision": decision.as_dict(),
            "guidance_coverage": {
                source: len(by_source[source]) for source in self.GUIDANCE_SOURCES
            },
        }

    @staticmethod
    def _thought(
        mode: str, content: Any, *, context: CognitiveContext | None,
        confidence: float, relevance: float, provenance: Iterable[str],
        metadata: Mapping[str, Any] | None,
    ) -> Thought:
        inherited = context.provenance if context else ()
        sources = tuple(dict.fromkeys(str(item)[:512] for item in (*inherited, *tuple(provenance))))[:64]
        return Thought(
            thought_id=uuid.uuid4().hex, mode=mode, content=content,
            confidence=_unit(confidence), relevance=_unit(relevance),
            provenance=sources, context_id=context.context_id if context else "",
            metadata=_bounded(metadata or {}),
        )


__all__ = ["Decision", "Thought", "ThoughtProcessor"]
