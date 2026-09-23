"""Bounded thought formation and evidence-based decision composition.

Thoughts remain medium-neutral records.  Non-linguistic input is never forced
through words, while linguistic input crosses the existing semantic-event
boundary before it can contribute to a decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping
import math
import uuid

from cognition_runtime.cognitive_context import CognitiveContext, _bounded
from expression_core import (
    INTERPRETATION_SCHEMA, REACTION_SCHEMA, create_expression_intent,
    text_expression_guidance,
)
from experience_cognition import plan_experience_cognition
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

    def revise_thought(
        self, parent: Thought, revision: Any, *, evidence: Iterable[Thought] = (),
        linguistic: bool | None = None, context: CognitiveContext | None = None,
        confidence: float | None = None, relevance: float | None = None,
        provenance: Iterable[str] = (),
    ) -> Thought:
        """Create an inspectable revision without overwriting the prior thought."""
        witnesses = tuple(evidence)[:32]
        use_language = parent.mode == "linguistic" if linguistic is None else bool(linguistic)
        metadata = {
            **dict(parent.metadata), "revision_of": parent.thought_id,
            "evidence_thought_ids": [item.thought_id for item in witnesses],
        }
        sources = tuple(dict.fromkeys((
            *parent.provenance,
            *(source for item in witnesses for source in item.provenance),
            *tuple(provenance),
        )))
        common = {
            "context": context, "confidence": parent.confidence if confidence is None else confidence,
            "relevance": parent.relevance if relevance is None else relevance,
            "provenance": sources, "metadata": metadata,
        }
        if use_language:
            revised = self.process_linguistic(str(revision or ""), **common)
        else:
            revised = self.process_non_linguistic(revision, **common)
        if context is None and parent.context_id:
            revised = replace(revised, context_id=parent.context_id)
        return revised

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

    def prepare_communication(
        self, purpose: str, thoughts: Iterable[Thought], *,
        audience_references: Iterable[str] = (), allowed_media: Iterable[str] | None = None,
        dimensions: Mapping[str, Any] | None = None, max_thoughts: int = 8,
        thought_ids: Iterable[str] | None = None,
        meaning_set: Mapping[str, Any] | None = None,
        cognition_event: Mapping[str, Any] | None = None,
        transient_candidates: Iterable[Mapping[str, Any]] = (),
        prediction_candidates: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """Select thought references and prepare an output-neutral expression intent."""
        bounded = tuple(thoughts)[:64]
        requested = set(str(item) for item in thought_ids or ())
        eligible = [item for item in bounded if not requested or item.thought_id in requested]
        selected = sorted(
            eligible, key=lambda item: (-(item.confidence * item.relevance), item.thought_id),
        )[:max(1, min(16, int(max_thoughts)))]
        if not selected:
            raise ValueError("communication requires at least one selected thought")
        thought_references = [f"thought:{item.thought_id}" for item in selected]
        linguistic = [f"semantic:{item.thought_id}" for item in selected if item.mode == "linguistic"]
        concepts = [f"concept:{item.thought_id}" for item in selected if item.mode != "linguistic"]
        affects = [
            f"affect:{item.thought_id}" for item in selected
            if item.metadata.get("guidance_source") == "emotion"
        ]
        uncertain = [
            {"thought_id": item.thought_id, "confidence": item.confidence}
            for item in selected if item.confidence < 1.0
        ]
        meaning_references = []
        if isinstance(meaning_set, Mapping):
            if meaning_set.get("schema") != "ina.communicative_meaning_set/V1":
                raise ValueError("a valid communicative meaning set is required")
            meaning_references = [
                f"meaning:{item.get('candidate_id')}"
                for item in list(meaning_set.get("candidates") or ())[:8]
                if isinstance(item, Mapping) and item.get("candidate_id")
            ]
        cognition_plan = None
        guidance = text_expression_guidance(None)
        if cognition_event is not None:
            cognition_plan = plan_experience_cognition(
                cognition_event,
                transient_candidates=transient_candidates,
                prediction_candidates=prediction_candidates,
            )
            guidance = text_expression_guidance(cognition_plan)
        intent = create_expression_intent(
            purpose, semantic_references=linguistic,
            meaning_references=meaning_references,
            concept_references=concepts, affect_references=affects,
            audience_references=audience_references, dimensions=dimensions,
            uncertainty={
                "thoughts": uncertain[:8],
                "epistemic_status": guidance["status"],
                "missing_evidence": guidance["missing_evidence"],
                "conflict_retained": guidance.get("conflict_retained", False),
            }, allowed_media=allowed_media,
            provenance=tuple(dict.fromkeys(
                (*thought_references, *(source for item in selected for source in item.provenance))
            )),
        )
        return {
            "schema": "ina.thought_communication_plan/V1",
            "plan_id": uuid.uuid4().hex, "purpose": str(purpose)[:500],
            "selected_thought_ids": [item.thought_id for item in selected],
            "expression_intent": intent,
            "communicative_meaning_set": dict(meaning_set) if isinstance(meaning_set, Mapping) else None,
            "experience_cognition": cognition_plan,
            "text_expression_guidance": guidance,
            "context_ids": list(dict.fromkeys(item.context_id for item in selected if item.context_id)),
        }

    def process_communication_feedback(
        self, plan: Mapping[str, Any], reaction: Mapping[str, Any],
        interpretation: Mapping[str, Any], *, context: CognitiveContext | None = None,
        relevance: float = 1.0,
    ) -> Thought:
        """Turn uncertain communication feedback into evidence for later revision."""
        if plan.get("schema") != "ina.thought_communication_plan/V1":
            raise ValueError("a valid thought communication plan is required")
        if reaction.get("schema") != REACTION_SCHEMA:
            raise ValueError("a valid expression reaction is required")
        if interpretation.get("schema") != INTERPRETATION_SCHEMA:
            raise ValueError("a valid reaction interpretation is required")
        if interpretation.get("reaction_id") != reaction.get("reaction_id"):
            raise ValueError("reaction interpretation does not match the reaction")
        observation = dict(reaction.get("observation") or {})
        if "reward" in observation:
            raise ValueError("communication feedback is evidence, not reward")
        candidates = [dict(item) for item in list(interpretation.get("candidates") or ())[:8]]
        causal = reaction.get("causal_confidence")
        causal_factor = 1.0 if causal is None else _unit(causal)
        confidence = round(_unit(
            _unit(max((item.get("confidence", 0.0) for item in candidates), default=0.0))
            * causal_factor
        ), 6)
        return self.process_non_linguistic(
            {
                "communication_plan_id": str(plan.get("plan_id") or ""),
                "expression_intent_id": str((plan.get("expression_intent") or {}).get("intent_id") or ""),
                "reaction_id": str(reaction.get("reaction_id") or ""),
                "observation": observation, "interpretations": candidates,
                "about_thought_ids": list(plan.get("selected_thought_ids") or ())[:16],
            },
            context=context, confidence=confidence, relevance=relevance,
            provenance=tuple(dict.fromkeys((
                *tuple(reaction.get("provenance") or ()),
                *tuple(interpretation.get("provenance") or ()),
                f"communication_plan:{plan.get('plan_id')}",
            ))),
            metadata={"role": "communication_feedback", "revision_candidate": True},
        )

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
