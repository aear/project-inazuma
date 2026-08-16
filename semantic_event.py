"""Small semantic-event bridge between discourse, recall, and rendering."""
from __future__ import annotations

from typing import Any, Mapping

from discourse_context import resolution_for
from language_intelligence import analyze_utterance


def build_semantic_event(text: str, discourse: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Describe meaning before native-symbol or English rendering choices."""
    discourse = discourse if isinstance(discourse, Mapping) else {}
    analysis = analyze_utterance(str(text or ""))
    events = []
    def semantic_argument(value: Any) -> dict[str, Any]:
        surface = str(value or "")
        head = surface.split()[0].casefold() if surface.split() else ""
        resolution = resolution_for(discourse, head) if head else None
        referents = [dict(item) for item in (resolution or {}).get("referents") or () if isinstance(item, Mapping)]
        return {
            "surface": surface,
            "referent": referents[0] if len(referents) == 1 and not (resolution or {}).get("ambiguous") else None,
            "alternatives": referents[:8] if (resolution or {}).get("ambiguous") else [],
            "role": (resolution or {}).get("role"),
        }
    for clause in list(analysis.get("clauses") or ())[:8]:
        if not isinstance(clause, Mapping):
            continue
        subject = clause.get("subject")
        subject_resolution = resolution_for(discourse, str(subject or ""))
        agent = None
        if subject_resolution and len(subject_resolution.get("referents") or ()) == 1:
            agent = dict(subject_resolution["referents"][0])
        events.append({
            "id": clause.get("id"),
            "predicate": clause.get("predicate"),
            "agent": agent or subject,
            "agent_surface": subject,
            "arguments": {
                str(role): semantic_argument(value)
                for role, value in dict(clause.get("arguments") or {}).items()
            },
            "tense": clause.get("tense"),
            "modality": clause.get("modality"),
            "negated": bool(clause.get("negated")),
            "negation_scope": clause.get("negation_scope"),
        })
    return {
        "version": 1,
        "source_text": str(text or ""),
        "events": events,
        "speech_act": dict(analysis.get("speech_act") or {}),
        "referent_uncertainty": dict((analysis.get("uncertainty") or {}).get("referents") or {}),
        "construction_features": sorted({
            feature
            for event in events
            for feature, present in (
                ("agent", bool(event.get("agent"))),
                ("arguments", bool(event.get("arguments"))),
                ("past", event.get("tense") == "past"),
                ("modality", bool(event.get("modality"))),
                ("negation", bool(event.get("negated"))),
            )
            if present
        }),
    }


def build_native_intent(semantic_event: Mapping[str, Any]) -> dict[str, Any]:
    """Turn event roles into a bounded, language-neutral symbol-selection plan."""
    concepts = []
    grammar = []
    for event in list(semantic_event.get("events") or ())[:8]:
        if not isinstance(event, Mapping):
            continue
        role_concepts = []
        agent_surface = str(event.get("agent_surface") or "").casefold()
        if agent_surface:
            role_concepts.append({"role": "agent", "surface": agent_surface})
        if event.get("predicate"):
            role_concepts.append({"role": "predicate", "surface": str(event["predicate"]).casefold()})
        for role, argument in dict(event.get("arguments") or {}).items():
            surface = str(argument.get("surface") if isinstance(argument, Mapping) else argument).casefold()
            for token in surface.split():
                role_concepts.append({"role": str(role), "surface": token})
        concepts.extend(role_concepts)
        if event.get("tense") and event.get("tense") != "present":
            grammar.append({"construction": "tense", "value": event.get("tense")})
        if event.get("modality"):
            grammar.append({"construction": "modality", "value": event.get("modality")})
        if event.get("negated"):
            grammar.append({"construction": "negation", "scope": event.get("negation_scope")})
    lexical = []
    for concept in concepts:
        surface = concept.get("surface")
        if surface and surface not in lexical:
            lexical.append(surface)
    return {
        "version": 1, "events": list(semantic_event.get("events") or ())[:8],
        "concepts": concepts[:32], "grammar": grammar[:16],
        "lexical_realizations": lexical[:32],
    }


__all__ = ["build_native_intent", "build_semantic_event"]
