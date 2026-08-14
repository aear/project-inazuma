"""Bounded, inspectable language structure and discourse analysis."""
from __future__ import annotations

import hashlib
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

from io_utils import atomic_write_json, load_json_dict
from origin_record import make_origin

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.!?,]")
_CONTRACTIONS = {
    "didn't": ("did", "not"), "doesn't": ("does", "not"), "don't": ("do", "not"),
    "isn't": ("is", "not"), "wasn't": ("was", "not"), "can't": ("can", "not"),
    "won't": ("will", "not"), "shouldn't": ("should", "not"), "wouldn't": ("would", "not"),
    "i'm": ("i", "am"), "you're": ("you", "are"), "that's": ("that", "is"),
}
_LEMMA = {"told": "tell", "said": "say", "stole": "steal", "gave": "give", "did": "do", "was": "be", "is": "be", "are": "be", "am": "be"}
_PAST = {"told", "said", "stole", "gave", "did", "was", "were", "had"}
_MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
_VERBS = {"tell", "say", "steal", "give", "be", "do", "have", "find", "remember", "think", "know", "open", "arrive"}
_PRONOUNS = {"i", "me", "my", "you", "your", "he", "him", "his", "she", "her", "hers", "they", "them", "their", "it", "its", "we", "us", "our"}


def _lemma_word(word: str) -> str:
    if word in _LEMMA:
        return _LEMMA[word]
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        return stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def morphology(text: str) -> list[dict[str, Any]]:
    result = []
    for surface in _TOKEN_RE.findall(text):
        lowered = surface.casefold()
        expanded = _CONTRACTIONS.get(lowered, (lowered,))
        for index, normalized in enumerate(expanded):
            if normalized in ".!?,":
                result.append({"surface": surface, "normalized": normalized, "lemma": normalized, "features": {"punctuation": True}})
                continue
            lemma = _lemma_word(normalized)
            features: dict[str, Any] = {}
            if lowered in _CONTRACTIONS:
                features.update({"contraction": lowered, "expansion_index": index})
            if normalized == "not": features["negation"] = True
            if normalized in _PAST: features["tense"] = "past"
            if normalized in _MODALS: features["modality"] = normalized
            result.append({"surface": surface, "normalized": normalized, "lemma": lemma, "features": features})
    return result


class DiscourseEntityMemory:
    """Small persistent referent ledger; callers decide if and where it is saved."""

    def __init__(self, max_entities: int = 64, state: Mapping[str, Any] | None = None) -> None:
        self.max_entities = max(4, int(max_entities))
        self.entities: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for item in list((state or {}).get("entities") or ())[-self.max_entities:]:
            if isinstance(item, Mapping) and item.get("id"):
                self.entities[str(item["id"])] = dict(item)

    def mention(self, name: str, *, role: str | None = None, turn: int | None = None) -> dict[str, Any]:
        entity_id = re.sub(r"\W+", "_", name.casefold()).strip("_") or "unknown"
        entity = self.entities.pop(entity_id, {"id": entity_id, "name": name, "mentions": 0})
        entity.update({"name": name, "last_role": role, "last_turn": turn, "mentions": int(entity.get("mentions", 0)) + 1})
        self.entities[entity_id] = entity
        while len(self.entities) > self.max_entities: self.entities.popitem(last=False)
        return dict(entity)

    def candidates(self, pronoun: str) -> list[dict[str, Any]]:
        values = list(self.entities.values())
        if pronoun.casefold() in {"i", "me", "my"}:
            values = [item for item in values if item.get("speaker") is True] or values
        return [dict(item) for item in reversed(values[-4:])]

    def snapshot(self) -> dict[str, Any]:
        return {"version": 1, "entities": list(self.entities.values())}


class ConstructionLearner:
    """Bounded counts for reusable multi-word predicate/argument patterns."""

    def __init__(self, path: Path | None = None, max_patterns: int = 256) -> None:
        self.path = Path(path) if path else None
        self.max_patterns = max(8, int(max_patterns))
        raw = load_json_dict(self.path) if self.path else {}
        self.counts = Counter({str(k): int(v) for k, v in (raw.get("counts") or {}).items()})

    def observe(self, clauses: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        matches = []
        for clause in clauses:
            roles = clause.get("arguments") or {}
            pattern = f"{clause.get('predicate')}({','.join(sorted(roles))})"
            self.counts[pattern] += 1
            matches.append({"pattern": pattern, "observations": self.counts[pattern]})
        self.counts = Counter(dict(self.counts.most_common(self.max_patterns)))
        if self.path:
            atomic_write_json(self.path, {"version": 1, "counts": dict(self.counts)})
        return matches


def _clause(words: list[str], *, clause_id: str, parent: str | None = None) -> dict[str, Any]:
    candidates = [i for i, word in enumerate(words) if _lemma_word(word) in _VERBS or word in _MODALS]
    verb_at = candidates[0] if candidates else None
    if len(candidates) > 1 and words[candidates[0]] in ({"do", "did", "does"} | _MODALS):
        verb_at = candidates[1]
    subject = words[0] if words and verb_at not in (None, 0) else None
    predicate_word = words[verb_at] if verb_at is not None else None
    predicate = _lemma_word(predicate_word or "") or None
    negated = "not" in words
    arguments: dict[str, Any] = {}
    tail = [word for word in words[(verb_at + 1 if verb_at is not None else len(words)):] if word not in {"not", ".", "!", "?", ","}]
    if predicate == "tell":
        if tail: arguments["addressee"] = tail[0]
    elif predicate == "give":
        if tail: arguments["recipient"] = tail[0]
        if len(tail) > 1: arguments["theme"] = " ".join(tail[1:])
    elif tail:
        arguments["theme"] = " ".join(tail)
    return {
        "id": clause_id, "parent_clause": parent, "subject": subject,
        "predicate": predicate, "arguments": arguments, "negated": negated,
        "negation_scope": clause_id if negated else None,
        "tense": "past" if any(word in _PAST or word.endswith("ed") for word in words) else "present",
        "modality": next((word for word in words if word in _MODALS), None),
        "surface_words": words,
    }


def _clauses(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words = [item["normalized"] for item in tokens if item["normalized"] not in ".!?,"]
    reporting = next((i for i, word in enumerate(words) if _lemma_word(word) in {"say", "tell"}), None)
    complement_start = None
    if reporting is not None:
        start = reporting + 1
        if _lemma_word(words[reporting]) == "tell" and start < len(words): start += 1
        if start + 1 < len(words) and any(_lemma_word(word) in _VERBS for word in words[start + 1:]):
            complement_start = start
    if complement_start is None:
        return [_clause(words, clause_id="c0")]
    return [_clause(words[:complement_start], clause_id="c0"), _clause(words[complement_start:], clause_id="c1", parent="c0")]


def _speech_act(text: str, context: Mapping[str, Any]) -> dict[str, Any]:
    lowered = text.casefold().strip()
    act = "question" if lowered.endswith("?") else "request" if lowered.split()[:1] and lowered.split()[0] in {"please", "could", "would"} else "assertion"
    tone = str(context.get("tone") or context.get("pragmatic_tone") or "").casefold()
    sarcastic = tone in {"sarcastic", "ironic"} or bool(context.get("sarcastic"))
    reading = "sarcastic_negative_evaluation" if sarcastic and "great" in lowered else "sincere_positive_evaluation" if "great" in lowered else act
    return {"act": act, "interpretation": reading, "tone": "sarcastic" if sarcastic else "literal", "confidence": 0.9 if tone else 0.7}


def analyze_utterance(text: str, *, context: Mapping[str, Any] | None = None, discourse: DiscourseEntityMemory | None = None, constructions: ConstructionLearner | None = None, turn: int | None = None) -> dict[str, Any]:
    context = context or {}
    discourse = discourse or DiscourseEntityMemory()
    tokens = morphology(text)
    clauses = _clauses(tokens)
    for clause in clauses:
        subject = clause.get("subject")
        if subject and subject not in _PRONOUNS: discourse.mention(subject, role="agent", turn=turn)
        for role, value in (clause.get("arguments") or {}).items():
            for name in re.findall(r"\b[A-Z][a-z]+\b", str(value)):
                discourse.mention(name, role=role, turn=turn)
    proper = [name for name in re.findall(r"\b[A-Z][a-z]+\b", text) if name.casefold() not in _PRONOUNS]
    for name in proper: discourse.mention(name, turn=turn)
    referents: list[dict[str, Any]] = []
    for clause in clauses:
        subject = str(clause.get("subject") or "").casefold()
        if subject in {"he", "she", "they", "him", "her", "them"}:
            candidates = discourse.candidates(subject)
            referents.append({
                "surface": subject, "relation": "subject", "candidates": candidates,
                "resolved": candidates[0]["id"] if len(candidates) == 1 else None,
            })
    possessive = re.search(r"\b(his|her|their|my|your|our|its)\s+(\w+)|\b(\w+)'s\s+(\w+)", text, re.I)
    if possessive:
        pronoun_owner, pronoun_thing, named_owner, named_thing = possessive.groups()
        owner, thing = (pronoun_owner, pronoun_thing) if pronoun_owner else (named_owner, named_thing)
        if owner.casefold() in {"his", "her", "their"}:
            candidates = discourse.candidates(owner)
        elif owner.casefold() in _PRONOUNS:
            candidates = discourse.candidates(owner)
        else:
            candidates = [discourse.mention(owner, role="possessor", turn=turn)]
        referents.append({"surface": owner, "relation": "possessor", "object": thing, "candidates": candidates, "resolved": candidates[0]["id"] if len(candidates) == 1 else None})
    pragmatic = _speech_act(text, context)
    matches = (constructions or ConstructionLearner()).observe(clauses)
    uncertainties = {
        "predicate_arguments": {"confidence": 0.9 if all(c.get("predicate") for c in clauses) else 0.35, "reasons": []},
        "negation_scope": {"confidence": 0.95 if any(c.get("negated") for c in clauses) else 1.0, "reasons": []},
        "referents": {"confidence": 1.0 if not referents else (1.0 if all(r.get("resolved") for r in referents) else 0.45), "reasons": ["multiple plausible possessors"] if any(len(r["candidates"]) > 1 for r in referents) else []},
        "pragmatics": {"confidence": pragmatic["confidence"], "reasons": [] if context.get("tone") else ["prosody/context not supplied"]},
        "morphology": {"confidence": 0.98, "reasons": []},
    }
    alternatives = []
    for ref in referents:
        for candidate in ref["candidates"][:4]:
            alternatives.append({"meaning": {"clauses": clauses, "possessor": candidate["id"], "pragmatics": pragmatic["interpretation"]}, "changes": ["possessor"], "confidence": round(1 / max(1, len(ref["candidates"])), 3)})
    if "great" in text.casefold():
        alternatives.extend([
            {"meaning": {"clauses": clauses, "pragmatics": "sincere_positive_evaluation"}, "changes": ["pragmatics"], "confidence": 0.5},
            {"meaning": {"clauses": clauses, "pragmatics": "sarcastic_negative_evaluation"}, "changes": ["pragmatics"], "confidence": 0.5},
        ])
    return {
        "version": 2, "text": text, "morphology": tokens, "clauses": clauses,
        "constructions": matches, "speech_act": pragmatic, "referents": referents,
        "discourse_state": discourse.snapshot(), "uncertainty": uncertainties,
        "whole_utterance_interpretations": alternatives[:8],
        "origins": [make_origin("LanguageIntelligence", "V2", inputs={"text": text}, trigger="utterance_received")],
    }


def reading_span_metadata(source: str, passage_index: int, passage_count: int, text: str, document_progress: Mapping[str, Any] | None = None) -> dict[str, Any]:
    document_id = "document_" + hashlib.sha256(str(source).encode("utf-8", errors="replace")).hexdigest()[:20]
    section_value = (document_progress or {}).get("section") or (document_progress or {}).get("entry") or "root"
    section_id = "section_" + hashlib.sha256(f"{source}\0{section_value}".encode()).hexdigest()[:20]
    passage_id = "passage_" + hashlib.sha256((str(source) + "\0" + text).encode("utf-8", errors="replace")).hexdigest()[:20]
    return {"document_id": document_id, "section_id": section_id, "passage_id": passage_id, "parent_ids": [document_id, section_id], "hierarchy": ["document", "section", "passage"], "passage_index": passage_index, "passage_count": passage_count, "relative_position": round((passage_index + 1) / max(1, passage_count), 6)}


__all__ = ["ConstructionLearner", "DiscourseEntityMemory", "analyze_utterance", "morphology", "reading_span_metadata"]
