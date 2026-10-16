"""Bounded, inspectable language structure and discourse analysis."""
from __future__ import annotations

import hashlib
import re
from collections import Counter, OrderedDict
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping

from io_utils import atomic_write_json, load_json_dict
from origin_record import make_origin

_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:['’][^\W\d_]+)?|[^\s]", re.UNICODE)
_CONTRACTIONS = {
    "didn't": ("did", "not"), "doesn't": ("does", "not"), "don't": ("do", "not"),
    "isn't": ("is", "not"), "wasn't": ("was", "not"), "can't": ("can", "not"),
    "won't": ("will", "not"), "shouldn't": ("should", "not"), "wouldn't": ("would", "not"),
    "i'm": ("i", "am"), "you're": ("you", "are"), "that's": ("that", "is"),
}
_LEMMA = {"told": "tell", "said": "say", "stole": "steal", "gave": "give", "did": "do", "was": "be", "is": "be", "are": "be", "am": "be", "were": "be", "does": "do", "has": "have", "had": "have", "ran": "run", "running": "run", "given": "give", "gone": "go", "went": "go", "seen": "see", "saw": "see", "found": "find", "felt": "feel", "written": "write", "wrote": "write", "read": "read", "taught": "teach", "thought": "think"}
_PAST = {"told", "said", "stole", "gave", "did", "was", "were", "had", "ran", "went", "saw", "found", "felt", "wrote", "taught", "thought"}
_MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}
_VERBS = {"tell", "say", "steal", "give", "be", "do", "have", "find", "remember", "think", "know", "open", "arrive", "run", "close", "go", "see", "feel", "write", "read", "teach", "learn", "revise", "understand", "recognize", "recognise", "keep", "change"}
MAX_ANALYSIS_TOKENS = 512
MAX_ANALYSIS_CHARACTERS = 16384
_PRONOUNS = {"i", "me", "my", "you", "your", "he", "him", "his", "she", "her", "hers", "they", "them", "their", "it", "its", "we", "us", "our"}


def _lemma_word(word: str) -> str:
    if word in _LEMMA:
        return _LEMMA[word]
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        return stem + "e" if stem + "e" in _VERBS else stem
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem + "e" if stem + "e" in _VERBS else stem
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def morphology(text: str) -> list[dict[str, Any]]:
    result = []
    for match in islice(_TOKEN_RE.finditer(str(text)[:MAX_ANALYSIS_CHARACTERS]), MAX_ANALYSIS_TOKENS):
        surface = match.group()
        lowered = surface.casefold().replace("’", "'")
        expanded = _CONTRACTIONS.get(lowered, (lowered,))
        for index, normalized in enumerate(expanded):
            if len(result) >= MAX_ANALYSIS_TOKENS:
                return result
            if normalized in ".!?,":
                result.append({"surface": surface, "normalized": normalized, "lemma": normalized, "features": {"punctuation": True}})
                continue
            if re.fullmatch(r"\d+(?:[.,]\d+)*", normalized):
                result.append({"surface": surface, "normalized": normalized, "lemma": normalized,
                               "features": {"quantity": True}})
                continue
            lemma = _lemma_word(normalized)
            features: dict[str, Any] = {}
            if not re.fullmatch(r"[a-z]+(?:'[a-z]+)?", normalized):
                features["unresolved_surface"] = True
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
    words = list(words)
    candidates = [i for i, word in enumerate(words) if _lemma_word(word) in _VERBS or word in _MODALS]
    verb_at = candidates[0] if candidates else None
    auxiliary = None
    question_word = words[0] if words and words[0] in {"who", "what", "where", "when", "why", "how"} else None
    front = 1 if question_word else 0
    inverted = bool(candidates and candidates[0] == front
                    and (_lemma_word(words[front]) in {"do", "be", "have"} or words[front] in _MODALS))
    if len(candidates) > 1:
        first, second = candidates[:2]
        is_aux = _lemma_word(words[first]) in {"do", "be", "have"} or words[first] in _MODALS
        between = words[first + 1:second]
        if is_aux and (inverted or all(word in {"not", "never"} for word in between)):
            auxiliary, verb_at = words[first], second
    predicate_word = words[verb_at] if verb_at is not None else None
    predicate = _lemma_word(predicate_word or "") or None
    if inverted and auxiliary is not None:
        subject_words = [word for word in words[front + 1:verb_at] if word not in {"not", "never"}]
    elif inverted and predicate == "be":
        # A short copular question has a subject followed by its complement.
        subject_words = words[front + 1:front + 2]
    else:
        subject_words = words[:candidates[0] if auxiliary else verb_at] if verb_at is not None else []
    subject = " ".join(subject_words) or None
    negated = any(word in {"not", "never"} for word in words)
    arguments: dict[str, Any] = {}
    tail_start = verb_at + 1 if verb_at is not None else len(words)
    if inverted and auxiliary is None and predicate == "be": tail_start = front + 2
    tail = [word for word in words[tail_start:] if word not in {"not", "never", ".", "!", "?", ","}]
    voice = "active"
    # Preserve semantic roles across passive/active forms; do not invent a by-agent.
    passive = bool(auxiliary and _lemma_word(auxiliary) == "be"
                   and (predicate_word in {"given", "seen", "written", "taught"} or str(predicate_word).endswith("ed")))
    if passive:
        voice = "passive"
        arguments["theme"] = subject
        if "by" in tail:
            at = tail.index("by"); subject = " ".join(tail[at + 1:]) or None; tail = tail[:at]
        else:
            subject = None
        if "to" in tail:
            at = tail.index("to")
            arguments["recipient"] = " ".join(tail[at + 1:])
    elif predicate in {"give", "tell", "teach"}:
        recipient_role = "addressee" if predicate == "tell" else "recipient"
        if "to" in tail:
            at = tail.index("to")
            if at: arguments["theme"] = " ".join(tail[:at])
            if at + 1 < len(tail): arguments[recipient_role] = " ".join(tail[at + 1:])
        elif tail and tail[0] not in {"a", "an", "the"} and (len(tail) > 1 or predicate == "tell"):
            arguments[recipient_role] = tail[0]
            if len(tail) > 1: arguments["theme"] = " ".join(tail[1:])
        elif tail: arguments["theme"] = " ".join(tail)
    elif tail:
        arguments["theme"] = " ".join(tail)
    tense_word = auxiliary or predicate_word or ""
    return {
        "id": clause_id, "parent_clause": parent, "subject": subject,
        "predicate": predicate, "arguments": arguments, "negated": negated,
        "negation_scope": clause_id if negated else None,
        "tense": "past" if tense_word in _PAST or tense_word.endswith("ed") else "present",
        "modality": next((word for word in words if word in _MODALS), None),
        "voice": voice, "question_word": question_word, "inverted": inverted,
        "surface_words": words,
    }


def _clauses(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments, current = [], []
    for item in tokens:
        word = item["normalized"]
        if word in {".", "!", "?", ";"}:
            if current: segments.append(current); current = []
        elif word not in {",", '"', "“", "”"}:
            current.append(word)
    if current: segments.append(current)
    result = []
    for words in segments[:8]:
        reporting = next((i for i, word in enumerate(words) if _lemma_word(word) in {"say", "tell"}), None)
        split = None
        if reporting is not None:
            start = reporting + 1
            if _lemma_word(words[reporting]) == "tell" and start < len(words): start += 1
            if start < len(words) and words[start] == "that": start += 1
            if start + 1 < len(words) and any(_lemma_word(word) in _VERBS or word in _MODALS for word in words[start + 1:]):
                split = start
        identifier = f"c{len(result)}"
        if split is None:
            result.append(_clause(words, clause_id=identifier))
        else:
            result.append(_clause(words[:split], clause_id=identifier))
            result.append(_clause(words[split:], clause_id=f"c{len(result)}", parent=identifier))
        if len(result) >= 8: break
    return result[:8] or [_clause([], clause_id="c0")]


def _speech_act(text: str, context: Mapping[str, Any]) -> dict[str, Any]:
    lowered = text.casefold().strip()
    first = lowered.split()[0] if lowered.split() else ""
    act = "question" if lowered.endswith("?") or first in {"who", "what", "where", "when", "why", "how"} else "request" if first in {"please", "could", "would"} else "assertion"
    tone = str(context.get("tone") or context.get("pragmatic_tone") or "").casefold()
    sarcastic = tone in {"sarcastic", "ironic"} or bool(context.get("sarcastic"))
    reading = "sarcastic_negative_evaluation" if sarcastic and "great" in lowered else "sincere_positive_evaluation" if "great" in lowered else act
    return {"act": act, "interpretation": reading, "tone": "sarcastic" if sarcastic else "literal", "confidence": 0.9 if tone else 0.7}


def analyze_utterance(text: str, *, context: Mapping[str, Any] | None = None, discourse: DiscourseEntityMemory | None = None, constructions: ConstructionLearner | None = None, turn: int | None = None) -> dict[str, Any]:
    context = context or {}
    discourse = discourse or DiscourseEntityMemory()
    text = str(text or "")
    tokens = morphology(text)
    incomplete = len(text) > MAX_ANALYSIS_CHARACTERS or len(tokens) >= MAX_ANALYSIS_TOKENS
    clauses = _clauses(tokens)
    incomplete = incomplete or len(clauses) >= 8
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
        "morphology": {
            "confidence": 0.45 if any(t["features"].get("unresolved_surface") for t in tokens)
                or not tokens or not all(c.get("predicate") for c in clauses) else 0.85,
            "reasons": ["unsupported surface or construction"] if any(t["features"].get("unresolved_surface") for t in tokens)
                or not tokens or not all(c.get("predicate") for c in clauses) else ["heuristic analysis; not empirically calibrated"],
        },
    }
    if incomplete:
        for dimension in uncertainties.values():
            dimension["confidence"] = min(dimension["confidence"], 0.35)
            dimension["reasons"].append("analysis budget reached; interpretation incomplete")
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
        "version": 4, "text": text[:MAX_ANALYSIS_CHARACTERS], "incomplete": incomplete, "morphology": tokens, "clauses": clauses,
        "constructions": matches, "speech_act": pragmatic, "referents": referents,
        "discourse_state": discourse.snapshot(), "uncertainty": uncertainties,
        "whole_utterance_interpretations": alternatives[:8],
        "origins": [make_origin("LanguageIntelligence", "V4", inputs={"text": text[:MAX_ANALYSIS_CHARACTERS], "incomplete": incomplete}, trigger="utterance_received")],
    }


def reading_span_metadata(source: str, passage_index: int, passage_count: int, text: str, document_progress: Mapping[str, Any] | None = None) -> dict[str, Any]:
    document_id = "document_" + hashlib.sha256(str(source).encode("utf-8", errors="replace")).hexdigest()[:20]
    section_value = (document_progress or {}).get("section") or (document_progress or {}).get("entry") or "root"
    section_id = "section_" + hashlib.sha256(f"{source}\0{section_value}".encode()).hexdigest()[:20]
    passage_id = "passage_" + hashlib.sha256((str(source) + "\0" + text).encode("utf-8", errors="replace")).hexdigest()[:20]
    return {"document_id": document_id, "section_id": section_id, "passage_id": passage_id, "parent_ids": [document_id, section_id], "hierarchy": ["document", "section", "passage"], "passage_index": passage_index, "passage_count": passage_count, "relative_position": round((passage_index + 1) / max(1, passage_count), 6)}


__all__ = ["ConstructionLearner", "DiscourseEntityMemory", "analyze_utterance", "morphology", "reading_span_metadata"]
