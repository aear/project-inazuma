"""Bounded, inspectable language structure and discourse analysis."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, OrderedDict
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Mapping

from io_utils import atomic_write_json, load_json_dict
from origin_record import make_origin
from discourse_context import resolution_for

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
# Grammar recognition only: these labels do not ground a desire or social intent.
_VERBS |= {"want", "like", "enjoy", "prefer", "need", "ask", "help", "show", "share", "explain", "listen", "talk", "try", "choose", "love", "hope"}
_VERBS |= {"believe", "wish", "suppose", "doubt"}
_COMPLEMENT_VERBS = {"say", "tell", "think", "know", "believe", "hope", "wish", "suppose", "doubt", "want"}
_PASSIVE_PARTICIPLES = {"given", "seen", "written", "taught", "found", "kept", "stolen", "told"}
_LEMMA.update({"kept": "keep", "stolen": "steal", "chose": "choose", "chosen": "choose"})
_PAST.update({"kept", "chose"})

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
            if all(unicodedata.category(c).startswith("P") for c in normalized):
                result.append({"surface": surface, "source_span": list(match.span()), "normalized": normalized, "lemma": normalized, "features": {"punctuation": True}})
                continue
            if re.fullmatch(r"\d+(?:[.,]\d+)*", normalized):
                result.append({"surface": surface, "source_span": list(match.span()), "normalized": normalized, "lemma": normalized,
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
            result.append({"surface": surface, "source_span": list(match.span()), "normalized": normalized, "lemma": lemma, "features": features})
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
        role = {"i": "speaker", "me": "speaker", "my": "speaker",
                "you": "addressee", "your": "addressee"}.get(pronoun.casefold())
        if role:
            # Missing conversational identity cannot be replaced by recency.
            values = [item for item in values if item.get(role) is True]
        elif pronoun.casefold() in {"we", "us", "our"}:
            return []  # A group needs explicit discourse evidence.
        return [dict(item) for item in reversed(values[-4:])]

    def snapshot(self) -> dict[str, Any]:
        return {"version": 1, "entities": list(self.entities.values())}


class ConstructionLearner:
    """Bounded counts for reusable multi-word predicate/argument patterns."""

    def __init__(self, path: Path | None = None, max_patterns: int = 256) -> None:
        self.path = Path(path) if path else None
        self.max_patterns = max(8, int(max_patterns))
        raw = load_json_dict(self.path) if self.path else {}
        self.counts = Counter({str(k): int(v) for k, v in islice((raw.get("counts") or {}).items(), self.max_patterns)})
        self.source_events = list(dict.fromkeys(str(v)[:160] for v in (raw.get("source_events") or ())[-512:]))

    def observe(self, clauses: Iterable[Mapping[str, Any]], *, source_event: str | None = None) -> list[dict[str, Any]]:
        """Count supplied examples, not independent corroboration or comprehension.

        Optional event IDs suppress replay within the retained 512-event window.
        They are caller provenance, not authenticated observations.
        """
        if source_event is not None and (not isinstance(source_event, str) or not 1 <= len(source_event) <= 160):
            raise ValueError("source_event must be 1..160 characters")
        replay = source_event is not None and source_event in self.source_events
        matches = []
        for clause in islice(clauses, 8):
            roles = clause.get("arguments") or {}
            pattern = f"{clause.get('predicate')}({','.join(sorted(roles))})"
            if not replay:
                self.counts[pattern] += 1
            matches.append({"pattern": pattern, "observations": self.counts[pattern],
                            "source_event": source_event, "replay": replay})
        self.counts = Counter(dict(self.counts.most_common(self.max_patterns)))
        if source_event is not None and not replay:
            self.source_events = (self.source_events + [source_event])[-512:]
        if self.path and not replay:
            atomic_write_json(self.path, {"version": 2, "counts": dict(self.counts), "source_events": self.source_events})
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
                   and (predicate_word in _PASSIVE_PARTICIPLES or str(predicate_word).endswith("ed")))
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
    alternatives = []
    choice_words = str(arguments.get("theme") or "").split()
    if not passive and choice_words.count("or") == 1:
        at = choice_words.index("or")
        left = choice_words[:at]
        if left and left[0] == "either": left = left[1:]
        if left and choice_words[at + 1:]:
            alternatives.append({"relation": "alternative", "argument_role": "theme",
                                 "options": [" ".join(left), " ".join(choice_words[at + 1:])],
                                 "exclusivity": "unresolved",
                                 "negation_scope": "unresolved" if negated else None})
    tense_word = auxiliary or predicate_word or ""
    return {
        "id": clause_id, "parent_clause": parent, "subject": subject,
        "predicate": predicate, "arguments": arguments, "negated": negated,
        "negation_operator": "never" if "never" in words else "not" if "not" in words else None,
        "negation_scope": clause_id if negated else None,
        "tense": "past" if tense_word in _PAST or tense_word.endswith("ed") else "present",
        "modality": next((word for word in words if word in _MODALS), None),
        "voice": voice, "question_word": question_word, "inverted": inverted,
        "surface_words": words, "alternatives": alternatives,
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
    def admit(words, parent=None, depth=0, relation=None):
        if len(result) >= 8: return
        identifier = f"c{len(result)}"
        matrix = next((i for i, word in enumerate(words) if _lemma_word(word) in _COMPLEMENT_VERBS), None)
        split = None
        if matrix is not None:
            start = matrix + 1
            if _lemma_word(words[matrix]) == "tell" and start < len(words): start += 1
            if start < len(words) and words[start] == "that": start += 1
            inner_verb = next((i for i in range(start + 1, len(words))
                               if _lemma_word(words[i]) in _VERBS or words[i] in _MODALS), None)
            prefix = words[start:inner_verb] if inner_verb is not None else []
            # A finite complement needs an explicit subject. Infinitives and
            # implied subjects remain outside this bounded recognizer.
            if prefix and "to" not in prefix and not any(word in {"not", "never", "or", "and"} for word in prefix):
                split = start
        if split is not None and depth < 4:
            outer_words = words[:split]
            if outer_words and outer_words[-1] == "that": outer_words = outer_words[:-1]
            outer = _clause(outer_words, clause_id=identifier, parent=parent)
            outer["clausal_relation"] = relation
            outer["embedded_content_is_fact"] = False
            result.append(outer)
            admit(words[split:], parent=identifier, depth=depth+1,
                  relation="reported_content" if outer["predicate"] in {"say", "tell"} else "attitude_content")
        else:
            clause = _clause(words, clause_id=identifier, parent=parent)
            clause["clausal_relation"] = relation
            clause["embedded_content_is_fact"] = False
            clause["scope_incomplete"] = split is not None or (matrix is not None and "to" in words[matrix+1:] and any(word in {"not", "never"} for word in words[matrix+1:]))
            result.append(clause)
    for words in segments[:8]:admit(words)
    return result[:8] or [_clause([], clause_id="c0")]


def _speech_act(text: str, context: Mapping[str, Any]) -> dict[str, Any]:
    lowered = text.casefold().strip()
    first = lowered.split()[0] if lowered.split() else ""
    act = "question" if lowered.endswith("?") or first in {"who", "what", "where", "when", "why", "how"} else "request" if first in {"please", "could", "would"} else "assertion"
    tone = str(context.get("tone") or context.get("pragmatic_tone") or "").casefold()
    sarcastic = tone in {"sarcastic", "ironic"} or bool(context.get("sarcastic"))
    reading = "sarcastic_negative_evaluation" if sarcastic and "great" in lowered else "sincere_positive_evaluation" if "great" in lowered else act
    return {"act": act, "interpretation": reading, "tone": "sarcastic" if sarcastic else "literal", "confidence": 0.9 if tone else 0.7}


def analyze_utterance(text: str, *, context: Mapping[str, Any] | None = None, discourse: DiscourseEntityMemory | None = None, constructions: ConstructionLearner | None = None, turn: int | None = None, source_event: str | None = None) -> dict[str, Any]:
    context = context or {}
    discourse = discourse or DiscourseEntityMemory()
    text = str(text or "")
    from character_language import analyze_characters
    character_analysis = analyze_characters(
        text, known_words=context.get("known_words") or (),
        expansions=context.get("surface_expansions") or (),
    )
    oversized = len(text) > MAX_ANALYSIS_CHARACTERS
    text = text[:MAX_ANALYSIS_CHARACTERS]
    tokens = morphology(text)
    incomplete = oversized or len(tokens) >= MAX_ANALYSIS_TOKENS
    clauses = _clauses(tokens)
    incomplete = incomplete or len(clauses) >= 8 or any(c.get("scope_incomplete") for c in clauses)
    for clause in clauses:
        subject = clause.get("subject")
        if subject and subject not in _PRONOUNS: discourse.mention(subject, role="agent", turn=turn)
        for role, value in (clause.get("arguments") or {}).items():
            for name in re.findall(r"\b[A-Z][a-z]+\b", str(value)):
                discourse.mention(name, role=role, turn=turn)
    proper = [name for name in re.findall(r"\b[A-Z][a-z]+\b", text) if name.casefold() not in _PRONOUNS]
    for name in proper: discourse.mention(name, turn=turn)
    authoritative = context.get("discourse")
    def referent_candidates(surface: str) -> list[dict[str, Any]]:
        route = resolution_for(authoritative, surface) if isinstance(authoritative, Mapping) else None
        if route is not None:
            return [dict(item) for item in (route.get("referents") or ())[:8]
                    if isinstance(item, Mapping) and item.get("id") not in {None, "unknown"}]
        return discourse.candidates(surface)

    def resolved_id(surface: str, candidates: list[dict[str, Any]]) -> str | None:
        route = resolution_for(authoritative, surface) if isinstance(authoritative, Mapping) else None
        return candidates[0]["id"] if len(candidates) == 1 and not (route or {}).get("ambiguous") else None

    referents: list[dict[str, Any]] = []
    for clause in clauses:
        subject = str(clause.get("subject") or "").casefold()
        if subject in _PRONOUNS:
            candidates = referent_candidates(subject)
            referents.append({
                "surface": subject, "relation": "subject", "candidates": candidates,
                "resolved": resolved_id(subject, candidates),
            })
    possessives = re.finditer(r"\b(his|her|their|my|your|our|its)\s+(\w+)|\b(\w+)['’]s\s+(\w+)", text[:MAX_ANALYSIS_CHARACTERS], re.I)
    possessive_matches = list(islice(possessives, 33))
    incomplete = incomplete or len(possessive_matches) > 32
    for possessive in possessive_matches[:32]:
        pronoun_owner, pronoun_thing, named_owner, named_thing = possessive.groups()
        owner, thing = (pronoun_owner, pronoun_thing) if pronoun_owner else (named_owner, named_thing)
        if named_owner and (named_owner.casefold() + "'s") in _CONTRACTIONS:
            continue  # "that's" is a contraction, not possession evidence.
        if owner.casefold() in {"his", "her", "their"}:
            candidates = referent_candidates(owner)
        elif owner.casefold() in _PRONOUNS:
            candidates = referent_candidates(owner)
        else:
            candidates = [discourse.mention(owner, role="possessor", turn=turn)]
        referents.append({"surface": owner, "relation": "possessor", "object": thing, "source_span": list(possessive.span()), "candidates": candidates, "resolved": resolved_id(owner, candidates)})
    pragmatic = _speech_act(text, context)
    surface_interpretations = []
    surface_interpretations_complete = character_analysis["complete"] and all(
        form["candidate_set_complete"] for form in character_analysis["forms"]
    )
    if character_analysis["bounds"]["source_complete"]:
        for form in character_analysis["forms"]:
            for candidate in form["candidates"]:
                if len(surface_interpretations) >= 8:
                    surface_interpretations_complete = False
                    break
                start, end = form["span"]
                variant = text[:start] + candidate["text"] + text[end:]
                surface_interpretations.append({
                    "text_candidate": variant, "source_span": [start, end],
                    "source_surface": form["surface"], "candidate_evidence": candidate,
                    "clauses": _clauses(morphology(variant)),
                    "speech_act": _speech_act(variant, context),
                    "status": "unconfirmed_alternative", "selected": False,
                    "meaning_certified": False,
                })
    matches = (constructions or ConstructionLearner()).observe(clauses, source_event=source_event)
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
            factor = "possessor" if ref["relation"] == "possessor" else "subject"
            alternatives.append({"meaning": {"clauses": clauses, factor: candidate["id"], "pragmatics": pragmatic["interpretation"]}, "changes": [factor], "confidence": round(1 / max(1, len(ref["candidates"])), 3)})
    if "great" in text.casefold():
        alternatives.extend([
            {"meaning": {"clauses": clauses, "pragmatics": "sincere_positive_evaluation"}, "changes": ["pragmatics"], "confidence": 0.5},
            {"meaning": {"clauses": clauses, "pragmatics": "sarcastic_negative_evaluation"}, "changes": ["pragmatics"], "confidence": 0.5},
        ])
    return {
        "version": 8, "character_analysis": character_analysis,
        "surface_interpretations": surface_interpretations,
        "surface_interpretations_complete": surface_interpretations_complete,
        "text": text[:MAX_ANALYSIS_CHARACTERS], "incomplete": incomplete, "morphology": tokens, "clauses": clauses,
        "constructions": matches, "speech_act": pragmatic, "referents": referents,
        "discourse_state": discourse.snapshot(), "uncertainty": uncertainties,
        "whole_utterance_interpretations": alternatives[:8],
        "origins": [make_origin("LanguageIntelligence", "V8", inputs={"text": text[:MAX_ANALYSIS_CHARACTERS], "incomplete": incomplete}, trigger="utterance_received")],
    }


def reading_span_metadata(source: str, passage_index: int, passage_count: int, text: str, document_progress: Mapping[str, Any] | None = None) -> dict[str, Any]:
    document_id = "document_" + hashlib.sha256(str(source).encode("utf-8", errors="replace")).hexdigest()[:20]
    section_value = (document_progress or {}).get("section") or (document_progress or {}).get("entry") or "root"
    section_id = "section_" + hashlib.sha256(f"{source}\0{section_value}".encode()).hexdigest()[:20]
    passage_id = "passage_" + hashlib.sha256((str(source) + "\0" + text).encode("utf-8", errors="replace")).hexdigest()[:20]
    return {"document_id": document_id, "section_id": section_id, "passage_id": passage_id, "parent_ids": [document_id, section_id], "hierarchy": ["document", "section", "passage"], "passage_index": passage_index, "passage_count": passage_count, "relative_position": round((passage_index + 1) / max(1, passage_count), 6)}


__all__ = ["ConstructionLearner", "DiscourseEntityMemory", "analyze_utterance", "morphology", "reading_span_metadata"]
