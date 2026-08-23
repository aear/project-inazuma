"""Bounded routing and evidence helpers for Ina's self-questions.

This module makes no autonomous calls.  It describes the next appropriate
source and lets the owning subsystem record evidence and evaluation results.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional


QUESTION_TYPES = {
    "semantic_grounding", "semantic_validation", "relation_learning",
    "self_reflection", "environment_grounding", "action_feedback",
    "system_fault", "open_philosophical",
}

_CONTRACTIONS = {
    "can't": "cannot", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "isn't": "is not",
    "wasn't": "was not", "weren't": "were not", "won't": "will not",
}
_OPAQUE_SOUND = re.compile(r"^(?:pair:)?(?:sym_snd_|sound_symbol_|combo_snd_)", re.I)


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("’", "'").replace("‘", "'")
    value = re.sub(r"\s+", " ", value).strip()
    for contraction, expanded in _CONTRACTIONS.items():
        value = re.sub(rf"\b{re.escape(contraction)}\b", expanded, value, flags=re.I)
    return value


def semantic_text_candidate(value: str) -> Optional[str]:
    """Return normalized lexical evidence, excluding opaque sound IDs."""
    candidate = normalize_text(value).strip(" '\".,?!")
    return None if not candidate or _OPAQUE_SOUND.match(candidate) else candidate


def classify_question(question: str) -> str:
    q = normalize_text(question).lower()
    if "what experience grounds" in q or "lack enough grounded" in q:
        return "semantic_grounding"
    if q.startswith("do i understand") or "confuse meaning" in q or "wrong word" in q:
        return "semantic_validation"
    if q.startswith("what links ") or "relationship" in q:
        return "relation_learning"
    if any(part in q for part in ("feels most like me", "what am i feeling", "who am i")):
        return "self_reflection"
    if "device called" in q or "which device" in q:
        return "environment_grounding"
    if any(part in q for part in ("what happened when", "did my action", "reversible")):
        return "action_feedback"
    if any(part in q for part in ("corrupted", "repair", "fault", "drifting", "re-cluster")):
        return "system_fault"
    return "open_philosophical"


def resolution_sources(question_type: str) -> List[str]:
    routes = {
        "semantic_grounding": ["own_memory", "current_runtime_state", "experiment_action", "human_operator", "external_learning_material"],
        "semantic_validation": ["own_memory", "experiment_action", "human_operator"],
        "relation_learning": ["own_memory", "own_source_code_self_read", "current_runtime_state", "experiment_action"],
        "self_reflection": ["own_memory", "current_runtime_state", "unresolved_open_reflection", "human_operator"],
        "environment_grounding": ["current_runtime_state", "own_source_code_self_read", "environment_device_probe", "human_operator"],
        "action_feedback": ["current_runtime_state", "experiment_action", "own_memory"],
        "system_fault": ["current_runtime_state", "own_source_code_self_read", "human_operator"],
        "open_philosophical": ["own_memory", "unresolved_open_reflection", "human_operator"],
    }
    return list(routes.get(question_type, routes["open_philosophical"]))


def question_key(question: str, question_type: Optional[str] = None) -> str:
    q = normalize_text(question).casefold()
    kind = question_type or classify_question(q)
    match = re.fullmatch(r"what links\s+(.+?)\s+and\s+(.+?)\??", q)
    if kind == "relation_learning" and match:
        left, right = sorted((match.group(1).strip(), match.group(2).strip()))
        q = f"what links {left} and {right}?"
    return f"{kind}:{q}"


def evidence_hash(evidence: Any = None, references: Optional[Iterable[Any]] = None) -> str:
    payload = {"evidence": evidence, "references": sorted(str(v) for v in (references or []))}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_resolution_plan(question: str, question_type: Optional[str] = None) -> Dict[str, Any]:
    kind = question_type if question_type in QUESTION_TYPES else classify_question(question)
    return {"question_type": kind, "sources": resolution_sources(kind), "next_source": resolution_sources(kind)[0]}


def formulate_help_request(entry: Dict[str, Any]) -> str:
    question = str(entry.get("question") or "").strip()
    kind = entry.get("question_type") or classify_question(question)
    quoted = re.search(r"['\"]([^'\"]+)['\"]", question)
    subject = quoted.group(1) if quoted else "this"
    if kind == "semantic_grounding":
        return f"I lack enough grounded experiences for '{subject}'. Could you show or describe an example and what makes it one, or a counterexample?"
    if kind == "semantic_validation":
        return f"I have a tentative interpretation of '{subject}'. Could you give an example where it would be wrong, or confirm it against a specific event?"
    if kind == "environment_grounding":
        return f"I could not resolve this from configuration or a device probe: {question} Could you identify the intended device?"
    if kind == "self_reflection":
        return f"I am reflecting on: {question} You may offer memories or observations as evidence, without deciding it for me."
    return f"I could not resolve this from my available evidence: {question} What specific evidence should I consider?"
