"""Small, bounded discourse/deixis resolution for conversational cognition."""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping
MODULE_NAME = "discourse"
MODULE_VERSION = "V3"


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_ROLE_WORDS = {
    "i": ("speaker", False), "me": ("speaker", False),
    "my": ("speaker", True), "mine": ("speaker", True),
    "you": ("addressee", False), "your": ("addressee", True), "yours": ("addressee", True),
    "we": ("speaker_group", False), "us": ("speaker_group", False),
    "our": ("speaker_group", True), "ours": ("speaker_group", True),
    "they": ("mentioned_entities", False), "them": ("mentioned_entities", False),
    "their": ("mentioned_entities", True), "theirs": ("mentioned_entities", True),
    "this": ("current_referent", False), "that": ("prior_referent", False),
}
DISCOURSE_TERMS = frozenset(_ROLE_WORDS)


def _entity(value: Any, *, fallback: str = "unknown") -> dict[str, Any]:
    if isinstance(value, Mapping):
        name = str(value.get("display_name") or value.get("name") or value.get("id") or fallback)[:80]
        is_self = bool(value.get("is_self"))
        identifier = "self" if is_self else str(value.get("id") or value.get("internal_id") or name).strip().casefold()[:128]
        return {"id": identifier or fallback, "name": name, "is_self": is_self}
    name = str(value or fallback).strip()[:80] or fallback
    return {"id": name.casefold(), "name": name, "is_self": False}


def _unique_entities(values: Iterable[Any], limit: int = 8) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        entity = _entity(value)
        if entity["id"] in seen or entity["id"] == "unknown":
            continue
        result.append(entity)
        seen.add(entity["id"])
        if len(result) >= limit:
            break
    return result


def build_discourse_context(
    text: str, *, speaker: Any, addressee: Any, self_identity: Any = None,
    current_subject: Any = None, mentioned_entities: Iterable[Any] = (), prior_referent: Any = None,
) -> dict[str, Any]:
    """Resolve explicit conversational roles while preserving ambiguity."""
    speaker_entity = _entity(speaker)
    addressee_entity = _entity(addressee)
    self_entity = _entity(self_identity or {"id": "self", "name": "self", "is_self": True})
    self_entity["is_self"] = True
    mentioned = _unique_entities(mentioned_entities)
    subject = _entity(current_subject) if current_subject else None
    prior = _entity(prior_referent) if prior_referent else subject
    resolutions = []
    for index, match in enumerate(_WORD_RE.finditer(str(text or ""))):
        surface = match.group(0).casefold()
        role_spec = _ROLE_WORDS.get(surface)
        if role_spec is None:
            continue
        role, possessive = role_spec
        ambiguous = False
        if role == "speaker":
            referents = [speaker_entity]
        elif role == "addressee":
            referents = [addressee_entity]
        elif role == "speaker_group":
            referents = _unique_entities((speaker_entity, addressee_entity))
            ambiguous = True
        elif role == "mentioned_entities":
            referents = mentioned
            ambiguous = len(referents) != 1
        else:
            referent = subject if role == "current_referent" else prior
            referents = [referent] if referent else []
            ambiguous = referent is None
        confidence = 1.0 if len(referents) == 1 and not ambiguous else (0.45 if referents else 0.0)
        retrieval_terms = []
        for referent in referents:
            if not isinstance(referent, Mapping):
                continue
            for value in (referent.get("name"), referent.get("id")):
                term = str(value or "").strip().casefold()
                if term and term not in {"self", "unknown"} and term not in retrieval_terms:
                    retrieval_terms.append(term)
        resolutions.append({
            "surface": surface, "token_index": index, "role": role,
            "possessive": possessive, "referents": referents[:8], "ambiguous": ambiguous,
            "confidence": confidence, "retrieval_terms": retrieval_terms[:8],
        })
    referent_table = {
        "speaker": speaker_entity,
        "addressee": addressee_entity,
        "self": self_entity,
        "current_referent": subject,
        "prior_referent": prior,
        "mentioned_entities": mentioned,
    }
    return {
        "version": 2, "speaker": speaker_entity, "addressee": addressee_entity,
        "self": self_entity, "current_subject": subject,
        "mentioned_entities": mentioned, "referent_table": referent_table,
        "resolutions": resolutions[:32],
    }


def retrieval_routes(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose bounded deictic routes without treating pronouns as concepts."""
    routes = []
    for item in list(context.get("resolutions") or ())[:32]:
        if not isinstance(item, Mapping):
            continue
        referents = [dict(ref) for ref in item.get("referents") or () if isinstance(ref, Mapping)]
        routes.append({
            "surface": str(item.get("surface") or ""),
            "token_index": item.get("token_index"),
            "role": item.get("role"),
            "possessive": bool(item.get("possessive")),
            "status": "resolved" if len(referents) == 1 and not item.get("ambiguous") else "ambiguous",
            "referents": referents[:8],
            "retrieval_terms": list(item.get("retrieval_terms") or ())[:8],
            "confidence": float(item.get("confidence") or 0.0),
        })
    return routes


def render_referent_gloss(gloss: str, resolution: Mapping[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
    """Mark an uncertain referent without pretending the guess is grounded."""
    if not isinstance(resolution, Mapping) or not resolution.get("ambiguous"):
        return str(gloss), None
    alternatives = [
        str(item.get("name") or item.get("id"))
        for item in resolution.get("referents") or ()
        if isinstance(item, Mapping) and (item.get("name") or item.get("id"))
    ][:4]
    suffix = "?" if not alternatives else "?=" + "/".join(alternatives)
    return f"{gloss}[{suffix}]", {
        "role": resolution.get("role"), "alternatives": alternatives,
        "confidence": float(resolution.get("confidence") or 0.0),
    }


def resolution_for(context: Mapping[str, Any], surface: str) -> dict[str, Any] | None:
    target = str(surface or "").casefold()
    for item in context.get("resolutions") or ():
        if isinstance(item, Mapping) and item.get("surface") == target:
            return dict(item)
    return None


def role_alignment(current: Mapping[str, Any], recalled: Mapping[str, Any], surface: str) -> dict[str, Any]:
    """Compare deictic roles without treating their words as lexical concepts."""
    present = resolution_for(current, surface)
    historical = resolution_for(recalled, surface)
    if not present or not historical:
        return {"available": False, "matched": False, "score": 0.0}
    present_ids = {str(item.get("id")) for item in present.get("referents") or () if isinstance(item, Mapping) and item.get("id")}
    historical_ids = {str(item.get("id")) for item in historical.get("referents") or () if isinstance(item, Mapping) and item.get("id")}
    matched = bool(present_ids and historical_ids and present_ids & historical_ids)
    return {
        "available": bool(present_ids and historical_ids), "matched": matched,
        "score": 0.5 if matched else -0.2, "role": present.get("role"),
        "present_referents": sorted(present_ids), "recalled_referents": sorted(historical_ids),
    }


__all__ = ["DISCOURSE_TERMS", "build_discourse_context", "render_referent_gloss", "resolution_for", "retrieval_routes", "role_alignment"]
