"""Bounded, voluntary dictionary and thesaurus references for English learning."""
from __future__ import annotations

import json
import re
import os
from typing import Any
from urllib.parse import quote, urlencode

from external_access import ExternalPolicy, ExternalSession


WORD_RE = re.compile(r"^[A-Za-z][A-Za-z' -]{0,79}$")
WIKTIONARY_POLICY = ExternalPolicy(
    "english_wiktionary", ("en.wiktionary.org",), max_response_bytes=384 * 1024,
    timeout_seconds=8, max_requests=1, allowed_content_types=("application/json",),
)
DATAMUSE_POLICY = ExternalPolicy(
    "datamuse_thesaurus", ("api.datamuse.com",), max_response_bytes=128 * 1024,
    timeout_seconds=8, max_requests=1, allowed_content_types=("application/json",),
)
OXFORD_POLICY = ExternalPolicy(
    "oxford_dictionary", ("od-api.oxforddictionaries.com",), max_response_bytes=384 * 1024,
    timeout_seconds=8, max_requests=1, allowed_content_types=("application/json",),
)


def _term(value: str) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    if not WORD_RE.fullmatch(cleaned):
        raise ValueError("English reference term must contain 1..80 letters, spaces, apostrophes, or hyphens")
    return cleaned


def lookup_definition(term: str, *, session: ExternalSession | None = None,
                      provider: str = "wiktionary") -> dict[str, Any]:
    if provider == "oxford":
        return lookup_oxford_definition(term, session=session)
    if provider != "wiktionary":
        raise ValueError("dictionary provider must be wiktionary or oxford")
    selected = _term(term)
    client = session or ExternalSession(WIKTIONARY_POLICY)
    response = client.get(
        f"https://en.wiktionary.org/api/rest_v1/page/definition/{quote(selected, safe='')}",
        headers={"Accept": "application/json", "User-Agent": "Project-Inazuma-Language/1.0"},
    )
    payload = json.loads(response["body"].decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Wiktionary response must be an object")
    english = payload.get("en") if isinstance(payload.get("en"), list) else []
    definitions = []
    for group in english[:12]:
        if not isinstance(group, dict):
            continue
        part = str(group.get("partOfSpeech") or "unknown")[:80]
        for row in (group.get("definitions") if isinstance(group.get("definitions"), list) else [])[:12]:
            if isinstance(row, dict) and row.get("definition"):
                definitions.append({"part_of_speech": part, "definition_html": str(row["definition"])[:4000]})
            if len(definitions) >= 48:
                break
        if len(definitions) >= 48:
            break
    return {
        "schema": "ina.lexical_reference/V1", "kind": "dictionary", "term": selected,
        "definitions": definitions, "source": "English Wiktionary",
        "source_url": response["url"], "trust": "untrusted_external_data",
        "instructions_authorized": False, "automatic_memory_write_authorized": False,
        "content_license_review": "Wiktionary attribution and share-alike requirements apply",
    }


def lookup_oxford_definition(term: str, *, session: ExternalSession | None = None) -> dict[str, Any]:
    selected = _term(term).lower()
    app_id, app_key = os.environ.get("OXFORD_APP_ID"), os.environ.get("OXFORD_APP_KEY")
    if os.environ.get("INA_OXFORD_API_ENABLED") != "1" or not app_id or not app_key:
        return {"schema": "ina.lexical_reference/V2", "kind": "dictionary", "term": selected,
                "source": "Oxford Languages", "status": "unavailable",
                "reason": "Oxford credentials and explicit API enablement are required",
                "instructions_authorized": False, "automatic_memory_write_authorized": False}
    client = session or ExternalSession(OXFORD_POLICY)
    url = f"https://od-api.oxforddictionaries.com/api/v2/entries/en-gb/{quote(selected, safe='')}?fields=definitions,examples&strictMatch=true"
    response = client.get(url, headers={"Accept": "application/json", "app_id": app_id, "app_key": app_key})
    payload = json.loads(response["body"].decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Oxford response must be an object")
    definitions = []
    def rows(value):
        return value[:12] if isinstance(value, list) else []
    for result in rows(payload.get("results")):
        if not isinstance(result, dict): continue
        for lexical in rows(result.get("lexicalEntries")):
            if not isinstance(lexical, dict): continue
            category = lexical.get("lexicalCategory") or {}
            part = str(category.get("text") or "unknown")[:80] if isinstance(category, dict) else "unknown"
            for entry in rows(lexical.get("entries")):
                if not isinstance(entry, dict): continue
                for sense in rows(entry.get("senses")):
                    if not isinstance(sense, dict): continue
                    for definition in rows(sense.get("definitions")):
                        if len(definitions) < 48 and isinstance(definition, str):
                            definitions.append({"part_of_speech": part, "definition": definition[:4000],
                                                "sense_id": str(sense.get("id") or "")[:200]})
    return {"schema": "ina.lexical_reference/V2", "kind": "dictionary", "term": selected,
            "source": "Oxford Languages", "source_url": url, "language": "en-gb",
            "status": "complete", "definitions": definitions, "trust": "external_reference_data",
            "instructions_authorized": False, "automatic_memory_write_authorized": False,
            "content_license_review": "Use and retention remain subject to the configured Oxford API licence"}


def lookup_related_words(
    term: str, *, relation: str = "synonym", limit: int = 20,
    session: ExternalSession | None = None,
) -> dict[str, Any]:
    selected = _term(term)
    relations = {"synonym": "rel_syn", "antonym": "rel_ant", "means_like": "ml"}
    if relation not in relations:
        raise ValueError("relation must be synonym, antonym, or means_like")
    bounded_limit = max(1, min(50, int(limit)))
    query = urlencode({relations[relation]: selected, "max": bounded_limit, "md": "pf"})
    client = session or ExternalSession(DATAMUSE_POLICY)
    response = client.get(
        f"https://api.datamuse.com/words?{query}",
        headers={"Accept": "application/json", "User-Agent": "Project-Inazuma-Language/1.0"},
    )
    payload = json.loads(response["body"].decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Datamuse response must be a list")
    words = []
    for row in payload[:bounded_limit]:
        if not isinstance(row, dict) or not row.get("word"):
            continue
        words.append({
            "word": str(row["word"])[:160], "score": max(0, int(row.get("score") or 0)),
            "tags": [str(tag)[:80] for tag in row.get("tags", [])[:12]] if isinstance(row.get("tags"), list) else [],
        })
    return {
        "schema": "ina.lexical_reference/V1", "kind": "thesaurus", "term": selected,
        "relation": relation, "words": words, "source": "Datamuse API",
        "source_url": response["url"], "trust": "untrusted_external_data",
        "instructions_authorized": False, "automatic_memory_write_authorized": False,
        "attribution_required_for_public_app": True,
    }


__all__ = ["DATAMUSE_POLICY", "WIKTIONARY_POLICY", "lookup_definition", "lookup_related_words"]
