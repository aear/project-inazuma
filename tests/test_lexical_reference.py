import json

import pytest

from lexical_reference import lookup_definition, lookup_related_words


class Session:
    def __init__(self, payload):
        self.payload = payload
        self.urls = []

    def get(self, url, headers=None):
        self.urls.append(url)
        return {"body": json.dumps(self.payload).encode(), "url": url}


def test_dictionary_result_is_bounded_untrusted_and_non_retaining():
    session = Session({"en": [{"partOfSpeech": "noun", "definitions": [{"definition": "<b>a test</b>"}]}]})
    result = lookup_definition("test", session=session)
    assert result["definitions"] == [{"part_of_speech": "noun", "definition_html": "<b>a test</b>"}]
    assert result["instructions_authorized"] is False
    assert result["automatic_memory_write_authorized"] is False
    assert session.urls == ["https://en.wiktionary.org/api/rest_v1/page/definition/test"]


def test_thesaurus_relations_are_allowlisted_and_bounded():
    result = lookup_related_words("ocean", relation="synonym", limit=2, session=Session([
        {"word": "sea", "score": 10, "tags": ["n"]}, {"word": "main", "score": 9}, {"word": "overflow"},
    ]))
    assert [row["word"] for row in result["words"]] == ["sea", "main"]
    assert result["relation"] == "synonym"
    with pytest.raises(ValueError, match="relation"):
        lookup_related_words("ocean", relation="execute", session=Session([]))


def test_reference_term_rejects_urls_and_instruction_payloads():
    for term in ("https://example.test", "word; rm -rf", "x" * 81):
        with pytest.raises(ValueError):
            lookup_definition(term, session=Session({}))


def test_definition_budget_applies_across_all_parts_of_speech():
    session = Session({'en': [{'definitions': [{'definition': 'definition'}] * 12}] * 12})
    assert len(lookup_definition('word', session=session)['definitions']) == 48
