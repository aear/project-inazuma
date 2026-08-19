import json

import text_memory as tm


def _disable_runtime_metrics(monkeypatch):
    monkeypatch.setattr(tm, "increment_inastate_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(tm, "set_inastate_metric", lambda *args, **kwargs: None)


class _MutableEmbedder:
    def __init__(self):
        self.prefer_b = False

    def embed_text(self, text, language=None):
        return ["word", str(text)]

    def cosine(self, word_embedding, symbol_embedding):
        marker = symbol_embedding[0]
        if marker == "a":
            return 0.8 if not self.prefer_b else 0.6
        if marker == "b":
            return 0.7 if not self.prefer_b else 0.95
        return 0.0


def test_contextual_human_guess_maps_only_the_changed_token():
    diagnostic = tm.diagnose_text_alignment(
        "°·'φam λ⊙··",
        "ina λ⊙··",
    )

    assert diagnostic["native_tokens"] == ["°·'φam", "λ⊙··"]
    assert diagnostic["gloss_tokens"] == ["ina", "λ⊙··"]
    assert diagnostic["accepted"] is True
    assert diagnostic["reason"] == "accepted_contextual"
    assert diagnostic["candidate_pairs"] == [
        {"native": "°·'φam", "english": "ina"}
    ]
    assert diagnostic["unchanged_context_count"] == 1


def test_history_evidence_accepts_contextual_replacement(tmp_path, monkeypatch):
    _disable_runtime_metrics(monkeypatch)
    monkeypatch.chdir(tmp_path)

    result = tm.review_text_evidence(
        [],
        [("°·'φam λ⊙··", "ina λ⊙··")],
        child="TestChild",
        source="test_diagnostics",
    )

    assert result["alignment_candidates"] == 1
    assert result["accepted_alignment_candidates"] == 1
    assert result["pairs"] == [{"native": "°·'φam", "english": "ina"}]
    assert result["alignment_rejections"] == []


def test_history_evidence_records_aligned_native_words_in_one_batch(tmp_path, monkeypatch):
    _disable_runtime_metrics(monkeypatch)
    monkeypatch.chdir(tmp_path)
    memory = tmp_path / "AI_Children" / "TestChild" / "memory"
    memory.mkdir(parents=True)

    result = tm.review_text_evidence(
        [{"text": "A fresh thought", "tags": ["discord", "history"]}],
        [("glyph_wave glyph_calm", "hello calm")],
        child="TestChild",
        source="test_review",
    )

    vocab = tm.load_text_vocab("TestChild")["vocab"]
    assert result["observed_messages"] == 2
    assert result["pairs"] == [
        {"native": "glyph_wave", "english": "hello"},
        {"native": "glyph_calm", "english": "calm"},
    ]
    assert vocab["hello"]["symbols"] == {"glyph_wave": 1}
    assert vocab["calm"]["symbols"] == {"glyph_calm": 1}


def test_existing_mapping_can_be_revisited_without_new_vocabulary(tmp_path, monkeypatch):
    _disable_runtime_metrics(monkeypatch)
    monkeypatch.chdir(tmp_path)
    memory = tmp_path / "AI_Children" / "TestChild" / "memory"
    memory.mkdir(parents=True)
    (memory / "symbol_to_token.json").write_text(
        json.dumps(
            {
                "sym_a": {
                    "word": "glyph_a",
                    "embedding": ["a"],
                    "confidence": 0.6,
                },
                "sym_b": {
                    "word": "glyph_b",
                    "embedding": ["b"],
                    "confidence": 0.6,
                },
            }
        ),
        encoding="utf-8",
    )
    tm.update_text_vocab("hello", child="TestChild", source="test")
    embedder = _MutableEmbedder()
    monkeypatch.setattr(tm, "_EMBEDDER", embedder)

    assert tm.build_text_symbol_links("TestChild", mapping_batch=4)
    first = json.loads((memory / "text_vocab_links.json").read_text(encoding="utf-8"))
    assert first["links"][0]["symbol"] == "sym_a"
    assert isinstance(first["evaluated"]["hello"], dict)

    embedder.prefer_b = True
    assert tm.build_text_symbol_links(
        "TestChild", mapping_batch=4, revisit_existing=1
    )
    revised = json.loads((memory / "text_vocab_links.json").read_text(encoding="utf-8"))
    assert revised["links"][0]["symbol"] == "sym_b"
    assert revised["last_batch"] == {
        "mode": "revisit",
        "new_mappings": 0,
        "revisited_mappings": 1,
    }


def test_mapping_queue_reports_source_and_new_batch(tmp_path, monkeypatch):
    _disable_runtime_metrics(monkeypatch)
    monkeypatch.chdir(tmp_path)
    memory = tmp_path / "AI_Children" / "TestChild" / "memory"
    memory.mkdir(parents=True)
    (memory / "symbol_to_token.json").write_text(
        json.dumps(
            {
                "sym_a": {
                    "word": "glyph_a",
                    "embedding": ["a"],
                    "confidence": 0.6,
                }
            }
        ),
        encoding="utf-8",
    )
    tm.update_text_vocab(
        "hello world",
        child="TestChild",
        tags=["discord", "history"],
        source="discord_history_review",
    )
    monkeypatch.setattr(tm, "_EMBEDDER", _MutableEmbedder())

    assert tm.build_text_symbol_links("TestChild", mapping_batch=1)
    payload = json.loads((memory / "text_vocab_links.json").read_text(encoding="utf-8"))

    assert payload["remaining"] == 1
    assert payload["queue_by_source"] == {"discord": 1}
    assert payload["last_batch"] == {
        "mode": "new",
        "new_mappings": 1,
        "revisited_mappings": 0,
    }


def test_mapper_retains_ranked_meanings_with_independent_metadata(tmp_path, monkeypatch):
    _disable_runtime_metrics(monkeypatch)
    monkeypatch.chdir(tmp_path)
    memory = tmp_path / "AI_Children" / "TestChild" / "memory"
    memory.mkdir(parents=True)
    (memory / "symbol_to_token.json").write_text(json.dumps({
        "sym_finance": {"word": "money", "embedding": ["a"], "confidence": 0.8},
        "sym_river": {"word": "river", "embedding": ["b"], "confidence": 0.7},
    }), encoding="utf-8")
    tm.update_text_vocab(
        "bank", child="TestChild", tags=["river", "money"],
        symbols=["sym_finance", "sym_river"], source="conversation",
    )
    monkeypatch.setattr(tm, "_EMBEDDER", _MutableEmbedder())

    assert tm.build_text_symbol_links("TestChild", mapping_batch=1)
    payload = json.loads((memory / "text_vocab_links.json").read_text(encoding="utf-8"))
    meanings = [link for link in payload["links"] if link["word"] == "bank"]

    assert payload["schema_version"] == 2
    assert {link["symbol"] for link in meanings} == {"sym_finance", "sym_river"}
    assert all(link["usage_count"] == 1 for link in meanings)
    assert all(link["reinforcement_count"] == 1 for link in meanings)
    assert all(link["last_reinforced"] for link in meanings)
    assert all(link["sources"] == {"conversation": 1} for link in meanings)
    assert all(set(link["contexts"]) == {"river", "money"} for link in meanings)
