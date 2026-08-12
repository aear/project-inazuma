import json

import logic_engine


class BatchTransformer:
    def __init__(self):
        self.calls = []

    def encode_many(self, fragments):
        self.calls.append(list(fragments))
        return [
            {"vector": [1.0, 0.0] if "trust" in fragment["summary"] else [0.0, 1.0]}
            for fragment in fragments
        ]


def test_logic_match_reuses_vectors_and_batches_missing_embeddings():
    transformer = BatchTransformer()
    prediction = {"predicted_vector": {"vector": [1.0, 0.0]}}
    words = [
        {"symbol_word_id": "stored", "summary": "other", "vector": [0.0, 1.0]},
        {"symbol_word_id": "encoded", "summary": "trust meaning"},
    ]

    word_id, similarity = logic_engine.test_prediction_against_logic(
        prediction, words, transformer
    )

    assert word_id == "encoded"
    assert similarity == 1.0
    assert len(transformer.calls) == 1
    assert len(transformer.calls[0]) == 1


def test_symbol_word_load_creates_and_reuses_compact_index(monkeypatch, tmp_path):
    memory = tmp_path / "AI_Children" / "Ina" / "memory"
    memory.mkdir(parents=True)
    source = memory / "symbol_words.json"
    source.write_text(json.dumps({
        "words": [{
            "symbol_word_id": "one",
            "components": [f"fragment-{index}" for index in range(5000)],
            "summary": "compact me",
            "tags": ["symbolic"],
            "vector": [1.0, 0.0],
        }]
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(logic_engine, "load_config", lambda: {
        "logic_engine_policy": {"symbol_candidate_limit": 64}
    })

    first = logic_engine.load_symbol_words("Ina")
    index = memory / "symbol_words.logic_index.json"
    assert first == [{
        "symbol_word_id": "one", "summary": "compact me",
        "tags": ["symbolic"], "vector": [1.0, 0.0]
    }]
    assert index.exists()
    assert index.stat().st_size < source.stat().st_size / 10

    monkeypatch.setattr(
        logic_engine,
        "iter_selected_array_objects",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should use index")),
    )
    assert logic_engine.load_symbol_words("Ina") == first
