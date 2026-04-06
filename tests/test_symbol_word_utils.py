from symbol_word_utils import score_symbol_word_candidates


class DummyTransformer:
    def encode(self, payload):
        summary = payload.get("summary", "")
        if "trust" in summary:
            return {"vector": [1.0, 0.0]}
        if "curious" in summary:
            return {"vector": [0.0, 1.0]}
        return {"vector": [0.2, 0.2]}



def test_score_symbol_word_candidates_prefers_multi_symbol_word_vector():
    transformer = DummyTransformer()
    word_state = {
        "words": [
            {
                "symbol_word_id": "sym_word_0001",
                "summary": "curious pulse",
                "vector": [0.0, 1.0],
                "symbol": "snd_c",
            }
        ],
        "proto_words": {},
        "multi_symbol_words": {
            "pair:snd_a_snd_b": {
                "sequence": ["snd_a", "snd_b"],
                "summary": "trust pair",
                "vector": [0.98, 0.02],
                "confidence": 0.72,
            }
        },
    }

    match = score_symbol_word_candidates([1.0, 0.0], transformer, word_state)

    assert match is not None
    assert match["kind"] == "multi_symbol_word"
    assert match["symbol_word_id"] == "pair:snd_a_snd_b"
    assert match["sequence"] == ["snd_a", "snd_b"]



def test_score_symbol_word_candidates_uses_summary_when_vector_is_missing():
    transformer = DummyTransformer()
    word_state = {
        "words": [],
        "proto_words": {
            "snd_a_snd_b": {
                "sequence": ["snd_a", "snd_b"],
                "summary": "trust pair",
                "confidence": 0.6,
            }
        },
        "multi_symbol_words": {},
    }

    match = score_symbol_word_candidates([1.0, 0.0], transformer, word_state)

    assert match is not None
    assert match["kind"] == "proto_word"
    assert match["symbol_word_id"] == "pair:snd_a_snd_b"
    assert match["summary"] == "trust pair"
