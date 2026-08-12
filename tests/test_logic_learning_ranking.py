import logic_engine


class Transformer:
    def encode_many(self, fragments):
        return [{"vector": [0.0, 1.0]} for _fragment in fragments]


def test_logic_ranking_keeps_alternatives_and_margin_evidence():
    ranked = logic_engine.rank_prediction_against_logic(
        {"predicted_vector": {"vector": [1.0, 0.0]}},
        [
            {"symbol_word_id": "best", "summary": "a", "vector": [1.0, 0.0]},
            {"symbol_word_id": "near", "summary": "b", "vector": [0.9, 0.1]},
            {"symbol_word_id": "far", "summary": "c", "vector": [0.0, 1.0]},
        ],
        Transformer(),
        limit=2,
    )
    assert [item["symbol_word_id"] for item in ranked] == ["best", "near"]
    assert ranked[0]["similarity"] > ranked[1]["similarity"]
