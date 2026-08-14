import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers.mycelial_transformer import MycelialTransformer


def test_weave_cross_domain():
    data = {
        "tags": ["forest"],
        "fragments": ["memory1"],
        "visuals": ["spiral"],
        "audio": ["note"],
        "text": ["poem"],
    }
    transformer = MycelialTransformer(max_links=2)
    result = transformer.weave(data)

    pathways = result["pathways"]
    assert pathways
    assert all(pathway["from"].split(":", 1)[0] != pathway["to"].split(":", 1)[0] for pathway in pathways)
    assert all(0.0 <= pathway["score"] <= 1.0 for pathway in pathways)
    assert all(set(pathway["factors"]) == {
        "novelty", "semantic_distance", "emotional_relevance", "historical_usefulness",
    } for pathway in pathways)
    sources = {pathway["from"] for pathway in pathways}
    assert all(sum(1 for pathway in pathways if pathway["from"] == source) <= 2 for source in sources)


