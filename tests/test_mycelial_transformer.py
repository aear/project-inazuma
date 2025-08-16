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
    assert any(p["from"].startswith("tags:") and p["to"].startswith("fragments:") for p in pathways)
    assert any(p["from"].startswith("visuals:") and p["to"].startswith("audio:") for p in pathways)


