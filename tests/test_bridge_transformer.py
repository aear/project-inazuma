import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import transformers.bridge_transformer as bridge_module
from transformers.bridge_transformer import BridgeTransformer


def test_bridge_creates_pause_file(tmp_path):
    flag = tmp_path / "pause.flag"
    transformer = BridgeTransformer(pause_flag=flag)
    prior = bridge_module.seed_self_question
    bridge_module.seed_self_question = lambda *args, **kwargs: None
    try:
        result = transformer.bridge("violence", "love", {"care": 0.9, "fear": 0.1})
    finally:
        bridge_module.seed_self_question = prior

    assert result["fused_truth"] == "violence as love"
    assert result["question"] == "How can violence be love?"
    assert result["emotion"] == "care"
    assert flag.exists()


def test_bridge_does_not_turn_arbitrary_tag_pair_into_contradiction(tmp_path):
    flag = tmp_path / "pause.flag"
    seeded = []
    prior = bridge_module.seed_self_question
    bridge_module.seed_self_question = seeded.append
    try:
        result = BridgeTransformer(pause_flag=flag).bridge(
            "text", "self_read", source_context={"relation_type": "consumes", "fragment_id": "frag-1"},
        )
    finally:
        bridge_module.seed_self_question = prior
    assert result["question"] is None
    assert result["contradiction_evidence"] is False
    assert result["relation_candidate"]["relation_type"] == "consumes"
    assert seeded == []
    assert not flag.exists()
