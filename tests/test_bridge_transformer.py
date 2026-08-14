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
