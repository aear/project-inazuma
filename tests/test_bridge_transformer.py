import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers.bridge_transformer import BridgeTransformer


def test_bridge_creates_pause_file(tmp_path):
    flag = tmp_path / "pause.flag"
    transformer = BridgeTransformer(pause_flag=flag)
    result = transformer.bridge("violence", "love", {"care": 0.9, "fear": 0.1})

    assert result["fused_truth"] == "violence as love"
    assert result["question"] == "How can violence be love?"
    assert result["emotion"] == "care"
    assert flag.exists()
