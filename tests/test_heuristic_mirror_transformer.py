import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from heuristic_mirror_transformer import HeuristicMirrorTransformer


def test_mirror_basic(tmp_path):
    transformer = HeuristicMirrorTransformer(child="tester", root_path=tmp_path)
    symbolic_state = {"tags": ["alpha", "beta"]}
    emotions = {"trust": 0.5, "care": 0.2}
    result = transformer.mirror(symbolic_state, emotions, perceived_audience="user")

    assert result["mirrored_symbols"] == ["alpha", "beta"]
    assert result["predicted_emotions"]["trust"] == pytest.approx(0.4)
    assert result["misalignment"]["trust"] == pytest.approx(0.1)
