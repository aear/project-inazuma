import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers.seedling_transformer import SeedlingTransformer


def test_germinate_clusters_by_grounded_geometry():
    symbols = ["alpha", "atom", "beta"]
    profiles = {
        "alpha": {"vector": [1.0, 0.0], "modality": "vision"},
        "atom": {"vector": [0.0, 1.0], "modality": "audio"},
        "beta": {"vector": [0.95, 0.05], "modality": "vision"},
    }
    result = SeedlingTransformer(seed=0, similarity_threshold=0.8).germinate(
        symbols, symbol_profiles=profiles,
    )

    mapping = result["symbol_clusters"]
    assert mapping["alpha"] == mapping["beta"]
    assert mapping["alpha"] != mapping["atom"]
    assert all(key.startswith("geometry_") for key in result["clusters"])
    assert set(result["seeds"]) == set(result["clusters"])
    assert result["origins"][0]["module_version"] == "V2"
