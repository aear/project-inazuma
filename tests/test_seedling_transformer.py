import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers.seedling_transformer import SeedlingTransformer


def test_germinate_clusters_and_seeds():
    symbols = ["alpha", "atom", "beta", "bloom"]
    transformer = SeedlingTransformer(seed=0)
    result = transformer.germinate(symbols)

    clusters = result["clusters"]
    seeds = result["seeds"]

    assert set(clusters.keys()) == {"a", "b"}
    assert set(seeds.keys()) == {"a", "b"}
    assert seeds["a"].startswith("a")
    assert seeds["b"].startswith("b")
