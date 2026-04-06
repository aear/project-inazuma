import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import language_processing as lp
import symbol_generator as sg


class _FakeFractalTransformer:
    def encode(self, payload):
        emotions = payload.get("emotions") or {}
        total = sum(float(v) for v in emotions.values()) or 1.0
        return {"vector": [total, total / 2.0, total / 4.0]}


def test_enrich_symbols_adds_seedling_transformer_insights():
    symbols = [
        {"symbol": "alpha", "components": {"emotion": "trust", "modulation": "sharp", "concept": "unknown"}},
        {"symbol": "atom", "components": {"emotion": "trust", "modulation": "soft", "concept": "known"}},
        {"symbol": "beta", "components": {"emotion": "fear", "modulation": "sharp", "concept": "known"}},
    ]

    enriched = sg.enrich_symbols(symbols, _FakeFractalTransformer())

    alpha = enriched[0]["transformer_insights"]
    beta = enriched[2]["transformer_insights"]
    assert alpha["seedling_cluster"] == "a"
    assert alpha["seedling_cluster_size"] == 2
    assert alpha["seedling_seed"]
    assert beta["seedling_cluster"] == "b"
    assert beta["seedling_cluster_size"] == 1


def test_load_generated_symbols_preserves_transformer_insights():
    temp_dir = Path(tempfile.mkdtemp(prefix="symbol_loader_"))
    try:
        child = "tester"
        identity_dir = temp_dir / child / "identity"
        identity_dir.mkdir(parents=True, exist_ok=True)
        (identity_dir / "self_reflection.json").write_text(
            '{"self_generated_symbols": [{"id": "sym_1", "symbol": "ae", "meaning": "seed", "transformer_insights": {"seedling_seed": "ae"}}]}',
            encoding="utf-8",
        )

        loaded = lp.load_generated_symbols(child, base_path=temp_dir)

        assert loaded[0]["transformer_insights"]["seedling_seed"] == "ae"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_generate_symbolic_reply_includes_transformer_insights():
    original_load = lp.load_symbol_to_token
    original_save = lp.save_symbol_to_token
    original_speak = lp.speak_symbolically
    original_seed = lp.seed_self_question
    try:
        vocab = {
            "sym_hello": {"word": "hello", "language": "en", "embedding": [1.0, 0.0]},
            "sym_echo": {"word": "echo", "language": "en", "embedding": [0.0, 1.0]},
        }
        lp.load_symbol_to_token = lambda child, base_path=None: dict(vocab)
        lp.save_symbol_to_token = lambda child, data, base_path=None: None
        lp.speak_symbolically = lambda symbols, child="Inazuma_Yagami", **kwargs: None
        lp.seed_self_question = lambda question: None

        result = lp.generate_symbolic_reply_from_text("hello echo", child="tester")

        assert result is not None
        assert result["symbols"] == ["sym_hello", "sym_echo"]
        assert result["transformer_insights"] is not None
        assert result["transformer_insights"]["seedling"]["cluster_count"] >= 1
        assert result["transformer_insights"]["mycelial"]["pathway_count"] >= 1
    finally:
        lp.load_symbol_to_token = original_load
        lp.save_symbol_to_token = original_save
        lp.speak_symbolically = original_speak
        lp.seed_self_question = original_seed
