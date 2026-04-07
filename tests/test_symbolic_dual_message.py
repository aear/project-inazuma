import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import language_processing as lp



def test_build_dual_symbolic_message_combines_native_and_guess(monkeypatch):
    monkeypatch.setattr(
        lp,
        "load_symbol_to_token",
        lambda child, base_path=None: {
            "sym_hello": {"word": "hello"},
            "sym_calm": {"word": "calm"},
        },
    )

    payload = lp.build_dual_symbolic_message(["sym_hello", "sym_calm"], child="TestChild")

    assert payload is not None
    assert payload["native_text"] == "sym_hello sym_calm"
    assert payload["gloss_text"] == "hello calm"
    assert payload["text"] == "Ina native: sym_hello sym_calm\nHuman guess: hello calm"



def test_build_dual_symbolic_message_prefers_supplied_human_text(monkeypatch):
    monkeypatch.setattr(
        lp,
        "load_symbol_to_token",
        lambda child, base_path=None: {"sym_wave": {"word": "wave"}},
    )

    payload = lp.build_dual_symbolic_message(
        ["sym_wave"],
        child="TestChild",
        human_text="hello there",
    )

    assert payload is not None
    assert payload["native_text"] == "sym_wave"
    assert payload["gloss_text"] == "hello there"
    assert payload["text"] == "Ina native: sym_wave\nHuman guess: hello there"


def test_build_dual_symbolic_message_uses_contextual_text_vocab_links(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "load_symbol_to_token", lambda child, base_path=None: {})
    memory_root = tmp_path / "TestChild" / "memory"
    memory_root.mkdir(parents=True)
    (memory_root / "text_vocab_links.json").write_text(
        """{
  "links": [
    {"word": "zero", "symbol": "sym_ambiguous", "count": 999, "similarity": 1.0},
    {"word": "heart", "symbol": "sym_ambiguous", "count": 1, "similarity": 1.0},
    {"word": "ina", "symbol": "sym_ina", "count": 10, "similarity": 1.0}
  ]
}
""",
        encoding="utf-8",
    )

    payload = lp.build_dual_symbolic_message(
        ["sym_ambiguous", "sym_ina"],
        child="TestChild",
        base_path=tmp_path,
        context={"tokens": ["heart", "ina"], "tags": ["discord"]},
        fallback_to_symbol_to_token=False,
    )

    assert payload is not None
    assert payload["native_text"] == "sym_ambiguous sym_ina"
    assert payload["gloss_text"] == "heart ina"
    assert payload["gloss_sources"] == {
        "sym_ambiguous": "text_vocab_links",
        "sym_ina": "text_vocab_links",
    }


def test_build_dual_symbolic_message_can_skip_symbol_to_token_fallback(tmp_path, monkeypatch):
    def fail_load_symbol_to_token(child, base_path=None):
        raise AssertionError("symbol_to_token fallback should stay lazy")

    monkeypatch.setattr(lp, "load_symbol_to_token", fail_load_symbol_to_token)
    memory_root = tmp_path / "TestChild" / "memory"
    memory_root.mkdir(parents=True)
    (memory_root / "text_vocab_links.json").write_text(
        '{"links": [{"word": "ina", "symbol": "sym_ina", "count": 1}]}',
        encoding="utf-8",
    )

    payload = lp.build_dual_symbolic_message(
        ["sym_unknown", "sym_ina"],
        child="TestChild",
        base_path=tmp_path,
        context={"tokens": ["ina"]},
        fallback_to_symbol_to_token=False,
    )

    assert payload is not None
    assert payload["gloss_text"] == "sym_unknown ina"
    assert payload["unresolved_symbols"] == ["sym_unknown"]


def test_build_dual_symbolic_message_can_use_symbol_word_as_native(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "load_symbol_to_token", lambda child, base_path=None: {})
    memory_root = tmp_path / "TestChild" / "memory"
    memory_root.mkdir(parents=True)
    (memory_root / "text_vocab_links.json").write_text(
        '{"links": [{"word": "ina", "symbol": "sym_ina", "symbol_word": "glyph_ina", "count": 1}]}',
        encoding="utf-8",
    )

    payload = lp.build_dual_symbolic_message(
        ["sym_ina"],
        child="TestChild",
        base_path=tmp_path,
        context={"tokens": ["ina"]},
        fallback_to_symbol_to_token=False,
        native_style="glyphs",
    )

    assert payload is not None
    assert payload["native_text"] == "glyph_ina"
    assert payload["gloss_text"] == "ina"
    assert payload["native_sources"] == {"sym_ina": "text_vocab_links"}


def test_generate_symbolic_reply_uses_text_vocab_links_without_symbol_to_token(tmp_path, monkeypatch):
    def fail_load_symbol_to_token(child, base_path=None):
        raise AssertionError("symbol_to_token fallback should stay lazy")

    monkeypatch.setattr(lp, "load_symbol_to_token", fail_load_symbol_to_token)
    monkeypatch.setattr(lp, "speak_symbolically", lambda *args, **kwargs: None)
    monkeypatch.setattr(lp, "_build_reply_transformer_insights", lambda *args, **kwargs: None)
    memory_root = tmp_path / "TestChild" / "memory"
    memory_root.mkdir(parents=True)
    (memory_root / "text_vocab_links.json").write_text(
        '{"links": [{"word": "ina", "symbol": "sym_ina", "symbol_word": "glyph_ina", "count": 1}, {"word": "3", "symbol": "sym_heart", "symbol_word": "glyph_heart", "count": 1}]}',
        encoding="utf-8",
    )

    payload = lp.generate_symbolic_reply_from_text(
        "Ina <3",
        child="TestChild",
        base_path=tmp_path,
    )

    assert payload is not None
    assert payload["symbols"] == ["sym_ina", "sym_heart"]
    assert payload["native_text"] == "glyph_ina glyph_heart"
    assert payload["gloss_text"] == "ina 3"
