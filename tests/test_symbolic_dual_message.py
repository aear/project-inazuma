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
