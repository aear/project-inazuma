import json

import text_vocab_store as tvs
from language_processing import load_text_vocab_links
from text_vocab_store import load_text_vocab_store, load_text_vocab_store_subset, write_text_vocab_store


def _payload():
    return {
        "schema_version": 2,
        "generated": "2026-08-30T00:00:00+00:00",
        "evaluated": {"music": {"evidence_revision": "abc"}},
        "evaluated_count": 1,
        "remaining": 0,
        "links": [
            {"word": "music", "symbol": "sym-song", "strength": 0.8},
            {"word": "music", "symbol": "sym-art", "strength": 0.7},
        ],
    }


def test_sqlite_text_vocab_round_trip(tmp_path):
    path = tmp_path / "text_vocab_links.sqlite"
    write_text_vocab_store(path, _payload())
    assert load_text_vocab_store(path) == _payload()


def test_sqlite_text_vocab_loads_only_requested_rows(tmp_path):
    path = tmp_path / "text_vocab_links.sqlite"
    payload = _payload()
    payload["links"].append({"word": "other", "symbol": "sym-other", "strength": 0.4})
    write_text_vocab_store(path, payload)

    by_word = load_text_vocab_store_subset(path, words=["music"])
    by_symbol = load_text_vocab_store_subset(path, symbols=["sym-other"])

    assert {row["word"] for row in by_word["links"]} == {"music"}
    assert {row["symbol"] for row in by_symbol["links"]} == {"sym-other"}
    assert by_word["subset"]["words"] == ["music"]


def test_sqlite_projection_uses_canonical_fast_index_seam(monkeypatch):
    calls = []
    expected = tvs.Path("/fast/index/text_vocab_links.sqlite")
    monkeypatch.setattr(
        tvs, "fast_runtime_path",
        lambda child, filename, fallback, **kwargs: calls.append(
            (child, filename, fallback, kwargs)
        ) or expected,
    )

    selected = tvs.sqlite_path_for(
        tvs.Path("AI_Children/Ina/memory/text_vocab_links.json")
    )

    assert selected == expected
    assert calls[0][0:2] == ("Ina", "text_vocab_links.sqlite")
    assert calls[0][3]["subdir"] == "index"


def test_text_vocab_storage_benchmark_v1_json_vs_v2_sqlite_preference(tmp_path):
    memory = tmp_path / "AI_Children" / "Ina" / "memory"
    memory.mkdir(parents=True)
    json_path = memory / "text_vocab_links.json"
    json_path.write_text(json.dumps({"links": [{"word": "old", "symbol": "old"}]}))
    write_text_vocab_store(memory / "text_vocab_links.sqlite", _payload())

    loaded = load_text_vocab_links("Ina", base_path=tmp_path / "AI_Children")

    assert loaded["links"][0]["word"] == "music"
    assert loaded["evaluated_count"] == 1
