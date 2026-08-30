import json

from language_processing import load_text_vocab_links
from text_vocab_store import load_text_vocab_store, write_text_vocab_store


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


def test_text_vocab_storage_benchmark_v1_json_vs_v2_sqlite_preference(tmp_path):
    memory = tmp_path / "AI_Children" / "Ina" / "memory"
    memory.mkdir(parents=True)
    json_path = memory / "text_vocab_links.json"
    json_path.write_text(json.dumps({"links": [{"word": "old", "symbol": "old"}]}))
    write_text_vocab_store(memory / "text_vocab_links.sqlite", _payload())

    loaded = load_text_vocab_links("Ina", base_path=tmp_path / "AI_Children")

    assert loaded["links"][0]["word"] == "music"
    assert loaded["evaluated_count"] == 1
