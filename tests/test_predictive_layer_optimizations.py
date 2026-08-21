import json
import sqlite3

from benchmarks.benchmark_predictive_memory import RESULTS_END, RESULTS_START, update_markdown_report

from predictive_layer import load_recent_fragments
from symbol_word_utils import (
    iter_symbol_word_candidates_from_path,
    load_compact_symbol_words,
    score_symbol_word_candidate_iter,
    score_symbol_word_candidates,
)


class _Transformer:
    def encode(self, entry):
        return {"vector": [1.0, 0.0] if "one" in entry.get("summary", "") else [0.0, 1.0]}


def test_streamed_symbol_scoring_matches_v1_loaded_scoring(tmp_path):
    payload = {
        "words": [
            {"symbol_word_id": "one", "summary": "one", "components": list(range(5000))},
            {"symbol_word_id": "two", "vector": [0.0, 1.0]},
        ],
        "proto_words": {"a_b": {"sequence": ["a", "b"], "confidence": 0.2}},
        "multi_symbol_words": {},
    }
    path = tmp_path / "symbol_words.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    transformer = _Transformer()
    expected = score_symbol_word_candidates([1.0, 0.0], transformer, payload)
    actual = score_symbol_word_candidate_iter(
        [1.0, 0.0], transformer, iter_symbol_word_candidates_from_path(path),
    )
    assert actual["symbol_word_id"] == expected["symbol_word_id"] == "one"
    assert actual["confidence"] == expected["confidence"]


def test_recent_fragment_selection_is_bounded_and_newest_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "AI_Children" / "Ina" / "memory" / "fragments"
    root.mkdir(parents=True)
    for index in range(20):
        path = root / f"frag_{index}.json"
        path.write_text(json.dumps({"id": index}), encoding="utf-8")
        path.touch()
        import os
        os.utime(path, (index, index))
    with sqlite3.connect(str(root.parent / "memory_map.sqlite")) as connection:
        connection.execute(
            "CREATE TABLE fragments(frag_id TEXT, tier TEXT, filename TEXT, mtime_ns INTEGER, tags_json TEXT)"
        )
        connection.executemany(
            "INSERT INTO fragments VALUES (?, ?, ?, ?, ?)",
            [(str(index), "", f"frag_{index}.json", index, "[]") for index in range(20)],
        )
    assert [item["id"] for item in load_recent_fragments("Ina", limit=3)] == [19, 18, 17]


def test_v2_compact_index_is_reused_without_reading_large_source(tmp_path, monkeypatch):
    source = tmp_path / "symbol_words.json"
    source.write_text(json.dumps({"words": [{
        "symbol_word_id": "one", "vector": [1.0, 0.0],
        "components": list(range(5000)),
    }]}), encoding="utf-8")
    assert load_compact_symbol_words(source) == [{
        "symbol_word_id": "one", "vector": [1.0, 0.0]
    }]
    monkeypatch.setattr(
        "symbol_word_utils.iter_selected_array_objects",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("source rescanned")),
    )
    assert load_compact_symbol_words(source)[0]["symbol_word_id"] == "one"


def test_benchmark_updates_only_marked_markdown_without_private_path(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        f"# Keep this\n\n{RESULTS_START}\nold\n{RESULTS_END}\n\nKeep this too.\n",
        encoding="utf-8",
    )
    result = {
        "version": "V2", "candidates": 7, "elapsed_seconds": 0.125,
        "peak_rss_kib": 20480, "source_bytes": 1_500_000_000,
        "index_bytes": 1400, "path": "/private/Ina/memory/symbol_words.json",
    }
    update_markdown_report(report, result)
    rendered = report.read_text(encoding="utf-8")
    assert "# Keep this" in rendered and "Keep this too." in rendered
    assert "20.0 MB" in rendered and "7" in rendered
    assert "/private/Ina" not in rendered
