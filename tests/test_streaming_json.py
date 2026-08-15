import json

from streaming_json import (
    count_top_level_array, iter_selected_array_objects, iter_selected_object_entries,
)


def test_selected_array_reader_skips_large_unselected_values(tmp_path):
    path = tmp_path / "words.json"
    payload = {
        "words": [
            {
                "symbol_word_id": "one",
                "components": [f"fragment-{index}" for index in range(2000)],
                "summary": "first",
                "vector": [1.0, 0.0],
            },
            {
                "symbol_word_id": "two",
                "components": ["x"],
                "summary": "second",
                "vector": [0.0, 1.0],
            },
        ],
        "other": {"large": list(range(1000))},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    rows = list(iter_selected_array_objects(
        path, "words", {"symbol_word_id", "summary", "vector"}
    ))

    assert rows == [
        {"symbol_word_id": "one", "summary": "first", "vector": [1.0, 0.0]},
        {"symbol_word_id": "two", "summary": "second", "vector": [0.0, 1.0]},
    ]
    assert all("components" not in row for row in rows)


def test_selected_array_reader_honours_limit(tmp_path):
    path = tmp_path / "words.json"
    path.write_text(json.dumps({"words": [{"id": i} for i in range(10)]}), encoding="utf-8")
    assert list(iter_selected_array_objects(path, "words", {"id"}, limit=3)) == [
        {"id": 0}, {"id": 1}, {"id": 2}
    ]


def test_selected_object_reader_and_bounded_array_count(tmp_path):
    path = tmp_path / "store.json"
    path.write_text(json.dumps({
        "words": [{"id": 1, "large": list(range(1000))}, {"id": 2}],
        "proto_words": {"pair:a_b": {"sequence": ["a", "b"], "large": list(range(1000))}},
    }), encoding="utf-8")
    assert count_top_level_array(path, "words") == 2
    assert list(iter_selected_object_entries(path, "proto_words", {"sequence"})) == [
        ("pair:a_b", {"sequence": ["a", "b"]})
    ]
