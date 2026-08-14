import json
import sqlite3

from memory_index import ensure_memory_index_db, index_is_current, iter_json_object_items, touch_fragments


def test_streaming_object_reader_handles_tiny_chunks(tmp_path):
    path = tmp_path / "map.json"
    payload = {f"frag_{index}": {"tags": ["symbolic"], "importance": index / 10} for index in range(20)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert dict(iter_json_object_items(path, chunk_chars=7)) == payload


def test_bounded_index_build_and_direct_recall_touch(tmp_path):
    json_path = tmp_path / "memory_map.json"
    db_path = tmp_path / "memory_map.sqlite"
    payload = {
        f"frag_{index}": {
            "tier": "long",
            "filename": f"frag_{index}.json",
            "last_seen": "2025-01-01T00:00:00+00:00",
            "importance": 0.5,
            "tags": ["symbolic", "unresolved"] if index % 10 == 0 else ["symbolic"],
        }
        for index in range(125)
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    assert ensure_memory_index_db(json_path, db_path, batch_size=11) is True
    assert index_is_current(json_path, db_path) is True
    assert touch_fragments(db_path, ["frag_2", "frag_7"], "2026-08-12T00:00:00+00:00") == 2

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
        touched = conn.execute(
            "SELECT last_seen FROM fragments WHERE frag_id = 'frag_7'"
        ).fetchone()[0]
        unresolved = [row[0] for row in conn.execute(
            "SELECT frag_id FROM fragment_tags WHERE tag = 'unresolved' ORDER BY frag_id"
        )]
        tag_index = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_fragment_tags_tag'"
        ).fetchone()
    assert count == 125
    assert touched == "2026-08-12T00:00:00+00:00"
    assert set(unresolved) == {f"frag_{index}" for index in range(0, 125, 10)}
    assert tag_index == (1,)
