import json
import sqlite3
from pathlib import Path

import fragment_health
import fragment_repair
import memory_mirror_db as mirror
from memory_index import indexed_fragment_count, indexed_fragment_rows


def _fragment_index(path, rows):
    with sqlite3.connect(str(path)) as connection:
        connection.execute(
            "CREATE TABLE fragments(frag_id TEXT PRIMARY KEY, tier TEXT, filename TEXT)"
        )
        connection.executemany("INSERT INTO fragments VALUES (?, ?, ?)", rows)


def test_integrity_scan_is_indexed_bounded_resumable_and_returns_queue_entries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory = tmp_path / "AI_Children" / "Ina" / "memory"
    fragments = memory / "fragments"
    fragments.mkdir(parents=True)
    rows = []
    for index in range(5):
        name = f"frag_{index}.json"
        (fragments / name).write_text(
            json.dumps({"id": f"f{index}"}) if index != 1 else "",
            encoding="utf-8",
        )
        rows.append((f"f{index}", "", name))
    _fragment_index(memory / "memory_map.sqlite", rows)

    first = fragment_health.scan_fragment_integrity("Ina", max_records=2, max_seconds=2)
    second = fragment_health.scan_fragment_integrity("Ina", max_records=2, max_seconds=2)

    assert first["checked_this_pass"] == 2
    assert first["corrupted_this_pass"] == 1
    assert first["corrupt_entries"][0]["id"] == "f1"
    assert second["checked_this_pass"] == 2
    assert second["cursor"] == "f3"


def test_repair_restores_verified_mirror_and_preserves_corrupt_original(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mirror.flush_mirror_writes(close=True)
    mirror._SESSION_CACHE.clear()
    memory = tmp_path / "AI_Children" / "Ina" / "memory"
    fragment = memory / "fragments" / "frag_one.json"
    fragment.parent.mkdir(parents=True)
    good = {"id": "one", "summary": "last good"}
    fragment.write_text(json.dumps(good), encoding="utf-8")
    config = {
        "memory_mirror_policy": {
            "enabled": True,
            "mirror_on_read": True,
            "db_root": str(tmp_path / "mirror"),
            "db_filename": "catalog.sqlite3",
            "batch_records": 1,
            "batch_bytes": 1024,
            "batch_seconds": 0,
            "remove_json_after_verified": False,
            "quarantine_json_after_verified": False,
        }
    }
    mirror.mirror_json_file("Ina", "fragment", fragment, payload=good, config=config)
    mirror.flush_mirror_writes(mirror.mirror_db_path("Ina", config))
    assert mirror.verified_payload_for_path("Ina", "fragment", fragment, config=config) == good
    fragment.write_text('{"id":"one","summary":', encoding="utf-8")
    monkeypatch.setattr(fragment_repair, "load_config", lambda: config)

    remaining, summary = fragment_repair.process_corrupt_queue(
        "Ina",
        [{"path": str(fragment), "reason": "invalid_json"}],
        {
            "mode": "repair",
            "max_actions_per_pass": 1,
            "max_repair_bytes": 1024,
            "quarantine_dir": "fragments/corrupt",
        },
    )

    assert remaining == []
    assert summary["counts"]["repaired"] == 1
    assert json.loads(fragment.read_text(encoding="utf-8")) == good
    backup = summary["actions"][0]["backup"]
    assert Path(backup).read_text(encoding="utf-8") == '{"id":"one","summary":'
    mirror.flush_mirror_writes(close=True)


def test_index_selection_and_threshold_count_are_bounded(tmp_path):
    db = tmp_path / "memory_map.sqlite"
    with sqlite3.connect(str(db)) as connection:
        connection.execute(
            "CREATE TABLE fragments(frag_id TEXT, tier TEXT, filename TEXT, mtime_ns INTEGER, tags_json TEXT)"
        )
        connection.executemany(
            "INSERT INTO fragments VALUES (?, '', ?, ?, '[]')",
            [(str(index), f"frag_{index}.json", index) for index in range(1000)],
        )

    rows = indexed_fragment_rows(db, limit=7)
    assert len(rows) == 7
    assert rows[0]["frag_id"] == "999"
    assert indexed_fragment_count(db, at_least=6) == 6
    assert indexed_fragment_rows(tmp_path / "missing.sqlite", limit=7) == []
