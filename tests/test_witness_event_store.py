import json
import sqlite3

from witness_event_store import backfill_witness_batch, record_witness_event


def test_witness_backfill_is_bounded_resumable_and_indexed(tmp_path):
    source = tmp_path / "emotion_log.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for index in range(5):
            handle.write(json.dumps({"timestamp": index, "mode": "calm", "values": {"attention": index / 5}}) + "\n")
    database = tmp_path / "witness.sqlite"
    first = backfill_witness_batch(source, store="emotion", database=database, max_records=2)
    second = backfill_witness_batch(source, store="emotion", database=database, max_records=10)
    assert first["imported"] == 2 and not first["complete"]
    assert second["imported"] == 3 and second["complete"]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM witness_events WHERE store='emotion'").fetchone()[0] == 5
        assert connection.execute("SELECT kind FROM witness_events ORDER BY sequence DESC LIMIT 1").fetchone() == ("calm",)


def test_live_witness_insert_deduplicates_stable_id(tmp_path):
    database = tmp_path / "witness.sqlite"
    payload = {"id": "precision-1", "timestamp": 1, "outcome": {"status": "stable"}}
    assert record_witness_event(store="precision", payload=payload, database=database) is True
    assert record_witness_event(store="precision", payload=payload, database=database) is False
