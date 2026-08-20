import json
import sqlite3

from outbox_event_store import backfill_jsonl_batch, record_event


def test_durable_event_precedes_rebuildable_hot_projection(tmp_path):
    durable = tmp_path / "hdd" / "events.sqlite"
    hot = tmp_path / "nvme" / "hot.sqlite"
    result = record_event(
        channel="typed", event_type="queued", payload={"id": "typed-1", "text": "hello"},
        durable_path=durable, hot_path=hot,
    )
    assert result["inserted"] is True
    with sqlite3.connect(durable) as connection:
        assert connection.execute("SELECT entry_id, status FROM outbox_events").fetchone() == ("typed-1", "pending")
    with sqlite3.connect(hot) as connection:
        assert connection.execute("SELECT entry_id, status FROM outbox_current").fetchone() == ("typed-1", "pending")


def test_backfill_is_bounded_resumable_and_idempotent(tmp_path):
    source = tmp_path / "history.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for index in range(5):
            handle.write(json.dumps({"id": f"entry-{index}", "status": "submitted"}) + "\n")
    durable = tmp_path / "events.sqlite"
    first = backfill_jsonl_batch(source, channel="github", event_type="history", durable_path=durable, max_records=2)
    second = backfill_jsonl_batch(source, channel="github", event_type="history", durable_path=durable, max_records=10)
    third = backfill_jsonl_batch(source, channel="github", event_type="history", durable_path=durable, max_records=10)
    assert first["imported"] == 2 and first["complete"] is False
    assert second["imported"] == 3 and second["complete"] is True
    assert third["imported"] == 0 and third["complete"] is True
    with sqlite3.connect(durable) as connection:
        assert connection.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0] == 5


def test_backfill_refuses_silent_replay_after_source_shrinks(tmp_path):
    source = tmp_path / "history.jsonl"
    source.write_text('{"id":"one"}\n{"id":"two"}\n', encoding="utf-8")
    durable = tmp_path / "events.sqlite"
    backfill_jsonl_batch(source, channel="typed", event_type="history", durable_path=durable)
    source.write_text('{"id":"one"}\n', encoding="utf-8")
    try:
        backfill_jsonl_batch(source, channel="typed", event_type="history", durable_path=durable)
    except RuntimeError as exc:
        assert "shrank" in str(exc)
    else:
        raise AssertionError("source shrink should require reconciliation")
