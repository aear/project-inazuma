"""V1/V2 benchmark for bounded, indexed outbox migration."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import time

from outbox_event_store import backfill_jsonl_batch


def main() -> int:
    with TemporaryDirectory(prefix="ina-outbox-db-benchmark-") as directory:
        root = Path(directory)
        source = root / "history.jsonl"
        with source.open("w", encoding="utf-8") as handle:
            for index in range(2000):
                handle.write(json.dumps({"id": f"entry-{index}", "status": "submitted"}) + "\n")
        database = root / "events.sqlite"
        first = backfill_jsonl_batch(
            source, channel="typed", event_type="history", durable_path=database,
            max_records=250, max_bytes=1024 * 1024,
        )
        while not first["complete"]:
            first = backfill_jsonl_batch(
                source, channel="typed", event_type="history", durable_path=database,
                max_records=250, max_bytes=1024 * 1024,
            )
        started = time.perf_counter()
        with sqlite3.connect(database) as connection:
            found = connection.execute(
                "SELECT status FROM outbox_events WHERE channel=? AND entry_id=? ORDER BY sequence DESC LIMIT 1",
                ("typed", "entry-1999"),
            ).fetchone()
            count = connection.execute("SELECT COUNT(*) FROM outbox_events").fetchone()[0]
        query_ms = (time.perf_counter() - started) * 1000.0
    result = {
        "V1_jsonl": {"bounded_resume": 0, "indexed_lookup": 0, "deduplicated_import": 0},
        "V2_sqlite": {
            "bounded_resume": int(first["complete"]),
            "indexed_lookup": int(found == ("submitted",)),
            "deduplicated_import": int(count == 2000),
            "query_ms": round(query_ms, 3),
        },
    }
    print(result)
    return 0 if all(result["V2_sqlite"][key] == 1 for key in (
        "bounded_resume", "indexed_lookup", "deduplicated_import"
    )) else 1


if __name__ == "__main__":
    raise SystemExit(main())
