"""V1/V2 benchmark for indexed witness-event migration."""
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from witness_event_store import backfill_witness_batch


def main() -> int:
    with TemporaryDirectory(prefix="ina-witness-db-benchmark-") as directory:
        root = Path(directory)
        source = root / "precision.jsonl"
        with source.open("w", encoding="utf-8") as handle:
            for index in range(500):
                handle.write(json.dumps({"id": f"precision-{index}", "timestamp": index, "type": "decision"}) + "\n")
        database = root / "witness.sqlite"
        result = backfill_witness_batch(source, store="precision", database=database, max_records=100)
        steps = 1
        while not result["complete"]:
            result = backfill_witness_batch(source, store="precision", database=database, max_records=100)
            steps += 1
        with sqlite3.connect(database) as connection:
            found = connection.execute("SELECT event_id FROM witness_events WHERE store=? AND event_id=?", ("precision", "precision-499")).fetchone()
            count = connection.execute("SELECT COUNT(*) FROM witness_events").fetchone()[0]
    v2 = {"resumable": int(steps == 5), "indexed": int(found == ("precision-499",)), "preserved": int(count == 500)}
    result = {"V1_jsonl": {"resumable": 0, "indexed": 0, "preserved": 1}, "V2_sqlite": v2}
    print(result)
    return 0 if all(v2.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
