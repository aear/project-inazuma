"""Indexed SQLite sidecar for chronological memory-witness JSONL streams."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


MAX_LINE_BYTES = 2 * 1024 * 1024


def witness_database_path(source: Path | str) -> Path:
    return Path(source).parent / "witness_events.sqlite"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=5.0)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS witness_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            store TEXT NOT NULL,
            event_id TEXT NOT NULL,
            event_at TEXT,
            kind TEXT,
            payload_json TEXT NOT NULL,
            source_path TEXT,
            source_offset INTEGER,
            UNIQUE(store, event_id),
            UNIQUE(source_path, source_offset)
        );
        CREATE INDEX IF NOT EXISTS idx_witness_store_time ON witness_events(store, event_at, sequence);
        CREATE INDEX IF NOT EXISTS idx_witness_store_kind ON witness_events(store, kind, sequence);
        CREATE TABLE IF NOT EXISTS witness_migration_cursors (
            source_path TEXT PRIMARY KEY,
            byte_offset INTEGER NOT NULL,
            imported_records INTEGER NOT NULL,
            invalid_records INTEGER NOT NULL,
            source_size INTEGER NOT NULL,
            source_mtime_ns INTEGER NOT NULL,
            complete INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    return connection


def _fields(store: str, payload: dict[str, Any], identity_hint: str = "") -> tuple[str, str | None, str | None, str]:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    event_id = str(payload.get("id") or payload.get("event_id") or "").strip()
    if not event_id:
        event_id = f"{store}_{hashlib.sha256((identity_hint + encoded).encode('utf-8')).hexdigest()}"
    event_at = payload.get("timestamp") or payload.get("created_at") or payload.get("recorded_at")
    kind = payload.get("type") or payload.get("kind") or payload.get("mode")
    return event_id, str(event_at) if event_at is not None else None, str(kind) if kind is not None else None, encoded


def record_witness_event(
    *, store: str, payload: dict[str, Any], database: Path,
    source_path: str | None = None, source_offset: int | None = None,
) -> bool:
    event_id, event_at, kind, encoded = _fields(store, payload, f"{source_path}:{source_offset}:")
    with _connect(Path(database)) as connection:
        cursor = connection.execute(
            """INSERT OR IGNORE INTO witness_events
               (store, event_id, event_at, kind, payload_json, source_path, source_offset)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (store, event_id, event_at, kind, encoded, source_path, source_offset),
        )
        return cursor.rowcount > 0


def backfill_witness_batch(
    source: Path | str, *, store: str, database: Path | None = None,
    max_records: int = 1000, max_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    source = Path(source).resolve()
    database = Path(database) if database is not None else witness_database_path(source)
    stat = source.stat()
    record_limit = max(1, min(10_000, int(max_records)))
    byte_limit = max(1024, min(64 * 1024 * 1024, int(max_bytes)))
    with _connect(database) as connection:
        row = connection.execute(
            "SELECT byte_offset, imported_records, invalid_records FROM witness_migration_cursors WHERE source_path=?",
            (str(source),),
        ).fetchone()
        offset, imported_total, invalid_total = (int(row[0]), int(row[1]), int(row[2])) if row else (0, 0, 0)
        if offset > stat.st_size:
            raise RuntimeError("Witness source shrank behind its migration cursor.")
        imported = invalid = consumed = 0
        with source.open("rb") as handle:
            handle.seek(offset)
            while imported + invalid < record_limit and consumed < byte_limit:
                line_offset = handle.tell()
                raw = handle.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    break
                consumed += len(raw)
                try:
                    if len(raw) > MAX_LINE_BYTES and not raw.endswith(b"\n"):
                        raise ValueError("oversized line")
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("record is not an object")
                    event_id, event_at, kind, encoded = _fields(store, payload, f"{source}:{line_offset}:")
                    connection.execute(
                        """INSERT OR IGNORE INTO witness_events
                           (store, event_id, event_at, kind, payload_json, source_path, source_offset)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (store, event_id, event_at, kind, encoded, str(source), line_offset),
                    )
                    imported += 1
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    invalid += 1
                if len(raw) > MAX_LINE_BYTES and not raw.endswith(b"\n"):
                    while raw and not raw.endswith(b"\n"):
                        raw = handle.readline(MAX_LINE_BYTES + 1)
            next_offset = handle.tell()
        latest = source.stat()
        complete = next_offset >= latest.st_size
        connection.execute(
            """INSERT INTO witness_migration_cursors
               (source_path, byte_offset, imported_records, invalid_records, source_size,
                source_mtime_ns, complete, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_path) DO UPDATE SET
                 byte_offset=excluded.byte_offset, imported_records=excluded.imported_records,
                 invalid_records=excluded.invalid_records, source_size=excluded.source_size,
                 source_mtime_ns=excluded.source_mtime_ns, complete=excluded.complete,
                 updated_at=excluded.updated_at""",
            (
                str(source), next_offset, imported_total + imported, invalid_total + invalid,
                latest.st_size, latest.st_mtime_ns, int(complete), datetime.now(timezone.utc).isoformat(),
            ),
        )
    return {
        "store": store, "source": str(source), "database": str(database),
        "start_offset": offset, "next_offset": next_offset, "source_size": latest.st_size,
        "imported": imported, "invalid": invalid, "complete": complete,
    }


__all__ = ["backfill_witness_batch", "record_witness_event", "witness_database_path"]
