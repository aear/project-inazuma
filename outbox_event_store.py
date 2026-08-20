"""Durable outbox event ledger with a rebuildable hot current-state projection."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_BACKFILL_RECORDS = 1000
DEFAULT_BACKFILL_BYTES = 8 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 2 * 1024 * 1024


def durable_database_path(child: str) -> Path:
    return Path("AI_Children") / str(child) / "memory" / "outbox_events.sqlite"


def hot_database_path(child: str, *, typed_path: Path | None = None) -> Path:
    if typed_path is not None:
        return Path(typed_path).expanduser().parent / "outbox_hot.sqlite"
    return Path("AI_Children") / str(child) / "memory" / "fast_runtime" / "outbox_hot.sqlite"


def record_configured_event(
    *, child: str, channel: str, event_type: str, payload: dict[str, Any],
    typed_path: Path | None = None,
) -> dict[str, Any]:
    return record_event(
        channel=channel, event_type=event_type, payload=payload,
        durable_path=durable_database_path(child),
        hot_path=hot_database_path(child, typed_path=typed_path),
    )


def _connect(path: Path, *, durable: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=5.0)
    # The durable HDD may be mounted through filesystems where WAL shared-memory
    # locking is slow or unavailable. The hot NVMe projection is rebuildable and
    # can safely use WAL; the authoritative ledger favours portable rollback
    # journaling and full sync.
    connection.execute(f"PRAGMA journal_mode={'DELETE' if durable else 'WAL'}")
    connection.execute(f"PRAGMA synchronous={'FULL' if durable else 'NORMAL'}")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _initialize_durable(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS outbox_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            status TEXT,
            event_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            source_path TEXT,
            source_offset INTEGER,
            UNIQUE(source_path, source_offset)
        );
        CREATE INDEX IF NOT EXISTS idx_outbox_events_entry
            ON outbox_events(channel, entry_id, sequence);
        CREATE INDEX IF NOT EXISTS idx_outbox_events_status
            ON outbox_events(channel, status, sequence);
        CREATE TABLE IF NOT EXISTS migration_cursors (
            source_path TEXT PRIMARY KEY,
            byte_offset INTEGER NOT NULL,
            source_size INTEGER NOT NULL,
            source_mtime_ns INTEGER NOT NULL,
            imported_records INTEGER NOT NULL,
            invalid_records INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            complete INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )


def _initialize_hot(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS outbox_current (
            channel TEXT NOT NULL,
            entry_id TEXT NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            durable_sequence INTEGER NOT NULL,
            PRIMARY KEY(channel, entry_id)
        );
        CREATE INDEX IF NOT EXISTS idx_outbox_current_status
            ON outbox_current(channel, status, updated_at);
        """
    )


def _event_fields(payload: dict[str, Any], event_type: str) -> tuple[str, str, str]:
    entry_id = str(payload.get("id") or payload.get("entry_id") or "").strip()
    if not entry_id:
        raise ValueError("Outbox event requires a stable entry id.")
    status = str(payload.get("status") or ("pending" if event_type == "queued" else event_type)).strip().lower()
    event_at = str(
        payload.get("timestamp") or payload.get("archived_at") or payload.get("created_at")
        or datetime.now(timezone.utc).isoformat()
    )
    return entry_id, status, event_at


def _insert_durable_event(
    connection: sqlite3.Connection, *, channel: str, event_type: str,
    payload: dict[str, Any], source_path: str | None, source_offset: int | None,
) -> tuple[bool, int, str, str, str, str]:
    entry_id, status, event_at = _event_fields(payload, event_type)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cursor = connection.execute(
        """INSERT OR IGNORE INTO outbox_events
           (channel, entry_id, event_type, status, event_at, payload_json, source_path, source_offset)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (str(channel), entry_id, str(event_type), status, event_at, encoded, source_path, source_offset),
    )
    inserted = cursor.rowcount > 0
    if inserted:
        sequence = int(cursor.lastrowid)
    else:
        row = connection.execute(
            "SELECT sequence FROM outbox_events WHERE source_path = ? AND source_offset = ?",
            (source_path, source_offset),
        ).fetchone()
        sequence = int(row[0]) if row else 0
    return inserted, sequence, entry_id, status, event_at, encoded


def _update_hot_event(
    connection: sqlite3.Connection, *, channel: str, entry_id: str, status: str,
    event_at: str, encoded: str, sequence: int,
) -> None:
    connection.execute(
        """INSERT INTO outbox_current
           (channel, entry_id, status, updated_at, payload_json, durable_sequence)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(channel, entry_id) DO UPDATE SET
             status=excluded.status, updated_at=excluded.updated_at,
             payload_json=excluded.payload_json, durable_sequence=excluded.durable_sequence
           WHERE excluded.durable_sequence >= outbox_current.durable_sequence""",
        (str(channel), entry_id, status, event_at, encoded, sequence),
    )


def record_event(
    *, channel: str, event_type: str, payload: dict[str, Any], durable_path: Path,
    hot_path: Path | None = None, source_path: str | None = None,
    source_offset: int | None = None,
) -> dict[str, Any]:
    """Commit durable evidence first, then update the rebuildable hot projection."""
    with _connect(Path(durable_path), durable=True) as durable:
        _initialize_durable(durable)
        inserted, sequence, entry_id, status, event_at, encoded = _insert_durable_event(
            durable, channel=channel, event_type=event_type, payload=payload,
            source_path=source_path, source_offset=source_offset,
        )
    hot_updated = False
    if hot_path is not None and sequence:
        with _connect(Path(hot_path), durable=False) as hot:
            _initialize_hot(hot)
            _update_hot_event(
                hot, channel=channel, entry_id=entry_id, status=status,
                event_at=event_at, encoded=encoded, sequence=sequence,
            )
            hot_updated = True
    return {"inserted": inserted, "sequence": sequence, "hot_updated": hot_updated, "entry_id": entry_id}


def backfill_jsonl_batch(
    source: Path | str, *, channel: str, event_type: str, durable_path: Path,
    hot_path: Path | None = None, max_records: int = DEFAULT_BACKFILL_RECORDS,
    max_bytes: int = DEFAULT_BACKFILL_BYTES,
) -> dict[str, Any]:
    """Resume a bounded JSONL import from its durable byte-offset cursor."""
    source = Path(source).resolve()
    stat = source.stat()
    record_limit = max(1, min(10_000, int(max_records)))
    byte_limit = max(1024, min(64 * 1024 * 1024, int(max_bytes)))
    with _connect(Path(durable_path), durable=True) as durable:
        _initialize_durable(durable)
        row = durable.execute(
            "SELECT byte_offset, imported_records, invalid_records FROM migration_cursors WHERE source_path = ?",
            (str(source),),
        ).fetchone()
    offset, imported_total, invalid_total = (int(row[0]), int(row[1]), int(row[2])) if row else (0, 0, 0)
    if offset > stat.st_size:
        raise RuntimeError("JSONL source shrank behind its migration cursor; explicit reconciliation is required.")
    imported = invalid = consumed = 0
    hot = _connect(Path(hot_path), durable=False) if hot_path is not None else None
    try:
        with _connect(Path(durable_path), durable=True) as durable:
            _initialize_durable(durable)
            if hot is not None:
                _initialize_hot(hot)
            with source.open("rb") as handle:
                handle.seek(offset)
                while imported + invalid < record_limit and consumed < byte_limit:
                    line_offset = handle.tell()
                    raw = handle.readline(MAX_JSONL_LINE_BYTES + 1)
                    if not raw:
                        break
                    consumed += len(raw)
                    if len(raw) > MAX_JSONL_LINE_BYTES and not raw.endswith(b"\n"):
                        invalid += 1
                        while raw and not raw.endswith(b"\n"):
                            raw = handle.readline(MAX_JSONL_LINE_BYTES + 1)
                        continue
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                        if not isinstance(payload, dict):
                            raise ValueError("record is not an object")
                        _, sequence, entry_id, status, event_at, encoded = _insert_durable_event(
                            durable, channel=channel, event_type=event_type, payload=payload,
                            source_path=str(source), source_offset=line_offset,
                        )
                        if hot is not None and sequence:
                            _update_hot_event(
                                hot, channel=channel, entry_id=entry_id, status=status,
                                event_at=event_at, encoded=encoded, sequence=sequence,
                            )
                        imported += 1
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                        invalid += 1
                next_offset = handle.tell()
            latest = source.stat()
            complete = next_offset >= latest.st_size
            durable.execute(
                """INSERT INTO migration_cursors
                   (source_path, byte_offset, source_size, source_mtime_ns, imported_records,
                    invalid_records, updated_at, complete)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_path) DO UPDATE SET
                     byte_offset=excluded.byte_offset, source_size=excluded.source_size,
                     source_mtime_ns=excluded.source_mtime_ns, imported_records=excluded.imported_records,
                     invalid_records=excluded.invalid_records, updated_at=excluded.updated_at,
                     complete=excluded.complete""",
                (
                    str(source), next_offset, latest.st_size, latest.st_mtime_ns,
                    imported_total + imported, invalid_total + invalid,
                    datetime.now(timezone.utc).isoformat(), int(complete),
                ),
            )
            if hot is not None:
                hot.commit()
    finally:
        if hot is not None:
            hot.close()
    return {
        "source": str(source), "start_offset": offset, "next_offset": next_offset,
        "source_size": latest.st_size, "imported": imported, "invalid": invalid,
        "complete": complete,
    }


__all__ = [
    "backfill_jsonl_batch", "durable_database_path", "hot_database_path",
    "record_configured_event", "record_event",
]
