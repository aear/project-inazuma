"""Bounded SQLite access to the legacy JSON memory index."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from io_utils import file_lock


_SELECT_COLUMNS = (
    "frag_id, tier, filename, last_seen, timestamp, importance, "
    "mtime_ns, size_bytes, tags_json"
)


def iter_json_object_items(path: Path, *, chunk_chars: int = 64 * 1024) -> Iterator[Tuple[str, Any]]:
    """Stream entries from a top-level JSON object without materialising it."""
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as handle:
        buffer = ""
        eof = False

        def fill() -> bool:
            nonlocal buffer, eof
            chunk = handle.read(max(1024, int(chunk_chars)))
            if chunk:
                buffer += chunk
                return True
            eof = True
            return False

        def skip_space(position: int) -> int:
            nonlocal buffer
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return position
                fill()

        fill()
        position = skip_space(0)
        if position >= len(buffer) or buffer[position] != "{":
            raise ValueError(f"Expected a JSON object in {path}")
        position += 1

        while True:
            position = skip_space(position)
            if position < len(buffer) and buffer[position] == "}":
                return
            while True:
                try:
                    key, key_end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if eof or not fill():
                        raise
            if not isinstance(key, str):
                raise ValueError(f"Expected a string key in {path}")
            position = skip_space(key_end)
            if position >= len(buffer):
                fill()
                position = skip_space(position)
            if position >= len(buffer) or buffer[position] != ":":
                raise ValueError(f"Expected ':' after {key!r} in {path}")
            position = skip_space(position + 1)
            value_start = position
            while True:
                try:
                    value, value_end = decoder.raw_decode(buffer, value_start)
                    break
                except json.JSONDecodeError:
                    if eof or not fill():
                        raise
            yield key, value
            position = skip_space(value_end)
            if position >= len(buffer):
                fill()
                position = skip_space(position)
            if position < len(buffer) and buffer[position] == "}":
                return
            if position >= len(buffer) or buffer[position] != ",":
                raise ValueError(f"Expected ',' after {key!r} in {path}")
            # Each metadata value is small; discard consumed text so memory use
            # is independent of the 449MB legacy index size.
            buffer = buffer[position + 1 :]
            position = 0


def _int_value(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fragments ("
        "frag_id TEXT PRIMARY KEY, tier TEXT, filename TEXT, last_seen TEXT, "
        "timestamp TEXT, importance REAL, mtime_ns INTEGER, size_bytes INTEGER, tags_json TEXT)"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS fragment_tags (tag TEXT, frag_id TEXT, PRIMARY KEY(tag, frag_id))")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fragment_tags_tag ON fragment_tags(tag)")


def index_is_current(json_path: Path, db_path: Path) -> bool:
    if not db_path.exists():
        return False
    if not json_path.exists():
        return True
    try:
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = 'source_mtime_ns'").fetchone()
            schema = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        return bool(row and schema and schema[0] == "2" and int(row[0]) == int(json_path.stat().st_mtime_ns))
    except Exception:
        return False


def ensure_memory_index_db(json_path: Path, db_path: Path, *, batch_size: int = 2048) -> bool:
    """Build a current sidecar with bounded memory; safe for multi-GB JSON."""
    json_path = Path(json_path)
    db_path = Path(db_path)
    if index_is_current(json_path, db_path):
        return True
    if not json_path.exists():
        return db_path.exists()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = db_path.with_suffix(db_path.suffix + ".build.lock")
    with file_lock(lock_path):
        if index_is_current(json_path, db_path):
            return True
        source_mtime = int(json_path.stat().st_mtime_ns)
        temp_path = db_path.with_name(f"{db_path.name}.{uuid.uuid4().hex}.tmp")
        conn = sqlite3.connect(str(temp_path))
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=NORMAL")
            _ensure_schema(conn)
            rows = []
            tag_rows = []
            for frag_id, meta in iter_json_object_items(json_path):
                if not isinstance(meta, dict):
                    continue
                try:
                    importance = float(meta.get("importance", 0.0) or 0.0)
                except (TypeError, ValueError):
                    importance = 0.0
                tags = [str(tag).lower() for tag in (meta.get("tags") or []) if tag]
                rows.append(
                    (
                        str(frag_id),
                        str(meta.get("tier") or ""),
                        str(meta.get("filename") or ""),
                        str(meta.get("last_seen") or ""),
                        str(meta.get("timestamp") or ""),
                        importance,
                        _int_value(meta.get("mtime_ns")),
                        _int_value(meta.get("size_bytes")),
                        json.dumps(meta.get("tags") or [], ensure_ascii=False),
                    )
                )
                tag_rows.extend((tag, str(frag_id)) for tag in tags)
                if len(rows) >= max(1, int(batch_size)):
                    conn.executemany(
                        f"INSERT OR REPLACE INTO fragments({_SELECT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )
                    if tag_rows:
                        conn.executemany("INSERT OR REPLACE INTO fragment_tags(tag, frag_id) VALUES (?, ?)", tag_rows)
                    conn.commit()
                    rows.clear()
                    tag_rows.clear()
            if rows:
                conn.executemany(
                    f"INSERT OR REPLACE INTO fragments({_SELECT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            if tag_rows:
                conn.executemany("INSERT OR REPLACE INTO fragment_tags(tag, frag_id) VALUES (?, ?)", tag_rows)
            if int(json_path.stat().st_mtime_ns) != source_mtime:
                raise RuntimeError("memory_map.json changed while its index was being built")
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('source_mtime_ns', ?)",
                (str(source_mtime),),
            )
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', '2')")
            conn.commit()
            conn.close()
            os.replace(temp_path, db_path)
            return True
        finally:
            try:
                conn.close()
            except Exception:
                pass
            try:
                temp_path.unlink()
            except OSError:
                pass


def touch_fragments(db_path: Path, fragment_ids: list[str], last_seen: str) -> int:
    ids = [str(fragment_id) for fragment_id in fragment_ids if fragment_id]
    if not ids or not Path(db_path).exists():
        return 0
    with sqlite3.connect(str(db_path)) as conn:
        _ensure_schema(conn)
        before = conn.total_changes
        conn.executemany(
            "UPDATE fragments SET last_seen = ? WHERE frag_id = ?",
            [(str(last_seen), fragment_id) for fragment_id in ids],
        )
        conn.commit()
        return int(conn.total_changes - before)
