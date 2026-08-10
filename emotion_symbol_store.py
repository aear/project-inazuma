"""Resumable SQLite store for the generated emotion-symbol vocabulary.

The legacy JSON remains the provenance source. Migration is incremental,
source-retaining, and writes only a rebuildable/queryable database to the
configured fast index tier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from storage_layout import fast_runtime_path, load_config

LSH_BITS = 16
DEFAULT_CANDIDATE_LIMIT = 20000

_LSH_PROJECTION_CACHE: Dict[int, tuple[tuple[float, ...], ...]] = {}
DEFAULT_MIGRATION_RECORDS = 4096
DEFAULT_MIGRATION_SECONDS = 2.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "emotion_symbol_map.json"


def state_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "emotion_symbol_db_migration_state.json"


def status_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "emotion_symbol_map_status.json"


def database_path(child: str, config: Optional[Dict[str, Any]] = None) -> Path:
    fallback = Path("AI_Children") / child / "memory" / "index" / "emotion_symbols.sqlite3"
    return fast_runtime_path(
        child,
        "emotion_symbols.sqlite3",
        fallback,
        subdir="index",
        root_keys=("fast_index_root", "fast_runtime_root", "fast_root"),
        config=config,
    )


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _source_signature(path: Path) -> Tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _lsh_signature(vector: list[Any]) -> int:
    values = []
    for value in vector:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(0.0)
    projections = _LSH_PROJECTION_CACHE.get(len(values))
    if projections is None:
        projections = tuple(
            tuple(
                1.0 if hashlib.blake2b(
                    f"{bit}:{dimension}".encode("ascii"), digest_size=1
                ).digest()[0] & 1 else -1.0
                for dimension in range(len(values))
            )
            for bit in range(LSH_BITS)
        )
        _LSH_PROJECTION_CACHE[len(values)] = projections
    signature = 0
    for bit, signs in enumerate(projections):
        projection = sum(value * sign for value, sign in zip(values, signs))
        if projection >= 0.0:
            signature |= 1 << bit
    return signature


def _nearby_lsh_keys(signature: int, radius: int = 2) -> list[int]:
    keys = {int(signature)}
    for first in range(LSH_BITS):
        keys.add(signature ^ (1 << first))
    if radius >= 2:
        for first in range(LSH_BITS):
            for second in range(first + 1, LSH_BITS):
                keys.add(signature ^ (1 << first) ^ (1 << second))
    return sorted(keys)



def _open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS emotion_symbols (
            symbol_word_id TEXT PRIMARY KEY,
            symbol TEXT,
            summary TEXT,
            average_emotion_json TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            lsh16 INTEGER,
            count INTEGER NOT NULL DEFAULT 0,
            birth_time TEXT,
            generated_word TEXT,
            confidence REAL NOT NULL DEFAULT 0,
            usage_count INTEGER NOT NULL DEFAULT 0,
            payload_json TEXT NOT NULL
        )
        """
    )
    columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(emotion_symbols)")
    }
    if "lsh16" not in columns:
        connection.execute("ALTER TABLE emotion_symbols ADD COLUMN lsh16 INTEGER")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS emotion_symbols_usage "
        "ON emotion_symbols(usage_count DESC, count DESC)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS emotion_symbols_lsh16 ON emotion_symbols(lsh16)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS emotion_symbol_metadata "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    if connection.execute(
        "SELECT 1 FROM emotion_symbol_metadata WHERE key = 'symbol_count'"
    ).fetchone() is None:
        total = int(connection.execute("SELECT COUNT(*) FROM emotion_symbols").fetchone()[0])
        connection.execute(
            "INSERT INTO emotion_symbol_metadata(key, value) VALUES ('symbol_count', ?)",
            (str(total),),
        )
    connection.commit()
    return connection
def _metadata_count(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT value FROM emotion_symbol_metadata WHERE key = 'symbol_count'"
    ).fetchone()
    return int(row[0]) if row else 0


def _set_metadata_count(connection: sqlite3.Connection, value: int) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO emotion_symbol_metadata(key, value) "
        "VALUES ('symbol_count', ?)",
        (str(max(0, int(value))),),
    )



def _object_stream(path: Path, offset: int = 0) -> Iterator[Tuple[Dict[str, Any], int]]:
    """Yield objects from the top-level symbols array using bounded memory."""
    with path.open("rb") as handle:
        if offset <= 0:
            header = handle.read(64 * 1024)
            key_at = header.find(b'"symbols"')
            array_at = header.find(b"[", key_at + 9) if key_at >= 0 else -1
            if array_at < 0:
                raise ValueError("symbols array not found")
            handle.seek(array_at + 1)
        else:
            handle.seek(offset)

        depth = 0
        in_string = False
        escaped = False
        payload: Optional[bytearray] = None
        while True:
            chunk_start = handle.tell()
            chunk = handle.read(1024 * 1024)
            if not chunk:
                if payload is not None:
                    raise ValueError("incomplete symbol object at end of JSON")
                return
            for index, byte in enumerate(chunk):
                absolute_offset = chunk_start + index
                if payload is None:
                    if byte == 93:  # ]
                        return
                    if byte != 123:  # {
                        continue
                    payload = bytearray((byte,))
                    depth = 1
                    in_string = False
                    escaped = False
                    continue

                payload.append(byte)
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == 92:  # backslash
                        escaped = True
                    elif byte == 34:  # quote
                        in_string = False
                    continue
                if byte == 34:
                    in_string = True
                elif byte == 123:
                    depth += 1
                elif byte == 125:
                    depth -= 1
                    if depth == 0:
                        decoded = json.loads(payload.decode("utf-8"))
                        if isinstance(decoded, dict):
                            yield decoded, absolute_offset + 1
                        payload = None


def _symbol_id(payload: Dict[str, Any]) -> str:
    value = str(payload.get("symbol_word_id") or "").strip()
    if value:
        return value
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"sym_emotion_imported_{digest}"


def _upsert(connection: sqlite3.Connection, payload: Dict[str, Any]) -> None:
    vector = payload.get("vector") or []
    connection.execute(
        """
        INSERT INTO emotion_symbols (
            symbol_word_id, symbol, summary, average_emotion_json, vector_json,
            lsh16, count, birth_time, generated_word, confidence, usage_count, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol_word_id) DO UPDATE SET
            symbol=excluded.symbol,
            summary=excluded.summary,
            average_emotion_json=excluded.average_emotion_json,
            vector_json=excluded.vector_json,
            lsh16=excluded.lsh16,
            count=excluded.count,
            birth_time=excluded.birth_time,
            generated_word=excluded.generated_word,
            confidence=excluded.confidence,
            usage_count=excluded.usage_count,
            payload_json=excluded.payload_json
        """,
        (
            _symbol_id(payload),
            str(payload.get("symbol") or ""),
            str(payload.get("summary") or ""),
            json.dumps(payload.get("average_emotion") or {}, separators=(",", ":")),
            json.dumps(vector, separators=(",", ":")),
            _lsh_signature(vector),
            int(payload.get("count") or 0),
            payload.get("birth_time"),
            payload.get("generated_word"),
            float(payload.get("confidence") or 0.0),
            int(payload.get("usage_count") or 0),
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        ),
    )


def migration_step(
    child: str,
    *,
    max_records: int = DEFAULT_MIGRATION_RECORDS,
    max_seconds: float = DEFAULT_MIGRATION_SECONDS,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source = source_path(child)
    if not source.is_file() or source.is_symlink():
        return {"status": "missing_source", "source": str(source)}
    cfg = config if isinstance(config, dict) else load_config()
    target = database_path(child, cfg)
    size, mtime_ns = _source_signature(source)
    state = _load_json(state_path(child))
    imported_before = int(state.get("imported_records") or 0)
    if imported_before and (
        int(state.get("source_size") or -1) != size
        or int(state.get("source_mtime_ns") or -1) != mtime_ns
    ):
        state.update(status="failed", error="source_changed_during_migration", updated_at=_now_iso())
        _atomic_write_json(state_path(child), state)
        return state

    offset = int(state.get("source_offset") or 0)
    record_limit = max(1, int(max_records))
    second_limit = max(0.05, float(max_seconds))
    started = time.monotonic()
    imported = 0
    exhausted = False
    connection = _open_database(target)
    try:
        iterator = _object_stream(source, offset)
        while imported < record_limit and time.monotonic() - started < second_limit:
            try:
                payload, next_offset = next(iterator)
            except StopIteration:
                exhausted = True
                break
            _upsert(connection, payload)
            imported += 1
            offset = next_offset
        connection.commit()
        total = int(connection.execute("SELECT COUNT(*) FROM emotion_symbols").fetchone()[0])
        _set_metadata_count(connection, total)
        connection.commit()
        verification = None
        if exhausted:
            verification = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            if verification.lower() != "ok":
                raise sqlite3.DatabaseError(f"quick_check failed: {verification}")
            metadata = {
                "source_size": str(size),
                "source_mtime_ns": str(mtime_ns),
                "source_path": str(source),
                "completed_at": _now_iso(),
            }
            connection.executemany(
                "INSERT OR REPLACE INTO emotion_symbol_metadata(key, value) VALUES (?, ?)",
                list(metadata.items()),
            )
            connection.commit()
    except Exception as exc:
        connection.rollback()
        state.update(
            status="failed",
            error=str(exc),
            source=str(source),
            target=str(target),
            source_offset=offset,
            imported_records=imported_before + imported,
            updated_at=_now_iso(),
        )
        _atomic_write_json(state_path(child), state)
        return state
    finally:
        connection.close()

    now = _now_iso()
    state = {
        "version": 1,
        "child": child,
        "status": "complete" if exhausted else "copying",
        "source": str(source),
        "target": str(target),
        "source_size": size,
        "source_mtime_ns": mtime_ns,
        "source_offset": offset,
        "imported_records": total,
        "last_step_records": imported,
        "progress": round(min(1.0, offset / max(1, size)), 6),
        "source_retained": True,
        "verification": verification,
        "updated_at": now,
    }
    if exhausted:
        state["progress"] = 1.0
        state["completed_at"] = now
        _atomic_write_json(status_path(child), {
            "version": 2,
            "backend": "sqlite",
            "symbol_count": total,
            "database_path": str(target),
            "source_size": size,
            "source_mtime_ns": mtime_ns,
            "updated_at": now,
        })
    _atomic_write_json(state_path(child), state)
    return state


def database_ready(child: str, config: Optional[Dict[str, Any]] = None) -> bool:
    state = _load_json(state_path(child))
    if state.get("status") != "complete":
        return False
    source = source_path(child)
    try:
        size, mtime_ns = _source_signature(source)
    except OSError:
        return False
    return bool(
        int(state.get("source_size") or -1) == size
        and int(state.get("source_mtime_ns") or -1) == mtime_ns
        and database_path(child, config).is_file()
    )


def symbol_count(child: str, config: Optional[Dict[str, Any]] = None) -> Optional[int]:
    if not database_ready(child, config):
        return None
    connection = sqlite3.connect(f"file:{database_path(child, config)}?mode=ro", uri=True)
    try:
        try:
            return _metadata_count(connection)
        except sqlite3.OperationalError:
            return int(connection.execute("SELECT COUNT(*) FROM emotion_symbols").fetchone()[0])
    finally:
        connection.close()


def backfill_lsh_step(
    child: str,
    *,
    max_records: int = 25000,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not database_ready(child, config):
        return {"status": "database_not_ready"}
    target = database_path(child, config)
    connection = _open_database(target)
    try:
        rows = connection.execute(
            "SELECT rowid, vector_json FROM emotion_symbols "
            "WHERE lsh16 IS NULL LIMIT ?",
            (max(1, int(max_records)),),
        ).fetchall()
        updates = []
        for rowid, vector_json in rows:
            try:
                vector = json.loads(vector_json)
            except Exception:
                vector = []
            updates.append((_lsh_signature(vector if isinstance(vector, list) else []), rowid))
        if updates:
            connection.executemany(
                "UPDATE emotion_symbols SET lsh16 = ? WHERE rowid = ?", updates
            )
        connection.commit()
        remaining = int(connection.execute(
            "SELECT COUNT(*) FROM emotion_symbols WHERE lsh16 IS NULL"
        ).fetchone()[0])
        verification = None
        if remaining == 0:
            verification = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        return {
            "status": "complete" if remaining == 0 else "indexing",
            "indexed_this_step": len(updates),
            "remaining": remaining,
            "verification": verification,
            "target": str(target),
            "updated_at": _now_iso(),
        }
    finally:
        connection.close()


def iter_candidate_payloads(
    child: str,
    vector: list[Any],
    *,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    hamming_radius: int = 2,
    config: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield a bounded approximate-nearest candidate set from indexed buckets."""
    if not database_ready(child, config):
        return
    keys = _nearby_lsh_keys(_lsh_signature(vector), radius=hamming_radius)
    placeholders = ",".join("?" for _ in keys)
    limit = max(1, int(candidate_limit))
    connection = sqlite3.connect(f"file:{database_path(child, config)}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            f"SELECT payload_json FROM emotion_symbols WHERE lsh16 IN ({placeholders}) "
            "ORDER BY usage_count DESC, count DESC LIMIT ?",
            (*keys, limit),
        ).fetchall()
        if not rows:
            rows = connection.execute(
                "SELECT payload_json FROM emotion_symbols "
                "ORDER BY usage_count DESC, count DESC LIMIT ?",
                (min(limit, 4096),),
            ).fetchall()
        for row in rows:
            try:
                payload = json.loads(row[0])
            except Exception:
                continue
            if isinstance(payload, dict):
                yield payload
    finally:
        connection.close()


def iter_symbol_payloads(
    child: str,
    *,
    batch_size: int = 512,
    config: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    if not database_ready(child, config):
        return
    connection = sqlite3.connect(f"file:{database_path(child, config)}?mode=ro", uri=True)
    try:
        cursor = connection.execute("SELECT payload_json FROM emotion_symbols ORDER BY rowid")
        while True:
            rows = cursor.fetchmany(max(1, int(batch_size)))
            if not rows:
                break
            for row in rows:
                try:
                    payload = json.loads(row[0])
                except Exception:
                    continue
                if isinstance(payload, dict):
                    yield payload
    finally:
        connection.close()


def upsert_symbols(
    child: str,
    symbols: list[Dict[str, Any]],
    *,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    if not database_ready(child, config):
        return 0
    connection = _open_database(database_path(child, config))
    try:
        total = _metadata_count(connection)
        for payload in symbols:
            if isinstance(payload, dict):
                symbol_id = _symbol_id(payload)
                exists = connection.execute(
                    "SELECT 1 FROM emotion_symbols WHERE symbol_word_id = ?", (symbol_id,)
                ).fetchone()
                _upsert(connection, payload)
                total += int(exists is None)
        _set_metadata_count(connection, total)
        connection.commit()
    finally:
        connection.close()
    source = source_path(child)
    try:
        size, mtime_ns = _source_signature(source)
    except OSError:
        size, mtime_ns = 0, 0
    _atomic_write_json(status_path(child), {
        "version": 2,
        "backend": "sqlite",
        "symbol_count": total,
        "database_path": str(database_path(child, config)),
        "source_size": size,
        "source_mtime_ns": mtime_ns,
        "updated_at": _now_iso(),
    })
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally migrate emotion symbols from JSON to SQLite.")
    parser.add_argument("--child", default=None)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MIGRATION_RECORDS)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MIGRATION_SECONDS)
    parser.add_argument("--backfill-index", action="store_true")
    args = parser.parse_args()
    config = load_config()
    child = str(args.child or config.get("current_child") or "Inazuma_Yagami")
    if args.backfill_index:
        result = backfill_lsh_step(child, max_records=args.max_records, config=config)
    else:
        result = migration_step(
            child,
            max_records=args.max_records,
            max_seconds=args.max_seconds,
            config=config,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

