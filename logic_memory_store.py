"""High-cap sparse SQLite graph for durable logic traces."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from emotion_symbol_store import _lsh_signature, _nearby_lsh_keys
from storage_layout import fast_runtime_path, load_config


DEFAULT_MAX_ENTRIES = 10_000_000
DEFAULT_EDGE_KNN = 8
DEFAULT_EDGE_MIN_SIMILARITY = 0.4
DEFAULT_CANDIDATE_LIMIT = 1024


def database_path(child: str, config: Optional[Dict[str, Any]] = None) -> Path:
    fallback = Path("AI_Children") / child / "memory" / "index" / "logic_memory.sqlite3"
    return fast_runtime_path(
        child,
        "logic_memory.sqlite3",
        fallback,
        subdir="index",
        root_keys=("fast_index_root", "fast_runtime_root", "fast_root"),
        config=config,
    )


def _policy(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = config.get("logic_store_policy") if isinstance(config, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    return {
        "max_entries": max(1000, int(raw.get("max_entries", DEFAULT_MAX_ENTRIES))),
        "edge_knn": max(1, int(raw.get("edge_knn", DEFAULT_EDGE_KNN))),
        "edge_min_similarity": max(
            -1.0, min(1.0, float(raw.get("edge_min_similarity", DEFAULT_EDGE_MIN_SIMILARITY)))
        ),
        "candidate_limit": max(32, int(raw.get("candidate_limit", DEFAULT_CANDIDATE_LIMIT))),
    }


def _open(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS logic_entries (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT,
            description TEXT,
            symbol_word_id TEXT,
            vector_json TEXT NOT NULL,
            lsh16 INTEGER NOT NULL,
            observation_count INTEGER NOT NULL DEFAULT 1,
            payload_json TEXT NOT NULL,
            inserted_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS logic_edges (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            weight REAL NOT NULL,
            relation TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(source, target)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS logic_entries_lsh16 ON logic_entries(lsh16)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS logic_entries_timestamp ON logic_entries(timestamp DESC)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS logic_edges_target ON logic_edges(target)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS logic_metadata "
        "(key TEXT PRIMARY KEY, value INTEGER NOT NULL)"
    )
    for key, table in (("entry_count", "logic_entries"), ("edge_count", "logic_edges")):
        if connection.execute(
            "SELECT 1 FROM logic_metadata WHERE key = ?", (key,)
        ).fetchone() is None:
            count = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            connection.execute(
                "INSERT INTO logic_metadata(key, value) VALUES (?, ?)", (key, count)
            )
    connection.commit()
    return connection
def _metadata_count(connection: sqlite3.Connection, key: str) -> int:
    row = connection.execute(
        "SELECT value FROM logic_metadata WHERE key = ?", (key,)
    ).fetchone()
    return int(row[0]) if row else 0


def _set_metadata_count(connection: sqlite3.Connection, key: str, value: int) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO logic_metadata(key, value) VALUES (?, ?)",
        (key, max(0, int(value))),
    )





def _event_id(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return "logic_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _coerce_vector(vector: Iterable[Any]) -> list[float]:
    values = []
    for value in vector:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return dot / (left_norm * right_norm + 1e-8)


def _candidate_rows(
    connection: sqlite3.Connection,
    vector: list[float],
    *,
    limit: int,
) -> list[tuple[str, str]]:
    keys = _nearby_lsh_keys(_lsh_signature(vector), radius=1)
    placeholders = ",".join("?" for _ in keys)
    rows = connection.execute(
        f"SELECT event_id, vector_json FROM logic_entries "
        f"WHERE lsh16 IN ({placeholders}) ORDER BY timestamp DESC LIMIT ?",
        (*keys, limit),
    ).fetchall()
    if rows:
        return rows
    return connection.execute(
        "SELECT event_id, vector_json FROM logic_entries "
        "ORDER BY timestamp DESC LIMIT ?",
        (min(limit, 256),),
    ).fetchall()


def store_logic_entry(
    child: str,
    payload: Dict[str, Any],
    vector: Iterable[Any],
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = config if isinstance(config, dict) else load_config()
    policy = _policy(cfg)
    values = _coerce_vector(vector)
    event_id = _event_id(payload)
    path = database_path(child, cfg)
    connection = _open(path)
    inserted = False
    edges_added = 0
    try:
        total = _metadata_count(connection, "entry_count")
        edge_total = _metadata_count(connection, "edge_count")
        exists = connection.execute(
            "SELECT 1 FROM logic_entries WHERE event_id = ?", (event_id,)
        ).fetchone()
        if exists:
            connection.execute(
                "UPDATE logic_entries SET observation_count = observation_count + 1 "
                "WHERE event_id = ?",
                (event_id,),
            )
        else:
            candidates = _candidate_rows(
                connection, values, limit=policy["candidate_limit"]
            )
            now = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                INSERT INTO logic_entries (
                    event_id, timestamp, description, symbol_word_id, vector_json,
                    lsh16, observation_count, payload_json, inserted_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    event_id,
                    payload.get("timestamp"),
                    str(payload.get("description") or ""),
                    str(payload.get("symbol_word_id") or ""),
                    json.dumps(values, separators=(",", ":")),
                    _lsh_signature(values),
                    json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                    now,
                ),
            )
            total += 1
            scored = []
            for candidate_id, candidate_json in candidates:
                try:
                    candidate_vector = _coerce_vector(json.loads(candidate_json))
                except Exception:
                    continue
                similarity = _cosine(values, candidate_vector)
                if similarity >= policy["edge_min_similarity"]:
                    scored.append((similarity, str(candidate_id)))
            scored.sort(reverse=True)
            for similarity, candidate_id in scored[: policy["edge_knn"]]:
                source, target = sorted((event_id, candidate_id))
                relation = "reinforces" if similarity >= 0.8 else "curious"
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO logic_edges(
                        source, target, weight, relation, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (source, target, round(similarity, 6), relation, now),
                )
                if cursor.rowcount:
                    edge_total += 1
                    edges_added += 1
                else:
                    connection.execute(
                        "UPDATE logic_edges SET weight = ?, relation = ?, updated_at = ? "
                        "WHERE source = ? AND target = ?",
                        (round(similarity, 6), relation, now, source, target),
                    )
            inserted = True

        if total > policy["max_entries"]:
            excess = total - policy["max_entries"]
            stale_ids = [
                row[0] for row in connection.execute(
                    "SELECT event_id FROM logic_entries "
                    "ORDER BY observation_count ASC, timestamp ASC LIMIT ?",
                    (excess,),
                ).fetchall()
            ]
            for stale_id in stale_ids:
                removed_edges = int(connection.execute(
                    "SELECT COUNT(*) FROM logic_edges WHERE source = ? OR target = ?",
                    (stale_id, stale_id),
                ).fetchone()[0])
                connection.execute(
                    "DELETE FROM logic_edges WHERE source = ? OR target = ?",
                    (stale_id, stale_id),
                )
                connection.execute(
                    "DELETE FROM logic_entries WHERE event_id = ?", (stale_id,)
                )
                total -= 1
                edge_total = max(0, edge_total - removed_edges)

        _set_metadata_count(connection, "entry_count", total)
        _set_metadata_count(connection, "edge_count", edge_total)
        connection.commit()
        return {
            "status": "stored",
            "event_id": event_id,
            "inserted": inserted,
            "edges_added": edges_added,
            "entries": total,
            "edges": edge_total,
            "database": str(path),
        }
    finally:
        connection.close()


def import_json_snapshot(
    child: str,
    entries: list[Dict[str, Any]],
    vector_fn,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    imported = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        result = store_logic_entry(
            child, entry, vector_fn(entry), config=config
        )
        imported += int(bool(result.get("inserted")))
    return {
        "status": "complete",
        "imported": imported,
        "entries": entry_count(child, config),
        "source_retained": True,
        "database": str(database_path(child, config)),
    }


def recent_entries(
    child: str,
    limit: int,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> list[Dict[str, Any]]:
    path = database_path(child, config)
    if not path.is_file():
        return []
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT payload_json FROM logic_entries "
            "ORDER BY timestamp DESC, rowid DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    finally:
        connection.close()
    result = []
    for row in reversed(rows):
        try:
            payload = json.loads(row[0])
        except Exception:
            continue
        if isinstance(payload, dict):
            result.append(payload)
    return result


def entry_count(child: str, config: Optional[Dict[str, Any]] = None) -> int:
    path = database_path(child, config)
    if not path.is_file():
        return 0
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        try:
            return _metadata_count(connection, "entry_count")
        except sqlite3.OperationalError:
            return int(connection.execute("SELECT COUNT(*) FROM logic_entries").fetchone()[0])
    finally:
        connection.close()


def graph_counts(child: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    path = database_path(child, config)
    if not path.is_file():
        return {"entries": 0, "edges": 0}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        try:
            return {
                "entries": _metadata_count(connection, "entry_count"),
                "edges": _metadata_count(connection, "edge_count"),
            }
        except sqlite3.OperationalError:
            return {
                "entries": int(connection.execute("SELECT COUNT(*) FROM logic_entries").fetchone()[0]),
                "edges": int(connection.execute("SELECT COUNT(*) FROM logic_edges").fetchone()[0]),
            }
    finally:
        connection.close()

