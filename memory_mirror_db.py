"""Incremental SQLite mirror for JSON memory records.

The mirror is deliberately conservative: a JSON record is only marked eligible
for removal after the SQLite copy has been written and verified. Removing or
quarantining the source JSON stays opt-in through config.
"""
from __future__ import annotations

import atexit
import argparse
import hashlib
import json
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from storage_layout import format_child_path, load_config, storage_layout


DEFAULT_MIRROR_POLICY: Dict[str, Any] = {
    "enabled": True,
    "mirror_on_read": True,
    "db_root": None,
    "db_filename": "memory_mirror.sqlite3",
    "max_record_bytes": 25 * 1024 * 1024,
    "remove_json_after_verified": False,
    "quarantine_json_after_verified": False,
    "quarantine_dir": "mirrored_json_quarantine",
    "batch_records": 256,
    "batch_bytes": 16 * 1024 * 1024,
    "batch_seconds": 2.0,
    "wal_autocheckpoint_pages": 4096,
    "synchronous": "NORMAL",
}

MIRROR_KINDS = {"fragment", "experience_event", "experience_episode"}
_SESSION_CACHE: set[tuple[str, str, int, int]] = set()
_MIRROR_SESSIONS: Dict[str, Dict[str, Any]] = {}
_MIRROR_SESSIONS_LOCK = threading.RLock()


def mirror_policy(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config if isinstance(config, dict) else load_config()
    raw = cfg.get("memory_mirror_policy") if isinstance(cfg, dict) else None
    policy = DEFAULT_MIRROR_POLICY.copy()
    if isinstance(raw, dict):
        policy.update({key: raw.get(key, policy[key]) for key in policy if key in raw})
    return policy


def mirror_db_path(child: str, config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config if isinstance(config, dict) else load_config()
    policy = mirror_policy(cfg)
    root = format_child_path(policy.get("db_root"), child)
    if root is None:
        layout = storage_layout(cfg)
        root = format_child_path(layout.get("cold_root"), child)
        if root is None:
            root = Path("AI_Children") / child / "memory"
        root = root / "mirror_db"
    return root / str(policy.get("db_filename") or DEFAULT_MIRROR_POLICY["db_filename"])


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _open_mirror_session(db_path: Path, policy: Dict[str, Any]) -> Dict[str, Any]:
    key = str(db_path.resolve())
    with _MIRROR_SESSIONS_LOCK:
        existing = _MIRROR_SESSIONS.get(key)
        if existing is not None:
            return existing

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        synchronous = str(policy.get("synchronous") or "NORMAL").strip().upper()
        if synchronous not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            synchronous = "NORMAL"
        checkpoint_pages = max(0, _coerce_int(policy.get("wal_autocheckpoint_pages"), 4096))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA synchronous={synchronous}")
        conn.execute(f"PRAGMA wal_autocheckpoint={checkpoint_pages}")
        _ensure_schema(conn, configure=False)
        conn.commit()

        session: Dict[str, Any] = {
            "conn": conn,
            "lock": threading.RLock(),
            "policy": dict(policy),
            "pending": [],
            "pending_bytes": 0,
            "last_flush": time.monotonic(),
            "closed": False,
        }
        _MIRROR_SESSIONS[key] = session
        return session


def _session_should_flush(session: Dict[str, Any]) -> bool:
    policy = session["policy"]
    record_limit = max(1, _coerce_int(policy.get("batch_records"), 256))
    byte_limit = max(1, _coerce_int(policy.get("batch_bytes"), 16 * 1024 * 1024))
    second_limit = max(0.0, _coerce_float(policy.get("batch_seconds"), 2.0))
    return bool(
        len(session["pending"]) >= record_limit
        or session["pending_bytes"] >= byte_limit
        or (second_limit > 0 and time.monotonic() - session["last_flush"] >= second_limit)
    )


def _flush_session(session: Dict[str, Any]) -> set[tuple[str, str, str]]:
    """Commit queued rows, then verify and mark them in one follow-up commit."""
    with session["lock"]:
        if session.get("closed") or not session["pending"]:
            return set()

        conn: sqlite3.Connection = session["conn"]
        pending = list(session["pending"])

        # Ingestion and verification deliberately use separate transactions.
        # A crash can lose at most one batch; source JSON is retained and will
        # simply be mirrored again on the next pass.
        conn.commit()

        verified_at = datetime.now(timezone.utc).isoformat()
        verified: set[tuple[str, str, str]] = set()
        for kind, item_id, source_path, source_hash, payload_hash, payload_json in pending:
            row = conn.execute(
                "SELECT source_sha256, payload_sha256, payload_json FROM mirrored_json "
                "WHERE kind = ? AND item_id = ? AND source_path = ?",
                (kind, item_id, source_path),
            ).fetchone()
            if row and row[0] == source_hash and row[1] == payload_hash and row[2] == payload_json:
                verified.add((kind, item_id, source_path))

        if verified:
            conn.executemany(
                "UPDATE mirrored_json SET verified_at = ?, removal_eligible = 1 "
                "WHERE kind = ? AND item_id = ? AND source_path = ?",
                [(verified_at, *key) for key in verified],
            )
        conn.commit()
        session["pending"].clear()
        session["pending_bytes"] = 0
        session["last_flush"] = time.monotonic()
        return verified


def flush_mirror_writes(db_path: Optional[Path] = None, *, close: bool = False) -> int:
    """Flush pending mirror batches, optionally closing reusable connections."""
    target = str(Path(db_path).resolve()) if db_path is not None else None
    with _MIRROR_SESSIONS_LOCK:
        sessions = [
            (key, session)
            for key, session in _MIRROR_SESSIONS.items()
            if target is None or key == target
        ]

    flushed = 0
    for key, session in sessions:
        flushed += len(_flush_session(session))
        if close:
            with session["lock"]:
                if not session.get("closed"):
                    session["conn"].close()
                    session["closed"] = True
            with _MIRROR_SESSIONS_LOCK:
                _MIRROR_SESSIONS.pop(key, None)
    return flushed


def _flush_mirror_writes_at_exit() -> None:
    try:
        flush_mirror_writes(close=True)
    except Exception:
        # Exit cleanup is best-effort. Originals remain authoritative, so an
        # interrupted final batch is safe to replay on the next run.
        pass


atexit.register(_flush_mirror_writes_at_exit)


def _managed_migration_active(child: str) -> bool:
    path = Path("AI_Children") / child / "memory" / "storage_migration_request.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and str(payload.get("status") or "").lower() in {"requested", "copying", "verifying"}


def mirror_json_file(
    child: str,
    kind: str,
    path: Path,
    *,
    payload: Optional[Dict[str, Any]] = None,
    item_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    remove_original: Optional[bool] = None,
) -> Dict[str, Any]:
    """Mirror one JSON memory record into SQLite and verify the stored copy."""

    if _managed_migration_active(child):
        return {"status": "migration_paused", "child": str(child)}

    cfg = config if isinstance(config, dict) else load_config()
    policy = mirror_policy(cfg)
    if not bool(policy.get("enabled", True)) or not bool(policy.get("mirror_on_read", True)):
        return {"status": "disabled"}

    kind = str(kind or "").strip()
    if kind not in MIRROR_KINDS:
        return {"status": "unsupported_kind", "kind": kind}

    source = Path(path)
    try:
        stat = source.stat()
    except OSError as exc:
        return {"status": "missing", "path": str(source), "reason": str(exc)}

    size_bytes = int(stat.st_size)
    max_bytes = _coerce_int(policy.get("max_record_bytes"), DEFAULT_MIRROR_POLICY["max_record_bytes"])
    if max_bytes > 0 and size_bytes > max_bytes:
        return {"status": "too_large", "path": str(source), "size_bytes": size_bytes}

    cache_key = (kind, str(source.resolve()), int(stat.st_mtime_ns), size_bytes)
    if cache_key in _SESSION_CACHE:
        return {"status": "cached", "path": str(source)}

    db_path = mirror_db_path(child, cfg)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    session = _open_mirror_session(db_path, policy)
    should_remove = (
        bool(policy.get("remove_json_after_verified", False))
        if remove_original is None
        else bool(remove_original)
    )

    # The caller normally already parsed the JSON and supplied its payload.
    # Use the durable mirror row as a restart checkpoint before reading and
    # hashing the source a second time. Size+mtime changes force a full replay.
    if isinstance(payload, dict):
        persisted_id = str(item_id or payload.get("id") or payload.get("frag_id") or source.stem)
        with session["lock"]:
            persisted = session["conn"].execute(
                "SELECT source_mtime_ns, source_size_bytes, verified_at "
                "FROM mirrored_json WHERE kind = ? AND item_id = ? AND source_path = ?",
                (kind, persisted_id, str(source)),
            ).fetchone()
        if (
            persisted
            and int(persisted[0]) == int(stat.st_mtime_ns)
            and int(persisted[1]) == size_bytes
            and persisted[2]
        ):
            status = "cached_verified"
            if should_remove:
                status = _remove_or_quarantine_source(
                    source, db_path, child, kind, persisted_id, policy
                )
            _SESSION_CACHE.add(cache_key)
            return {
                "status": status,
                "verified": True,
                "removal_eligible": True,
                "kind": kind,
                "item_id": persisted_id,
                "path": str(source),
                "db_path": str(db_path),
            }

    try:
        raw_text = source.read_text(encoding="utf-8")
    except Exception as exc:
        return {"status": "unreadable", "path": str(source), "reason": str(exc)}

    if payload is None:
        try:
            loaded = json.loads(raw_text)
        except Exception as exc:
            return {"status": "invalid_json", "path": str(source), "reason": str(exc)}
        if not isinstance(loaded, dict):
            return {"status": "unsupported_payload", "path": str(source)}
        payload = loaded

    record_id = str(item_id or payload.get("id") or payload.get("frag_id") or source.stem)
    source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    metadata = _metadata_for(kind, payload, source)
    source_path = str(source)
    record_key = (kind, record_id, source_path)
    with session["lock"]:
        session["conn"].execute(
            "INSERT OR REPLACE INTO mirrored_json("
            "kind, item_id, child, source_path, source_sha256, payload_sha256, "
            "source_size_bytes, source_mtime_ns, payload_json, summary, tags_json, "
            "importance, record_timestamp, mirrored_at, verified_at, removal_eligible, "
            "json_removed_at, remove_status"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, NULL, ?)",
            (
                kind,
                record_id,
                str(child),
                source_path,
                source_hash,
                payload_hash,
                size_bytes,
                int(stat.st_mtime_ns),
                payload_json,
                metadata.get("summary"),
                json.dumps(metadata.get("tags") or [], ensure_ascii=True),
                metadata.get("importance"),
                metadata.get("record_timestamp"),
                now,
                "json_retained",
            ),
        )
        session["pending"].append(
            (kind, record_id, source_path, source_hash, payload_hash, payload_json)
        )
        session["pending_bytes"] += len(raw_text.encode("utf-8")) + len(payload_json.encode("utf-8"))

        should_flush = should_remove or _session_should_flush(session)

    verified_keys = _flush_session(session) if should_flush else set()
    verified = record_key in verified_keys

    status = "mirrored" if verified else "queued_for_verification"
    if should_remove:
        if not verified:
            return {"status": "verification_failed", "path": source_path, "db_path": str(db_path)}
        status = _remove_or_quarantine_source(source, db_path, child, kind, record_id, policy)

    _SESSION_CACHE.add(cache_key)
    return {
        "status": status,
        "verified": verified,
        "removal_eligible": verified,
        "kind": kind,
        "item_id": record_id,
        "path": str(source),
        "db_path": str(db_path),
    }


def mirror_status(child: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    db_path = mirror_db_path(child, config)
    flush_mirror_writes(db_path)
    if not db_path.exists():
        return {"status": "missing", "db_path": str(db_path), "records": 0}
    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            "SELECT kind, COUNT(*), SUM(removal_eligible), SUM(CASE WHEN json_removed_at IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM mirrored_json GROUP BY kind ORDER BY kind"
        ).fetchall()
    finally:
        conn.close()
    return {
        "status": "ok",
        "db_path": str(db_path),
        "records": sum(int(row[1] or 0) for row in rows),
        "kinds": {
            str(row[0]): {
                "records": int(row[1] or 0),
                "removal_eligible": int(row[2] or 0),
                "json_removed": int(row[3] or 0),
            }
            for row in rows
        },
    }


def experience_catalog_candidates(
    child: str,
    *,
    limit: int,
    min_age_hours: float,
    max_importance: float,
    min_size_bytes: int,
    now: Optional[datetime] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[list[Dict[str, Any]]]:
    """Select verified experience candidates from SQLite without walking files."""
    db_path = mirror_db_path(child, config)
    flush_mirror_writes(db_path)
    if not db_path.exists():
        return None
    if limit <= 0:
        return []

    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    cutoff = datetime.fromtimestamp(
        now_dt.timestamp() - max(0.0, float(min_age_hours)) * 3600.0,
        timezone.utc,
    ).isoformat()

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT kind, item_id, source_path, source_size_bytes, importance, "
            "record_timestamp, summary, tags_json FROM mirrored_json "
            "WHERE child = ? AND kind IN ('experience_event', 'experience_episode') "
            "AND verified_at IS NOT NULL AND source_size_bytes >= ? "
            "AND COALESCE(importance, 0.0) <= ? "
            "AND (record_timestamp = '' OR record_timestamp <= ?) "
            "ORDER BY source_size_bytes DESC, record_timestamp ASC LIMIT ?",
            (str(child), max(0, int(min_size_bytes)), float(max_importance), cutoff, int(limit)),
        ).fetchall()
    finally:
        conn.close()

    candidates: list[Dict[str, Any]] = []
    for kind, item_id, source_path, size_bytes, importance, timestamp, summary, tags_json in rows:
        try:
            tags = json.loads(tags_json) if tags_json else []
        except Exception:
            tags = []
        age_hours = 0.0
        if timestamp:
            try:
                parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                age_hours = max(0.0, (now_dt - parsed).total_seconds() / 3600.0)
            except Exception:
                age_hours = 0.0
        candidates.append({
            "kind": str(kind),
            "id": str(item_id),
            "path": str(source_path),
            "size_bytes": int(size_bytes or 0),
            "age_hours": round(age_hours, 3),
            "importance": round(float(importance or 0.0), 4),
            "tags": [str(tag) for tag in tags if tag] if isinstance(tags, list) else [],
            "summary": str(summary or ""),
            "recommended_action": "compact_experience_to_cold_stub",
            "reason": "Verified catalogue candidate selected without a filesystem traversal.",
        })
    return candidates

def catalog_path_known(
    child: str,
    kind: str,
    path: Path,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Check path registration without touching the source inode."""
    db_path = mirror_db_path(child, config)
    if not db_path.exists():
        return False
    session = _open_mirror_session(db_path, mirror_policy(config))
    with session["lock"]:
        row = session["conn"].execute(
            "SELECT 1 FROM mirrored_json WHERE child = ? AND kind = ? AND source_path = ? "
            "AND verified_at IS NOT NULL LIMIT 1",
            (str(child), str(kind), str(path)),
        ).fetchone()
    return row is not None

def catalog_path_is_current(
    child: str,
    kind: str,
    path: Path,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return whether a source path already has a verified matching catalogue row."""
    source = Path(path)
    try:
        stat = source.stat()
    except OSError:
        return False
    db_path = mirror_db_path(child, config)
    if not db_path.exists():
        return False
    session = _open_mirror_session(db_path, mirror_policy(config))
    with session["lock"]:
        row = session["conn"].execute(
            "SELECT 1 FROM mirrored_json WHERE child = ? AND kind = ? AND source_path = ? "
            "AND source_mtime_ns = ? AND source_size_bytes = ? AND verified_at IS NOT NULL LIMIT 1",
            (str(child), str(kind), str(source), int(stat.st_mtime_ns), int(stat.st_size)),
        ).fetchone()
    return row is not None


def _ensure_schema(conn: sqlite3.Connection, *, configure: bool = True) -> None:
    if configure:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS mirrored_json ("
        "kind TEXT NOT NULL, "
        "item_id TEXT NOT NULL, "
        "child TEXT NOT NULL, "
        "source_path TEXT NOT NULL, "
        "source_sha256 TEXT NOT NULL, "
        "payload_sha256 TEXT NOT NULL, "
        "source_size_bytes INTEGER NOT NULL, "
        "source_mtime_ns INTEGER NOT NULL, "
        "payload_json TEXT NOT NULL, "
        "summary TEXT, "
        "tags_json TEXT, "
        "importance REAL, "
        "record_timestamp TEXT, "
        "mirrored_at TEXT NOT NULL, "
        "verified_at TEXT, "
        "removal_eligible INTEGER NOT NULL DEFAULT 0, "
        "json_removed_at TEXT, "
        "remove_status TEXT, "
        "PRIMARY KEY(kind, item_id, source_path)"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mirrored_json_child_kind ON mirrored_json(child, kind)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mirrored_json_source ON mirrored_json(source_path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mirrored_json_eligible ON mirrored_json(removal_eligible, json_removed_at)")


def _remove_or_quarantine_source(
    source: Path,
    db_path: Path,
    child: str,
    kind: str,
    item_id: str,
    policy: Dict[str, Any],
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    status = "json_removed"
    try:
        if bool(policy.get("quarantine_json_after_verified", False)):
            quarantine_root = db_path.parent / str(policy.get("quarantine_dir") or "mirrored_json_quarantine")
            target = _quarantine_path(source, quarantine_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            status = "json_quarantined"
        else:
            source.unlink()
    except Exception as exc:
        status = f"remove_failed: {exc}"
        now = ""

    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_schema(conn)
        with conn:
            conn.execute(
                "UPDATE mirrored_json SET json_removed_at = ?, remove_status = ? "
                "WHERE kind = ? AND item_id = ? AND source_path = ?",
                (now or None, status, kind, item_id, str(source)),
            )
    finally:
        conn.close()
    return status


def _quarantine_path(source: Path, quarantine_root: Path) -> Path:
    parts = list(source.parts)
    rel = Path(source.name)
    if "AI_Children" in parts:
        index = parts.index("AI_Children")
        rel = Path(*parts[index:])
    return quarantine_root / rel


def _metadata_for(kind: str, payload: Dict[str, Any], source: Path) -> Dict[str, Any]:
    tags = _tags_from_payload(payload)
    return {
        "summary": _summary_from_payload(payload),
        "tags": tags,
        "importance": _importance_from_payload(payload),
        "record_timestamp": _timestamp_from_payload(payload),
    }


def _summary_from_payload(payload: Dict[str, Any]) -> str:
    for key in ("summary", "narrative", "text", "content", "intent", "description"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:500]
    for key in ("thought", "observation", "stimulus", "response"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:500]
    tags = _tags_from_payload(payload)
    if tags:
        return "Tagged " + ", ".join(tags[:10])
    return ""


def _tags_from_payload(payload: Dict[str, Any]) -> list[str]:
    found: list[str] = []
    for key in ("tags", "situation_tags", "keywords"):
        _extend_tags(found, payload.get(key))
    for key in ("metadata", "internal_state", "outcome", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            _extend_tags(found, value.get("tags"))
            _extend_tags(found, value.get("flags"))
    seen = set()
    unique: list[str] = []
    for tag in found:
        normalized = str(tag).strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _extend_tags(target: list[str], value: Any) -> None:
    if isinstance(value, list):
        target.extend(str(item) for item in value if item)
    elif isinstance(value, str) and value.strip():
        target.append(value.strip())


def _importance_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    values: list[float] = []
    for key in ("importance", "salience", "priority", "novelty"):
        _append_float(values, payload.get(key))
    for key in ("metadata", "internal_state", "outcome", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            for nested_key in ("importance", "salience", "priority", "novelty", "risk", "stress"):
                _append_float(values, value.get(nested_key))
    if not values:
        return None
    return max(0.0, min(1.0, max(abs(value) for value in values)))


def _append_float(values: list[float], value: Any) -> None:
    try:
        values.append(float(value))
    except (TypeError, ValueError):
        return


def _timestamp_from_payload(payload: Dict[str, Any]) -> str:
    for key in ("timestamp", "start_time", "end_time", "created_at", "last_seen"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mirror one memory JSON file or inspect mirror status.")
    parser.add_argument("--child", default=None)
    parser.add_argument("--kind", choices=sorted(MIRROR_KINDS), default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config()
    child = args.child or str(cfg.get("current_child") or "Inazuma_Yagami")
    if args.status:
        print(json.dumps(mirror_status(child, cfg), indent=2, ensure_ascii=True))
        return 0
    if not args.kind or not args.path:
        parser.error("--kind and --path are required unless --status is used")
    result = mirror_json_file(child, args.kind, Path(args.path), config=cfg)
    if result.get("status") == "queued_for_verification":
        result["flushed_verified"] = flush_mirror_writes(mirror_db_path(child, cfg))
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result.get("verified") or result.get("status") in {
        "cached", "cached_verified", "disabled", "queued_for_verification"
    } else 1


if __name__ == "__main__":
    raise SystemExit(_main())
