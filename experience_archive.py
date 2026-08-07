"""Lossless, incremental archive for experience JSON records.

The archive reduces both inode pressure and payload size while retaining the
exact original JSON bytes.  Source files are removed only after the compressed
row has been committed, decompressed, and checksum-verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from storage_layout import format_child_path, load_config, storage_layout


DEFAULT_POLICY: Dict[str, Any] = {
    "enabled": True,
    "batch_files": 2000,
    "max_seconds": 30.0,
    "min_age_hours": 24.0 * 7.0,
    "compression_level": 6,
    "archive_path": None,
    "state_path": "AI_Children/{child}/memory/experience_archive_state.json",
}


def archive_policy(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config if isinstance(config, dict) else load_config()
    policy = dict(DEFAULT_POLICY)
    raw = cfg.get("experience_archive_policy") if isinstance(cfg, dict) else None
    if isinstance(raw, dict):
        policy.update({key: raw[key] for key in policy if key in raw})
    return policy


def archive_path(child: str, config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config if isinstance(config, dict) else load_config()
    policy = archive_policy(cfg)
    explicit = format_child_path(policy.get("archive_path"), child)
    if explicit is not None:
        return explicit
    layout = storage_layout(cfg)
    cold = format_child_path(layout.get("cold_storage_root"), child)
    if cold is None:
        cold = Path("AI_Children") / child / "memory" / "cold_storage"
    return cold / "experiences" / "experience_archive.sqlite3"


def state_path(child: str, config: Optional[Dict[str, Any]] = None) -> Path:
    policy = archive_policy(config)
    raw = str(policy.get("state_path") or DEFAULT_POLICY["state_path"]).format(child=child)
    return Path(raw).expanduser()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS experience_archive ("
        "kind TEXT NOT NULL, item_id TEXT NOT NULL, source_path TEXT NOT NULL, "
        "payload_zlib BLOB NOT NULL, source_sha256 TEXT NOT NULL, "
        "raw_size_bytes INTEGER NOT NULL, compressed_size_bytes INTEGER NOT NULL, "
        "record_timestamp TEXT, archived_at TEXT NOT NULL, "
        "PRIMARY KEY(kind, item_id))"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_experience_archive_recent "
        "ON experience_archive(kind, record_timestamp DESC)"
    )


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_archived_experience(child: str, kind: str, item_id: str, *, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    path = archive_path(child, config)
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
    try:
        row = conn.execute(
            "SELECT payload_zlib, source_sha256 FROM experience_archive WHERE kind = ? AND item_id = ?",
            (str(kind), str(item_id)),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    try:
        raw = zlib.decompress(row[0])
        if hashlib.sha256(raw).hexdigest() != str(row[1]):
            return None
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def recent_archived_experiences(child: str, limit: int, *, config: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
    path = archive_path(child, config)
    bounded = max(0, int(limit))
    if bounded <= 0 or not path.exists():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
    try:
        rows = conn.execute(
            "SELECT payload_zlib, source_sha256 FROM experience_archive "
            "WHERE kind = 'experience_event' ORDER BY record_timestamp DESC LIMIT ?",
            (bounded,),
        ).fetchall()
    finally:
        conn.close()
    result: list[Dict[str, Any]] = []
    for compressed, expected_hash in rows:
        try:
            raw = zlib.decompress(compressed)
            if hashlib.sha256(raw).hexdigest() != str(expected_hash):
                continue
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            result.append(payload)
    return result


def _archive_one(conn: sqlite3.Connection, path: Path, level: int) -> Dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return {"status": "unreadable", "reason": str(exc)}
    if not isinstance(payload, dict):
        return {"status": "unreadable", "reason": "JSON root is not an object"}
    item_id = str(payload.get("id") or path.stem)
    digest = hashlib.sha256(raw).hexdigest()
    compressed = zlib.compress(raw, level)
    timestamp = str(payload.get("timestamp") or payload.get("start_time") or payload.get("end_time") or "")
    now = datetime.now(timezone.utc).isoformat()
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO experience_archive("
                "kind,item_id,source_path,payload_zlib,source_sha256,raw_size_bytes,"
                "compressed_size_bytes,record_timestamp,archived_at) VALUES(?,?,?,?,?,?,?,?,?)",
                ("experience_event", item_id, str(path), compressed, digest, len(raw), len(compressed), timestamp, now),
            )
        row = conn.execute(
            "SELECT payload_zlib, source_sha256 FROM experience_archive WHERE kind='experience_event' AND item_id=?",
            (item_id,),
        ).fetchone()
        if not row or hashlib.sha256(zlib.decompress(row[0])).hexdigest() != digest:
            return {"status": "verification_failed", "raw_bytes": len(raw)}
        path.unlink()
    except Exception as exc:
        return {"status": "failed", "reason": str(exc), "raw_bytes": len(raw)}
    return {
        "status": "archived",
        "raw_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "saved_bytes": max(0, len(raw) - len(compressed)),
    }


def archive_step(child: str, *, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config if isinstance(config, dict) else load_config()
    policy = archive_policy(cfg)
    state_file = state_path(child, cfg)
    prior: Dict[str, Any] = {}
    try:
        loaded = json.loads(state_file.read_text(encoding="utf-8"))
        prior = loaded if isinstance(loaded, dict) else {}
    except Exception:
        pass
    if not bool(policy.get("enabled", True)):
        result = {"status": "disabled", "updated_at": datetime.now(timezone.utc).isoformat()}
        _atomic_json(state_file, result)
        return result

    events_dir = Path("AI_Children") / child / "memory" / "experiences" / "events"
    db_path = archive_path(child, cfg)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    batch = max(1, min(int(policy.get("batch_files") or 2000), 50_000))
    deadline = time.monotonic() + max(1.0, min(float(policy.get("max_seconds") or 30.0), 120.0))
    cutoff = time.time() - max(0.0, float(policy.get("min_age_hours") or 0.0)) * 3600.0
    level = max(1, min(int(policy.get("compression_level") or 6), 9))
    run = {"scanned": 0, "eligible": 0, "archived": 0, "failed": 0, "raw_bytes": 0, "compressed_bytes": 0, "saved_bytes": 0}
    exhausted = True
    try:
        metadata_before = int(events_dir.stat().st_size)
    except OSError:
        metadata_before = 0
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    try:
        _ensure_schema(conn)
        if events_dir.is_dir():
            with os.scandir(events_dir) as entries:
                for entry in entries:
                    if time.monotonic() >= deadline or run["eligible"] >= batch:
                        exhausted = False
                        break
                    if not entry.name.startswith("evt_") or not entry.name.endswith(".json"):
                        continue
                    run["scanned"] += 1
                    try:
                        if not entry.is_file(follow_symlinks=False) or entry.stat(follow_symlinks=False).st_mtime > cutoff:
                            continue
                    except OSError:
                        run["failed"] += 1
                        continue
                    run["eligible"] += 1
                    outcome = _archive_one(conn, Path(entry.path), level)
                    if outcome.get("status") == "archived":
                        run["archived"] += 1
                        for key in ("raw_bytes", "compressed_bytes", "saved_bytes"):
                            run[key] += int(outcome.get(key) or 0)
                    else:
                        run["failed"] += 1
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(raw_size_bytes),0), COALESCE(SUM(compressed_size_bytes),0) "
            "FROM experience_archive WHERE kind='experience_event'"
        ).fetchone()
    finally:
        conn.close()

    cumulative = dict(prior.get("cumulative") or {})
    for key in ("scanned", "eligible", "archived", "failed", "raw_bytes", "compressed_bytes", "saved_bytes"):
        cumulative[key] = int(cumulative.get(key) or 0) + int(run[key])
    try:
        directory_metadata_bytes = int(events_dir.stat().st_size)
    except OSError:
        directory_metadata_bytes = 0
    history = list(prior.get("history") or []) if isinstance(prior.get("history"), list) else []
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "directory_metadata_before": metadata_before,
        "directory_metadata_after": directory_metadata_bytes,
        "directory_metadata_delta": directory_metadata_bytes - metadata_before,
    })
    history = history[-48:]
    result = {
        "status": "idle" if exhausted and run["eligible"] == 0 else "active",
        "phase": "legacy_flat_events",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "cumulative": cumulative,
        "history": history,
        "archive": {
            "path": str(db_path),
            "records": int(row[0] or 0),
            "raw_bytes": int(row[1] or 0),
            "compressed_bytes": int(row[2] or 0),
        },
        "source": {
            "path": str(events_dir),
            "directory_metadata_bytes": directory_metadata_bytes,
            "directory_metadata_before": metadata_before,
            "directory_metadata_delta": directory_metadata_bytes - metadata_before,
            "exhausted_this_pass": exhausted,
        },
        "note": "Counts are incremental and index-backed; no recursive full-tree count is performed.",
    }
    _atomic_json(state_file, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one bounded experience archive pass.")
    parser.add_argument("--child", default=None)
    args = parser.parse_args()
    cfg = load_config()
    child = str(args.child or cfg.get("current_child") or "Inazuma_Yagami")
    print(json.dumps(archive_step(child, config=cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
