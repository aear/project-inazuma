"""Losslessly pack legacy flat live-media files into a verified SQLite archive."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from storage_layout import load_config


def _paths(child: str, config: Dict[str, Any]) -> tuple[Path, Path]:
    policy = config.get("experience_archive_policy") if isinstance(config, dict) else {}
    policy = policy if isinstance(policy, dict) else {}
    raw_archive = policy.get("media_archive_path")
    if raw_archive:
        archive = Path(str(raw_archive).format(child=child)).expanduser()
    else:
        archive = Path("AI_Children") / child / "memory" / "cold_storage" / "experiences" / "live_media_archive.sqlite3"
    raw_state = policy.get("media_state_path") or "AI_Children/{child}/memory/experience_media_archive_state.json"
    return archive, Path(str(raw_state).format(child=child)).expanduser()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS media_archive ("
        "filename TEXT PRIMARY KEY, event_id TEXT NOT NULL, source_path TEXT NOT NULL, "
        "payload_blob BLOB NOT NULL, encoding TEXT NOT NULL, source_sha256 TEXT NOT NULL, "
        "raw_size_bytes INTEGER NOT NULL, stored_size_bytes INTEGER NOT NULL, "
        "source_mtime_ns INTEGER NOT NULL, archived_at TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_archive_event ON media_archive(event_id)")


def load_archived_media(child: str, filename: str, *, config: Optional[Dict[str, Any]] = None) -> Optional[bytes]:
    cfg = config if isinstance(config, dict) else load_config()
    archive, _state = _paths(child, cfg)
    if not archive.exists():
        return None
    conn = sqlite3.connect(f"file:{archive}?mode=ro", uri=True, timeout=2.0)
    try:
        row = conn.execute(
            "SELECT payload_blob, encoding, source_sha256 FROM media_archive WHERE filename=?",
            (str(filename),),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    try:
        raw = zlib.decompress(row[0]) if row[1] == "zlib" else bytes(row[0])
    except Exception:
        return None
    return raw if hashlib.sha256(raw).hexdigest() == str(row[2]) else None


def _archive_one(conn: sqlite3.Connection, path: Path, level: int) -> Dict[str, Any]:
    try:
        stat = path.stat()
        raw = path.read_bytes()
    except Exception as exc:
        return {"status": "unreadable", "reason": str(exc)}
    digest = hashlib.sha256(raw).hexdigest()
    candidate = zlib.compress(raw, level)
    stored, encoding = (candidate, "zlib") if len(candidate) < len(raw) else (raw, "raw")
    event_id = path.stem
    for marker in ("_screen", "_dialogue", "_motor"):
        if marker in event_id:
            event_id = event_id.split(marker, 1)[0]
            break
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO media_archive("
                "filename,event_id,source_path,payload_blob,encoding,source_sha256,"
                "raw_size_bytes,stored_size_bytes,source_mtime_ns,archived_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (path.name, event_id, str(path), stored, encoding, digest, len(raw), len(stored), int(stat.st_mtime_ns), datetime.now(timezone.utc).isoformat()),
            )
        row = conn.execute(
            "SELECT payload_blob, encoding, source_sha256 FROM media_archive WHERE filename=?",
            (path.name,),
        ).fetchone()
        restored = zlib.decompress(row[0]) if row and row[1] == "zlib" else bytes(row[0]) if row else b""
        if not row or hashlib.sha256(restored).hexdigest() != digest:
            return {"status": "verification_failed", "raw_bytes": len(raw)}
        path.unlink()
    except Exception as exc:
        return {"status": "failed", "reason": str(exc), "raw_bytes": len(raw)}
    return {"status": "archived", "raw_bytes": len(raw), "stored_bytes": len(stored), "saved_bytes": max(0, len(raw) - len(stored))}


def media_archive_step(child: str, *, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config if isinstance(config, dict) else load_config()
    policy = cfg.get("experience_archive_policy") if isinstance(cfg, dict) else {}
    policy = policy if isinstance(policy, dict) else {}
    media_dir = Path("AI_Children") / child / "memory" / "experiences" / "live_media"
    archive, state_path = _paths(child, cfg)
    prior: Dict[str, Any] = {}
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
        prior = value if isinstance(value, dict) else {}
    except Exception:
        pass
    archive.parent.mkdir(parents=True, exist_ok=True)
    batch = max(1, min(int(policy.get("batch_files") or 2000) // 2, 25_000))
    deadline = time.monotonic() + max(1.0, min(float(policy.get("max_seconds") or 30.0), 120.0))
    cutoff = time.time() - max(0.0, float(policy.get("min_age_hours") or 168.0)) * 3600.0
    level = max(1, min(int(policy.get("compression_level") or 6), 9))
    run = {"scanned": 0, "eligible": 0, "archived": 0, "failed": 0, "raw_bytes": 0, "stored_bytes": 0, "saved_bytes": 0}
    try:
        metadata_before = int(media_dir.stat().st_size)
    except OSError:
        metadata_before = 0
    exhausted = True
    conn = sqlite3.connect(str(archive), timeout=10.0)
    try:
        _ensure_schema(conn)
        if media_dir.is_dir():
            with os.scandir(media_dir) as entries:
                for entry in entries:
                    if time.monotonic() >= deadline or run["eligible"] >= batch:
                        exhausted = False
                        break
                    if not entry.name.startswith("evt_"):
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
                        for key in ("raw_bytes", "stored_bytes", "saved_bytes"):
                            run[key] += int(outcome.get(key) or 0)
                    else:
                        run["failed"] += 1
        row = conn.execute("SELECT COUNT(*),COALESCE(SUM(raw_size_bytes),0),COALESCE(SUM(stored_size_bytes),0) FROM media_archive").fetchone()
    finally:
        conn.close()
    try:
        metadata_after = int(media_dir.stat().st_size)
    except OSError:
        metadata_after = 0
    cumulative = dict(prior.get("cumulative") or {})
    for key, value in run.items():
        cumulative[key] = int(cumulative.get(key) or 0) + int(value)
    history = list(prior.get("history") or []) if isinstance(prior.get("history"), list) else []
    history.append({"timestamp": datetime.now(timezone.utc).isoformat(), "run": run, "directory_metadata_before": metadata_before, "directory_metadata_after": metadata_after, "directory_metadata_delta": metadata_after - metadata_before})
    result = {
        "status": "idle" if exhausted and run["eligible"] == 0 else "active",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "cumulative": cumulative,
        "history": history[-48:],
        "archive": {"path": str(archive), "records": int(row[0] or 0), "raw_bytes": int(row[1] or 0), "stored_bytes": int(row[2] or 0)},
        "source": {"path": str(media_dir), "directory_metadata_bytes": metadata_after, "directory_metadata_before": metadata_before, "directory_metadata_delta": metadata_after - metadata_before, "exhausted_this_pass": exhausted},
        "note": "Already-compressed media stays raw; compressible metadata/arrays use zlib. All bytes are checksum-verified before source retirement.",
    }
    _atomic_json(state_path, result)
    return result
