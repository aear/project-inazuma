"""Incremental SQLite mirror for JSON memory records.

The mirror is deliberately conservative: a JSON record is only marked eligible
for removal after the SQLite copy has been written and verified. Removing or
quarantining the source JSON stays opt-in through config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
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
}

MIRROR_KINDS = {"fragment", "experience_event", "experience_episode"}
_SESSION_CACHE: set[tuple[str, str, int, int]] = set()


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
    db_path = mirror_db_path(child, cfg)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    verified = False

    conn = sqlite3.connect(str(db_path))
    try:
        _ensure_schema(conn)
        with conn:
            conn.execute(
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
                    str(source),
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
            row = conn.execute(
                "SELECT source_sha256, payload_sha256, payload_json FROM mirrored_json "
                "WHERE kind = ? AND item_id = ? AND source_path = ?",
                (kind, record_id, str(source)),
            ).fetchone()
            verified = bool(row and row[0] == source_hash and row[1] == payload_hash and row[2] == payload_json)
            if verified:
                conn.execute(
                    "UPDATE mirrored_json SET verified_at = ?, removal_eligible = 1 "
                    "WHERE kind = ? AND item_id = ? AND source_path = ?",
                    (now, kind, record_id, str(source)),
                )
    finally:
        conn.close()

    if not verified:
        return {"status": "verification_failed", "path": str(source), "db_path": str(db_path)}

    status = "mirrored"
    should_remove = bool(policy.get("remove_json_after_verified", False)) if remove_original is None else bool(remove_original)
    if should_remove:
        status = _remove_or_quarantine_source(source, db_path, child, kind, record_id, policy)

    _SESSION_CACHE.add(cache_key)
    return {
        "status": status,
        "verified": True,
        "removal_eligible": True,
        "kind": kind,
        "item_id": record_id,
        "path": str(source),
        "db_path": str(db_path),
    }


def mirror_status(child: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    db_path = mirror_db_path(child, config)
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


def _ensure_schema(conn: sqlite3.Connection) -> None:
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
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result.get("verified") or result.get("status") in {"cached", "disabled"} else 1


if __name__ == "__main__":
    raise SystemExit(_main())
