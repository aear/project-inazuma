#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from experience_storage import sharded_event_path


READ_CHUNK_SIZE = 1024 * 1024
MANAGED_COPY_CHUNK_SIZE = 64 * 1024 * 1024


def _load_config() -> Dict[str, Any]:
    path = Path("config.json")
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _format_child_path(value: str, child: str) -> str:
    return value.replace("{child}", child)


def _hash_file(path: Path) -> Optional[str]:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(READ_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    tmp_path.replace(path)


def migration_history_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "storage_migration_history.jsonl"


def _record_migration_summary(child: Optional[str], operation: str, report: Dict[str, Any]) -> None:
    if not child or not bool(report.get("apply")):
        return
    summary = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "child": str(child),
        "operation": str(operation),
        "status": str(report.get("status") or "unknown"),
        "failed": int(report.get("failed") or 0),
        "conflicts": int(report.get("conflicts") or 0),
        "verified": int(report.get("verified") or 0),
        "copied": int(report.get("copied") or 0),
        "moved": int(report.get("moved") or 0),
        "bytes": int(report.get("bytes") or 0),
        "cutover": bool(report.get("cutover", False)),
        "rolled_back": bool(report.get("rolled_back", False)),
        "manifest": str(report.get("manifest") or "") or None,
    }
    path = migration_history_path(str(child))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=True) + "\n")
    except OSError:
        pass



def migrate_tree_and_link(source: Path, target: Path, *, apply: bool, relative_link: bool = True, child: Optional[str] = None) -> Dict[str, Any]:
    """Checksum-copy a tree, then retain its old name as a compatibility link."""
    source, target = Path(source).expanduser(), Path(target).expanduser()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = source.with_name(f"{source.name}.ina-migration-backup-{stamp}")
    report: Dict[str, Any] = {"source": str(source), "target": str(target), "backup": str(backup), "apply": apply, "files": [], "copied": 0, "verified": 0, "failed": 0, "bytes": 0, "cutover": False}
    if source.is_symlink():
        report.update(status="already_linked", link_target=os.readlink(source)); return report
    if not source.exists(): report["status"] = "missing_source"; return report
    if source.is_file():
        return migrate_file_and_link(source, target, apply=apply, relative_link=relative_link, child=child)
    if target.exists() and not target.is_dir(): report["status"] = "target_not_directory"; return report
    try:
        target.resolve().relative_to(source.resolve())
        report["status"] = "source_contains_target"; return report
    except ValueError: pass
    for src in sorted(source.rglob("*")):
        rel, dst = src.relative_to(source), target / src.relative_to(source)
        if src.is_symlink():
            item = {"rel_path": rel.as_posix(), "source": str(src), "target": str(dst), "kind": "symlink", "link_target": os.readlink(src)}
            try:
                if apply:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.is_symlink() and os.readlink(dst) == item["link_target"]:
                        item["status"] = "verified"
                    elif dst.exists() or dst.is_symlink():
                        raise OSError("target conflicts with source symlink")
                    else:
                        dst.symlink_to(item["link_target"], target_is_directory=src.is_dir())
                        item["status"] = "verified"
                    report["verified"] += 1
                else: item["status"] = "planned"
            except OSError as exc:
                item.update(status="failed", error=str(exc)); report["failed"] += 1
            report["files"].append(item); continue
        if src.is_dir():
            if apply: dst.mkdir(parents=True, exist_ok=True)
            continue
        if not src.is_file(): continue
        item = {"rel_path": rel.as_posix(), "source": str(src), "target": str(dst), "kind": "file"}
        try:
            item["size"] = src.stat().st_size; report["bytes"] += item["size"]
            source_hash = _hash_file(src); item["sha256"] = source_hash
            if not source_hash: raise OSError("source checksum failed")
            if apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists() or _hash_file(dst) != source_hash:
                    shutil.copy2(src, dst); report["copied"] += 1
                if _hash_file(dst) != source_hash: raise OSError("target checksum failed")
                report["verified"] += 1; item["status"] = "verified"
            else: item["status"] = "planned"
        except OSError as exc:
            item.update(status="failed", error=str(exc)); report["failed"] += 1
        report["files"].append(item)
    if apply and report["failed"] == 0:
        target.mkdir(parents=True, exist_ok=True)
        try:
            source.rename(backup)
            link_value = os.path.relpath(target, source.parent) if relative_link else str(target.resolve())
            pending = source.with_name(f".{source.name}.ina-link-{stamp}")
            pending.symlink_to(link_value, target_is_directory=True); pending.replace(source)
            report.update(cutover=True, link_target=link_value)
        except Exception as exc:
            report.update(failed=report["failed"] + 1, cutover_error=str(exc))
            try:
                if source.is_symlink(): source.unlink()
                if backup.exists() and not source.exists(): backup.rename(source)
                report["rolled_back"] = True
            except Exception as rollback_exc: report["rollback_error"] = str(rollback_exc)
    report["status"] = "ok" if report["failed"] == 0 else "failed"
    if apply:
        manifest = target / "migration_manifests" / f"tree_migration_{stamp}.json"
        _atomic_write_json(manifest, report); report["manifest"] = str(manifest)
        _record_migration_summary(child, "tree_migration", report)
    return report
def migrate_file_and_link(
    source: Path,
    target: Path,
    *,
    apply: bool,
    relative_link: bool = True,
    child: Optional[str] = None,
) -> Dict[str, Any]:
    """Checksum-copy one file, retain a backup, then link its original path."""
    source, target = Path(source).expanduser(), Path(target).expanduser()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = source.with_name(f"{source.name}.ina-migration-backup-{stamp}")
    report: Dict[str, Any] = {
        "source": str(source),
        "target": str(target),
        "backup": str(backup),
        "apply": apply,
        "kind": "file",
        "copied": 0,
        "verified": 0,
        "failed": 0,
        "bytes": 0,
        "cutover": False,
    }
    if source.is_symlink():
        report.update(status="already_linked", link_target=os.readlink(source))
        return report
    if not source.is_file():
        report["status"] = "missing_source" if not source.exists() else "source_not_file"
        return report
    if target.exists() and not target.is_file():
        report["status"] = "target_not_file"
        return report

    source_hash = _hash_file(source)
    if not source_hash:
        report.update(status="failed", failed=1, error="source_checksum_failed")
        return report
    report.update(bytes=int(source.stat().st_size), sha256=source_hash)
    if not apply:
        report["status"] = "ok"
        return report

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.ina-copy-{stamp}")
    try:
        if target.exists():
            if _hash_file(target) != source_hash:
                raise OSError("target conflicts with source file")
        else:
            shutil.copy2(source, partial)
            if _hash_file(partial) != source_hash:
                raise OSError("target checksum failed")
            os.replace(partial, target)
            report["copied"] = 1
        if _hash_file(target) != source_hash:
            raise OSError("target verification failed")
        report["verified"] = 1

        source.rename(backup)
        link_value = os.path.relpath(target, source.parent) if relative_link else str(target.resolve())
        pending = source.with_name(f".{source.name}.ina-link-{stamp}")
        pending.symlink_to(link_value, target_is_directory=False)
        pending.replace(source)
        report.update(cutover=True, link_target=link_value, status="ok")
    except Exception as exc:
        report.update(status="failed", failed=1, error=str(exc))
        try:
            if partial.exists():
                partial.unlink()
            if source.is_symlink():
                source.unlink()
            if backup.exists() and not source.exists():
                backup.rename(source)
            report["rolled_back"] = True
        except Exception as rollback_exc:
            report["rollback_error"] = str(rollback_exc)

    manifest = target.parent / "migration_manifests" / f"file_migration_{stamp}.json"
    _atomic_write_json(manifest, report)
    report["manifest"] = str(manifest)
    _record_migration_summary(child, "file_migration", report)
    return report



def _resolve_cold_storage_paths(child: str, cfg: Dict[str, Any]) -> tuple[Path, Path]:
    source = Path("AI_Children") / child / "memory" / "cold_storage"
    layout = cfg.get("storage_layout") if isinstance(cfg, dict) else {}
    cold_policy = cfg.get("cold_storage_policy") if isinstance(cfg, dict) else {}
    target_raw = None
    if isinstance(cold_policy, dict):
        target_raw = cold_policy.get("storage_root")
    if not target_raw and isinstance(layout, dict):
        target_raw = layout.get("cold_storage_root")
    if not isinstance(target_raw, str) or not target_raw.strip():
        raise SystemExit("No cold_storage_policy.storage_root or storage_layout.cold_storage_root configured.")
    target = Path(_format_child_path(target_raw.strip(), child)).expanduser()
    return source, target


def copy_and_verify_cold_storage(child: str, *, apply: bool) -> Dict[str, Any]:
    cfg = _load_config()
    source, target = _resolve_cold_storage_paths(child, cfg)
    report: Dict[str, Any] = {
        "child": child,
        "source": str(source),
        "target": str(target),
        "apply": apply,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "copied": 0,
        "verified": 0,
        "skipped_existing": 0,
        "failed": 0,
        "bytes": 0,
    }
    if not source.exists():
        report["status"] = "missing_source"
        return report

    for src in _iter_files(source):
        rel = src.relative_to(source)
        dst = target / rel
        item: Dict[str, Any] = {"rel_path": rel.as_posix(), "source": str(src), "target": str(dst)}
        try:
            item["size"] = src.stat().st_size
            report["bytes"] += int(item["size"])
        except OSError:
            item["status"] = "failed_stat"
            report["failed"] += 1
            report["files"].append(item)
            continue

        source_hash = _hash_file(src)
        if not source_hash:
            item["status"] = "failed_hash_source"
            report["failed"] += 1
            report["files"].append(item)
            continue
        item["sha256"] = source_hash

        if dst.exists() and _hash_file(dst) == source_hash:
            item["status"] = "already_verified"
            report["skipped_existing"] += 1
            report["verified"] += 1
            report["files"].append(item)
            continue

        if apply:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            except Exception as exc:
                item["status"] = "copy_failed"
                item["error"] = str(exc)
                report["failed"] += 1
                report["files"].append(item)
                continue
            report["copied"] += 1
            if _hash_file(dst) == source_hash:
                item["status"] = "copied_verified"
                report["verified"] += 1
            else:
                item["status"] = "copy_verification_failed"
                report["failed"] += 1
        else:
            item["status"] = "planned"
        report["files"].append(item)

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "ok" if report["failed"] == 0 else "failed"
    manifest = target / "migration_manifests" / f"cold_storage_{report['finished_at'].replace(':', '').replace('+', 'Z')}.json"
    if apply:
        _atomic_write_json(manifest, report)
        report["manifest"] = str(manifest)
        _record_migration_summary(child, "cold_storage_copy", report)
    return report


def shard_experience_events(
    child: str,
    *,
    apply: bool,
    limit: Optional[int] = None,
    keep_legacy: bool = False,
    detail_limit: int = 200,
) -> Dict[str, Any]:
    events_dir = Path("AI_Children") / child / "memory" / "experiences" / "events"
    report: Dict[str, Any] = {
        "child": child,
        "events_dir": str(events_dir),
        "apply": apply,
        "keep_legacy": keep_legacy,
        "limit": limit,
        "detail_limit": detail_limit,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
        "file_details_truncated": False,
        "planned": 0,
        "moved": 0,
        "copied": 0,
        "removed_legacy_duplicates": 0,
        "already_sharded": 0,
        "conflicts": 0,
        "failed": 0,
        "bytes": 0,
    }
    if not events_dir.exists():
        report["status"] = "missing_source"
        return report

    try:
        max_details = max(0, int(detail_limit))
    except Exception:
        max_details = 200

    def _record_item(item: Dict[str, Any]) -> None:
        if len(report["files"]) < max_details:
            report["files"].append(item)
        else:
            report["file_details_truncated"] = True

    try:
        max_files = None if limit is None else max(0, int(limit))
    except Exception:
        max_files = None

    seen = 0
    for src in events_dir.glob("evt_*.json"):
        if not src.is_file():
            continue
        if max_files is not None and seen >= max_files:
            break
        seen += 1
        dst = sharded_event_path(events_dir, src.stem)
        if src == dst:
            continue

        item: Dict[str, Any] = {
            "event_id": src.stem,
            "source": str(src),
            "target": str(dst),
        }
        try:
            item["size"] = src.stat().st_size
            report["bytes"] += int(item["size"])
        except OSError as exc:
            item["status"] = "failed_stat"
            item["error"] = str(exc)
            report["failed"] += 1
            _record_item(item)
            continue

        source_hash = _hash_file(src)
        if not source_hash:
            item["status"] = "failed_hash_source"
            report["failed"] += 1
            _record_item(item)
            continue
        item["sha256"] = source_hash

        if dst.exists():
            target_hash = _hash_file(dst)
            if target_hash == source_hash:
                report["already_sharded"] += 1
                item["status"] = "already_sharded"
                if apply and not keep_legacy:
                    try:
                        src.unlink()
                        item["status"] = "legacy_duplicate_removed"
                        report["removed_legacy_duplicates"] += 1
                    except OSError as exc:
                        item["status"] = "failed_remove_legacy_duplicate"
                        item["error"] = str(exc)
                        report["failed"] += 1
                _record_item(item)
                continue
            item["status"] = "conflict_existing_target"
            item["target_sha256"] = target_hash
            report["conflicts"] += 1
            _record_item(item)
            continue

        report["planned"] += 1
        if apply:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if keep_legacy:
                    shutil.copy2(src, dst)
                    report["copied"] += 1
                else:
                    src.replace(dst)
                    report["moved"] += 1
            except Exception as exc:
                item["status"] = "move_failed"
                item["error"] = str(exc)
                report["failed"] += 1
                _record_item(item)
                continue
            if _hash_file(dst) == source_hash:
                item["status"] = "copied_verified" if keep_legacy else "moved_verified"
            else:
                item["status"] = "verification_failed"
                report["failed"] += 1
        else:
            item["status"] = "planned"
        _record_item(item)

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "ok" if report["failed"] == 0 and report["conflicts"] == 0 else "failed"
    if apply:
        manifest = (
            events_dir.parent
            / "migration_manifests"
            / f"experience_event_shards_{report['finished_at'].replace(':', '').replace('+', 'Z')}.json"
        )
        _atomic_write_json(manifest, report)
        report["manifest"] = str(manifest)
        _record_migration_summary(child, "experience_event_sharding", report)
    return report


def managed_migration_request_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "storage_migration_request.json"


def managed_migration_state_path(child: str) -> Path:
    return Path("AI_Children") / child / "memory" / "storage_migration_state.json"


def _load_json_dict(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def managed_migration_active(child: str) -> bool:
    request = _load_json_dict(managed_migration_request_path(child))
    return str(request.get("status") or "").lower() in {"requested", "copying", "verifying"}


def _sqlite_quick_check(path: Path) -> tuple[bool, str]:
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        try:
            row = conn.execute("PRAGMA quick_check").fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return False, str(exc)
    result = str(row[0] if row else "missing_result")
    return result.lower() == "ok", result


def _activate_mirror_target(config_path: Path, target: Path) -> None:
    config = _load_json_dict(config_path)
    policy = config.get("memory_mirror_policy")
    policy = dict(policy) if isinstance(policy, dict) else {}
    policy["db_root"] = str(target.parent)
    policy["db_filename"] = target.name
    config["memory_mirror_policy"] = policy
    _atomic_write_json(config_path, config)


def managed_migration_step(child: str, *, chunk_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Advance an Ina-owned storage migration by one verified copy chunk."""
    request_path = managed_migration_request_path(child)
    state_path = managed_migration_state_path(child)
    request = _load_json_dict(request_path)
    status = str(request.get("status") or "").lower()
    if status not in {"requested", "copying", "verifying"}:
        return {"status": "idle", "request_status": status or "missing"}

    def fail(error: str, **extra: Any) -> Dict[str, Any]:
        failed_at = datetime.now(timezone.utc).isoformat()
        payload = _load_json_dict(state_path)
        payload.update({"status": "failed", "error": str(error), "updated_at": failed_at, **extra})
        request.update({"status": "failed", "error": str(error), "updated_at": failed_at})
        _atomic_write_json(state_path, payload)
        _atomic_write_json(request_path, request)
        return payload
    if str(request.get("operation") or "") != "promote_mirror_database":
        return fail("unsupported_operation")

    source = Path(str(request.get("source") or "")).expanduser()
    target = Path(str(request.get("target") or "")).expanduser()
    partial = target.with_suffix(target.suffix + ".partial")
    if not source.is_file() or source.is_symlink():
        return fail("source_unavailable", source=str(source))
    if source.resolve() == target.resolve():
        return fail("source_equals_target")

    source_stat = source.stat()
    state = _load_json_dict(state_path)
    expected_size = int(state.get("source_size") or source_stat.st_size)
    expected_mtime_ns = int(state.get("source_mtime_ns") or source_stat.st_mtime_ns)
    if source_stat.st_size != expected_size or source_stat.st_mtime_ns != expected_mtime_ns:
        state.update({"status": "failed", "error": "source_changed_during_migration", "updated_at": datetime.now(timezone.utc).isoformat()})
        _atomic_write_json(state_path, state)
        request.update({"status": "failed", "error": state["error"], "updated_at": state["updated_at"]})
        _atomic_write_json(request_path, request)
        return state

    target.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        return fail("partial_larger_than_source", offset=offset)
    bounded_chunk = max(1024 * 1024, int(chunk_bytes or request.get("chunk_bytes") or MANAGED_COPY_CHUNK_SIZE))
    copied = 0
    if offset < expected_size:
        with source.open("rb") as source_handle:
            source_handle.seek(offset)
            data = source_handle.read(min(bounded_chunk, expected_size - offset))
        mode = "r+b" if partial.exists() else "wb"
        with partial.open(mode) as target_handle:
            target_handle.seek(offset)
            target_handle.write(data)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        with partial.open("rb") as verify_handle:
            verify_handle.seek(offset)
            copied_data = verify_handle.read(len(data))
        if hashlib.sha256(copied_data).digest() != hashlib.sha256(data).digest():
            return fail("chunk_verification_failed", offset=offset)
        copied = len(data)
        offset += copied

    now = datetime.now(timezone.utc).isoformat()
    state.update({
        "child": child,
        "operation": "promote_mirror_database",
        "source": str(source),
        "target": str(target),
        "partial": str(partial),
        "source_size": expected_size,
        "source_mtime_ns": expected_mtime_ns,
        "bytes_copied": offset,
        "progress": round(offset / max(1, expected_size), 6),
        "last_chunk_bytes": copied,
        "status": "copying" if offset < expected_size else "verifying",
        "updated_at": now,
    })
    request.update({"status": state["status"], "updated_at": now})
    _atomic_write_json(state_path, state)
    _atomic_write_json(request_path, request)
    if offset < expected_size:
        return state

    ok, detail = _sqlite_quick_check(partial)
    if not ok:
        state.update({"status": "failed", "error": "sqlite_quick_check_failed", "verification": detail, "updated_at": datetime.now(timezone.utc).isoformat()})
        request.update({"status": "failed", "error": state["error"], "updated_at": state["updated_at"]})
        _atomic_write_json(state_path, state)
        _atomic_write_json(request_path, request)
        return state

    os.replace(partial, target)
    _activate_mirror_target(Path("config.json"), target)
    completed_at = datetime.now(timezone.utc).isoformat()
    state.update({"status": "complete", "verification": detail, "completed_at": completed_at, "updated_at": completed_at, "source_retained": True})
    request.update({"status": "complete", "completed_at": completed_at, "updated_at": completed_at, "source_retained": True})
    _atomic_write_json(state_path, state)
    _atomic_write_json(request_path, request)
    _record_migration_summary(child, "managed_mirror_database_promotion", {
        "apply": True, "status": "ok", "verified": 1, "copied": 1, "bytes": expected_size, "cutover": True,
    })
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy Ina cold storage or shard experience event files with checksum verification.")
    parser.add_argument("--child", default=None, help="Child name; defaults to config current_child.")
    parser.add_argument("--apply", action="store_true", help="Copy files. Without this, only plan and count.")
    parser.add_argument("--details", action="store_true", help="Print per-file details instead of only the summary.")
    parser.add_argument("--shard-experience-events", action="store_true", help="Move flat experience event JSON files into deterministic directory shards.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of flat event files considered.")
    parser.add_argument("--keep-legacy", action="store_true", help="Copy into shards but keep the original flat event files.")
    parser.add_argument("--detail-limit", type=int, default=200, help="Maximum per-file entries retained in sharding reports.")
    parser.add_argument("--migrate-source", type=Path, help="File or tree to move while preserving this path as a symlink.")
    parser.add_argument("--migrate-target", type=Path, help="Destination path on an NVMe or new-system mount.")
    parser.add_argument("--absolute-link", action="store_true", help="Use an absolute compatibility link; relative is portable by default.")
    parser.add_argument("--resume-managed", action="store_true", help="Advance Ina owned managed migration by one verified chunk.")
    args = parser.parse_args()

    cfg = _load_config()
    child = args.child or cfg.get("current_child") or "Inazuma_Yagami"
    if bool(args.migrate_source) != bool(args.migrate_target):
        parser.error("--migrate-source and --migrate-target must be supplied together")
    if args.resume_managed:
        report = managed_migration_step(str(child))
    elif args.migrate_source:
        report = migrate_tree_and_link(args.migrate_source, args.migrate_target, apply=bool(args.apply), relative_link=not args.absolute_link, child=str(child))
    elif args.shard_experience_events:
        report = shard_experience_events(
            str(child),
            apply=bool(args.apply),
            limit=args.limit,
            keep_legacy=bool(args.keep_legacy),
            detail_limit=args.detail_limit,
        )
    else:
        report = copy_and_verify_cold_storage(str(child), apply=bool(args.apply))
    if args.details:
        print_payload = report
    else:
        print_payload = {key: value for key, value in report.items() if key != "files"}
        if "file_details_truncated" in report:
            print_payload["file_detail_count"] = len(report.get("files", []))
        else:
            print_payload["file_count"] = len(report.get("files", []))
    print(json.dumps(print_payload, indent=2, ensure_ascii=True))
    return 0 if report.get("status") in {"ok", "idle", "copying", "verifying", "complete"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
