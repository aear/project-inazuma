#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from experience_storage import sharded_event_path


READ_CHUNK_SIZE = 1024 * 1024


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
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy Ina cold storage or shard experience event files with checksum verification.")
    parser.add_argument("--child", default=None, help="Child name; defaults to config current_child.")
    parser.add_argument("--apply", action="store_true", help="Copy files. Without this, only plan and count.")
    parser.add_argument("--details", action="store_true", help="Print per-file details instead of only the summary.")
    parser.add_argument("--shard-experience-events", action="store_true", help="Move flat experience event JSON files into deterministic directory shards.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of flat event files considered.")
    parser.add_argument("--keep-legacy", action="store_true", help="Copy into shards but keep the original flat event files.")
    parser.add_argument("--detail-limit", type=int, default=200, help="Maximum per-file entries retained in sharding reports.")
    args = parser.parse_args()

    cfg = _load_config()
    child = args.child or cfg.get("current_child") or "Inazuma_Yagami"
    if args.shard_experience_events:
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
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
