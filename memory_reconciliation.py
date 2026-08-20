"""Bounded, resumable discovery of uncatalogued Ina memory files.

The filesystem remains authoritative for orphan discovery. This worker walks it
in small, interruptible passes while the memory graph consumes verified SQLite
rows. Known verified paths are rejected through SQLite before their source
inode is touched; metadata/hash validation remains a separate audit phase.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple

from io_utils import atomic_write_json
from memory_mirror_db import (
    catalog_path_known,
    flush_mirror_writes,
    mirror_db_path,
    mirror_json_file,
)
from storage_layout import load_config


def _memory_root(child: str) -> Path:
    return Path("AI_Children") / child / "memory"


def _state_path(child: str) -> Path:
    return _memory_root(child) / "reconciliation_state.json"


def _load_state(child: str) -> Dict[str, Any]:
    path = _state_path(child)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(child: str, state: Dict[str, Any]) -> None:
    atomic_write_json(_state_path(child), state, indent=2)


def _reconciliation_policy(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = config.get("memory_reconciliation_policy") if isinstance(config, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    return {
        # Flat legacy directories can contain millions of files. They require an
        # explicit migration/audit pass; walking them in the live cognitive lane
        # can block in the kernel before a Python time budget can be checked.
        "include_legacy_events": bool(raw.get("include_legacy_events", False)),
        # Fragments already have memory_map.json/SQLite as their authoritative
        # catalogue and are mirrored on access. Avoid a second filesystem walk.
        "scan_fragments": bool(raw.get("scan_fragments", False)),
        "max_directory_entries": max(100, int(raw.get("max_directory_entries", 10000) or 10000)),
    }


class ReconciliationDirectoryTooLarge(RuntimeError):
    pass


def _source_specs(
    child: str,
    *,
    include_legacy_events: bool = False,
    scan_fragments: bool = False,
) -> list[Tuple[str, str, Path, bool, Callable[[str], bool]]]:
    memory = _memory_root(child)
    events = memory / "experiences" / "events"
    specs: list[Tuple[str, str, Path, bool, Callable[[str], bool]]] = []
    if include_legacy_events:
        specs.append(("events_legacy", "experience_event", events, False, lambda name: name.startswith("evt_") and name.endswith(".json")))
    specs.extend([
        ("events_time", "experience_event", events / "by_time", True, lambda name: name.startswith("evt_") and name.endswith(".json")),
        ("events_hash", "experience_event", events / "by_hash", True, lambda name: name.endswith(".json")),
        ("episodes", "experience_episode", memory / "experiences" / "episodes", False, lambda name: name.endswith(".json")),
    ])
    fragments = memory / "fragments"
    if scan_fragments:
        matcher = lambda name: name.startswith("frag_") and name.endswith(".json")
        specs.append(("fragments_root", "fragment", fragments, False, matcher))
        for tier in ("short", "working", "long", "cold"):
            specs.append((f"fragments_{tier}", "fragment", fragments / tier, False, matcher))
    return specs


def _bounded_sorted_entries(path: Path, limit: int) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(path) as iterator:
            entries = []
            for entry in iterator:
                entries.append(entry)
                if len(entries) > limit:
                    raise ReconciliationDirectoryTooLarge(f"directory exceeds {limit} direct entries: {path}")
    except FileNotFoundError:
        return []
    return sorted(entries, key=lambda entry: entry.name)


def _iter_source_paths(
    root: Path,
    *,
    recursive: bool,
    matches: Callable[[str], bool],
    resume_after: Optional[str],
    max_directory_entries: int,
) -> Iterator[Path]:
    resume_parts = Path(resume_after).parts if resume_after else ()

    def walk(directory: Path, relative_parts: Tuple[str, ...], on_resume_branch: bool) -> Iterator[Path]:
        target = resume_parts[len(relative_parts)] if on_resume_branch and len(relative_parts) < len(resume_parts) else None
        for entry in _bounded_sorted_entries(directory, max_directory_entries):
            if target is not None and entry.name < target:
                continue
            child_parts = relative_parts + (entry.name,)
            child_on_branch = bool(target is not None and entry.name == target)
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError:
                continue
            if is_dir and recursive:
                yield from walk(Path(entry.path), child_parts, child_on_branch)
            elif is_file and matches(entry.name):
                relative = Path(*child_parts).as_posix()
                if resume_after is None or relative > resume_after:
                    yield Path(entry.path)

    if root.is_dir():
        yield from walk(root, (), bool(resume_parts))


def _legacy_cursor(child: str, last_path: Any, specs: list[Tuple[str, str, Path, bool, Callable[[str], bool]]]) -> Dict[str, str]:
    text = str(last_path or "").strip()
    if not text:
        return {}
    candidate = Path(text)
    for source, _kind, root, _recursive, _matches in specs:
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError:
            continue
        return {"source": source, "relative_path": relative}
    return {}


def _sources(
    child: str,
    *,
    cursor: Optional[Dict[str, Any]] = None,
    include_legacy_events: bool = False,
    scan_fragments: bool = False,
    max_directory_entries: int = 10000,
) -> Iterator[Tuple[str, str, Path]]:
    specs = _source_specs(child, include_legacy_events=include_legacy_events, scan_fragments=scan_fragments)
    cursor = cursor if isinstance(cursor, dict) else {}
    cursor_source = str(cursor.get("source") or "")
    reached_cursor = not cursor_source
    for source, kind, root, recursive, matches in specs:
        if not reached_cursor:
            if source != cursor_source:
                continue
            reached_cursor = True
        resume_after = str(cursor.get("relative_path") or "") if source == cursor_source else None
        for path in _iter_source_paths(
            root,
            recursive=recursive,
            matches=matches,
            resume_after=resume_after or None,
            max_directory_entries=max_directory_entries,
        ):
            yield source, kind, path


def reconcile_step(
    child: str,
    *,
    max_new_records: int = 1000,
    max_seconds: float = 30.0,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Discover and mirror a bounded number of unknown/changed records."""
    cfg = config if isinstance(config, dict) else load_config()
    policy = _reconciliation_policy(cfg)
    started = time.monotonic()
    now_iso = datetime.now(timezone.utc).isoformat()
    prior = _load_state(child)
    generation = int(prior.get("generation") or 0) + (1 if prior.get("completed") else 0)
    if generation <= 0:
        generation = 1

    specs = _source_specs(child, include_legacy_events=policy["include_legacy_events"], scan_fragments=policy["scan_fragments"])
    cursor = {} if prior.get("completed") else (prior.get("cursor") if isinstance(prior.get("cursor"), dict) else {})
    if not cursor:
        cursor = _legacy_cursor(child, prior.get("last_path"), specs)
    stats: Dict[str, Any] = {
        "generation": generation,
        "status": "running",
        "started_at": prior.get("started_at") if not prior.get("completed") else now_iso,
        "updated_at": now_iso,
        "completed": False,
        "paths_seen_this_step": 0,
        "catalogued_this_step": 0,
        "unchanged_this_step": 0,
        "invalid_this_step": 0,
        "last_path": prior.get("last_path"),
        "cursor": cursor or None,
    }

    exhausted = True
    try:
        source_iterator = _sources(child, cursor=cursor, **policy)
        for source, kind, path in source_iterator:
            stats["paths_seen_this_step"] += 1
            if max_seconds > 0 and time.monotonic() - started >= float(max_seconds):
                exhausted = False
                break

            relative = next((path.relative_to(root).as_posix() for spec_source, _kind, root, _recursive, _matches in specs if spec_source == source), path.name)
            stats["last_path"] = str(path)
            stats["cursor"] = {"source": source, "relative_path": relative}
            if catalog_path_known(child, kind, path, config=cfg):
                stats["unchanged_this_step"] += 1
                continue

            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                stats["invalid_this_step"] += 1
                continue
            if not isinstance(payload, dict):
                stats["invalid_this_step"] += 1
                continue

            result = mirror_json_file(child, kind, path, payload=payload, config=cfg)
            if result.get("status") not in {"missing", "invalid_json", "unreadable", "too_large"}:
                stats["catalogued_this_step"] += 1

            if stats["catalogued_this_step"] >= max(1, int(max_new_records)):
                exhausted = False
                break
            if max_seconds > 0 and time.monotonic() - started >= float(max_seconds):
                exhausted = False
                break
    except ReconciliationDirectoryTooLarge as exc:
        exhausted = False
        stats["blocked_reason"] = "directory_too_large"
        stats["blocked_detail"] = str(exc)

    verified = flush_mirror_writes(mirror_db_path(child, cfg))
    stats["verified_this_step"] = verified
    stats["elapsed_seconds"] = round(time.monotonic() - started, 3)
    stats["updated_at"] = datetime.now(timezone.utc).isoformat()
    stats["completed"] = exhausted
    stats["status"] = "completed" if exhausted else "paused"
    if exhausted:
        stats["cursor"] = None

    totals = dict(prior.get("totals") or {}) if not prior.get("completed") else {}
    for key in ("paths_seen", "catalogued", "unchanged", "invalid", "verified"):
        step_key = f"{key}_this_step"
        totals[key] = int(totals.get(key) or 0) + int(stats.get(step_key) or 0)
    stats["totals"] = totals
    _save_state(child, stats)
    return stats


def _main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Incrementally reconcile Ina memory files with SQLite.")
    parser.add_argument("--child", default=None)
    parser.add_argument("--max-new-records", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = load_config()
    child = args.child or str(cfg.get("current_child") or "Inazuma_Yagami")
    if args.status:
        print(json.dumps(_load_state(child), indent=2))
        return 0
    result = reconcile_step(
        child,
        max_new_records=args.max_new_records,
        max_seconds=args.max_seconds,
        config=cfg,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
