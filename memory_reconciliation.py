"""Bounded, resumable discovery of uncatalogued Ina memory files.

The filesystem remains authoritative for orphan discovery. This worker walks it
in small, interruptible passes while the memory graph consumes verified SQLite
rows. Known verified paths are rejected through SQLite before their source
inode is touched; metadata/hash validation remains a separate audit phase.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from experience_storage import iter_event_paths
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


def _sources(child: str) -> Iterator[Tuple[str, Path]]:
    memory = _memory_root(child)
    events = memory / "experiences" / "events"
    if events.exists():
        for path in iter_event_paths(events):
            yield "experience_event", path
    episodes = memory / "experiences" / "episodes"
    if episodes.exists():
        for path in episodes.glob("*.json"):
            yield "experience_episode", path
    fragments = memory / "fragments"
    if fragments.exists():
        for path in fragments.glob("frag_*.json"):
            yield "fragment", path
        for tier in ("short", "working", "long", "cold"):
            tier_root = fragments / tier
            if tier_root.exists():
                for path in tier_root.glob("frag_*.json"):
                    yield "fragment", path


def reconcile_step(
    child: str,
    *,
    max_new_records: int = 1000,
    max_seconds: float = 30.0,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Discover and mirror a bounded number of unknown/changed records."""
    cfg = config if isinstance(config, dict) else load_config()
    started = time.monotonic()
    now_iso = datetime.now(timezone.utc).isoformat()
    prior = _load_state(child)
    generation = int(prior.get("generation") or 0) + (1 if prior.get("completed") else 0)
    if generation <= 0:
        generation = 1

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
    }

    exhausted = True
    for kind, path in _sources(child):
        stats["paths_seen_this_step"] += 1
        if max_seconds > 0 and time.monotonic() - started >= float(max_seconds):
            exhausted = False
            break

        stats["last_path"] = str(path)
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

    verified = flush_mirror_writes(mirror_db_path(child, cfg))
    stats["verified_this_step"] = verified
    stats["elapsed_seconds"] = round(time.monotonic() - started, 3)
    stats["updated_at"] = datetime.now(timezone.utc).isoformat()
    stats["completed"] = exhausted
    stats["status"] = "completed" if exhausted else "paused"

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
