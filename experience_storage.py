"""Filesystem layout helpers for experience event records.

Experience events can grow into millions of tiny JSON files. Keeping them all
in one directory can exhaust filesystem directory indexes, so new event files
are placed under deterministic shards while legacy flat files remain readable.
"""
from __future__ import annotations

import hashlib
import heapq
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


TIME_SHARD_ROOT = "by_time"
HASH_SHARD_ROOT = "by_hash"
_EVENT_TIME_RE = re.compile(r"^evt_(\d{4})(\d{2})(\d{2})T(\d{2})")


def event_filename(event_id: str) -> str:
    clean = str(event_id).strip().replace("\\", "_").replace("/", "_")
    if not clean:
        clean = "evt_unknown"
    if clean.endswith(".json"):
        return clean
    return f"{clean}.json"


def legacy_event_path(events_dir: Path, event_id: str) -> Path:
    return events_dir / event_filename(event_id)


def sharded_event_path(events_dir: Path, event_id: str) -> Path:
    event_id = str(event_id)
    filename = event_filename(event_id)
    match = _EVENT_TIME_RE.match(event_id)
    if match:
        year, month, day, hour = match.groups()
        return events_dir / TIME_SHARD_ROOT / year / month / day / hour / filename

    digest = hashlib.sha256(event_id.encode("utf-8", "replace")).hexdigest()
    return events_dir / HASH_SHARD_ROOT / digest[:2] / digest[2:4] / filename


def candidate_event_paths(events_dir: Path, event_id: str) -> List[Path]:
    sharded = sharded_event_path(events_dir, event_id)
    legacy = legacy_event_path(events_dir, event_id)
    return [sharded, legacy] if sharded != legacy else [legacy]


def resolve_event_path(events_dir: Path, event_id: str) -> Path:
    for path in candidate_event_paths(events_dir, event_id):
        if path.exists():
            return path
    return sharded_event_path(events_dir, event_id)


def iter_event_paths(events_dir: Path, *, include_legacy: bool = True) -> Iterator[Path]:
    if not events_dir.exists():
        return
    if include_legacy:
        yield from _iter_files(events_dir.glob("evt_*.json"))

    time_root = events_dir / TIME_SHARD_ROOT
    if time_root.exists():
        yield from _iter_files(time_root.rglob("evt_*.json"))

    hash_root = events_dir / HASH_SHARD_ROOT
    if hash_root.exists():
        yield from _iter_files(hash_root.rglob("*.json"))


def newest_event_paths(events_dir: Path, limit: int) -> List[Path]:
    try:
        bounded_limit = max(0, int(limit))
    except Exception:
        bounded_limit = 0
    if bounded_limit <= 0:
        return []

    def candidates() -> Iterator[tuple[float, Path]]:
        for path in iter_event_paths(events_dir):
            try:
                yield path.stat().st_mtime, path
            except OSError:
                continue

    return [path for _mtime, path in heapq.nlargest(bounded_limit, candidates(), key=lambda item: item[0])]


def count_flat_event_files(events_dir: Path, *, limit: Optional[int] = None) -> int:
    total = 0
    for _path in _iter_files(events_dir.glob("evt_*.json")):
        total += 1
        if limit is not None and total >= limit:
            break
    return total


def _iter_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_file():
            yield path
