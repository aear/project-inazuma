"""Bounded retention helpers for Discord runtime buffers and delivery history."""
from __future__ import annotations

import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Iterator

from io_utils import should_fsync


class BoundedIdSet:
    """Insertion-ordered set that forgets the oldest IDs at a fixed bound."""

    def __init__(self, max_entries: int = 10_000, values: Iterable[str] = ()) -> None:
        self.max_entries = max(1, int(max_entries))
        self._values: OrderedDict[str, None] = OrderedDict()
        self.update(values)

    def add(self, value: Any) -> None:
        key = str(value)
        if not key:
            return
        self._values.pop(key, None)
        self._values[key] = None
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)

    def update(self, values: Iterable[Any]) -> None:
        for value in values:
            self.add(value)

    def clear(self) -> None:
        self._values.clear()

    def __contains__(self, value: object) -> bool:
        return str(value) in self._values

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)


def _tail_lines(path: Path, *, max_lines: int, max_tail_bytes: int) -> list[bytes]:
    if not path.exists() or max_lines <= 0 or max_tail_bytes <= 0:
        return []
    size = path.stat().st_size
    start = max(0, size - max_tail_bytes)
    with path.open("rb") as handle:
        handle.seek(start)
        payload = handle.read(max_tail_bytes)
    lines = payload.splitlines(keepends=True)
    if start and lines:
        lines = lines[1:]
    return lines[-max_lines:]


def tail_jsonl_entries(path: Path, *, max_lines: int = 10_000, max_tail_bytes: int = 8 * 1024 * 1024) -> list[dict[str, Any]]:
    """Parse a bounded tail of a JSONL file without scanning its full history."""
    result: list[dict[str, Any]] = []
    for raw in _tail_lines(Path(path), max_lines=max_lines, max_tail_bytes=max_tail_bytes):
        try:
            item = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


def compact_jsonl_tail(path: Path, *, max_bytes: int = 64 * 1024 * 1024, keep_lines: int = 10_000, tail_bytes: int = 8 * 1024 * 1024) -> dict[str, int | bool]:
    """Atomically replace an oversized operational history with its bounded tail."""
    path = Path(path)
    if not path.exists():
        return {"compacted": False, "old_bytes": 0, "new_bytes": 0, "kept_lines": 0}
    old_size = path.stat().st_size
    if old_size <= max(1, int(max_bytes)):
        return {"compacted": False, "old_bytes": old_size, "new_bytes": old_size, "kept_lines": 0}
    lines = _tail_lines(path, max_lines=max(1, int(keep_lines)), max_tail_bytes=max(1024, int(tail_bytes)))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".compact.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            for line in lines:
                handle.write(line if line.endswith(b"\n") else line + b"\n")
            handle.flush()
            if should_fsync(path):
                os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    new_size = path.stat().st_size
    return {"compacted": True, "old_bytes": old_size, "new_bytes": new_size, "kept_lines": len(lines)}


def prune_buffer_files(directory: Path, *, max_files: int = 256, max_bytes: int = 512 * 1024 * 1024, max_age_hours: float = 24.0) -> dict[str, int]:
    """Remove oldest runtime buffer files until age, count and byte bounds hold."""
    directory = Path(directory)
    if not directory.exists():
        return {"removed_files": 0, "removed_bytes": 0, "remaining_files": 0, "remaining_bytes": 0}
    rows = []
    for path in directory.iterdir():
        try:
            stat = path.stat()
        except OSError:
            continue
        if path.is_file():
            rows.append([path, stat.st_mtime, stat.st_size])
    rows.sort(key=lambda row: row[1])
    cutoff = time.time() - max(0.0, float(max_age_hours)) * 3600.0
    total = sum(row[2] for row in rows)
    removed_files = removed_bytes = 0
    while rows and (rows[0][1] < cutoff or len(rows) > max(1, int(max_files)) or total > max(1, int(max_bytes))):
        path, _, size = rows.pop(0)
        try:
            path.unlink()
        except OSError:
            continue
        total -= size
        removed_files += 1
        removed_bytes += size
    return {"removed_files": removed_files, "removed_bytes": removed_bytes, "remaining_files": len(rows), "remaining_bytes": total}


__all__ = ["BoundedIdSet", "compact_jsonl_tail", "prune_buffer_files", "tail_jsonl_entries"]
