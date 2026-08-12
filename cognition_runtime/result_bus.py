"""Bounded structured result bus with preserved provenance."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from itertools import islice
from threading import Condition, RLock
from typing import Any, Iterable, Mapping
import uuid

from .contracts import Contribution


def _inline_size(value: Any, limit: int) -> int:
    """Estimate reachable inline payload without serialising or copying it."""
    total = 0
    seen = set()
    stack = [value]
    while stack and total <= limit:
        item = stack.pop()
        marker = id(item)
        if marker in seen:
            continue
        seen.add(marker)
        if item is None or isinstance(item, (bool, int, float)):
            total += 16
        elif isinstance(item, str):
            total += len(item.encode("utf-8", errors="replace"))
        elif isinstance(item, (bytes, bytearray, memoryview)):
            total += len(item)
        elif isinstance(item, Mapping):
            total += len(item) * 16
            remaining = max(0, (limit - total) // 16 + 1)
            for key, nested in islice(item.items(), min(65536, remaining)):
                stack.append(key)
                stack.append(nested)
        elif isinstance(item, (list, tuple, set, frozenset, deque)):
            total += len(item) * 8
            remaining = max(0, (limit - total) // 8 + 1)
            stack.extend(islice(item, min(65536, remaining)))
        else:
            nbytes = getattr(item, "nbytes", None)
            if isinstance(nbytes, int):
                total += max(0, nbytes)
            else:
                total += 256
    return total


class ResultBus:
    def __init__(self, max_contributions: int = 1024, max_inline_bytes: int = 8 * 1024 * 1024) -> None:
        self.max_contributions = max(1, int(max_contributions))
        self.max_inline_bytes = max(1024, int(max_inline_bytes))
        self._items: deque[Contribution] = deque(maxlen=self.max_contributions)
        self._lock = RLock()
        self._changed = Condition(self._lock)

    def publish(self, contribution: Contribution) -> Contribution:
        inline = (contribution.value, contribution.cost, contribution.metadata, contribution.provenance)
        if _inline_size(inline, self.max_inline_bytes) > self.max_inline_bytes:
            raise ValueError(
                f"contribution exceeds {self.max_inline_bytes} inline bytes; publish a bounded result and durable reference"
            )
        item = contribution if contribution.contribution_id else replace(contribution, contribution_id=uuid.uuid4().hex)
        with self._changed:
            self._items.append(item)
            self._changed.notify_all()
        return item

    def snapshot(self, *, capabilities: Iterable[str] = (), context_id: str | None = None) -> tuple[Contribution, ...]:
        wanted = {str(item) for item in capabilities}
        with self._lock:
            items = tuple(self._items)
        if wanted:
            items = tuple(item for item in items if item.capability in wanted)
        if context_id is not None:
            items = tuple(item for item in items if item.metadata.get("context_id") == context_id)
        return items

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
