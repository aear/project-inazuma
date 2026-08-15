"""Bounded-memory readers for selected fields in large pretty or compact JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional, Set, TextIO


class _Reader:
    def __init__(self, handle: TextIO) -> None:
        self.handle = handle
        self.pushed: Optional[str] = None

    def get(self) -> str:
        if self.pushed is not None:
            value, self.pushed = self.pushed, None
            return value
        return self.handle.read(1)

    def push(self, value: str) -> None:
        self.pushed = value

    def nonspace(self) -> str:
        value = self.get()
        while value and value.isspace():
            value = self.get()
        return value

    def value_text(self, first: str, *, capture: bool) -> str:
        output = [first] if capture else []
        if first == '"':
            escaped = False
            while True:
                char = self.get()
                if not char:
                    raise ValueError("unterminated JSON string")
                if capture:
                    output.append(char)
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    break
            return "".join(output)
        if first in "[{":
            stack = [first]
            in_string = False
            escaped = False
            pairs = {"]": "[", "}": "{"}
            while stack:
                char = self.get()
                if not char:
                    raise ValueError("unterminated JSON container")
                if capture:
                    output.append(char)
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char in "[{":
                    stack.append(char)
                elif char in "]}":
                    if not stack or stack.pop() != pairs[char]:
                        raise ValueError("malformed JSON container")
            return "".join(output)
        while True:
            char = self.get()
            if not char or char in ",}]":
                if char:
                    self.push(char)
                return "".join(output).strip()
            if capture:
                output.append(char)


def _read_string(reader: _Reader, first: Optional[str] = None) -> str:
    first = reader.nonspace() if first is None else first
    if first != '"':
        raise ValueError("expected JSON string")
    return str(json.loads(reader.value_text(first, capture=True)))


def _read_selected_object(reader: _Reader, fields: Set[str]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    first = reader.nonspace()
    if first != "{":
        raise ValueError("expected JSON object")
    while True:
        token = reader.nonspace()
        if token == "}":
            return selected
        key = _read_string(reader, token)
        if reader.nonspace() != ":":
            raise ValueError("expected ':'")
        start = reader.nonspace()
        capture = key in fields
        raw = reader.value_text(start, capture=capture)
        if capture:
            selected[key] = json.loads(raw)
        separator = reader.nonspace()
        if separator == "}":
            return selected
        if separator != ",":
            raise ValueError("expected object separator")


def iter_selected_array_objects(
    path: Path,
    array_key: str,
    fields: Set[str],
    *,
    limit: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    """Yield selected direct fields from objects in one top-level array.

    Unselected values are scanned but never retained, so a multi-megabyte
    ``components`` list does not become a multi-megabyte Python object.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = _Reader(handle)
        if reader.nonspace() != "{":
            raise ValueError("expected top-level JSON object")
        while True:
            token = reader.nonspace()
            if token == "}":
                return
            key = _read_string(reader, token)
            if reader.nonspace() != ":":
                raise ValueError("expected ':'")
            start = reader.nonspace()
            if key != array_key:
                reader.value_text(start, capture=False)
            else:
                if start != "[":
                    raise ValueError(f"{array_key!r} is not an array")
                count = 0
                while True:
                    item = reader.nonspace()
                    if item == "]":
                        return
                    reader.push(item)
                    yield _read_selected_object(reader, fields)
                    count += 1
                    if limit is not None and count >= max(0, int(limit)):
                        return
                    separator = reader.nonspace()
                    if separator == "]":
                        return
                    if separator != ",":
                        raise ValueError("expected array separator")
            separator = reader.nonspace()
            if separator == "}":
                return
            if separator != ",":
                raise ValueError("expected top-level separator")


def iter_selected_object_entries(
    path: Path,
    object_key: str,
    fields: Set[str],
    *,
    limit: Optional[int] = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield keys and selected fields from one top-level object value.

    This is the object-map counterpart to :func:`iter_selected_array_objects`.
    Large unselected fields inside each mapped value are scanned, not retained.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = _Reader(handle)
        if reader.nonspace() != "{":
            raise ValueError("expected top-level JSON object")
        while True:
            token = reader.nonspace()
            if token == "}":
                return
            key = _read_string(reader, token)
            if reader.nonspace() != ":":
                raise ValueError("expected ':'")
            start = reader.nonspace()
            if key != object_key:
                reader.value_text(start, capture=False)
            else:
                if start != "{":
                    raise ValueError(f"{object_key!r} is not an object")
                count = 0
                while True:
                    item = reader.nonspace()
                    if item == "}":
                        return
                    entry_key = _read_string(reader, item)
                    if reader.nonspace() != ":":
                        raise ValueError("expected ':'")
                    reader.push(reader.nonspace())
                    yield entry_key, _read_selected_object(reader, fields)
                    count += 1
                    if limit is not None and count >= max(0, int(limit)):
                        return
                    separator = reader.nonspace()
                    if separator == "}":
                        return
                    if separator != ",":
                        raise ValueError("expected object separator")
            separator = reader.nonspace()
            if separator == "}":
                return
            if separator != ",":
                raise ValueError("expected top-level separator")


def count_top_level_array(path: Path, array_key: str) -> int:
    """Count direct items in a top-level array with constant retained memory."""
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        reader = _Reader(handle)
        if reader.nonspace() != "{":
            raise ValueError("expected top-level JSON object")
        while True:
            token = reader.nonspace()
            if token == "}":
                return count
            key = _read_string(reader, token)
            if reader.nonspace() != ":":
                raise ValueError("expected ':'")
            start = reader.nonspace()
            if key != array_key:
                reader.value_text(start, capture=False)
            else:
                if start != "[":
                    raise ValueError(f"{array_key!r} is not an array")
                while True:
                    item = reader.nonspace()
                    if item == "]":
                        return count
                    reader.value_text(item, capture=False)
                    count += 1
                    separator = reader.nonspace()
                    if separator == "]":
                        return count
                    if separator != ",":
                        raise ValueError("expected array separator")
            separator = reader.nonspace()
            if separator == "}":
                return count
            if separator != ",":
                raise ValueError("expected top-level separator")
