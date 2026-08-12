"""Thread-safe metadata registry; capability reasoning stays in specialists."""
from __future__ import annotations

from dataclasses import replace
from threading import RLock
from typing import Iterable

from .contracts import CapabilitySpec


class CapabilityRegistry:
    def __init__(self, specs: Iterable[CapabilitySpec] = ()) -> None:
        self._lock = RLock()
        self._specs: dict[str, CapabilitySpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: CapabilitySpec, *, replace_existing: bool = False) -> None:
        with self._lock:
            if spec.name in self._specs and not replace_existing:
                raise ValueError(f"capability already registered: {spec.name}")
            self._specs[spec.name] = spec

    def get(self, name: str) -> CapabilitySpec | None:
        with self._lock:
            return self._specs.get(str(name or "").strip())

    def require(self, name: str) -> CapabilitySpec:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"unknown capability: {name}")
        return spec

    def set_availability(self, name: str, available: bool) -> CapabilitySpec:
        with self._lock:
            spec = self.require(name)
            updated = replace(spec, available=bool(available))
            self._specs[spec.name] = updated
            return updated

    def list(self, *, available_only: bool = False) -> tuple[CapabilitySpec, ...]:
        with self._lock:
            values = tuple(self._specs[key] for key in sorted(self._specs))
        return tuple(spec for spec in values if spec.available) if available_only else values

    def describe(self, *, available_only: bool = False) -> list[dict]:
        return [spec.as_dict() for spec in self.list(available_only=available_only)]
