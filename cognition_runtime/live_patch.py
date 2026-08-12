"""Atomic generation swaps for specialist handlers without global module reloads."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
from threading import RLock
from typing import Any, Callable
import hashlib

from .capability_registry import CapabilityRegistry

Handler = Callable[..., Any]
Validator = Callable[[Handler], None]


@dataclass(frozen=True)
class HandlerGeneration:
    capability: str
    generation: int
    handler: Handler
    source: str
    installed_at: str
    module_name: str = ""


class LivePatchManager:
    """Swap immutable handler slots; leases keep old generations alive safely."""

    def __init__(self, registry: CapabilityRegistry, history_limit: int = 8) -> None:
        self.registry = registry
        self.history_limit = max(1, int(history_limit))
        self._lock = RLock()
        self._active: dict[str, HandlerGeneration] = {}
        self._history: dict[str, list[HandlerGeneration]] = {}

    def install(
        self, capability: str, handler: Handler, *, source: str = "runtime",
        validator: Validator | None = None, module_name: str = "",
    ) -> HandlerGeneration:
        name = self.registry.require(capability).name
        if not callable(handler):
            raise TypeError("capability handler must be callable")
        if validator is not None:
            validator(handler)
        with self._lock:
            previous = self._active.get(name)
            generation = 1 if previous is None else previous.generation + 1
            slot = HandlerGeneration(
                capability=name, generation=generation, handler=handler, source=str(source),
                installed_at=datetime.now(timezone.utc).isoformat(),
                module_name=str(module_name),
            )
            discarded: list[HandlerGeneration] = []
            if previous is not None:
                history = self._history.setdefault(name, [])
                history.append(previous)
                discarded = history[:-self.history_limit]
                del history[:-self.history_limit]
            self._active[name] = slot
            for old in discarded:
                self._release_module_if_unused(old.module_name)
            return slot

    def _release_module_if_unused(self, module_name: str) -> None:
        if not module_name:
            return
        retained = tuple(self._active.values()) + tuple(
            item for history in self._history.values() for item in history
        )
        if all(item.module_name != module_name for item in retained):
            sys.modules.pop(module_name, None)

    def lease(self, capability: str) -> HandlerGeneration | None:
        with self._lock:
            return self._active.get(str(capability or "").strip())

    def rollback(self, capability: str) -> HandlerGeneration:
        name = self.registry.require(capability).name
        with self._lock:
            history = self._history.get(name) or []
            if not history:
                raise LookupError(f"no rollback generation for capability: {name}")
            target = history.pop()
            current = self._active.get(name)
            generation = (current.generation if current else target.generation) + 1
            restored = HandlerGeneration(
                capability=name, generation=generation, handler=target.handler,
                source=f"rollback:{target.source}",
                installed_at=datetime.now(timezone.utc).isoformat(),
                module_name=target.module_name,
            )
            discarded: list[HandlerGeneration] = []
            if current is not None:
                history.append(current)
                discarded = history[:-self.history_limit]
                del history[:-self.history_limit]
            self._active[name] = restored
            for old in discarded:
                self._release_module_if_unused(old.module_name)
            return restored

    def install_from_path(
        self, capability: str, path: Path | str, attribute: str, *,
        validator: Validator | None = None, allowed_root: Path | str | None = None,
    ) -> HandlerGeneration:
        source_path = Path(path).resolve()
        root = Path(allowed_root).resolve() if allowed_root is not None else None
        if root is not None and source_path != root and root not in source_path.parents:
            raise ValueError("patch module is outside the allowed root")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if source_path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("patch module exceeds the 2 MiB live-patch limit")
        hasher = hashlib.sha256()
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()[:16]
        current = self.lease(capability)
        next_generation = 1 if current is None else current.generation + 1
        module_name = f"_ina_live_{capability}_{next_generation}_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load patch module: {source_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            handler = getattr(module, str(attribute), None)
            return self.install(
                capability, handler, source=f"{source_path}#{digest}",
                validator=validator, module_name=module_name,
            )
        except BaseException:
            sys.modules.pop(module_name, None)
            raise

    def status(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: {
                "generation": slot.generation, "source": slot.source,
                "installed_at": slot.installed_at,
                "rollback_generations": len(self._history.get(name) or []),
            } for name, slot in self._active.items()}
