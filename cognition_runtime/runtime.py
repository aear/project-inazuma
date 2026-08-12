"""Composition root for heterogeneous, failure-isolated cognition."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from threading import RLock
import time
from typing import Any, Iterable, Mapping

from .capability_registry import CapabilityRegistry
from .cognitive_context import CognitiveContext
from .contracts import Contribution
from .live_patch import LivePatchManager
from .result_bus import ResultBus
from .scheduler import ExistingSchedulerAdapter


class CognitionRuntime:
    def __init__(
        self, registry: CapabilityRegistry, result_bus: ResultBus, *,
        scheduler: ExistingSchedulerAdapter | None = None, max_parallel: int = 4,
    ) -> None:
        self.registry = registry
        self.result_bus = result_bus
        self.scheduler = scheduler
        self.max_parallel = max(1, min(16, int(max_parallel)))
        self.live_patches = LivePatchManager(registry)
        self._group_guard = RLock()
        self._group_locks: dict[str, RLock] = {}

    def install_handler(self, capability: str, handler, *, source: str = "runtime", validator=None):
        return self.live_patches.install(capability, handler, source=source, validator=validator)

    def route(
        self, capability: str, context: CognitiveContext, *, payload: Any = None,
        reason: str = "cognition_runtime", priority: int | None = None,
        metadata: Mapping[str, Any] | None = None, measured: Mapping[str, Any] | None = None,
    ) -> Contribution:
        spec = self.registry.require(capability)
        slot = self.live_patches.lease(spec.name)
        if slot is None:
            if self.scheduler is None:
                return self.result_bus.publish(Contribution(
                    capability=spec.name, value={"status": "unavailable"}, relevance=0.0,
                    source="cognition_runtime", provenance=context.provenance,
                    metadata={"context_id": context.context_id, "reason": "no_handler_or_scheduler"},
                ))
            return self.scheduler.submit(
                spec.name, context.view(spec.supported_context), reason=reason, priority=priority,
                metadata=metadata, measured=measured,
            )
        started = time.perf_counter()
        try:
            with self._group_guard:
                locks = [self._group_locks.setdefault(group, RLock()) for group in sorted(spec.concurrency_groups)]
            with ExitStack() as stack:
                for lock in locks:
                    stack.enter_context(lock)
                value = slot.handler(context.view(spec.supported_context), payload)
            elapsed = time.perf_counter() - started
            if isinstance(value, Contribution):
                contribution = Contribution(
                    capability=value.capability or spec.name, value=value.value,
                    confidence=value.confidence, relevance=value.relevance,
                    cost={**dict(value.cost), "elapsed_seconds": elapsed},
                    timestamp=value.timestamp, source=value.source or slot.source,
                    provenance=value.provenance or context.provenance,
                    metadata={**dict(value.metadata), "context_id": context.context_id, "generation": slot.generation},
                    contribution_id=value.contribution_id,
                )
            else:
                contribution = Contribution(
                    capability=spec.name, value=value, relevance=1.0,
                    cost={"elapsed_seconds": elapsed}, source=slot.source,
                    provenance=context.provenance,
                    metadata={"context_id": context.context_id, "generation": slot.generation},
                )
        except Exception as exc:
            contribution = Contribution(
                capability=spec.name, value={"status": "failed", "error": str(exc)},
                confidence=0.0, relevance=0.0, cost={"elapsed_seconds": time.perf_counter() - started},
                source=slot.source, provenance=context.provenance,
                metadata={"context_id": context.context_id, "generation": slot.generation,
                          "failure_isolated": True, "error_type": type(exc).__name__},
            )
        try:
            return self.result_bus.publish(contribution)
        except ValueError as exc:
            return self.result_bus.publish(Contribution(
                capability=spec.name,
                value={"status": "failed", "error": str(exc)},
                confidence=0.0, relevance=0.0, source="cognition_runtime.result_bus",
                provenance=context.provenance,
                metadata={"context_id": context.context_id, "generation": slot.generation,
                          "failure_isolated": True, "error_type": type(exc).__name__},
            ))

    def route_many(
        self, capabilities: Iterable[str], context: CognitiveContext, *,
        payloads: Mapping[str, Any] | None = None, max_parallel: int | None = None,
    ) -> tuple[Contribution, ...]:
        names = tuple(dict.fromkeys(str(name) for name in capabilities))
        if not names:
            return ()
        workers = max(1, min(len(names), int(max_parallel or self.max_parallel)))
        payloads = payloads or {}
        if workers == 1:
            return tuple(self.route(name, context, payload=payloads.get(name)) for name in names)
        output: dict[str, Contribution] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ina-cognition") as pool:
            futures = {pool.submit(self.route, name, context, payload=payloads.get(name)): name for name in names}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    output[name] = future.result()
                except Exception as exc:  # registry failures are isolated too
                    output[name] = self.result_bus.publish(Contribution(
                        capability=name, value={"status": "failed", "error": str(exc)},
                        confidence=0.0, relevance=0.0, source="cognition_runtime",
                        provenance=context.provenance,
                        metadata={"context_id": context.context_id, "failure_isolated": True},
                    ))
        return tuple(output[name] for name in names)
