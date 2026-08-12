"""Capability scheduling adapter over the existing Project Inazuma task queue."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from .capability_registry import CapabilityRegistry
from .cognitive_context import CognitiveContext
from .contracts import CapabilitySpec, Contribution, CostEstimate
from .resource_budget import ResourceBudget
from .result_bus import ResultBus

_MEMORY_COST_GB = {"low": 0.75, "medium": 3.0, "high": 12.0}
_CPU_COST = {"low": 10.0, "medium": 35.0, "high": 70.0}


def capability_specs_from_task_profiles(
    profiles: Mapping[str, Mapping[str, Any]], limits: Mapping[str, Any] | None = None,
) -> tuple[CapabilitySpec, ...]:
    limits = limits or {}
    specs = []
    for name, profile in profiles.items():
        memory_class = str(profile.get("memory_class") or "low").lower()
        memory_gb = float(profile.get("memory_estimate_gb") or limits.get(f"memory_estimate_{memory_class}_gb") or _MEMORY_COST_GB.get(memory_class, 0.75))
        cpu_class = str(profile.get("cpu_class") or "low").lower()
        module = str(profile.get("module") or name)
        command = profile.get("command") if isinstance(profile.get("command"), list) else []
        io_class = str(profile.get("io_class") or ("high" if module in {"memory_graph", "memory_reconciliation", "meaning_map", "neural_graph", "logic_map", "emotion_map"} else "low"))
        group = str(profile.get("exclusive_group") or "").strip()
        specs.append(CapabilitySpec(
            name=str(name), description=f"Schedule the {module} specialist through the existing resource-aware queue.",
            accepts={"context": "CognitiveContext", "reason": "string", "priority": "integer"},
            returns={"task_id": "string|null", "status": "scheduled|deferred"},
            expected_cost=CostEstimate(
                ram_bytes=max(0, int(memory_gb * (1024 ** 3))),
                cpu_percent=_CPU_COST.get(cpu_class, 10.0), io_class=io_class,
                elapsed_seconds=float(profile.get("max_runtime_sec") or 0.0),
            ),
            supported_context=frozenset({"observations", "goals", "active_state", "references", "metadata"}),
            confidence_semantics="scheduler acceptance is not epistemic confidence",
            backend="python-subprocess" if profile.get("kind") == "subprocess" else "python-step",
            implementation=module, concurrency_groups=(group,) if group else (),
            metadata={"task_profile": str(name), "command": tuple(str(part) for part in command),
                      "memory_class": memory_class, "cpu_class": cpu_class, "io_class": io_class},
        ))
    return tuple(specs)


class ExistingSchedulerAdapter:
    def __init__(
        self, registry: CapabilityRegistry, resource_budget: ResourceBudget, result_bus: ResultBus,
        enqueue: Callable[..., str | None],
    ) -> None:
        self.registry = registry
        self.resource_budget = resource_budget
        self.result_bus = result_bus
        self._enqueue = enqueue

    def submit(
        self, capability: str, context: CognitiveContext, *, reason: str = "cognition_runtime",
        priority: int | None = None, metadata: Mapping[str, Any] | None = None,
        measured: Mapping[str, Any] | None = None,
    ) -> Contribution:
        spec = self.registry.require(capability)
        if not spec.available:
            return self.result_bus.publish(Contribution(
                capability=spec.name, value={"status": "unavailable", "task_id": None},
                confidence=None, relevance=0.0, source="cognition_runtime.scheduler",
                provenance=context.provenance, metadata={"context_id": context.context_id, "reason": "capability_unavailable"},
            ))
        decision = self.resource_budget.assess(spec.expected_cost, measured)
        if not decision.allowed:
            return self.result_bus.publish(Contribution(
                capability=spec.name, value={"status": "deferred", "task_id": None},
                confidence=None, relevance=0.0, cost={"expected": spec.expected_cost.as_dict(), "measured": decision.snapshot.as_dict()},
                source="cognition_runtime.scheduler", provenance=context.provenance,
                metadata={"context_id": context.context_id, "reason": decision.reason},
            ))
        scheduler_metadata = dict(metadata or {})
        scheduler_metadata.update({"context_id": context.context_id, "context_references": len(context.references)})
        task_id = self._enqueue(spec.name, reason=reason, priority=priority, metadata=scheduler_metadata)
        status = "scheduled" if task_id else "deferred"
        return self.result_bus.publish(Contribution(
            capability=spec.name, value={"status": status, "task_id": task_id},
            confidence=None, relevance=1.0 if task_id else 0.0,
            cost={"expected": spec.expected_cost.as_dict(), "measured": decision.snapshot.as_dict()},
            source="cognition_runtime.scheduler", provenance=context.provenance,
            metadata={"context_id": context.context_id, "reason": reason},
        ))
