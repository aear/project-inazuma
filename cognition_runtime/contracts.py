"""Low-dependency contracts shared by cognition-runtime components."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CostEstimate:
    ram_bytes: int = 0
    swap_bytes: int = 0
    cpu_percent: float = 0.0
    io_class: str = "low"
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ram_bytes": max(0, int(self.ram_bytes)),
            "swap_bytes": max(0, int(self.swap_bytes)),
            "cpu_percent": max(0.0, float(self.cpu_percent)),
            "io_class": str(self.io_class or "low"),
            "elapsed_seconds": max(0.0, float(self.elapsed_seconds)),
        }


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    description: str
    accepts: Mapping[str, str] = field(default_factory=dict)
    returns: Mapping[str, str] = field(default_factory=dict)
    expected_cost: CostEstimate = field(default_factory=CostEstimate)
    supported_context: frozenset[str] = field(default_factory=frozenset)
    confidence_semantics: str = "unspecified"
    backend: str = "python"
    implementation: str = ""
    available: bool = True
    concurrency_groups: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ValueError("capability name is required")
        object.__setattr__(self, "name", name)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "accepts": dict(self.accepts), "returns": dict(self.returns),
            "expected_cost": self.expected_cost.as_dict(),
            "supported_context": sorted(self.supported_context),
            "confidence_semantics": self.confidence_semantics,
            "backend": self.backend, "implementation": self.implementation,
            "available": bool(self.available),
            "concurrency_groups": list(self.concurrency_groups),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Contribution:
    capability: str
    value: Any
    confidence: float | None = None
    relevance: float | None = None
    cost: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)
    source: str = ""
    provenance: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    contribution_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "contribution_id": self.contribution_id,
            "capability": self.capability, "value": self.value,
            "confidence": self.confidence, "relevance": self.relevance,
            "cost": dict(self.cost), "timestamp": self.timestamp,
            "source": self.source, "provenance": list(self.provenance),
            "metadata": dict(self.metadata),
        }
