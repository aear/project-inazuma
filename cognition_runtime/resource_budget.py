"""Cognition-facing budgets derived from the actual kernel resource envelope."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from resource_envelope import cgroup_status, desired_limits
from .contracts import CostEstimate

_GIB = float(1024 ** 3)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _boolean(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(default)


@dataclass(frozen=True)
class BudgetSnapshot:
    enforced: bool
    required: bool
    ram_limit_bytes: int
    ram_current_bytes: int
    swap_limit_bytes: int
    swap_current_bytes: int
    cpu_percent: float = 0.0
    io_pressure: str = "clear"
    elapsed_seconds: float = 0.0
    source: str = "cgroup_v2"
    verification: str = "unverified"

    @property
    def ram_ratio(self) -> float:
        return self.ram_current_bytes / self.ram_limit_bytes if self.ram_limit_bytes > 0 else 0.0

    @property
    def swap_ratio(self) -> float:
        return self.swap_current_bytes / self.swap_limit_bytes if self.swap_limit_bytes > 0 else 0.0

    @property
    def pressure(self) -> str:
        ratio = max(self.ram_ratio, self.swap_ratio)
        if ratio >= 0.9 or self.io_pressure == "hard":
            return "hard"
        if ratio >= 0.75 or self.io_pressure == "soft":
            return "soft"
        return "normal"

    def as_dict(self) -> dict[str, Any]:
        return {
            "enforced": self.enforced, "required": self.required,
            "ram_limit_bytes": self.ram_limit_bytes, "ram_current_bytes": self.ram_current_bytes,
            "swap_limit_bytes": self.swap_limit_bytes, "swap_current_bytes": self.swap_current_bytes,
            "ram_ratio": self.ram_ratio, "swap_ratio": self.swap_ratio,
            "cpu_percent": self.cpu_percent, "io_pressure": self.io_pressure,
            "elapsed_seconds": self.elapsed_seconds, "pressure": self.pressure,
            "source": self.source, "verification": self.verification,
        }


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str
    snapshot: BudgetSnapshot
    expected: CostEstimate


class ResourceBudget:
    """Plans graceful degradation; it never replaces kernel enforcement."""

    def __init__(
        self, *, config_loader: Callable[[], Mapping[str, Any]] | None = None,
        envelope_reader: Callable[..., Mapping[str, Any]] = cgroup_status,
        desired_reader: Callable[..., Mapping[str, Any]] = desired_limits,
    ) -> None:
        self._config_loader = config_loader or (lambda: {})
        self._envelope_reader = envelope_reader
        self._desired_reader = desired_reader

    def snapshot(self, measured: Mapping[str, Any] | None = None) -> BudgetSnapshot:
        config = dict(self._config_loader() or {})
        envelope = dict(self._envelope_reader(config) or {})
        measured = measured or {}
        ram_limit = envelope.get("kernel_ram_limit_bytes")
        if ram_limit is None:
            ram_limit = envelope.get("ram_limit_bytes", 0)
        swap_limit = envelope.get("kernel_swap_limit_bytes")
        if swap_limit is None:
            swap_limit = envelope.get("swap_limit_bytes", 0)
        ram_current = envelope.get("ram_current_bytes")
        if ram_current is None:
            ram_current = measured.get("ram_bytes", 0)
        swap_current = envelope.get("swap_current_bytes")
        if swap_current is None:
            swap_current = measured.get("swap_bytes", 0)
        return BudgetSnapshot(
            enforced=bool(envelope.get("enforced")), required=bool(envelope.get("required", True)),
            ram_limit_bytes=max(0, int(ram_limit or 0)),
            ram_current_bytes=max(0, int(ram_current or 0)),
            swap_limit_bytes=max(0, int(swap_limit or 0)),
            swap_current_bytes=max(0, int(swap_current or 0)),
            cpu_percent=max(0.0, _number(measured.get("cpu_percent"))),
            io_pressure=str(measured.get("io_pressure") or "clear").lower(),
            elapsed_seconds=max(0.0, _number(measured.get("elapsed_seconds"))),
            verification=str(envelope.get("verification") or "unverified"),
        )

    def assess(self, expected: CostEstimate, measured: Mapping[str, Any] | None = None) -> BudgetDecision:
        snapshot = self.snapshot(measured)
        if snapshot.required and not snapshot.enforced:
            return BudgetDecision(False, "hard_limit_unverified", snapshot, expected)
        if snapshot.ram_limit_bytes and snapshot.ram_current_bytes + max(0, expected.ram_bytes) > snapshot.ram_limit_bytes:
            return BudgetDecision(False, "ram_budget_exceeded", snapshot, expected)
        if snapshot.swap_limit_bytes and snapshot.swap_current_bytes + max(0, expected.swap_bytes) > snapshot.swap_limit_bytes:
            return BudgetDecision(False, "swap_budget_exceeded", snapshot, expected)
        if snapshot.pressure == "hard" and expected.ram_bytes > 0:
            return BudgetDecision(False, "resource_pressure_hard", snapshot, expected)
        if snapshot.pressure == "soft" and expected.ram_bytes >= max(1, snapshot.ram_limit_bytes // 10):
            return BudgetDecision(False, "resource_pressure_soft", snapshot, expected)
        if snapshot.cpu_percent >= 90.0 and expected.cpu_percent >= 25.0:
            return BudgetDecision(False, "cpu_pressure_hard", snapshot, expected)
        if snapshot.io_pressure == "soft" and str(expected.io_class).lower() == "high":
            return BudgetDecision(False, "io_pressure_soft", snapshot, expected)
        return BudgetDecision(True, "ok", snapshot, expected)

    def scheduler_limits(self, config: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict[str, Any]:
        raw = config.get("process_scheduler") if isinstance(config, Mapping) else None
        raw = raw if isinstance(raw, Mapping) else {}
        limits = dict(defaults)
        for key in ("enabled", "memory_budget_enabled", "terminate_over_budget_tasks", "track_gpu"):
            if key in raw:
                limits[key] = _boolean(raw[key], bool(limits[key]))
        for key in ("max_queue_slots", "max_parallel_tasks", "max_memory_heavy_tasks", "max_cpu_heavy_tasks", "max_gpu_tasks", "history_limit", "decision_limit"):
            if key in raw:
                limits[key] = max(1, int(_number(raw[key], limits[key])))
        for key in ("cpu_soft_percent", "cpu_hard_percent", "gpu_soft_percent", "gpu_hard_percent"):
            if key in raw:
                limits[key] = min(100.0, max(0.0, _number(raw[key], limits[key])))
        for key in ("history_window_hours", "max_total_rss_gb", "max_managed_rss_gb", "min_available_gb", "memory_estimate_low_gb", "memory_estimate_medium_gb", "memory_estimate_high_gb", "terminate_grace_sec"):
            if key in raw:
                limits[key] = max(0.0, _number(raw[key], limits[key]))
        limits["max_memory_heavy_tasks"] = min(limits["max_memory_heavy_tasks"], limits["max_parallel_tasks"])
        limits["max_cpu_heavy_tasks"] = min(limits["max_cpu_heavy_tasks"], limits["max_parallel_tasks"])
        limits["max_gpu_tasks"] = min(limits["max_gpu_tasks"], limits["max_parallel_tasks"])
        limits["cpu_soft_percent"] = min(limits["cpu_soft_percent"], limits["cpu_hard_percent"])
        limits["gpu_soft_percent"] = min(limits["gpu_soft_percent"], limits["gpu_hard_percent"])
        envelope = dict(self._desired_reader(dict(config)) or {})
        envelope_ram_gb = _number(envelope.get("ram_limit_bytes")) / _GIB
        if envelope.get("enabled") and envelope_ram_gb > 0:
            configured = _number(limits.get("max_total_rss_gb"))
            limits["max_total_rss_gb"] = min(configured, envelope_ram_gb) if configured > 0 else envelope_ram_gb
            if _number(limits.get("max_managed_rss_gb")) <= 0:
                limits["max_managed_rss_gb"] = envelope_ram_gb
        if limits["max_total_rss_gb"] > 0 and limits["max_managed_rss_gb"] > 0:
            limits["max_managed_rss_gb"] = min(limits["max_managed_rss_gb"], limits["max_total_rss_gb"])
        limits["history_window_hours"] = max(0.25, _number(limits.get("history_window_hours"), 24.0))
        limits["terminate_grace_sec"] = max(1.0, _number(limits.get("terminate_grace_sec"), 10.0))
        limits["resource_envelope_required"] = bool(envelope.get("required", True))
        limits["resource_envelope_ram_gb"] = round(envelope_ram_gb, 3)
        limits["resource_envelope_swap_gb"] = round(_number(envelope.get("swap_limit_bytes")) / _GIB, 3)
        return limits
