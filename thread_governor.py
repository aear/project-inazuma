"""Bounded, observation-driven thread-count governor.

The governor never runs a learning loop on its own. Call record_observation from
an explicitly invoked benchmark, then use environment_for at process launch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

from io_utils import atomic_write_json


NUMERICAL_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
DEFAULT_CANDIDATES = (1, 2, 4, 8)
DEFAULT_EXPLORATION_BUDGET = 4
_STATE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ThreadObservation:
    module: str
    workload: str
    hardware: str
    threads: int
    capability_score: float
    interference_score: float
    measured_at: str

    @classmethod
    def create(
        cls,
        module: str,
        workload: str,
        hardware: str,
        threads: int,
        capability_score: float,
        interference_score: float,
    ) -> "ThreadObservation":
        return cls(
            module=_name(module),
            workload=_name(workload),
            hardware=_name(hardware),
            threads=max(1, int(threads)),
            capability_score=float(capability_score),
            interference_score=max(0.0, float(interference_score)),
            measured_at=datetime.now(timezone.utc).isoformat(),
        )


@dataclass(frozen=True)
class ThreadDecision:
    module: str
    workload: str
    hardware: str
    threads: int
    reason: str
    explored: int
    budget: int


def _name(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("Governor module, workload, and hardware names must be non-empty.")
    return normalized


def hardware_profile() -> str:
    logical = max(1, int(os.cpu_count() or 1))
    return f"logical-cpus:{logical}"


def default_state_path(project_root: Path | str) -> Path:
    return Path(project_root).resolve() / "logs" / "thread_governor_state.json"


class AdaptiveThreadGovernor:
    """Select the smallest measured count that meets bounded requirements."""

    def __init__(
        self,
        state_path: Path | str,
        *,
        candidates: Sequence[int] = DEFAULT_CANDIDATES,
        exploration_budget: int = DEFAULT_EXPLORATION_BUDGET,
        minimum_capability: float = 1.0,
        maximum_interference: float = 1.0,
        conservative_default: int = 2,
        hard_ceiling: int | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        cpu_ceiling = max(1, int(os.cpu_count() or 1))
        requested_ceiling = cpu_ceiling if hard_ceiling is None else int(hard_ceiling)
        self.hard_ceiling = max(1, min(cpu_ceiling, requested_ceiling))
        bounded = {max(1, min(self.hard_ceiling, int(value))) for value in candidates}
        self.candidates = tuple(sorted(bounded or {1}))
        self.exploration_budget = max(1, min(16, int(exploration_budget)))
        self.minimum_capability = float(minimum_capability)
        self.maximum_interference = max(0.0, float(maximum_interference))
        self.conservative_default = max(1, min(self.hard_ceiling, int(conservative_default)))

    def _load(self) -> dict[str, Any]:
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _save(self, state: Mapping[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.state_path, dict(state), indent=2, ensure_ascii=False)

    @staticmethod
    def _key(module: str, workload: str, hardware: str) -> str:
        return "\u241f".join((_name(module), _name(workload), _name(hardware)))

    def observations(
        self, module: str, workload: str = "default", hardware: str | None = None,
    ) -> list[ThreadObservation]:
        profile = hardware or hardware_profile()
        key = self._key(module, workload, profile)
        state = self._load()
        raw = state.get("profiles", {}).get(key, {}).get("observations", [])
        observations: list[ThreadObservation] = []
        for item in raw[-self.exploration_budget:]:
            try:
                observations.append(ThreadObservation(**item))
            except (TypeError, ValueError):
                continue
        return observations

    def decide(
        self, module: str, workload: str = "default", hardware: str | None = None,
    ) -> ThreadDecision:
        module = _name(module)
        workload = _name(workload)
        profile = hardware or hardware_profile()
        observations = self.observations(module, workload, profile)
        eligible = [
            item for item in observations
            if item.capability_score >= self.minimum_capability
            and item.interference_score <= self.maximum_interference
        ]
        if eligible:
            chosen = min(eligible, key=lambda item: (item.threads, item.interference_score))
            return ThreadDecision(
                module, workload, profile, chosen.threads, "measured_smallest_sufficient",
                len(observations), self.exploration_budget,
            )
        if observations:
            previous = min(
                observations,
                key=lambda item: (
                    item.interference_score > self.maximum_interference,
                    -item.capability_score,
                    item.interference_score,
                    item.threads,
                ),
            )
            return ThreadDecision(
                module, workload, profile,
                max(1, min(self.hard_ceiling, previous.threads)),
                "bounded_best_observed", len(observations), self.exploration_budget,
            )
        return ThreadDecision(
            module, workload, profile, self.conservative_default,
            "conservative_unmeasured_default", 0, self.exploration_budget,
        )

    def next_candidate(
        self, module: str, workload: str = "default", hardware: str | None = None,
    ) -> int | None:
        observations = self.observations(module, workload, hardware)
        if len(observations) >= self.exploration_budget:
            return None
        measured = {item.threads for item in observations}
        return next((value for value in self.candidates if value not in measured), None)

    def record_observation(self, observation: ThreadObservation) -> ThreadDecision:
        if observation.threads > self.hard_ceiling:
            raise ValueError("Observed thread count exceeds the governor hard ceiling.")
        key = self._key(observation.module, observation.workload, observation.hardware)
        with _STATE_LOCK:
            state = self._load()
            profiles = state.setdefault("profiles", {})
            profile = profiles.setdefault(key, {
                "module": observation.module,
                "workload": observation.workload,
                "hardware": observation.hardware,
                "observations": [],
            })
            records = list(profile.get("observations") or [])
            if len(records) >= self.exploration_budget:
                raise RuntimeError("Explicit exploration budget is exhausted.")
            records.append(asdict(observation))
            profile["observations"] = records[-self.exploration_budget:]
            profile["updated_at"] = observation.measured_at
            state["schema_version"] = 1
            self._save(state)
        return self.decide(observation.module, observation.workload, observation.hardware)

    def environment_for(
        self,
        module: str,
        *,
        base: Mapping[str, str] | None = None,
        workload: str = "default",
        hardware: str | None = None,
    ) -> dict[str, str]:
        decision = self.decide(module, workload, hardware)
        environment = dict(os.environ if base is None else base)
        for variable in NUMERICAL_THREAD_VARIABLES:
            environment[variable] = str(decision.threads)
        environment["INA_THREAD_GOVERNOR_MODULE"] = decision.module
        environment["INA_THREAD_GOVERNOR_THREADS"] = str(decision.threads)
        environment["INA_THREAD_GOVERNOR_REASON"] = decision.reason
        return environment


def governed_environment(
    module: str,
    *,
    project_root: Path | str,
    base: Mapping[str, str] | None = None,
    workload: str = "default",
    interactive: bool = False,
) -> dict[str, str]:
    """Return module-scoped pool limits without starting any learning activity."""
    ceiling = 2 if interactive else 4
    governor = AdaptiveThreadGovernor(
        default_state_path(project_root),
        conservative_default=1 if interactive else 2,
        hard_ceiling=ceiling,
    )
    return governor.environment_for(module, base=base, workload=workload)


__all__ = [
    "AdaptiveThreadGovernor",
    "DEFAULT_CANDIDATES",
    "DEFAULT_EXPLORATION_BUDGET",
    "NUMERICAL_THREAD_VARIABLES",
    "ThreadDecision",
    "ThreadObservation",
    "default_state_path",
    "governed_environment",
    "hardware_profile",
]
