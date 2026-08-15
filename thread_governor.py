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
    baseline_threads: int | None = None
    direction: str = "baseline"
    settled: bool = True
    constraint_violations: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        module: str,
        workload: str,
        hardware: str,
        threads: int,
        capability_score: float,
        interference_score: float,
        *,
        baseline_threads: int | None = None,
        direction: str = "baseline",
        settled: bool = True,
        constraint_violations: Sequence[str] = (),
    ) -> "ThreadObservation":
        direction = str(direction or "baseline").strip().lower()
        if direction not in {"baseline", "lower", "higher"}:
            raise ValueError("Observation direction must be baseline, lower, or higher.")
        baseline = None if baseline_threads is None else max(1, int(baseline_threads))
        return cls(
            module=_name(module),
            workload=_name(workload),
            hardware=_name(hardware),
            threads=max(1, int(threads)),
            capability_score=float(capability_score),
            interference_score=max(0.0, float(interference_score)),
            measured_at=datetime.now(timezone.utc).isoformat(),
            baseline_threads=baseline,
            direction=direction,
            settled=bool(settled),
            constraint_violations=tuple(sorted({str(item) for item in constraint_violations if str(item)})),
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


@dataclass(frozen=True)
class ThreadChallenge:
    module: str
    workload: str
    hardware: str
    centre_threads: int
    candidate_threads: int
    direction: str
    explored: int
    budget_remaining: int


@dataclass(frozen=True)
class OperatingEnvelope:
    minimum_capability: float
    maximum_interference: float
    deadband: float
    hysteresis: float
    requires_settled_measurement: bool = True


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
        deadband: float = 0.03,
        hysteresis: float = 0.02,
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
        self.deadband = max(0.0, min(0.5, float(deadband)))
        self.hysteresis = max(0.0, min(0.5, float(hysteresis)))
        self.envelope = OperatingEnvelope(
            self.minimum_capability,
            self.maximum_interference,
            self.deadband,
            self.hysteresis,
        )

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

    def _within_envelope(
        self, observation: ThreadObservation, *, require_capability: bool = True,
    ) -> bool:
        return (
            observation.settled
            and not observation.constraint_violations
            and observation.interference_score <= self.maximum_interference
            and (not require_capability or observation.capability_score >= self.minimum_capability)
        )

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
        stored = self._load().get("profiles", {}).get(self._key(module, workload, profile), {})
        try:
            accepted_threads = int(stored.get("accepted_threads"))
        except (TypeError, ValueError):
            accepted_threads = 0
        eligible = [
            item for item in observations
            if self._within_envelope(item)
        ]
        if eligible:
            accepted = [item for item in eligible if item.threads == accepted_threads]
            if accepted:
                chosen = accepted[-1]
                reason = "control_envelope_hold"
            else:
                chosen = min(eligible, key=lambda item: (item.threads, item.interference_score))
                reason = "measured_smallest_sufficient"
            return ThreadDecision(
                module, workload, profile, chosen.threads, reason,
                len(observations), self.exploration_budget,
            )
        if observations:
            safe = [
                item for item in observations
                if self._within_envelope(item, require_capability=False)
            ]
            if not safe:
                return ThreadDecision(
                    module, workload, profile, self.conservative_default,
                    "hold_conservative_no_safe_observation",
                    len(observations), self.exploration_budget,
                )
            previous = min(
                safe,
                key=lambda item: (
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

    def next_challenge(
        self, module: str, workload: str = "default", hardware: str | None = None,
    ) -> ThreadChallenge | None:
        """Return one sequential lower/higher probe around the active centre."""
        module = _name(module)
        workload = _name(workload)
        profile = hardware or hardware_profile()
        observations = self.observations(module, workload, profile)
        if len(observations) >= self.exploration_budget:
            return None

        decision = self.decide(module, workload, profile)
        centre = decision.threads
        measured = {item.threads for item in observations}

        # Finish the opposite half before recentering on a successful lower probe.
        for item in reversed(observations):
            if item.direction != "lower" or item.baseline_threads is None:
                continue
            pending_centre = max(1, min(self.hard_ceiling, item.baseline_threads))
            step = max(1, pending_centre // 2)
            higher = min(self.hard_ceiling, pending_centre + step)
            higher_done = any(
                candidate.direction == "higher"
                and candidate.baseline_threads == pending_centre
                for candidate in observations
            )
            if higher != pending_centre and not higher_done and higher not in measured:
                return ThreadChallenge(
                    module, workload, profile, pending_centre, higher, "higher",
                    len(observations), self.exploration_budget - len(observations),
                )
            break

        # Measure the conservative operating point before moving either way.
        if centre not in measured:
            return ThreadChallenge(
                module, workload, profile, centre, centre, "baseline",
                len(observations), self.exploration_budget - len(observations),
            )

        step = max(1, centre // 2)
        lower = max(1, centre - step)
        higher = min(self.hard_ceiling, centre + step)
        for candidate, direction in ((lower, "lower"), (higher, "higher")):
            if candidate != centre and candidate not in measured:
                return ThreadChallenge(
                    module, workload, profile, centre, candidate, direction,
                    len(observations), self.exploration_budget - len(observations),
                )
        return None

    def next_candidate(
        self, module: str, workload: str = "default", hardware: str | None = None,
    ) -> int | None:
        """Compatibility view of the next sequential differential challenge."""
        challenge = self.next_challenge(module, workload, hardware)
        return None if challenge is None else challenge.candidate_threads

    def _apply_transition(
        self,
        profile: dict[str, Any],
        observation: ThreadObservation,
        prior_records: Sequence[Mapping[str, Any]],
    ) -> None:
        """Apply one settled differential without allowing limits to be traded away."""
        transition: dict[str, Any] = {
            "direction": observation.direction,
            "candidate_threads": observation.threads,
            "centre_threads": observation.baseline_threads,
            "measured_at": observation.measured_at,
        }
        if not observation.settled:
            transition["outcome"] = "hold_unsettled"
        elif observation.constraint_violations:
            transition["outcome"] = "reject_hard_limit"
            transition["violations"] = list(observation.constraint_violations)
        elif observation.interference_score > self.maximum_interference:
            transition["outcome"] = "reject_interference_envelope"
        elif observation.direction == "baseline" and observation.baseline_threads is not None:
            profile["accepted_threads"] = observation.threads
            transition["outcome"] = "establish_centre"
        elif observation.direction in {"lower", "higher"} and observation.baseline_threads is not None:
            baseline = None
            for item in reversed(prior_records):
                if int(item.get("threads", 0) or 0) == observation.baseline_threads:
                    baseline = item
                    break
            if baseline is None:
                transition["outcome"] = "hold_missing_centre"
            else:
                baseline_capability = float(baseline.get("capability_score", 0.0) or 0.0)
                denominator = max(abs(baseline_capability), 1e-9)
                capability_delta = (
                    observation.capability_score - baseline_capability
                ) / denominator
                transition["capability_delta"] = round(capability_delta, 6)
                transition["interference_delta"] = round(
                    observation.interference_score
                    - float(baseline.get("interference_score", 0.0) or 0.0),
                    6,
                )
                if observation.direction == "lower":
                    acceptable_loss = capability_delta >= -self.deadband
                    accepted = (
                        observation.capability_score >= self.minimum_capability
                        and acceptable_loss
                    )
                    transition["outcome"] = (
                        "accept_negative_differential" if accepted
                        else "hold_below_negative_deadband"
                    )
                else:
                    threshold = self.deadband + self.hysteresis
                    accepted = (
                        observation.capability_score >= self.minimum_capability
                        and capability_delta >= threshold
                    )
                    transition["outcome"] = (
                        "accept_positive_differential" if accepted
                        else "hold_inside_positive_deadband"
                    )
                if accepted:
                    profile["accepted_threads"] = observation.threads
        else:
            transition["outcome"] = "legacy_observation"

        profile["last_transition"] = transition

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
            self._apply_transition(profile, observation, records)
            records.append(asdict(observation))
            profile["observations"] = records[-self.exploration_budget:]
            profile["updated_at"] = observation.measured_at
            state["schema_version"] = 2
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




def observation_from_interference(
    challenge: ThreadChallenge,
    *,
    capability_score: float,
    benchmark_result: Mapping[str, Any],
    interference_score: float = 0.0,
    settled: bool = True,
) -> ThreadObservation:
    """Convert one bounded interference result into a differential observation."""
    comparison = benchmark_result.get("comparison", {})
    regressions = comparison.get("regressions", {}) if isinstance(comparison, Mapping) else {}
    violations = tuple(
        sorted(str(name) for name, failed in regressions.items() if bool(failed))
    )
    return ThreadObservation.create(
        challenge.module,
        challenge.workload,
        challenge.hardware,
        challenge.candidate_threads,
        capability_score,
        interference_score,
        baseline_threads=challenge.centre_threads,
        direction=challenge.direction,
        settled=settled,
        constraint_violations=violations,
    )
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
    "OperatingEnvelope",
    "ThreadChallenge",
    "ThreadDecision",
    "ThreadObservation",
    "default_state_path",
    "governed_environment",
    "hardware_profile",
    "observation_from_interference",
]
