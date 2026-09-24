"""Modality-neutral counting by observing one discrete unit at a time.

No API accepts a declared total.  A count advances only through ``observe``.
This makes the same primitive usable for beats, pixels, spiral wraps, versions,
or any later domain that can expose discrete observations and a boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping
import uuid


SCHEMA = "ina.experiential_count/V1"
CHECKPOINT_SCHEMA = "ina.experiential_count_checkpoint/V1"
MODES = frozenset({"occurrence", "unique"})
MAX_CHECKPOINT_IDENTITIES = 100_000
MAX_GROUPS = 256


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _label(value: Any, *, name: str, maximum: int = 160) -> str:
    result = str(value or "").strip()
    if not result or len(result) > maximum:
        raise ValueError(f"{name} must be 1..{maximum} characters")
    return result


def _identity(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _checkpoint_digest(payload: Mapping[str, Any]) -> str:
    body = {str(key): value for key, value in payload.items() if str(key) != "integrity_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CountObservation:
    sequence: int
    admitted: bool
    counted: bool
    identity: str | None
    group: str | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence, "admitted": self.admitted,
            "counted": self.counted, "identity": self.identity,
            "group": self.group, "reason": self.reason,
        }


class ExperientialCounter:
    """Increment a cardinality only for individually admitted observations."""

    def __init__(
        self, unit: str, *, rule: str, mode: str = "occurrence",
        identity_of: Callable[[Any], Any] | None = None,
        admit: Callable[[Any], bool] | None = None,
        group_of: Callable[[Any], Any] | None = None,
        trace_limit: int = 32,
    ) -> None:
        self.count_id = _id("count")
        self.unit = _label(unit, name="unit", maximum=80)
        self.rule = _label(rule, name="rule", maximum=500)
        self.mode = str(mode or "").strip().lower()
        if self.mode not in MODES:
            raise ValueError("mode must be occurrence or unique")
        if self.mode == "unique" and identity_of is None:
            raise ValueError("unique counting requires identity_of")
        self.identity_of = identity_of
        self.admit = admit or (lambda _observation: True)
        self.group_of = group_of
        self.trace_limit = max(0, min(256, int(trace_limit)))
        self.observed = 0
        self.admitted = 0
        self.count = 0
        self.rejected = 0
        self.duplicates = 0
        self.groups: dict[str, int] = {}
        self._seen: set[str] = set()
        self._trace: list[CountObservation] = []
        self._boundary_complete = False

    def observe(self, observation: Any) -> CountObservation:
        """Inspect exactly one candidate unit and increment at most once."""
        if self._boundary_complete:
            raise RuntimeError("cannot observe after the counting boundary is complete")
        self.observed += 1
        sequence = self.observed
        try:
            admitted = bool(self.admit(observation))
        except Exception as exc:
            result = CountObservation(sequence, False, False, None, None, f"admission_failed:{type(exc).__name__}")
            self.rejected += 1
            self._remember(result)
            return result
        if not admitted:
            result = CountObservation(sequence, False, False, None, None, "rule_rejected")
            self.rejected += 1
            self._remember(result)
            return result
        self.admitted += 1
        identity = None
        if self.identity_of is not None:
            try:
                identity = _identity(self.identity_of(observation))[:1000]
            except Exception as exc:
                result = CountObservation(sequence, True, False, None, None, f"identity_failed:{type(exc).__name__}")
                self.rejected += 1
                self._remember(result)
                return result
        if self.mode == "unique":
            if identity in self._seen:
                result = CountObservation(sequence, True, False, identity, None, "duplicate")
                self.duplicates += 1
                self._remember(result)
                return result
            if len(self._seen) >= MAX_CHECKPOINT_IDENTITIES:
                result = CountObservation(sequence, True, False, identity, None, "identity_capacity_reached")
                self.rejected += 1
                self._remember(result)
                return result
            self._seen.add(str(identity))
        group = None
        if self.group_of is not None:
            try:
                group = _label(self.group_of(observation), name="group", maximum=120)
            except Exception as exc:
                result = CountObservation(sequence, True, False, identity, None, f"group_failed:{type(exc).__name__}")
                self.rejected += 1
                self._remember(result)
                return result
            if group not in self.groups and len(self.groups) >= MAX_GROUPS:
                result = CountObservation(sequence, True, False, identity, group, "group_capacity_reached")
                self.rejected += 1
                self._remember(result)
                return result
        self.count += 1
        if group is not None:
            self.groups[group] = self.groups.get(group, 0) + 1
        result = CountObservation(sequence, True, True, identity, group, "counted")
        self._remember(result)
        return result

    def observe_many(self, observations: Iterable[Any], *, budget: int | None = None) -> dict[str, Any]:
        """Count a stream; a budget stop is explicitly incomplete."""
        limit = None if budget is None else max(0, int(budget))
        processed = 0
        iterator = iter(observations)
        while limit is None or processed < limit:
            try:
                observation = next(iterator)
            except StopIteration:
                self._boundary_complete = True
                return self.result()
            self.observe(observation)
            processed += 1
        # Reaching a budget never claims exhaustion, even if the source happened
        # to end at exactly this point.  A later empty resume establishes it.
        return self.result()

    def complete_boundary(self) -> dict[str, Any]:
        """Declare that the caller observed the end of the defined source."""
        self._boundary_complete = True
        return self.result()

    def result(self) -> dict[str, Any]:
        complete = self._boundary_complete
        exact = complete and self.rejected == 0
        return {
            "schema": SCHEMA, "count_id": self.count_id,
            "unit": self.unit, "rule": self.rule, "mode": self.mode,
            "value": self.count, "observed": self.observed,
            "admitted": self.admitted, "rejected": self.rejected,
            "duplicates": self.duplicates, "groups": dict(sorted(self.groups.items())),
            "status": "exact" if exact else ("complete_with_unresolved" if complete else "incomplete"),
            "lower_bound": None if exact else self.count,
            "boundary_complete": complete,
            "counted_by_observation": True,
            "trace": [item.as_dict() for item in self._trace],
        }

    def checkpoint(self) -> dict[str, Any]:
        """Export bounded progress; call sites own persistence and source cursors."""
        payload = {
            "schema": CHECKPOINT_SCHEMA, "count_id": self.count_id,
            "unit": self.unit, "rule": self.rule, "mode": self.mode,
            "observed": self.observed, "admitted": self.admitted,
            "count": self.count, "rejected": self.rejected,
            "duplicates": self.duplicates, "groups": dict(self.groups),
            "seen_identities": sorted(self._seen),
            "boundary_complete": self._boundary_complete,
        }
        payload["integrity_sha256"] = _checkpoint_digest(payload)
        return payload

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported counting checkpoint")
        if checkpoint.get("unit") != self.unit or checkpoint.get("rule") != self.rule or checkpoint.get("mode") != self.mode:
            raise ValueError("checkpoint does not match this counting rule")
        if checkpoint.get("integrity_sha256") != _checkpoint_digest(checkpoint):
            raise ValueError("counting checkpoint integrity mismatch")
        identities = list(checkpoint.get("seen_identities") or ())
        if len(identities) > MAX_CHECKPOINT_IDENTITIES:
            raise ValueError("checkpoint identity capacity exceeded")
        restored_count = max(0, int(checkpoint.get("count", 0)))
        restored_observed = max(0, int(checkpoint.get("observed", 0)))
        restored_admitted = max(0, int(checkpoint.get("admitted", 0)))
        restored_duplicates = max(0, int(checkpoint.get("duplicates", 0)))
        if restored_count > restored_admitted or restored_admitted > restored_observed:
            raise ValueError("counting checkpoint totals are inconsistent")
        if self.mode == "unique" and restored_count != len(set(str(item) for item in identities)):
            raise ValueError("unique checkpoint count does not match observed identities")
        groups = {str(key): max(0, int(value)) for key, value in dict(checkpoint.get("groups") or {}).items()}
        if groups and sum(groups.values()) != restored_count:
            raise ValueError("checkpoint groups do not match count")
        self.count_id = str(checkpoint.get("count_id") or self.count_id)
        self.observed = restored_observed
        self.admitted = restored_admitted
        self.count = restored_count
        self.rejected = max(0, int(checkpoint.get("rejected", 0)))
        self.duplicates = restored_duplicates
        self.groups = groups
        self._seen = {str(item) for item in identities}
        self._boundary_complete = bool(checkpoint.get("boundary_complete"))

    def _remember(self, observation: CountObservation) -> None:
        if self.trace_limit <= 0:
            return
        self._trace.append(observation)
        if len(self._trace) > self.trace_limit:
            del self._trace[0]


def verify_by_recount(
    source_factory: Callable[[], Iterable[Any]], *, unit: str, rule: str,
    mode: str = "occurrence", identity_of: Callable[[Any], Any] | None = None,
    admit: Callable[[Any], bool] | None = None,
    group_of: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Perform two separate enumerations and compare their complete results."""
    results = []
    for _run in range(2):
        counter = ExperientialCounter(
            unit, rule=rule, mode=mode, identity_of=identity_of,
            admit=admit, group_of=group_of,
        )
        results.append(counter.observe_many(source_factory()))
    first, second = results
    comparable = all(row["status"] == "exact" for row in results)
    agrees = comparable and first["value"] == second["value"] and first["groups"] == second["groups"]
    return {
        "schema": "ina.experiential_count_verification/V1",
        "status": "verified" if agrees else ("disagreed" if comparable else "unavailable"),
        "value": first["value"] if agrees else None,
        "runs": results,
        "independent_enumerations": 2,
    }


def count_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Bounded cognition-runtime adapter; still counts each supplied observation."""
    if not isinstance(payload, Mapping):
        raise TypeError("counting payload must be a mapping")
    if "declared_total" in payload or "count" in payload:
        raise ValueError("declared totals are not observations and cannot be counted")
    observations = payload.get("observations")
    if observations is None or isinstance(observations, (str, bytes, Mapping)):
        raise ValueError("observations must be a finite iterable of discrete units")
    identity_key = str(payload.get("identity_key") or "").strip()
    group_key = str(payload.get("group_key") or "").strip()
    include_key = str(payload.get("include_key") or "").strip()

    def field(key: str, observation: Any) -> Any:
        if not isinstance(observation, Mapping) or key not in observation:
            raise KeyError(key)
        return observation[key]

    counter = ExperientialCounter(
        payload.get("unit"), rule=payload.get("rule"),
        mode=payload.get("mode", "occurrence"),
        identity_of=(lambda item: field(identity_key, item)) if identity_key else None,
        admit=(lambda item: bool(field(include_key, item))) if include_key else None,
        group_of=(lambda item: field(group_key, item)) if group_key else None,
        trace_limit=payload.get("trace_limit", 32),
    )
    budget = max(1, min(100_000, int(payload.get("observation_budget", 10_000))))
    result = counter.observe_many(observations, budget=budget)
    result["observation_budget"] = budget
    return result


__all__ = [
    "CHECKPOINT_SCHEMA", "ExperientialCounter", "MODES", "SCHEMA",
    "CountObservation", "count_payload", "verify_by_recount",
]
