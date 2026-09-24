"""Bounded learning of exact counting cadence from completed experiences."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from experiential_counting import ExperientialCounter


SCHEMA = "ina.counting_cadence/V2"
DEFAULT_STRIDES = (1, 2, 5, 10)
MAX_STRIDE = 10_000


@dataclass
class CadenceLearner:
    """An inspectable finite multi-armed bandit over exact grouping strides."""

    candidates: tuple[int, ...] = DEFAULT_STRIDES
    rewards: dict[int, float] = field(default_factory=dict)
    trials: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned = tuple(sorted({int(stride) for stride in self.candidates}))
        if not cleaned or cleaned[0] != 1 or cleaned[-1] > MAX_STRIDE:
            raise ValueError("candidate strides must include 1 and stay within the bounded maximum")
        self.candidates = cleaned
        self.rewards = {stride: float(self.rewards.get(stride, 0.0)) for stride in cleaned}
        self.trials = {stride: max(0, int(self.trials.get(stride, 0))) for stride in cleaned}

    def choose(self, *, grouping_reliable: bool, maximum_stride: int = 10) -> int:
        candidates = [s for s in self.candidates if s <= maximum_stride and (s == 1 or grouping_reliable)]
        unexplored = [s for s in candidates if self.trials[s] == 0]
        if unexplored:
            return unexplored[0]
        return max(candidates, key=lambda s: (self.rewards[s], -s))

    def learn(self, stride: int, *, exact: bool, recount_agreed: bool, steps: int,
              observations: int) -> dict[str, Any]:
        if stride not in self.candidates:
            raise ValueError("stride was not offered as a learning candidate")
        # Accuracy dominates efficiency. A failed or unverified cadence cannot
        # be rewarded into preference merely because it used fewer steps.
        efficiency = 1.0 - (max(0, steps) / max(1, observations))
        reward = (1.0 if exact else -2.0) + (0.5 if recount_agreed else -1.0) + 0.25 * efficiency
        n = self.trials[stride] + 1
        self.rewards[stride] += (reward - self.rewards[stride]) / n
        self.trials[stride] = n
        return {"stride": stride, "reward": reward, "mean_reward": self.rewards[stride], "trials": n}

    def state(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "rewards": {str(k): v for k, v in self.rewards.items()},
                "trials": {str(k): v for k, v in self.trials.items()}}


def count_with_cadence(observations: Iterable[Any], *, unit: str, rule: str,
                       stride: int = 1) -> dict[str, Any]:
    """Count every observation, while chunking the spoken/mental accumulator."""
    if isinstance(stride, bool) or int(stride) != stride or not 1 <= int(stride) <= MAX_STRIDE:
        raise ValueError(f"stride must be an integer from 1 to {MAX_STRIDE}")
    stride = int(stride)
    counter = ExperientialCounter(unit, rule=rule)
    groups = []
    current = []
    for observation in observations:
        result = counter.observe(observation)
        if result.counted:
            current.append(result.sequence)
            if len(current) == stride:
                groups.append({"size": len(current), "through_observation": current[-1]})
                current = []
    if current:
        groups.append({"size": len(current), "through_observation": current[-1], "remainder": True})
    result = counter.complete_boundary()
    result.update({
        "cadence_schema": SCHEMA, "stride": stride,
        "cadence": "linear" if stride == 1 else "grouped_skip_counting",
        "accumulator_steps": len(groups), "verified_groups": groups,
        "approximate": False,
    })
    return result


def compare_linear_and_grouped(observations: Iterable[Any], *, unit: str,
                               rule: str, group_size: int) -> dict[str, Any]:
    """Run linear and grouped accumulators concurrently over one observation stream.

    This corroborates accumulation, not detection: both paths deliberately share
    the same admitted observations and are not an independent recount.
    """
    if isinstance(group_size, bool) or int(group_size) != group_size or not 2 <= int(group_size) <= MAX_STRIDE:
        raise ValueError(f"group_size must be an integer from 2 to {MAX_STRIDE}")
    group_size = int(group_size)
    linear = 0
    grouped_total = 0
    pending = 0
    observed = 0
    groups = []
    counter = ExperientialCounter(unit, rule=rule)
    for observation in observations:
        outcome = counter.observe(observation)
        observed += 1
        if not outcome.counted:
            continue
        linear += 1
        pending += 1
        if pending == group_size:
            grouped_total += group_size
            groups.append({"size": group_size, "through_observation": observed})
            pending = 0
    base = counter.complete_boundary()
    grouped_with_remainder = grouped_total + pending
    agrees = linear == grouped_with_remainder == base["value"]
    if pending:
        groups.append({"size": pending, "through_observation": observed, "remainder": True})
    return {
        "schema": "ina.concurrent_count_comparison/V1",
        "status": "agreed" if agrees else "disagreed",
        "value": linear if agrees else None,
        "linear_value": linear,
        "grouped_value": grouped_with_remainder,
        "closed_group_total": grouped_total,
        "remainder": pending,
        "group_size": group_size,
        "groups": groups,
        "observed": observed,
        "base_count": base,
        "concurrent_accumulators": 2,
        "shared_observation_stream": True,
        "independent_enumerations": 1,
        "validates": "accumulation_agreement",
        "does_not_validate": "observation_completeness",
    }


__all__ = [
    "CadenceLearner", "DEFAULT_STRIDES", "MAX_STRIDE", "SCHEMA",
    "compare_linear_and_grouped", "count_with_cadence",
]
