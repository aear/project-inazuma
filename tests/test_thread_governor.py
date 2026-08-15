import json
import os

import pytest

from thread_governor import (
    AdaptiveThreadGovernor,
    NUMERICAL_THREAD_VARIABLES,
    ThreadObservation,
    governed_environment,
    observation_from_interference,
)


def observation(threads, capability, interference):
    return ThreadObservation.create(
        "meaning_map", "background", "test-hardware",
        threads, capability, interference,
    )


def test_unmeasured_module_uses_conservative_cap(tmp_path):
    governor = AdaptiveThreadGovernor(
        tmp_path / "state.json",
        conservative_default=2,
        hard_ceiling=4,
    )

    decision = governor.decide("meaning_map", "background", "test-hardware")
    environment = governor.environment_for(
        "meaning_map", base={"PATH": "test"}, workload="background",
        hardware="test-hardware",
    )

    assert decision.threads == 2
    assert decision.reason == "conservative_unmeasured_default"
    assert all(environment[name] == "2" for name in NUMERICAL_THREAD_VARIABLES)


def test_selects_smallest_count_that_meets_capability_and_interference(tmp_path):
    governor = AdaptiveThreadGovernor(
        tmp_path / "state.json",
        exploration_budget=4,
        hard_ceiling=4,
    )
    governor.record_observation(observation(1, 0.7, 0.2))
    governor.record_observation(observation(2, 1.0, 0.4))
    decision = governor.record_observation(observation(4, 1.5, 0.8))

    assert decision.threads == 2
    assert decision.reason == "measured_smallest_sufficient"


def test_learning_budget_is_explicit_and_hard(tmp_path):
    governor = AdaptiveThreadGovernor(
        tmp_path / "state.json",
        exploration_budget=2,
        hard_ceiling=4,
    )
    governor.record_observation(observation(1, 0.5, 0.1))
    governor.record_observation(observation(2, 1.0, 0.5))

    assert governor.next_candidate("meaning_map", "background", "test-hardware") is None
    with pytest.raises(RuntimeError, match="budget is exhausted"):
        governor.record_observation(observation(4, 1.2, 0.7))


def test_state_is_per_module_workload_and_hardware(tmp_path):
    path = tmp_path / "state.json"
    governor = AdaptiveThreadGovernor(path, hard_ceiling=4)
    governor.record_observation(observation(1, 1.0, 0.2))

    other = governor.decide("discord_bridge", "background", "test-hardware")
    payload = json.loads(path.read_text())

    assert other.reason == "conservative_unmeasured_default"
    assert len(payload["profiles"]) == 1


def test_interactive_launch_has_lower_unmeasured_ceiling(tmp_path):
    environment = governed_environment(
        "virtual_workspace_viewer",
        project_root=tmp_path,
        base={"PATH": os.environ.get("PATH", "")},
        interactive=True,
    )

    assert environment["INA_THREAD_GOVERNOR_THREADS"] == "1"
    assert environment["OMP_NUM_THREADS"] == "1"


def test_safe_popen_applies_module_scoped_environment(monkeypatch):
    import safe_popen

    captured = {}

    class FakeProcess:
        pass

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(safe_popen.subprocess, "Popen", fake_popen)
    process = safe_popen.safe_popen(
        ["python", "worker.py"],
        governor_module="discord_bridge",
    )

    assert isinstance(process, FakeProcess)
    assert captured["env"]["INA_THREAD_GOVERNOR_MODULE"] == "discord_bridge"

    assert int(captured["env"]["OMP_NUM_THREADS"]) <= 4

def test_differential_challenges_run_lower_then_opposing_higher(tmp_path):
    governor = AdaptiveThreadGovernor(
        tmp_path / "state.json",
        exploration_budget=4,
        conservative_default=4,
        hard_ceiling=8,
    )
    baseline = governor.next_challenge("meaning_map", "background", "test-hardware")
    assert (baseline.candidate_threads, baseline.direction) == (4, "baseline")
    governor.record_observation(ThreadObservation.create(
        "meaning_map", "background", "test-hardware",
        4, 1.0, 0.4, direction="baseline", baseline_threads=4,
    ))

    lower = governor.next_challenge("meaning_map", "background", "test-hardware")
    assert (lower.centre_threads, lower.candidate_threads, lower.direction) == (4, 2, "lower")
    governor.record_observation(ThreadObservation.create(
        "meaning_map", "background", "test-hardware",
        2, 1.0, 0.2, direction=lower.direction,
        baseline_threads=lower.centre_threads,
    ))

    higher = governor.next_challenge("meaning_map", "background", "test-hardware")
    assert (higher.centre_threads, higher.candidate_threads, higher.direction) == (4, 6, "higher")


def test_interference_constraint_cannot_be_bought_with_throughput(tmp_path):
    governor = AdaptiveThreadGovernor(
        tmp_path / "state.json",
        exploration_budget=4,
        conservative_default=4,
        hard_ceiling=8,
        maximum_interference=1.0,
    )
    governor.record_observation(ThreadObservation.create(
        "meaning_map", "background", "test-hardware", 4, 1.0, 0.4,
    ))
    governor.record_observation(ThreadObservation.create(
        "meaning_map", "background", "test-hardware", 6, 3.0, 1.01,
        direction="higher", baseline_threads=4,
    ))

    decision = governor.decide("meaning_map", "background", "test-hardware")
    assert decision.threads == 4


def test_changed_workload_gets_fresh_bounded_search_without_erasing_prior(tmp_path):
    path = tmp_path / "state.json"
    governor = AdaptiveThreadGovernor(
        path, exploration_budget=2, conservative_default=2, hard_ceiling=4,
    )
    governor.record_observation(observation(1, 0.5, 0.1))
    governor.record_observation(observation(2, 1.0, 0.3))
    assert governor.next_challenge("meaning_map", "background", "test-hardware") is None

    changed = governor.next_challenge("meaning_map", "video", "test-hardware")
    assert changed.direction == "baseline"
    assert changed.budget_remaining == 2
    assert len(json.loads(path.read_text())["profiles"]) == 1


def controlled_observation(
    threads, capability, interference, *, direction, centre,
    settled=True, violations=(),
):
    return ThreadObservation.create(
        "meaning_map", "background", "control-hardware",
        threads, capability, interference,
        direction=direction, baseline_threads=centre,
        settled=settled, constraint_violations=violations,
    )


def test_negative_differential_moves_only_outside_noise_and_within_envelope(tmp_path):
    governor = AdaptiveThreadGovernor(
        tmp_path / "state.json", conservative_default=4, hard_ceiling=8,
        deadband=0.03, hysteresis=0.02,
    )
    governor.record_observation(controlled_observation(
        4, 100.0, 0.4, direction="baseline", centre=4,
    ))
    decision = governor.record_observation(controlled_observation(
        2, 98.0, 0.2, direction="lower", centre=4,
    ))

    assert decision.threads == 2
    state = json.loads((tmp_path / "state.json").read_text())
    transition = next(iter(state["profiles"].values()))["last_transition"]
    assert transition["outcome"] == "accept_negative_differential"


def test_positive_differential_requires_deadband_plus_hysteresis(tmp_path):
    governor = AdaptiveThreadGovernor(
        tmp_path / "state.json", conservative_default=4, hard_ceiling=8,
        deadband=0.03, hysteresis=0.02,
    )
    baseline = controlled_observation(
        4, 100.0, 0.4, direction="baseline", centre=4,
    )
    governor.record_observation(baseline)
    neutral = governor.record_observation(controlled_observation(
        6, 104.9, 0.5, direction="higher", centre=4,
    ))
    assert neutral.threads == 4

    second = AdaptiveThreadGovernor(
        tmp_path / "gain.json", conservative_default=4, hard_ceiling=8,
        deadband=0.03, hysteresis=0.02,
    )
    second.record_observation(baseline)
    positive = second.record_observation(controlled_observation(
        6, 105.1, 0.5, direction="higher", centre=4,
    ))
    assert positive.threads == 6


def test_unsettled_or_hard_limit_probe_cannot_change_allocation(tmp_path):
    governor = AdaptiveThreadGovernor(
        tmp_path / "state.json", conservative_default=4, hard_ceiling=8,
    )
    governor.record_observation(controlled_observation(
        4, 1.0, 0.3, direction="baseline", centre=4,
    ))
    unsettled = governor.record_observation(controlled_observation(
        2, 1.0, 0.1, direction="lower", centre=4, settled=False,
    ))
    rejected = governor.record_observation(controlled_observation(
        6, 4.0, 0.2, direction="higher", centre=4,
        violations=("audio_xrun",),
    ))

    assert unsettled.threads == 4
    assert rejected.threads == 4
    state = json.loads((tmp_path / "state.json").read_text())
    transition = next(iter(state["profiles"].values()))["last_transition"]
    assert transition["outcome"] == "reject_hard_limit"
    assert transition["violations"] == ["audio_xrun"]


def test_interference_result_supplies_named_hard_limit_witnesses(tmp_path):
    governor = AdaptiveThreadGovernor(
        tmp_path / "state.json", conservative_default=4, hard_ceiling=8,
    )
    baseline = governor.next_challenge("meaning_map", "background", "control-hardware")
    governor.record_observation(observation_from_interference(
        baseline,
        capability_score=1.0,
        benchmark_result={"comparison": {"regressions": {}}},
    ))
    lower = governor.next_challenge("meaning_map", "background", "control-hardware")
    measured = observation_from_interference(
        lower,
        capability_score=3.0,
        benchmark_result={"comparison": {"regressions": {
            "audio_xruns": True,
            "writeback_pressure": True,
            "input_latency": False,
        }}},
    )
    decision = governor.record_observation(measured)

    assert measured.constraint_violations == ("audio_xruns", "writeback_pressure")
    assert decision.threads == 4

def test_only_unsafe_observation_falls_back_to_conservative_allocation(tmp_path):
    governor = AdaptiveThreadGovernor(
        tmp_path / "state.json", conservative_default=2, hard_ceiling=8,
    )
    decision = governor.record_observation(ThreadObservation.create(
        "meaning_map", "background", "unsafe-hardware",
        8, 10.0, 0.1,
        settled=True,
        constraint_violations=("audio_xruns",),
    ))

    assert decision.threads == 2
    assert decision.reason == "hold_conservative_no_safe_observation"
