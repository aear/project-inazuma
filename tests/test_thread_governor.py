import json
import os

import pytest

from thread_governor import (
    AdaptiveThreadGovernor,
    NUMERICAL_THREAD_VARIABLES,
    ThreadObservation,
    governed_environment,
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
