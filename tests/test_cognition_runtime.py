from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import model_manager as mm
from cognition_runtime import (
    CapabilityRegistry, CapabilitySpec, CognitionRuntime, CognitiveContext,
    Contribution, CostEstimate, ExistingSchedulerAdapter, ResourceBudget, ResultBus,
)


def spec(name, **kwargs):
    return CapabilitySpec(
        name=name, description=f"{name} specialist",
        accepts={"context": "CognitiveContext"}, returns={"value": "structured"},
        supported_context=frozenset({"observations", "goals", "references", "metadata"}),
        **kwargs,
    )


def enforced_envelope(_config=None):
    return {
        "enforced": True, "required": True, "verification": "verified",
        "kernel_ram_limit_bytes": 1000, "ram_current_bytes": 100,
        "kernel_swap_limit_bytes": 500, "swap_current_bytes": 10,
    }


def desired(_config=None):
    return {"enabled": True, "required": True, "ram_limit_bytes": 1000, "swap_limit_bytes": 500}


def test_registry_holds_metadata_without_reasoning_logic():
    registry = CapabilityRegistry([spec("logic", backend="python", confidence_semantics="cosine margin")])
    described = registry.describe()[0]
    assert described["name"] == "logic"
    assert described["version"] == "V1"
    assert described["confidence_semantics"] == "cosine margin"
    assert not hasattr(registry, "route")


def test_context_is_bounded_and_references_do_not_load_durable_store(monkeypatch, tmp_path):
    durable = tmp_path / "multi-gigabyte-memory.json"
    opened = []
    original_open = Path.open
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: opened.append(self) or original_open(self, *a, **k))
    context = CognitiveContext.build(
        observations=["x" * 10000] * 100, goals=range(100),
        discourse={"speaker": {"id": "sakura"}, "resolutions": list(range(100))},
        references=[str(durable)], metadata={str(i): i for i in range(100)},
    )
    assert len(context.observations) == 32
    assert len(context.observations[0]) == 4096
    assert len(context.goals) == 16
    assert context.references[0].uri == str(durable)
    assert context.discourse["speaker"]["id"] == "sakura"
    assert len(context.discourse["resolutions"]) == 32
    assert opened == []


def test_result_bus_is_thread_safe_and_bounded():
    bus = ResultBus(max_contributions=20)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: bus.publish(Contribution("logic", {"i": i}, provenance=("test",))), range(100)))
    snapshot = bus.snapshot(capabilities=["logic"])
    assert len(snapshot) == 20
    assert all(item.contribution_id for item in snapshot)
    assert all(item.provenance == ("test",) for item in snapshot)


def test_resource_budget_fails_closed_and_uses_actual_cgroup_values():
    blocked = ResourceBudget(
        config_loader=lambda: {},
        envelope_reader=lambda _cfg=None: {"required": True, "enforced": False, "ram_limit_bytes": 1000},
        desired_reader=desired,
    ).assess(CostEstimate(ram_bytes=1))
    assert not blocked.allowed
    assert blocked.reason == "hard_limit_unverified"

    budget = ResourceBudget(config_loader=lambda: {}, envelope_reader=enforced_envelope, desired_reader=desired)
    decision = budget.assess(CostEstimate(ram_bytes=950))
    assert not decision.allowed
    assert decision.reason == "ram_budget_exceeded"
    assert decision.snapshot.ram_current_bytes == 100


def test_scheduler_adapter_uses_existing_queue_and_preserves_provenance():
    registry = CapabilityRegistry([spec("logic", expected_cost=CostEstimate(ram_bytes=10))])
    bus = ResultBus()
    budget = ResourceBudget(config_loader=lambda: {}, envelope_reader=enforced_envelope, desired_reader=desired)
    calls = []
    adapter = ExistingSchedulerAdapter(
        registry, budget, bus,
        enqueue=lambda name, **kwargs: calls.append((name, kwargs)) or "task-1",
    )
    context = CognitiveContext.build(provenance=["observation:7"], references=["memory://fragment/7"])
    result = adapter.submit("logic", context, reason="test")
    assert result.value == {"status": "scheduled", "task_id": "task-1"}
    assert result.provenance == ("observation:7",)
    assert calls[0][1]["metadata"]["context_references"] == 1


def test_runtime_routes_concurrently_and_isolates_specialist_failure():
    registry = CapabilityRegistry([spec("logic"), spec("math")])
    bus = ResultBus()
    runtime = CognitionRuntime(registry, bus, max_parallel=2)
    barrier = threading.Barrier(2)
    runtime.install_handler("logic", lambda context, payload: barrier.wait() or {"logic": payload})
    def broken(context, payload):
        barrier.wait()
        raise RuntimeError("solver unavailable")
    runtime.install_handler("math", broken)
    context = CognitiveContext.build(observations=[{"signal": 1}], provenance=["sensor:a"])
    logic, math = runtime.route_many(["logic", "math"], context, payloads={"logic": 3})
    assert logic.value == {"logic": 3}
    assert math.value["status"] == "failed"
    assert math.metadata["failure_isolated"] is True
    assert logic.metadata["context_id"] == math.metadata["context_id"]


def test_live_patch_keeps_old_generation_lease_and_supports_rollback():
    registry = CapabilityRegistry([spec("logic")])
    runtime = CognitionRuntime(registry, ResultBus())
    first = runtime.install_handler("logic", lambda _context, _payload: "old", source="v1")
    leased = runtime.live_patches.lease("logic")
    second = runtime.install_handler("logic", lambda _context, _payload: "new", source="v2")
    assert first.generation == 1 and second.generation == 2
    assert leased.handler(None, None) == "old"
    assert runtime.route("logic", CognitiveContext.build()).value == "new"
    restored = runtime.live_patches.rollback("logic")
    assert restored.generation == 3
    assert runtime.route("logic", CognitiveContext.build()).value == "old"


def test_model_manager_facade_routes_multiple_capabilities_compatibly(monkeypatch):
    registry = CapabilityRegistry([spec("logic"), spec("math")])
    runtime = CognitionRuntime(registry, ResultBus(), max_parallel=2)
    runtime.install_handler("logic", lambda _context, payload: {"symbol": payload})
    runtime.install_handler("math", lambda _context, payload: payload * 2)
    monkeypatch.setattr(mm, "_COGNITION_RUNTIME", runtime)
    context = mm.build_cognitive_context(observations=["signal"], provenance=["test"])
    results = mm.route_cognitive_work_many(
        ["logic", "math"], context=context, payloads={"logic": "hope", "math": 4},
    )
    assert [item.value for item in results] == [{"symbol": "hope"}, 8]
    assert all(item.provenance == ("test",) for item in results)


def test_result_bus_rejects_unbounded_inline_payload():
    bus = ResultBus(max_inline_bytes=1024)
    try:
        bus.publish(Contribution("logic", b"x" * 2048))
    except ValueError as exc:
        assert "durable reference" in str(exc)
    else:
        raise AssertionError("oversized result should not enter the active result bus")


def test_live_patch_loads_versioned_module_from_allowed_root(tmp_path):
    registry = CapabilityRegistry([spec("logic")])
    runtime = CognitionRuntime(registry, ResultBus())
    patch = tmp_path / "logic_patch.py"
    patch.write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class PatchedResult:\n"
        "    value: int\n"
        "def handle(context, payload):\n"
        "    return PatchedResult(payload)\n",
        encoding="utf-8",
    )
    installed = runtime.live_patches.install_from_path(
        "logic", patch, "handle", allowed_root=tmp_path,
    )
    assert installed.generation == 1
    assert installed.module_name
    result = runtime.route("logic", CognitiveContext.build(), payload=9).value
    assert result.value == 9
    assert result.__class__.__module__ == installed.module_name


def test_concurrency_group_serializes_live_handlers():
    guarded = spec("logic", concurrency_groups=("shared_lane",))
    also_guarded = spec("math", concurrency_groups=("shared_lane",))
    runtime = CognitionRuntime(CapabilityRegistry([guarded, also_guarded]), ResultBus(), max_parallel=2)
    active = 0
    peak = 0
    lock = threading.Lock()
    def handler(_context, payload):
        nonlocal active, peak
        import time
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return payload
    runtime.install_handler("logic", handler)
    runtime.install_handler("math", handler)
    results = runtime.route_many(["logic", "math"], CognitiveContext.build())
    assert len(results) == 2
    assert peak == 1


def test_model_manager_process_capability_restart_marks_running_for_requeue(monkeypatch, tmp_path):
    original_path = mm._PROCESS_SCHEDULER_STATE_PATH
    mm._PROCESS_SCHEDULER_STATE_PATH = tmp_path / "scheduler.json"
    stopped = []
    try:
        state = mm._new_process_scheduler_state()
        state["running"] = [{
            "id": "task-live", "task_key": "logic_engine_run", "pid": 123,
            "priority": 76, "status": "running",
        }]
        mm._save_process_scheduler_state(state, mm._process_scheduler_limits())
        def stop(entry, state, limits, reason, force=False):
            stopped.append((entry["task_key"], reason, force))
            entry["status"] = "stopping"
            return True
        monkeypatch.setattr(mm, "_scheduler_request_task_stop", stop)
        result = mm.restart_cognitive_capability("logic_engine_run")
        saved = mm._load_process_scheduler_state()
        assert result["status"] == "restart_pending"
        assert saved["running"][0]["restart_after_exit"] is True
        assert stopped == [("logic_engine_run", "operator_restart", False)]
    finally:
        mm._PROCESS_SCHEDULER_STATE_PATH = original_path
