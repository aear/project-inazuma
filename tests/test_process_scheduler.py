import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import model_manager as mm


def test_process_scheduler_limits_parse_overrides():
    cfg = {
        "process_scheduler": {
            "enabled": True,
            "max_queue_slots": 7,
            "max_parallel_tasks": 3,
            "max_memory_heavy_tasks": 1,
            "max_cpu_heavy_tasks": 2,
            "max_gpu_tasks": 1,
            "cpu_soft_percent": 61,
            "cpu_hard_percent": 83,
            "gpu_soft_percent": 52,
            "gpu_hard_percent": 91,
            "history_limit": 40,
            "history_window_hours": 6,
            "decision_limit": 12,
            "track_gpu": False,
        }
    }
    limits = mm._process_scheduler_limits(cfg)
    assert limits["enabled"] is True
    assert limits["max_queue_slots"] == 7
    assert limits["max_parallel_tasks"] == 3
    assert limits["max_memory_heavy_tasks"] == 1
    assert limits["cpu_soft_percent"] == 61
    assert limits["cpu_hard_percent"] == 83
    assert limits["gpu_soft_percent"] == 52
    assert limits["gpu_hard_percent"] == 91
    assert limits["history_limit"] == 40
    assert limits["history_window_hours"] == 6
    assert limits["decision_limit"] == 12
    assert limits["track_gpu"] is False


def test_enqueue_process_task_dedupes_and_caps_queue():
    limits = mm._process_scheduler_limits({"process_scheduler": {"max_queue_slots": 1, "history_limit": 10}})
    state = mm._new_process_scheduler_state()
    first_id = mm._enqueue_process_task(state, "deep_recall_step", limits=limits, priority=10, reason="first")
    second_id = mm._enqueue_process_task(state, "deep_recall_step", limits=limits, priority=25, reason="bumped")
    assert first_id == second_id
    assert len(state["queue"]) == 1
    assert state["queue"][0]["priority"] == 25
    mm._enqueue_process_task(state, "memory_graph_neural", limits=limits, priority=90, reason="heavy")
    assert len(state["queue"]) == 1
    assert state["queue"][0]["task_key"] == "memory_graph_neural"
    assert state["history"][-1]["status"] == "dropped"


def test_scheduler_blocks_memory_graph_while_deep_recall_running():
    limits = mm._process_scheduler_limits({"process_scheduler": {"max_parallel_tasks": 2, "max_memory_heavy_tasks": 1}})
    state = mm._new_process_scheduler_state()
    state["running"] = [{"task_key": "deep_recall_step", "id": "task_running"}]
    resources = {"memory_guard_level": "ok", "cpu_percent": 20.0, "gpu_utilization_percent": 0.0, "gpu_memory_percent": 0.0}
    allowed, reason = mm._scheduler_can_start_task({"task_key": "memory_graph_neural", "id": "task_next"}, state, resources, limits)
    assert allowed is False
    assert reason == "exclusive_group_busy"


def test_scheduler_blocks_memory_heavy_task_on_soft_guard():
    limits = mm._process_scheduler_limits()
    state = mm._new_process_scheduler_state()
    resources = {"memory_guard_level": "soft", "cpu_percent": 10.0, "gpu_utilization_percent": 0.0, "gpu_memory_percent": 0.0}
    allowed, reason = mm._scheduler_can_start_task({"task_key": "memory_graph_neural", "id": "task_next"}, state, resources, limits)
    assert allowed is False
    assert reason == "memory_guard_soft"


def test_build_process_scheduler_summary_reports_learning_hint_for_exclusive_memory_lane():
    limits = mm._process_scheduler_limits({"process_scheduler": {"max_queue_slots": 10, "max_parallel_tasks": 2}})
    state = mm._new_process_scheduler_state()
    state["running"] = [{"task_key": "deep_recall_step", "id": "task_running", "status": "running", "priority": 70}]
    state["queue"] = [{"task_key": "memory_graph_neural", "id": "task_next", "priority": 90, "request_reason": "deferred_resume"}]
    state["resources"] = {
        "memory_guard_level": "normal",
        "cpu_percent": 22.5,
        "gpu_available": False,
        "gpu_utilization_percent": 0.0,
    }
    state["last_decisions"] = [{
        "task_key": "memory_graph_neural",
        "decision": "blocked",
        "reason": "exclusive_group_busy",
        "priority": 90,
    }]

    summary = mm._build_process_scheduler_summary(state, limits)

    assert summary["queue_depth"] == 1
    assert summary["running_count"] == 1
    assert summary["blocked_count"] == 1
    assert "high-memory lane" in summary["learning_hint"]
    assert "memory graph neural" in summary["summary"]


def test_extract_resource_context_includes_scheduler_payload():
    payload = {
        "pressure_level": "soft",
        "summary": "RAM is rising.",
        "optimization_hint": "Start with memory_graph.py.",
        "trend": {
            "samples": 4,
            "summary": "RAM trend is rising.",
            "short": {"direction": "rising", "ram_delta_bytes": 512 * 1024 * 1024, "system_ram_delta_percent": 1.2},
            "long": {"direction": "stable", "ram_delta_bytes": 1024 * 1024 * 1024, "system_ram_delta_percent": 2.4},
        },
        "process_scheduler": {
            "summary": "Running 1/2 task(s) | queue 1/10 | guard normal | CPU 22.5%.",
            "learning_hint": "Deep recall and the neural memory graph share the same high-memory lane.",
            "running": [{"task_key": "deep_recall_step", "label": "deep recall step"}],
            "next_slots": [{"task_key": "memory_graph_neural", "label": "memory graph neural"}],
            "last_decisions": [{"task_key": "memory_graph_neural", "decision": "blocked", "reason": "exclusive_group_busy"}],
            "queue_depth": 1,
            "running_count": 1,
            "blocked_count": 1,
            "memory_guard_level": "normal",
        },
    }

    resource_context = mm._extract_resource_context(payload)

    assert resource_context["scheduler_available"] is True
    assert resource_context["scheduler_queue_depth"] == 1
    assert resource_context["scheduler_blocked_count"] == 1
    assert resource_context["scheduler_last_block_reason"] == "exclusive_group_busy"
    assert resource_context["scheduler_next_slots"][0]["label"] == "memory graph neural"


def test_request_scheduler_task_enqueues_generic_runtime_module():
    original_path = mm._PROCESS_SCHEDULER_STATE_PATH
    temp_dir = tempfile.mkdtemp(prefix='scheduler_test_')
    mm._PROCESS_SCHEDULER_STATE_PATH = Path(temp_dir) / 'process_scheduler_state.json'
    try:
        task_id = mm.request_scheduler_task('logic_engine_run', reason='unit_test', priority=81)
        state = mm._load_process_scheduler_state()
        assert task_id
        assert state['queue']
        assert state['queue'][0]['task_key'] == 'logic_engine_run'
        assert state['planner']['queue_depth'] == 1
    finally:
        mm._PROCESS_SCHEDULER_STATE_PATH = original_path
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_scheduler_blocks_meditation_while_dreamstate_running():
    limits = mm._process_scheduler_limits({"process_scheduler": {"max_parallel_tasks": 2, "max_memory_heavy_tasks": 1}})
    state = mm._new_process_scheduler_state()
    state['running'] = [{"task_key": "dreamstate_run", "id": "task_running"}]
    resources = {"memory_guard_level": "normal", "cpu_percent": 18.0, "gpu_utilization_percent": 0.0, "gpu_memory_percent": 0.0}
    allowed, reason = mm._scheduler_can_start_task({"task_key": "meditation_state_run", "id": "task_next"}, state, resources, limits)
    assert allowed is False
    assert reason == 'exclusive_group_busy'


def test_attention_allocation_suppresses_vision_for_memory_lane():
    original_get = mm.get_inastate
    try:
        state = {
            "world_connected": False,
            "dreaming": False,
            "meditating": False,
            "emotion_snapshot": {"values": {"focus": 0.2}},
            "process_scheduler": {
                "planner": {
                    "running": [{"task_key": "memory_graph_neural"}],
                    "next_slots": [],
                    "queue_depth": 1,
                    "cpu_percent": 48.0,
                }
            },
        }
        mm.get_inastate = lambda key, default=None: state.get(key, default)
        plan = mm._derive_attention_allocation(memory_guard={"level": "normal"})
        assert plan["vision_mode"] == "suppressed"
        assert "memory_recall_lane_active" in plan["reasons"]
    finally:
        mm.get_inastate = original_get


def test_attention_allocation_suppresses_audio_on_hard_guard():
    original_get = mm.get_inastate
    try:
        state = {
            "world_connected": False,
            "dreaming": False,
            "meditating": False,
            "emotion_snapshot": {"values": {"focus": 0.1}},
            "process_scheduler": {
                "planner": {
                    "running": [],
                    "next_slots": [{"task_key": "logic_engine_run"}],
                    "queue_depth": 4,
                    "cpu_percent": 86.0,
                }
            },
        }
        mm.get_inastate = lambda key, default=None: state.get(key, default)
        plan = mm._derive_attention_allocation(memory_guard={"level": "hard"})
        assert plan["audio_mode"] == "suppressed"
        assert "memory_guard_hard" in plan["reasons"]
    finally:
        mm.get_inastate = original_get


def test_attention_allocation_uses_attention_when_focus_missing():
    original_get = mm.get_inastate
    try:
        state = {
            "world_connected": False,
            "dreaming": False,
            "meditating": False,
            "emotion_snapshot": {"values": {"attention": 0.81}},
            "process_scheduler": {
                "planner": {
                    "running": [{"task_key": "logic_engine_run"}],
                    "next_slots": [],
                    "queue_depth": 1,
                    "cpu_percent": 42.0,
                }
            },
        }
        mm.get_inastate = lambda key, default=None: state.get(key, default)
        plan = mm._derive_attention_allocation(memory_guard={"level": "normal"})
        assert plan["vision_mode"] == "suppressed"
        assert "deep_focus_allocation" in plan["reasons"]
    finally:
        mm.get_inastate = original_get


def test_attention_allocation_honors_explicit_request():
    original_get = mm.get_inastate
    try:
        state = {
            "world_connected": False,
            "dreaming": False,
            "meditating": False,
            "emotion_snapshot": {"values": {"focus": 0.05}},
            "attention_request": {
                "suppress_vision": True,
                "suppress_audio": False,
                "source": "unit_test",
                "reason": "focus_window",
                "expires_at": "2999-01-01T00:00:00+00:00",
            },
            "process_scheduler": {
                "planner": {
                    "running": [],
                    "next_slots": [],
                    "queue_depth": 0,
                    "cpu_percent": 12.0,
                }
            },
        }
        mm.get_inastate = lambda key, default=None: state.get(key, default)
        plan = mm._derive_attention_allocation(memory_guard={"level": "normal"})
        assert plan["vision_mode"] == "suppressed"
        assert "attention_request_vision" in plan["reasons"]
        assert plan["attention_request"]["source"] == "unit_test"
    finally:
        mm.get_inastate = original_get


def test_request_attention_allocation_clamps_ttl_and_persists_payload():
    original_update = mm.update_inastate
    captured = {}
    try:
        mm.update_inastate = lambda key, value: captured.update({"key": key, "value": value})
        payload = mm.request_attention_allocation(
            suppress_vision=True,
            duration_sec=999999,
            reason="focus_window",
            source="unit_test",
        )
        assert payload is not None
        assert captured["key"] == "attention_request"
        assert captured["value"] == payload
        requested_at = datetime.fromisoformat(payload["requested_at"])
        expires_at = datetime.fromisoformat(payload["expires_at"])
        assert (expires_at - requested_at).total_seconds() <= 1800.5
        assert payload["suppress_vision"] is True
        assert payload["suppress_audio"] is False
    finally:
        mm.update_inastate = original_update

def test_process_scheduler_limits_parse_memory_overrides():
    cfg = {
        "process_scheduler": {
            "memory_budget_enabled": True,
            "max_total_rss_gb": 88,
            "max_managed_rss_gb": 44,
            "min_available_gb": 12,
            "memory_estimate_low_gb": 1.25,
            "memory_estimate_medium_gb": 4.5,
            "memory_estimate_high_gb": 18,
            "terminate_over_budget_tasks": False,
            "terminate_grace_sec": 22,
        }
    }
    limits = mm._process_scheduler_limits(cfg)
    assert limits["memory_budget_enabled"] is True
    assert limits["max_total_rss_gb"] == 88
    assert limits["max_managed_rss_gb"] == 44
    assert limits["min_available_gb"] == 12
    assert limits["memory_estimate_low_gb"] == 1.25
    assert limits["memory_estimate_medium_gb"] == 4.5
    assert limits["memory_estimate_high_gb"] == 18
    assert limits["terminate_over_budget_tasks"] is False
    assert limits["terminate_grace_sec"] == 22


def test_scheduler_blocks_task_when_total_rss_budget_would_be_exceeded():
    limits = mm._process_scheduler_limits({
        "process_scheduler": {
            "max_total_rss_gb": 32,
            "max_managed_rss_gb": 0,
            "min_available_gb": 0,
            "memory_estimate_high_gb": 12,
        }
    })
    state = mm._new_process_scheduler_state()
    resources = {
        "memory_guard_level": "normal",
        "ina_rss_gb": 24.5,
        "ram_available_gb": 64.0,
        "cpu_percent": 10.0,
        "gpu_utilization_percent": 0.0,
        "gpu_memory_percent": 0.0,
    }
    allowed, reason = mm._scheduler_can_start_task({"task_key": "memory_graph_neural", "id": "task_next"}, state, resources, limits)
    assert allowed is False
    assert reason == "scheduler_total_rss_limit"


def test_scheduler_blocks_task_when_managed_budget_would_be_exceeded():
    limits = mm._process_scheduler_limits({
        "process_scheduler": {
            "max_total_rss_gb": 0,
            "max_managed_rss_gb": 24,
            "min_available_gb": 0,
            "max_memory_heavy_tasks": 2,
            "memory_estimate_high_gb": 12,
            "memory_estimate_medium_gb": 3,
        }
    })
    state = mm._new_process_scheduler_state()
    state["running"] = [{"task_key": "dreamstate_run", "id": "task_running", "rss_gb": 18.0, "priority": 84, "pid": 4321}]
    resources = {
        "memory_guard_level": "normal",
        "ina_rss_gb": 20.0,
        "ram_available_gb": 64.0,
        "cpu_percent": 10.0,
        "gpu_utilization_percent": 0.0,
        "gpu_memory_percent": 0.0,
    }
    allowed, reason = mm._scheduler_can_start_task({"task_key": "logic_map_refresh", "id": "task_next"}, state, resources, limits)
    assert allowed is False
    assert reason == "scheduler_managed_rss_limit"



def test_scheduler_allows_high_cpu_reasoning_when_only_medium_cpu_task_is_running():
    limits = mm._process_scheduler_limits({"process_scheduler": {"max_parallel_tasks": 3, "max_cpu_heavy_tasks": 1}})
    state = mm._new_process_scheduler_state()
    state["running"] = [{"task_key": "emotion_engine_run", "id": "task_running"}]
    resources = {
        "memory_guard_level": "normal",
        "cpu_percent": 18.0,
        "gpu_utilization_percent": 0.0,
        "gpu_memory_percent": 0.0,
    }
    allowed, reason = mm._scheduler_can_start_task({"task_key": "logic_engine_run", "id": "task_next"}, state, resources, limits)
    assert allowed is True
    assert reason == "ok"


def test_scheduler_enforce_memory_limits_requests_stop_for_low_priority_task():
    original_stop = mm._scheduler_request_task_stop
    calls = []
    try:
        def fake_stop(entry, state, limits, reason, force=False):
            calls.append((entry.get("task_key"), reason, force))
            entry["status"] = "stopping"
            entry["stop_reason"] = reason
            return True
        mm._scheduler_request_task_stop = fake_stop
        limits = mm._process_scheduler_limits({
            "process_scheduler": {
                "max_total_rss_gb": 60,
                "max_managed_rss_gb": 50,
                "min_available_gb": 0,
                "terminate_over_budget_tasks": True,
            }
        })
        state = mm._new_process_scheduler_state()
        state["running"] = [
            {"task_key": "dreamstate_run", "id": "task_dream", "priority": 84, "pid": 111, "rss_gb": 18.0, "status": "running"},
            {"task_key": "memory_graph_neural", "id": "task_graph", "priority": 90, "pid": 222, "rss_gb": 40.0, "status": "running"},
        ]
        resources = {"ina_rss_gb": 70.0, "ram_available_gb": 20.0}
        mm._scheduler_enforce_memory_limits(state, resources, limits)
        assert calls
        assert calls[0][0] == "dreamstate_run"
        assert calls[0][1] in {"scheduler_total_rss_limit", "scheduler_managed_rss_limit"}
        assert calls[0][2] is False
    finally:
        mm._scheduler_request_task_stop = original_stop


def test_request_discord_outbox_flush_persists_payload():
    original_update = mm.update_inastate
    captured = {}
    try:
        mm.update_inastate = lambda key, value: captured.update({"key": key, "value": value})
        payload = mm.request_discord_outbox_flush(reason="backlog", burst=40, stale_mode="drop", source="unit_test")
        assert captured["key"] == "discord_outbox_flush"
        assert captured["value"] == payload
        assert payload["status"] == "requested"
        assert payload["reason"] == "backlog"
        assert payload["burst"] == 40
        assert payload["stale_mode"] == "drop"
        assert payload["source"] == "unit_test"
    finally:
        mm.update_inastate = original_update


def test_request_discord_outbox_flush_normalizes_values():
    original_update = mm.update_inastate
    captured = {}
    try:
        mm.update_inastate = lambda key, value: captured.update({"key": key, "value": value})
        payload = mm.request_discord_outbox_flush(reason="", burst=9999, stale_mode="weird", source="")
        assert payload["reason"] == "manual_flush"
        assert payload["burst"] == 512
        assert payload["stale_mode"] == "drop"
        assert payload["source"] == "internal"
    finally:
        mm.update_inastate = original_update



def test_remove_process_task_records_queue_cancellation():
    limits = mm._process_scheduler_limits({"process_scheduler": {"history_limit": 10, "history_window_hours": 24}})
    state = mm._new_process_scheduler_state()

    task_id = mm._enqueue_process_task(state, "deep_recall_step", limits=limits, reason="unit_test")
    assert task_id
    mm._remove_process_task(state, "deep_recall_step", limits=limits, reason="unit_test_cancel")

    assert state["queue"] == []
    assert state["history"][-1]["status"] == "cancelled"
    assert state["history"][-1]["reason"] == "unit_test_cancel"


def test_build_process_scheduler_summary_includes_recent_module_history():
    limits = mm._process_scheduler_limits({"process_scheduler": {"history_limit": 20, "history_window_hours": 24}})
    now = datetime.now(timezone.utc)
    stale = (now - timedelta(hours=30)).isoformat()
    recent = now.isoformat()
    state = mm._new_process_scheduler_state()
    state["history"] = [
        {"task_key": "memory_graph_neural", "status": "queued", "timestamp": recent},
        {"task_key": "memory_graph_neural", "status": "started", "timestamp": recent},
        {"task_key": "memory_graph_neural", "status": "cancelled", "timestamp": recent, "reason": "scheduler_total_rss_limit"},
        {"task_key": "logic_engine_run", "status": "completed", "timestamp": recent},
        {"task_key": "dreamstate_run", "status": "queued", "timestamp": stale},
    ]
    state["resources"] = {
        "memory_guard_level": "normal",
        "cpu_percent": 18.0,
        "gpu_available": False,
        "gpu_utilization_percent": 0.0,
    }

    summary = mm._build_process_scheduler_summary(state, limits)

    assert summary["history_window_hours"] == 24.0
    assert summary["cancelled_count"] == 1
    assert all(item["module"] != "dreamstate" for item in summary["recent_activity"])
    memory_graph = next(item for item in summary["module_history"] if item["module"] == "memory_graph")
    assert memory_graph["queued_count"] == 1
    assert memory_graph["started_count"] == 1
    assert memory_graph["cancelled_count"] == 1
    assert "cancelled" in memory_graph["status_spectrum"]
