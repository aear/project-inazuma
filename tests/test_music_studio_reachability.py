from types import SimpleNamespace

import pytest

import model_manager as mm


def _install_state(monkeypatch, initial):
    store = dict(initial)
    append_calls = []

    def fake_get(key, default=None):
        return store.get(key, default)

    def fake_update(key, value):
        store[key] = value

    def fake_append(key, item, *, queue_limit, child=None):
        append_calls.append((key, item, queue_limit, child))
        raw = store.get(key)
        if raw in (None, ""):
            queue = []
        elif isinstance(raw, dict):
            queue = [raw]
        else:
            queue = list(raw)
        queued = len(queue) < queue_limit
        if queued:
            queue.append(item)
        store[key] = queue
        return {
            "queued": queued,
            "remaining": len(queue),
            "dropped": 0 if queued else 1,
            "invalid": False,
        }

    monkeypatch.setattr(mm, "get_inastate", fake_get)
    monkeypatch.setattr(mm, "update_inastate", fake_update)
    monkeypatch.setattr(mm, "append_inastate_queue", fake_append)
    return store, append_calls


def test_music_studio_has_a_bounded_creative_scheduler_profile():
    profile = mm._scheduler_task_profile("daw_window_open")

    assert profile is not None
    assert profile["command"] == [
        "python",
        "daw_window.py",
        "--child",
        str(mm.CHILD),
    ]
    assert profile["module"] == "daw_window"
    assert profile["memory_class"] == "low"
    assert profile["cpu_class"] == "medium"
    assert profile["exclusive_group"] == "creative_ui"


def test_music_seed_uses_atomic_append_and_only_one_bounded_preview(monkeypatch):
    store, append_calls = _install_state(monkeypatch, {"daw_command_queue": []})
    monkeypatch.setattr(mm.time, "time", lambda: 1234.5)

    queued = mm._queue_autonomous_music_seed(
        {
            "values": {
                "curiosity": 0.8,
                "joy": 0.7,
                "intensity": 0.4,
                "calm": 0.65,
            }
        }
    )

    commands = store["daw_command_queue"]
    actions = [command["action"] for command in commands]
    assert queued == 4
    assert len(commands) == 4
    assert len(append_calls) == 4
    assert all(call[0] == "daw_command_queue" for call in append_calls)
    assert all(call[2] == mm._DAW_COMMAND_QUEUE_LIMIT for call in append_calls)
    assert all(call[3] == str(mm.CHILD) for call in append_calls)
    assert actions == ["set_track", "set_step", "preview_note", "inspect"]
    assert commands[1]["enabled"] is True
    assert actions.count("preview_note") == 1
    assert not {"play", "start_recording", "stop_recording", "generate_vocal"}.intersection(actions)


def test_music_seed_preserves_an_existing_queue(monkeypatch):
    pending = {"id": "already_waiting", "action": "save"}
    store = {"daw_command_queue": [pending]}
    monkeypatch.setattr(mm, "get_inastate", lambda key, default=None: store.get(key, default))

    def unexpected_append(*args, **kwargs):
        raise AssertionError("an existing DAW queue must not be changed")

    monkeypatch.setattr(mm, "append_inastate_queue", unexpected_append)

    assert mm._queue_autonomous_music_seed({}) == 0
    assert store["daw_command_queue"] == [pending]


def test_music_request_seeds_and_schedules_once(monkeypatch):
    store, _append_calls = _install_state(
        monkeypatch,
        {
            "music_studio_request": {
                "requested": True,
                "source": "music_self_read",
                "reason": "curious melody pass",
            },
            "daw_window_open": False,
            "daw_command_queue": [],
            "emotion_snapshot": {"values": {"curiosity": 0.75, "joy": 0.55}},
            "dreaming": False,
            "meditating": False,
        },
    )
    scheduler_calls = []

    def fake_schedule(task_key, **kwargs):
        scheduler_calls.append((task_key, kwargs))
        return "task_music_1"

    monkeypatch.setattr(mm, "request_scheduler_task", fake_schedule)
    monkeypatch.setattr(mm, "_music_studio_process_running", lambda child: False)
    monkeypatch.setattr(mm, "_last_music_studio_launch", 0.0)
    monkeypatch.setattr(mm.time, "time", lambda: 1000.0)
    monkeypatch.setattr(mm, "log_to_statusbox", lambda message: None)

    assert mm.music_studio_check() == "task_music_1"
    assert mm.music_studio_check() is None
    assert len(scheduler_calls) == 1
    assert scheduler_calls[0][0] == "daw_window_open"
    assert scheduler_calls[0][1]["reason"] == "music_practice"
    assert scheduler_calls[0][1]["metadata"]["child"] == str(mm.CHILD)
    assert store["music_studio_request"] is False
    assert len(store["daw_command_queue"]) == 4
    assert store["last_music_studio_trigger"]["status"] == "queued"
    assert store["last_music_studio_trigger"]["seeded_commands"] == 4
    assert store["last_music_studio_trigger"]["child"] == str(mm.CHILD)


def test_open_music_studio_consumes_request_without_duplicate_launch(monkeypatch):
    store, _append_calls = _install_state(
        monkeypatch,
        {
            "music_studio_request": True,
            "daw_window_open": True,
            "daw_command_queue": [],
            "emotion_snapshot": {},
            "dreaming": False,
            "meditating": False,
        },
    )
    monkeypatch.setattr(mm, "_music_studio_process_running", lambda child: True)
    monkeypatch.setattr(mm, "_last_music_studio_launch", 0.0)
    monkeypatch.setattr(mm.time, "time", lambda: 1000.0)
    monkeypatch.setattr(mm, "log_to_statusbox", lambda message: None)
    monkeypatch.setattr(
        mm,
        "request_scheduler_task",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("duplicate launch")),
    )

    assert mm.music_studio_check() == "already_running"
    assert mm.music_studio_check() is None
    assert store["music_studio_request"] is False
    assert store["last_music_studio_trigger"]["status"] == "already_open"


def test_music_request_waits_through_cooldown_without_consuming(monkeypatch):
    store, append_calls = _install_state(
        monkeypatch,
        {
            "music_studio_request": True,
            "daw_window_open": False,
            "daw_command_queue": [],
        },
    )
    monkeypatch.setattr(mm, "_last_music_studio_launch", 900.0)
    monkeypatch.setattr(mm.time, "time", lambda: 1000.0)
    monkeypatch.setattr(
        mm,
        "request_scheduler_task",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cooldown launch")),
    )

    assert mm.music_studio_check() is None
    assert store["music_studio_request"] is True
    assert append_calls == []


@pytest.mark.parametrize("launch_succeeds", [False, True])
def test_music_self_read_requests_practice_only_after_process_launch(monkeypatch, launch_succeeds):
    store = {
        "emotion_snapshot": {
            "values": {
                "curiosity": 0.9,
                "novelty": 0.7,
                "attention": 0.8,
                "clarity": 0.75,
            }
        },
        "meta_arbitration": {},
    }

    monkeypatch.setattr(mm, "get_inastate", lambda key, default=None: store.get(key, default))
    monkeypatch.setattr(mm, "update_inastate", lambda key, value: store.__setitem__(key, value))
    monkeypatch.setattr(mm, "_last_self_read_launch", 0.0)
    monkeypatch.setattr(mm.time, "time", lambda: 2000.0)
    monkeypatch.setattr(mm, "_raw_file_manager_active", lambda: (False, {}))
    monkeypatch.setattr(mm, "_pick_self_read_source", lambda **kwargs: "music")
    monkeypatch.setattr(
        mm,
        "safe_popen",
        lambda *args, **kwargs: SimpleNamespace(pid=4321) if launch_succeeds else None,
    )
    monkeypatch.setattr(mm, "_record_raw_file_manager_launch", lambda *args, **kwargs: None)
    monkeypatch.setattr(mm, "_record_exploration_nudge", lambda *args, **kwargs: None)
    monkeypatch.setattr(mm, "log_to_statusbox", lambda message: None)
    monkeypatch.setattr(mm.threading, "Thread", lambda *args, **kwargs: SimpleNamespace(start=lambda: None))

    mm._maybe_self_read()

    if launch_succeeds:
        request = store["music_studio_request"]
        assert request["requested"] is True
        assert request["source"] == "music_self_read"
        assert request["read_focus"] == "new"
    else:
        assert "music_studio_request" not in store
        assert store["last_self_read_trigger"]["launch_failed"] is True


def test_music_studio_process_arguments_are_child_specific():
    command = ["python", "daw_window.py", "--child", "Ina_A"]

    assert mm._daw_command_targets_child(command, "Ina_A") is True
    assert mm._daw_command_targets_child(command, "Ina_B") is False
    assert mm._daw_command_targets_child(["python", "daw_window.py"], "Ina_A") is False
    assert mm._daw_command_targets_child(["python", "other.py", "--child", "Ina_A"], "Ina_A") is False


def test_music_studio_running_scan_does_not_confuse_another_child(tmp_path, monkeypatch):
    class FakePsutil:
        @staticmethod
        def process_iter(_attributes):
            return [SimpleNamespace(info={"cmdline": ["python", "daw_window.py", "--child", "Ina_A"]})]

    monkeypatch.setattr(mm, "psutil", FakePsutil())
    monkeypatch.chdir(tmp_path)

    assert mm._music_studio_process_running("Ina_A") is True
    assert mm._music_studio_process_running("Ina_B") is False
