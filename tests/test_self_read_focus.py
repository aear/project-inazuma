import model_manager as mm

from self_read_policy import self_read_focus_from_emotions


def test_focus_policy_prefers_new_seen_and_balanced_lanes():
    new_hint = self_read_focus_from_emotions(
        {
            "values": {
                "curiosity": 0.9,
                "novelty": 0.8,
                "attention": 0.7,
                "clarity": 0.8,
            }
        }
    )
    seen_hint = self_read_focus_from_emotions(
        {
            "values": {
                "familiarity": 0.9,
                "fuzziness": 0.8,
                "clarity": 0.1,
            }
        }
    )
    balanced_hint = self_read_focus_from_emotions({})

    assert new_hint["focus"] == "new"
    assert new_hint["new_score"] > new_hint["seen_score"]
    assert seen_hint["focus"] == "seen"
    assert seen_hint["seen_score"] > seen_hint["new_score"]
    assert balanced_hint["focus"] == "balanced"


def test_maybe_self_read_passes_focus_and_source_in_same_environment(monkeypatch):
    store = {
        "emotion_snapshot": {
            "values": {
                "curiosity": 0.9,
                "novelty": 0.85,
                "attention": 0.8,
                "clarity": 0.8,
                "fuzziness": 0.0,
                "familiarity": 0.1,
            }
        },
        "meta_arbitration": {},
    }
    captured = {}

    def fake_get(key, default=None):
        return store.get(key, default)

    def fake_update(key, value):
        store[key] = value

    class FakeProcess:
        pid = 4321

    class NoopThread:
        def start(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(mm, "get_inastate", fake_get)
    monkeypatch.setattr(mm, "update_inastate", fake_update)
    monkeypatch.setattr(mm, "_last_self_read_launch", 0.0)
    monkeypatch.setattr(mm, "_raw_file_manager_active", lambda: (False, {}))
    monkeypatch.setattr(mm, "_pick_self_read_source", lambda **kwargs: "code")
    monkeypatch.setattr(mm, "safe_popen", fake_popen)
    monkeypatch.setattr(mm, "_record_raw_file_manager_launch", lambda *args, **kwargs: None)
    monkeypatch.setattr(mm, "_record_exploration_nudge", lambda *args, **kwargs: None)
    monkeypatch.setattr(mm.threading, "Thread", lambda *args, **kwargs: NoopThread())

    mm._maybe_self_read()

    env = captured["kwargs"]["env"]
    trigger = store["last_self_read_trigger"]
    assert captured["command"] == [
        "python",
        "raw_file_manager.py",
        "--child",
        str(mm.CHILD),
    ]
    assert env["SELF_READ_SOURCE"] == "code"
    assert env[mm.SELF_READ_FOCUS_ENV] == "new"
    assert trigger["read_focus"] == "new"
    assert trigger["read_focus_scores"]["new"] > trigger["read_focus_scores"]["seen"]
    assert trigger["drivers"]["novelty"] == 0.85



def test_waiter_preserves_explicit_failed_state_on_zero_exit(monkeypatch):
    class Process:
        pid = 4321

        @staticmethod
        def wait():
            return 0

    stored_state = {
        "pid": 4321,
        "status": "failed",
        "error": "history_load_failed",
        "source": "code",
    }
    written = []
    published = []

    monkeypatch.setattr(
        mm,
        "_load_raw_file_manager_state",
        lambda: dict(stored_state),
    )
    monkeypatch.setattr(
        mm,
        "_write_raw_file_manager_state",
        lambda value: written.append(dict(value)),
    )
    monkeypatch.setattr(
        mm,
        "update_inastate",
        lambda key, value: published.append((key, dict(value))),
    )

    mm._wait_for_raw_file_manager(Process(), source="code", reason="test")

    assert written[-1]["status"] == "failed"
    assert written[-1]["returncode"] == 0
    raw_publication = next(
        value for key, value in published if key == "raw_file_manager_state"
    )
    exit_publication = next(
        value for key, value in published if key == "last_raw_file_manager_exit"
    )
    assert raw_publication["status"] == "failed"
    assert exit_publication["status"] == "failed"
