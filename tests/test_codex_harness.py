import os
from pathlib import Path

import pytest

from codex_harness import (
    AppServerClient,
    BLOCKED_BILLING_ENV,
    BoundedEvents,
    HarnessConfig,
    MAX_IMAGES,
    MAX_EVENT_CHARS,
    SubscriptionAuthError,
    image_inputs,
    subscription_environment,
)


def test_subscription_environment_removes_usage_billing_credentials():
    source = {
        "PATH": os.environ.get("PATH", ""),
        "OPENAI_API_KEY": "do-not-use",
        "AZURE_OPENAI_API_KEY": "do-not-use",
        "OPENAI_BASE_URL": "https://metered.example",
    }
    environment = subscription_environment(source)

    assert not (BLOCKED_BILLING_ENV & environment.keys())
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["INA_CODEX_HARNESS"] == "1"


def test_event_history_bounds_count_and_payload_size():
    events = BoundedEvents(maximum=20)
    for index in range(30):
        events.append("tool", {"index": index, "payload": "x" * (MAX_EVENT_CHARS + 100)})

    retained = events.wait_after(0, 0)
    assert len(retained) == 20
    assert retained[-1]["payload"]["truncated"] is True
    assert len(retained[-1]["payload"]["preview"]) == MAX_EVENT_CHARS


def test_account_gate_accepts_only_chatgpt_subscription():
    client = object.__new__(AppServerClient)
    client.request = lambda *_args, **_kwargs: {
        "account": {"type": "apiKey"},
        "requiresOpenaiAuth": True,
    }

    with pytest.raises(SubscriptionAuthError):
        client.account()


def test_account_gate_returns_no_identity_details():
    client = object.__new__(AppServerClient)
    client.request = lambda *_args, **_kwargs: {
        "account": {
            "type": "chatgpt",
            "planType": "plus",
            "email": "private@example.test",
        },
        "requiresOpenaiAuth": True,
    }

    assert client.account() == {
        "type": "chatgpt",
        "plan_type": "plus",
        "requires_openai_auth": True,
    }


def test_app_server_command_forces_chatgpt_and_has_no_auto_approval(tmp_path, monkeypatch):
    import codex_harness
    captured = {}

    class FakeProcess:
        stdin = None
        stdout = None
        stderr = None
        pid = 1

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(codex_harness.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match="stop after capture"):
        AppServerClient(HarnessConfig(Path(tmp_path), "/usr/bin/codex"))

    assert 'forced_login_method="chatgpt"' in captured["command"]
    assert "--approve-for-me" not in captured["command"]
    assert not (BLOCKED_BILLING_ENV & captured["env"].keys())


def test_gui_is_local_asset_with_explicit_approval_and_no_api_key_field():
    source = Path("codex_harness_ui.html").read_text(encoding="utf-8")

    assert "/api/approval" in source
    assert "Accept for session" in source
    assert "Browser login" in source
    assert "Device code" in source
    assert "API-key billing is disabled" in source
    assert 'name="api_key"' not in source
    assert 'id="steering" type="checkbox" checked' in source
    assert "Raw protocol details" in source
    assert "MAX_DOM_EVENTS" in source
    assert 'id="workStatus"' in source
    assert 'id="imagePicker"' in source
    assert "clipboardData" in source
    assert "MAX_IMAGE_TOTAL_BYTES" in source


def test_client_disconnect_benchmark_v1_raises_and_v2_is_quiet(monkeypatch):
    """V2 makes an abandoned long poll quiet without hiding other failures."""
    import codex_harness

    handler = object.__new__(codex_harness.HarnessHandler)

    def abandoned_request(_handler):
        raise BrokenPipeError(32, "Broken pipe")

    # V1: BaseHTTPRequestHandler.handle exposed the socket error to socketserver.
    with pytest.raises(BrokenPipeError):
        abandoned_request(handler)

    # V2: the harness boundary recognizes the same error as client cancellation.
    monkeypatch.setattr(codex_harness.BaseHTTPRequestHandler, "handle", abandoned_request)
    assert handler.handle() is None


def test_client_disconnect_handler_does_not_hide_unexpected_errors(monkeypatch):
    import codex_harness

    handler = object.__new__(codex_harness.HarnessHandler)

    def broken_handler(_handler):
        raise RuntimeError("real handler failure")

    monkeypatch.setattr(codex_harness.BaseHTTPRequestHandler, "handle", broken_handler)
    with pytest.raises(RuntimeError, match="real handler failure"):
        handler.handle()


def _notification_client(tmp_path):
    client = object.__new__(AppServerClient)
    client.config = HarnessConfig(Path(tmp_path), "/usr/bin/codex")
    client.events = BoundedEvents()
    client.thread_id = "thread-1"
    client.turn_id = "turn-1"
    client.active_model = "codex-test"
    client.running_turn = False
    client.thread_status = "idle"
    client.turn_status = "idle"
    client.work_status = "Ready"
    client.last_test_status = "not observed"
    client.last_benchmark_status = "not observed"
    client.diff_seen = False
    return client


def test_reasoning_summary_is_live_status_and_raw_payload_stays_lazy(tmp_path):
    client = _notification_client(tmp_path)
    params = {"threadId": "thread-1", "turnId": "turn-1", "delta": "Checking tests"}
    client._handle_notification("item/reasoning/summaryTextDelta", params)

    event = client.events.wait_after(0, 0)[-1]
    assert client.work_status == "Checking tests"
    assert event["kind"] == "work_status"
    assert event["payload"]["summary"] == "Checking tests"
    assert event["payload"]["raw"]["params"] == params


def test_assistant_message_benchmark_v1_delta_is_ignored_and_v2_item_is_authoritative(tmp_path):
    client = _notification_client(tmp_path)

    client._handle_notification("item/agentMessage/delta", {
        "itemId": "message-1", "delta": "draft fragment",
    })
    assert not [event for event in client.events.wait_after(0, 0) if event["kind"] == "assistant"]

    client._handle_notification("item/completed", {
        "item": {
            "id": "message-1", "type": "agentMessage",
            "text": "Authoritative complete message", "phase": "final_answer",
        },
    })
    assistant = [event for event in client.events.wait_after(0, 0) if event["kind"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["payload"]["summary"] == "Authoritative complete message"


def test_image_inputs_are_bounded_native_app_server_payloads():
    tiny_png = "data:image/png;base64,iVBORw0KGgo="
    assert image_inputs([{"url": tiny_png, "detail": "high"}]) == [{
        "type": "image", "url": tiny_png, "detail": "high",
    }]

    with pytest.raises(ValueError, match=f"At most {MAX_IMAGES}"):
        image_inputs([{"url": tiny_png}] * (MAX_IMAGES + 1))
    with pytest.raises(ValueError, match="PNG, JPEG, WebP, or GIF"):
        image_inputs([{"url": "data:text/plain;base64,aGk="}])
    with pytest.raises(ValueError, match="invalid base64"):
        image_inputs([{"url": "data:image/png;base64,not-valid!"}])


def test_image_only_prompt_reaches_turn_start_without_persistence(tmp_path):
    client = _notification_client(tmp_path)
    captured = {}
    tiny_png = "data:image/png;base64,iVBORw0KGgo="

    def request(method, params):
        captured.update(method=method, params=params)
        return {"turn": {"id": "turn-2", "status": "inProgress"}}

    client.request = request
    client.status = lambda: {"turn_running": True}
    client.send_prompt("", images=[{"url": tiny_png}])

    assert captured["method"] == "turn/start"
    assert captured["params"]["input"] == [{
        "type": "image", "url": tiny_png, "detail": "auto",
    }]


def test_thread_and_turn_notifications_are_authoritative_for_completion(tmp_path):
    client = _notification_client(tmp_path)
    client._handle_notification("thread/status/changed", {
        "threadId": "thread-1", "status": {"type": "active", "activeFlags": []},
    })
    assert client.running_turn is True
    client._handle_notification("turn/completed", {
        "threadId": "thread-1", "turn": {"id": "turn-1", "status": "completed"},
    })
    assert client.running_turn is False
    assert client.turn_status == "completed"


def test_steering_off_preserves_prompt_and_omits_collaboration_framing(tmp_path):
    client = _notification_client(tmp_path)
    captured = {}

    def request(method, params):
        captured.update(method=method, params=params)
        return {"turn": {"id": "turn-2", "status": "inProgress"}}

    client.request = request
    client.status = lambda: {"turn_running": True}
    client.send_prompt("  exact prompt\n", steering=False, collaboration_mode="plan")
    assert captured["method"] == "turn/start"
    assert captured["params"]["input"][0]["text"] == "  exact prompt\n"
    assert "collaborationMode" not in captured["params"]


def test_workspace_excludes_heavy_runtime_trees():
    import json

    payload = json.loads(Path("Project Inazuma.code-workspace").read_text(encoding="utf-8"))
    settings = payload["settings"]
    assert settings["python.analysis.indexing"] is False
    assert settings["python.analysis.diagnosticMode"] == "openFilesOnly"
    assert settings["files.watcherExclude"]["**/AI_Children/**"] is True
    assert settings["search.exclude"]["**/benchmark_results/**"] is True
