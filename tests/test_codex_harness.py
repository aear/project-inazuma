import os
from pathlib import Path

import pytest

from codex_harness import (
    AppServerClient,
    BLOCKED_BILLING_ENV,
    BoundedEvents,
    HarnessConfig,
    MAX_EVENT_CHARS,
    SubscriptionAuthError,
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


def test_workspace_excludes_heavy_runtime_trees():
    import json

    payload = json.loads(Path("Project Inazuma.code-workspace").read_text(encoding="utf-8"))
    settings = payload["settings"]
    assert settings["python.analysis.indexing"] is False
    assert settings["python.analysis.diagnosticMode"] == "openFilesOnly"
    assert settings["files.watcherExclude"]["**/AI_Children/**"] is True
    assert settings["search.exclude"]["**/benchmark_results/**"] is True
