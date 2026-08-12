import signal
from pathlib import Path
from types import SimpleNamespace

import runtime_services as services


def test_ensure_supervisor_replaces_legacy_bridges_before_launch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(services, "supervisor_status_path", lambda child: tmp_path / "missing.json")
    monkeypatch.setattr(services, "_is_supervisor_process", lambda pid: False)
    monkeypatch.setattr(
        services,
        "stop_runtime_services",
        lambda root: calls.append(("stop", Path(root))) or {"matched": ["world_server.py"], "errors": []},
    )
    monkeypatch.setattr(
        services,
        "safe_popen",
        lambda command, **kwargs: calls.append(("launch", command, kwargs)) or SimpleNamespace(pid=321, poll=lambda: None),
    )
    monkeypatch.setattr(services, "log_to_statusbox", lambda message: None)
    monkeypatch.setattr(services, "_safe_json", lambda path: {"supervisor_pid": 321})

    assert services.ensure_runtime_service_supervisor("Ina") == 321
    assert calls[0][0] == "stop"
    assert calls[1][0] == "launch"
    assert calls[1][1][1:] == ["runtime_services.py", "--supervise", "--child", "Ina"]


def test_gui_restart_request_signals_the_supervisor(monkeypatch, tmp_path):
    monkeypatch.setattr(
        services,
        "_safe_json",
        lambda path: {"supervisor_pid": 777},
    )
    monkeypatch.setattr(services, "_is_supervisor_process", lambda pid: pid == 777)
    calls = []
    monkeypatch.setattr(services.os, "kill", lambda pid, sig: calls.append((pid, sig)))

    result = services.request_service_restart("Ina", "world_server")
    assert result["ok"] is True
    assert calls == [(777, services.SERVICE_SIGNALS["world_server"])]
    if hasattr(signal, "SIGUSR1"):
        assert calls[0][1] == signal.SIGUSR1


def test_crash_exit_requests_bounded_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "load_config", lambda: {"runtime_services": {}})
    supervisor = services.RuntimeServiceSupervisor("Ina")
    supervisor.status_path = tmp_path / "status.json"
    process = SimpleNamespace(pid=44)
    supervisor.processes["world_server"] = process
    supervisor.started_at["world_server"] = services.time.monotonic()
    calls = []
    monkeypatch.setattr(supervisor, "_publish", lambda: None)
    monkeypatch.setattr(
        supervisor,
        "_schedule_restart",
        lambda name, immediate=False: calls.append((name, immediate)),
    )

    supervisor._handle_exit("world_server", {"process": process, "returncode": 1})

    assert supervisor.failures["world_server"] == 1
    assert calls == [("world_server", False)]


def test_operator_restart_is_immediate_and_does_not_count_as_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "load_config", lambda: {"runtime_services": {}})
    supervisor = services.RuntimeServiceSupervisor("Ina")
    supervisor.status_path = tmp_path / "status.json"

    class Process:
        pid = 55

        def poll(self):
            return None

        def terminate(self):
            return None

    process = Process()
    supervisor.processes["discord_bridge"] = process
    supervisor.started_at["discord_bridge"] = services.time.monotonic()
    monkeypatch.setattr(supervisor, "_publish", lambda: None)
    calls = []
    monkeypatch.setattr(
        supervisor,
        "_schedule_restart",
        lambda name, immediate=False: calls.append((name, immediate)),
    )

    supervisor._restart("discord_bridge")
    supervisor._handle_exit("discord_bridge", {"process": process, "returncode": 0})

    assert supervisor.failures["discord_bridge"] == 0
    assert calls == [("discord_bridge", True)]
