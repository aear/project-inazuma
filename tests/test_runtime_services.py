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


def test_gui_shutdown_waits_for_supervisor_then_cleans_stragglers(monkeypatch):
    calls = []

    class Process:
        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout):
            calls.append(("wait", timeout))
            return 0

    monkeypatch.setattr(services, "_safe_json", lambda path: {"supervisor_pid": 777})
    monkeypatch.setattr(services, "_is_supervisor_process", lambda pid: pid == 777)
    monkeypatch.setattr(services.psutil, "Process", lambda pid: Process())
    monkeypatch.setattr(
        services,
        "stop_runtime_services",
        lambda root, grace_seconds: calls.append(("cleanup", grace_seconds))
        or {"matched": [], "errors": []},
    )

    result = services.shutdown_runtime_service_supervisor("Ina", grace_seconds=6.0)

    assert result["ok"] is True
    assert result["graceful"] is True
    assert calls == ["terminate", ("wait", 6.0), ("cleanup", 3.0)]


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


def test_dependency_failure_blocks_without_spawn(monkeypatch):
    monkeypatch.setattr(services, 'load_config', lambda: {})
    s=services.RuntimeServiceSupervisor('Fixture')
    monkeypatch.setattr(s, '_publish', lambda: None)
    monkeypatch.setattr(services, 'check_environment', lambda *a: {'ok':False,'reason':'missing'})
    monkeypatch.setattr(services, 'safe_popen', lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not spawn')))
    monkeypatch.setattr(services, 'log_to_statusbox', lambda *a: None)
    s._launch('discord_bridge')
    assert s.state['services']['discord_bridge']['status']=='blocked_environment'
    assert not s.timers


def test_versioned_startup_fixtures():
    from benchmarks.benchmark_runtime_startup import cases
    rows=cases(Path(services.__file__).read_text())
    assert rows and all(row['correct'] for row in rows), rows


def test_stale_exit_does_not_replace_current_process(monkeypatch):
    monkeypatch.setattr(services, 'load_config', lambda: {})
    s=services.RuntimeServiceSupervisor('Fixture')
    current=SimpleNamespace(pid=7);s.processes['discord_bridge']=current
    s._handle_exit('discord_bridge',{'process':SimpleNamespace(pid=6),'returncode':1})
    assert s.processes['discord_bridge'] is current
    assert s.failures['discord_bridge']==0


def test_shutdown_clears_waiting_service_state(monkeypatch):
    monkeypatch.setattr(services, 'load_config', lambda: {})
    s=services.RuntimeServiceSupervisor('Fixture')
    monkeypatch.setattr(s,'_publish',lambda:None)
    monkeypatch.setattr(s,'_launch',lambda name:s._service_state(name,status='restart_wait',restart_in_sec=30))
    s.request_stop();s.run()
    assert s.state['status']=='stopped'
    assert all(v['status']=='stopped' and v['restart_in_sec'] is None for v in s.state['services'].values())


def test_live_operator_restart_renews_budget(monkeypatch):
    monkeypatch.setattr(services,'load_config',lambda:{})
    s=services.RuntimeServiceSupervisor('Fixture')
    monkeypatch.setattr(s,'_publish',lambda:None)
    s.failures['discord_bridge']=4
    s.processes['discord_bridge']=SimpleNamespace(pid=7,poll=lambda:None,terminate=lambda:None)
    s._restart('discord_bridge')
    assert s.failures['discord_bridge']==0


def test_bridge_launch_uses_supported_interpreter(monkeypatch):
    monkeypatch.setattr(services,'load_config',lambda:{})
    s=services.RuntimeServiceSupervisor('Fixture')
    monkeypatch.setattr(s,'_publish',lambda:None)
    seen=[]
    monkeypatch.setattr(services,'project_python',lambda root,service=None:'/fixture/bridge314/python')
    monkeypatch.setattr(services,'check_environment',lambda exe,*a:seen.append(('probe',exe)) or {'ok':True})
    monkeypatch.setattr(services,'safe_popen',lambda command,**k:seen.append(('launch',command[0])) or None)
    s._launch('discord_bridge')
    assert seen==[('probe','/fixture/bridge314/python'),('launch','/fixture/bridge314/python')]
