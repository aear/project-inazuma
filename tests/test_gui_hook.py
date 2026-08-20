import errno
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gui_hook


def test_log_retention_environment_values_are_bounded(monkeypatch):
    monkeypatch.setenv("INA_TEST_LOG_LIMIT", "not-a-number")
    assert gui_hook._bounded_env_int("INA_TEST_LOG_LIMIT", 6, 1, 24) == 6
    monkeypatch.setenv("INA_TEST_LOG_LIMIT", "999")
    assert gui_hook._bounded_env_int("INA_TEST_LOG_LIMIT", 6, 1, 24) == 24


def test_disk_log_rotates_with_bounded_backups(tmp_path, monkeypatch):
    log_path = tmp_path / "ina_status.log"
    monkeypatch.setattr(gui_hook, "STATUS_LOG_PATH", log_path)
    monkeypatch.setattr(gui_hook, "STATUS_LOG_MAX_BYTES", 80)
    monkeypatch.setattr(gui_hook, "STATUS_LOG_BACKUPS", 2)

    for index in range(8):
        gui_hook._write_disk_log(f"bounded message {index}")

    assert log_path.exists()
    assert log_path.with_name("ina_status.log.1").exists()
    assert log_path.with_name("ina_status.log.2").exists()
    assert not log_path.with_name("ina_status.log.3").exists()


def test_fallback_does_not_duplicate_primary_disk_log(tmp_path, monkeypatch):
    log_path = tmp_path / "ina_status.log"
    monkeypatch.setattr(gui_hook, "STATUS_LOG_PATH", log_path)
    monkeypatch.setattr(gui_hook, "STATUS_PIPE_PATH", str(tmp_path / "missing.pipe"))
    monkeypatch.setattr(gui_hook, "_status_pipe_disabled_until", 0.0)
    monkeypatch.setattr(gui_hook, "IS_WINDOWS", False)
    monkeypatch.setattr(gui_hook, "STATUS_LOG_MAX_BYTES", 1024 * 1024)
    monkeypatch.setattr(gui_hook, "STATUS_LOG_BACKUPS", 2)

    gui_hook.log_to_statusbox("one retained line")

    assert log_path.read_text(encoding="utf-8").count("one retained line") == 1


def _quiet_logs(monkeypatch):
    monkeypatch.setattr(gui_hook, "IS_WINDOWS", False)
    monkeypatch.setattr(gui_hook, "STATUS_PIPE_PATH", "/tmp/ina_status.pipe")
    monkeypatch.setattr(gui_hook, "_status_pipe_disabled_until", 0.0)
    monkeypatch.setattr(gui_hook, "_last_status_pipe_report_at", None)
    monkeypatch.setattr(gui_hook, "_write_disk_log", lambda message: None)


def test_log_to_statusbox_reports_self_read_broken_pipe_once(monkeypatch):
    _quiet_logs(monkeypatch)
    reports = []
    fallbacks = []

    def fake_report(**kwargs):
        reports.append(kwargs)
        return {
            "explanation": "scan can continue without live status output",
            "issue_entry_id": "github_test_entry",
            "duplicate_within_cooldown": False,
        }

    def raise_broken_pipe(message):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(gui_hook, "report_self_read_broken_pipe", fake_report)
    monkeypatch.setattr(gui_hook.os.path, "exists", lambda path: True)
    monkeypatch.setattr(gui_hook, "_write_posix_status_pipe", raise_broken_pipe)
    monkeypatch.setattr(gui_hook, "fallback_log", lambda message, announce=True: fallbacks.append((message, announce)))

    gui_hook.log_to_statusbox("[SelfRead] PROCESSING demo.py [text]")
    gui_hook.log_to_statusbox("[SelfRead] PROCESSING other.py [text]")

    assert len(reports) == 1
    assert reports[0]["component"] == "status_pipe"
    assert reports[0]["operation"] == "status_log_write"
    assert len(fallbacks) == 2
    assert all(announce is False for _, announce in fallbacks)


def test_log_to_statusbox_no_fifo_reader_does_not_queue_report(monkeypatch):
    _quiet_logs(monkeypatch)
    reports = []
    fallbacks = []

    def no_reader(message):
        raise OSError(errno.ENXIO, "No such device or address")

    monkeypatch.setattr(gui_hook, "report_self_read_broken_pipe", lambda **kwargs: reports.append(kwargs))
    monkeypatch.setattr(gui_hook.os.path, "exists", lambda path: True)
    monkeypatch.setattr(gui_hook, "_write_posix_status_pipe", no_reader)
    monkeypatch.setattr(gui_hook, "fallback_log", lambda message, announce=True: fallbacks.append((message, announce)))

    gui_hook.log_to_statusbox("[SelfRead] PROCESSING demo.py [text]")

    assert reports == []
    assert fallbacks == [("[SelfRead] PROCESSING demo.py [text]", False)]
