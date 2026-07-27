import os
from pathlib import Path

import io_utils


def test_auto_policy_skips_btrfs_and_keeps_other_filesystems(monkeypatch):
    monkeypatch.delenv("INA_FSYNC_MODE", raising=False)
    monkeypatch.setattr(io_utils, "filesystem_type", lambda path: "btrfs")
    assert io_utils.should_fsync(Path("state.json")) is False
    monkeypatch.setattr(io_utils, "filesystem_type", lambda path: "xfs")
    assert io_utils.should_fsync(Path("state.json")) is True


def test_never_override_preserves_atomic_replace_without_fsync(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    monkeypatch.setenv("INA_FSYNC_MODE", "never")
    calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd))
    io_utils.atomic_write_json(target, {"ok": True})
    assert io_utils.load_json_dict(target) == {"ok": True}
    assert calls == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_always_override_requests_fsync(monkeypatch):
    monkeypatch.setenv("INA_FSYNC_MODE", "always")
    assert io_utils.should_fsync(Path("state.json")) is True
