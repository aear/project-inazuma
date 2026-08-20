from datetime import datetime, timedelta, timezone
from pathlib import Path

from io_pressure import active_pressure, classify_latency, pressure_signal
from storage_migration import (
    managed_migration_step,
    migrate_file_and_link,
    migrate_tree_and_link,
    request_managed_file_move,
)

def test_tree_migration_keeps_relative_link_and_backup(tmp_path: Path):
    source = tmp_path / "old" / "memory"
    target = tmp_path / "nvme" / "memory"
    source.mkdir(parents=True)
    (source / "nested").mkdir()
    (source / "nested" / "fact.txt").write_text("continuous")
    report = migrate_tree_and_link(source, target, apply=True)
    assert report["status"] == "ok" and report["cutover"]
    assert source.is_symlink()
    assert not Path(report["link_target"]).is_absolute()
    assert (source / "nested" / "fact.txt").read_text() == "continuous"
    assert (Path(report["backup"]) / "nested" / "fact.txt").read_text() == "continuous"

def test_tree_migration_dry_run_does_not_change_paths(tmp_path: Path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir(); (source / "x").write_text("x")
    report = migrate_tree_and_link(source, target, apply=False)
    assert report["status"] == "ok"
    assert source.is_dir() and not source.is_symlink() and not target.exists()

def test_file_migration_keeps_verified_target_link_and_backup(tmp_path: Path):
    source = tmp_path / "old" / "emotion_map.json"
    target = tmp_path / "nvme" / "emotion_map.json"
    source.parent.mkdir(parents=True)
    source.write_text("symbol-state", encoding="utf-8")

    report = migrate_file_and_link(source, target, apply=True)

    assert report["status"] == "ok"
    assert report["cutover"] is True
    assert report["verified"] == 1
    assert source.is_symlink()
    assert source.read_text(encoding="utf-8") == "symbol-state"
    assert target.read_text(encoding="utf-8") == "symbol-state"
    assert Path(report["backup"]).read_text(encoding="utf-8") == "symbol-state"


def test_file_migration_dry_run_leaves_source_and_target_unchanged(tmp_path: Path):
    source, target = tmp_path / "source.json", tmp_path / "nvme" / "target.json"
    source.write_text("state", encoding="utf-8")
    report = migrate_file_and_link(source, target, apply=False)
    assert report["status"] == "ok" and not target.exists() and not source.is_symlink()


def test_managed_move_is_choice_backed_capability_scoped_and_resumable(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "AI_Children" / "Ina" / "memory" / "large.bin"
    target_root = tmp_path / "nvme"
    target = target_root / "large.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"bounded move payload" * 100_000)
    cfg = {"storage_migration_policy": {"move_target_roots": [str(target_root)]}}

    inspected = request_managed_file_move("Ina", source, target, choice="inspect", cfg=cfg)
    assert inspected["status"] == "planned" and not target.exists()
    requested = request_managed_file_move(
        "Ina", source, target, choice="move_and_link", chunk_bytes=1024 * 1024, cfg=cfg,
    )
    assert requested["status"] == "requested"
    state = managed_migration_step("Ina", chunk_bytes=1024 * 1024)
    while state["status"] in {"copying", "verifying"}:
        state = managed_migration_step("Ina", chunk_bytes=1024 * 1024)
    assert state["status"] == "complete"
    assert state["verification"] == "sha256_match"
    assert source.is_symlink() and source.resolve() == target.resolve()


def test_managed_move_decline_and_capability_rejection_are_inert(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "AI_Children" / "Ina" / "memory" / "item.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"item")
    cfg = {"storage_migration_policy": {"move_target_roots": [str(tmp_path / "nvme")]}}
    assert request_managed_file_move(
        "Ina", source, tmp_path / "nvme" / "item.bin", choice="decline", cfg=cfg,
    )["status"] == "declined"
    rejected = request_managed_file_move(
        "Ina", source, tmp_path / "outside" / "item.bin", choice="move_and_link", cfg=cfg,
    )
    assert rejected["status"] == "target_outside_capability"
    assert not (tmp_path / "AI_Children" / "Ina" / "memory" / "storage_migration_request.json").exists()


def test_latency_pressure_levels_and_expiry():
    assert classify_latency(0.1) == "clear"
    assert classify_latency(0.5) == "soft"
    assert classify_latency(2.0) == "hard"
    now = datetime.now(timezone.utc)
    signal = pressure_signal("discord", 2.0, observed_at=(now - timedelta(seconds=21)).isoformat())
    assert active_pressure(signal, now=now) == "clear"
