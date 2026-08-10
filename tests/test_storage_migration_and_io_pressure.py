from datetime import datetime, timedelta, timezone
from pathlib import Path

from io_pressure import active_pressure, classify_latency, pressure_signal
from storage_migration import migrate_file_and_link, migrate_tree_and_link

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


def test_latency_pressure_levels_and_expiry():
    assert classify_latency(0.1) == "clear"
    assert classify_latency(0.5) == "soft"
    assert classify_latency(2.0) == "hard"
    now = datetime.now(timezone.utc)
    signal = pressure_signal("discord", 2.0, observed_at=(now - timedelta(seconds=21)).isoformat())
    assert active_pressure(signal, now=now) == "clear"
