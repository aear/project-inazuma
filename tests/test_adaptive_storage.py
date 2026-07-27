from pathlib import Path

import adaptive_storage
import storage_layout


def _config(tmp_path: Path):
    return {
        "current_child": "Ina",
        "adaptive_storage_policy": {
            "enabled": True,
            "apply_recommendations": True,
            "min_samples_before_switch": 3,
            "switch_margin": 0.05,
            "state_path": str(tmp_path / "{child}_adaptive.json"),
        },
    }


def test_online_observations_can_change_rebuildable_placement(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    vitals = {"roles": {"fast_runtime": {"path": str(tmp_path / "fast")}, "project": {"path": str(tmp_path / "durable")}}}

    def slow_fast_probe(path, _size):
        is_fast = Path(path).name == "fast"
        return {
            "success": True,
            "latency_seconds": 1.0 if is_fast else 0.001,
            "throughput_bytes_per_second": 1000.0 if is_fast else 500_000_000.0,
            "free_ratio": 0.5,
        }

    monkeypatch.setattr(adaptive_storage, "_probe", slow_fast_probe)
    for _ in range(3):
        state = adaptive_storage.update_from_storage_vitals("Ina", cfg, vitals, force=True)
    assert state["decisions"]["runtime"]["tier"] == "durable"
    assert adaptive_storage.recommend_rebuildable_tier("Ina", "runtime", cfg, fast_available=True) == "durable"


def test_insufficient_samples_keep_safe_fast_default(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    cfg["adaptive_storage_policy"]["min_samples_before_switch"] = 4
    vitals = {"roles": {"fast_runtime": {"path": str(tmp_path / "fast")}, "project": {"path": str(tmp_path / "durable")}}}
    monkeypatch.setattr(adaptive_storage, "_probe", lambda path, size: {"success": Path(path).name != "fast", "latency_seconds": 0.001, "throughput_bytes_per_second": 1_000_000.0, "free_ratio": 0.5})
    state = adaptive_storage.update_from_storage_vitals("Ina", cfg, vitals, force=True)
    assert state["decisions"]["runtime"]["tier"] == "fast"
    assert state["decisions"]["runtime"]["reason"] == "hysteresis_or_insufficient_samples"


def test_storage_layout_applies_decision_only_to_rebuildable_path(tmp_path, monkeypatch):
    fast_mount = tmp_path / "nvme"
    fast_root = fast_mount / "runtime"
    fallback = tmp_path / "hdd" / "index.sqlite"
    fast_root.mkdir(parents=True)
    fallback.parent.mkdir(parents=True)
    cfg = _config(tmp_path)
    cfg["storage_layout"] = {
        "fast_runtime_enabled": True,
        "fast_mount": str(fast_mount),
        "fast_runtime_root": str(fast_root),
    }
    state = adaptive_storage.load_state("Ina", cfg)
    state["decisions"]["index"] = {"tier": "durable", "reason": "test"}
    adaptive_storage.save_state("Ina", state, cfg)
    original_is_mount = Path.is_mount
    monkeypatch.setattr(Path, "is_mount", lambda self: True if self == fast_mount else original_is_mount(self))
    selected = storage_layout.fast_runtime_path("Ina", "index.sqlite", fallback, subdir="index", config=cfg)
    assert selected == fallback
