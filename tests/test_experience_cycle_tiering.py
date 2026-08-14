from pathlib import Path

from experience_cycle_storage import CycleTierPolicy
from experience_engine import ExperienceCycleEngine


def _config(tmp_path, *, max_hot_bytes=1024 * 1024):
    fast_runtime = tmp_path / "nvme" / "AI_Children" / "{child}" / "memory" / "fast_runtime"
    fast_index = tmp_path / "nvme" / "AI_Children" / "{child}" / "memory" / "fast_index"
    (tmp_path / "nvme").mkdir(exist_ok=True)
    return {
        "current_child": "Ina",
        "storage_layout": {
            "fast_runtime_enabled": True,
            "fast_runtime_current_child_only": True,
            "fast_runtime_root": str(fast_runtime),
            "fast_index_root": str(fast_index),
        },
        "experience_cycle_storage": {
            "enabled": True,
            "max_hot_bytes": max_hot_bytes,
            "max_hot_files": 100,
            "min_free_bytes": 1024 * 1024 * 1024,
            "max_index_rows": 1000,
        },
    }


def test_hot_tier_has_hard_quota_and_durable_fallback(tmp_path):
    durable = tmp_path / "hdd" / "cycles"
    policy = CycleTierPolicy("Ina", durable, config=_config(tmp_path), enable_hot=True)

    assert policy.hot_root is not None
    assert policy.choose_write_root(64 * 1024) == policy.hot_root
    assert policy.choose_write_root(2 * 1024 * 1024) == durable


def test_cycle_moves_from_hot_workspace_to_durable_store_and_index_stays_quick(tmp_path):
    base = tmp_path / "hdd" / "AI_Children"
    engine = ExperienceCycleEngine("Ina", base_path=base, enable_hot=True, config=_config(tmp_path))
    cycle = engine.start_cycle("one hot attempt", domain="drawing", payload_references=["canvas-1"])
    engine.complete_attempt(cycle["cycle_id"], attempt_reference="stroke-1", choice="keep")

    assert engine.storage.hot_root is not None
    assert (engine.storage.hot_root / "manifests" / f"{cycle['cycle_id']}.json").exists()
    assert engine.recent_cycles(domain="drawing")[0]["cycle_id"] == cycle["cycle_id"]

    result = engine.drain_hot_tier(max_files=16, max_bytes=1024 * 1024)

    assert result["moved_files"] >= 2
    assert (engine.root / "manifests" / f"{cycle['cycle_id']}.json").exists()
    assert not (engine.storage.hot_root / "manifests" / f"{cycle['cycle_id']}.json").exists()
    indexed = engine.recent_cycles(domain="drawing")[0]
    assert indexed["manifest_path"] == str(engine.root / "manifests" / f"{cycle['cycle_id']}.json")


def test_cycle_index_is_bounded_navigation_not_embedded_domain_history(tmp_path):
    engine = ExperienceCycleEngine("Ina", base_path=tmp_path, enable_hot=False, config={})
    cycle = engine.start_cycle("find this without replay", domain="text", payload_references=[{"id": "draft-9"}])

    row = engine.recent_cycles(limit=1)[0]
    assert row["cycle_id"] == cycle["cycle_id"]
    assert row["payload_references"] == [{"id": "draft-9"}]
    assert set(row) == {
        "cycle_id", "parent_cycle_id", "domain", "stage", "intent", "payload_references",
        "manifest_path", "created_at", "updated_at",
    }
