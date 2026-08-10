import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from continuity_manager import ContinuityManager


def _write_fragment(memory_root, fragment_id, summary, tags):
    path = memory_root / "fragments" / f"{fragment_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "id": fragment_id,
        "summary": summary,
        "tags": tags,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "source": "test",
    }), encoding="utf-8")
    return path


def test_continuity_manager_proposes_minimum_core_map_with_limits(tmp_path):
    manager = ContinuityManager("TestIna", memory_root=tmp_path / "memory", max_fragments=42)

    proposal = manager.propose_minimum_core_map_integration(
        limit_rules={
            "max_total_rss_gb": 96,
            "max_managed_rss_gb": 90,
            "min_available_gb": 8,
            "memory_estimate_high_gb": 12,
        },
        trigger={"largest_module": "meaning_map.py", "largest_module_ram_gb": 140},
    )

    assert proposal["status"] == "proposal_only"
    assert proposal["review_required"] is True
    assert proposal["limit_rules"]["max_total_rss_gb"] == 96
    assert proposal["minimum_core_bounds"]["max_fragments_sampled"] == 42
    assert len(proposal["options"]) == 3
    assert all("memory_profile" in option for option in proposal["options"])


def test_continuity_reports_dimensions_deltas_and_bounded_boot_core(tmp_path):
    memory = tmp_path / "memory"
    identity = _write_fragment(memory, "frag_identity", "I know who I am.", ["identity", "self", "core"])
    _write_fragment(memory, "frag_goal", "I am tending the garden.", ["goal", "active_thread"])
    manager = ContinuityManager("TestIna", memory_root=memory, max_fragments=10)

    baseline = manager.run()
    assert baseline["overall_continuity"] is None
    assert baseline["dimensions"]["identity_preferences"]["state"] == "baseline"

    stable = manager.run()
    assert stable["dimensions"]["identity_preferences"]["score"] == 1.0
    assert stable["dimensions"]["active_goals"]["score"] == 1.0
    assert stable["overall_delta"] is None
    core = manager.load_minimum_boot_core()
    assert core["requires_fragment_scan_on_boot"] is False
    assert core["status"] in {"ready", "partial"}
    assert len(core["anchors"]) <= core["bounds"]["max_anchors"]

    payload = json.loads(identity.read_text(encoding="utf-8"))
    payload["summary"] = "My account of myself has changed."
    identity.write_text(json.dumps(payload), encoding="utf-8")
    changed = manager.run()

    identity_score = changed["dimensions"]["identity_preferences"]
    assert identity_score["score"] == 0.0
    assert identity_score["delta"] == -1.0
    assert changed["overall_delta"] < 0.0


def test_minimum_boot_core_load_does_not_scan_fragments(tmp_path, monkeypatch):
    memory = tmp_path / "memory"
    _write_fragment(memory, "frag_identity", "I know who I am.", ["identity", "self", "core"])
    manager = ContinuityManager("TestIna", memory_root=memory)
    manager.run()
    monkeypatch.setattr(manager, "_fragment_paths", lambda: (_ for _ in ()).throw(AssertionError("scan")))

    core = manager.load_minimum_boot_core()

    assert core["anchors"]
